#!/usr/bin/env python3
"""
任务队列模块
支持多用户后台排队、异步执行、任务记录管理
"""

import os
import json
import asyncio
import uuid
import heapq
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
import threading
import time

from backend.logger_config import system_logger
from backend.config import OUTPUTS_DIR


class ThreadSafeProgressUpdater:
    """线程安全的进度更新器，用于在线程池中更新任务进度"""

    def __init__(self):
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._callbacks: List[tuple] = []
        self._lock = threading.Lock()

    def set_loop(self, loop: asyncio.AbstractEventLoop):
        """设置事件循环"""
        self._loop = loop
        # 处理积压的回调
        with self._lock:
            pending = self._callbacks[:]
            self._callbacks.clear()
        for callback, args, kwargs in pending:
            self._call_in_loop(callback, args, kwargs)

    def _call_in_loop(self, callback, args, kwargs):
        """在事件循环中调用回调"""
        if self._loop and self._loop.is_running():
            try:
                asyncio.run_coroutine_threadsafe(
                    self._async_callback(callback, args, kwargs),
                    self._loop
                )
            except Exception as e:
                system_logger.warning(f"【进度更新】调度回调失败: {e}")
        else:
            with self._lock:
                self._callbacks.append((callback, args, kwargs))

    async def _async_callback(self, callback, args, kwargs):
        """异步执行回调"""
        try:
            callback(*args, **kwargs)
        except Exception as e:
            system_logger.warning(f"【进度更新】执行回调失败: {e}")

    def update_progress(self, callback, *args, **kwargs):
        """线程安全地调用进度更新回调"""
        self._call_in_loop(callback, args, kwargs)


# 全局线程安全进度更新器
progress_updater = ThreadSafeProgressUpdater()


class TaskStatus(Enum):
    """任务状态"""
    PENDING = "pending"         # 等待中
    QUEUED = "queued"           # 已入队
    PROCESSING = "processing"   # 执行中
    COMPLETED = "completed"     # 已完成
    FAILED = "failed"           # 失败
    CANCELLED = "cancelled"     # 已取消
    RETRIED = "retried"         # 已被重试（指向新任务）


@dataclass
class TaskRecord:
    """任务记录"""
    task_id: str
    user_id: str                    # 用户标识（会话ID或用户ID）
    model: str                      # TTS模型
    mode: str                       # 生成模式
    text: str                       # 合成文本
    params: Dict[str, Any]          # 其他参数
    status: str = "pending"
    priority: int = 0               # 优先级（数字越小优先级越高）
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    audio_file: Optional[str] = None
    audio_url: Optional[str] = None
    error_message: Optional[str] = None
    error_code: Optional[str] = None  # 失败类型码（如 AUDIO_VERIFY_FAILED），便于前端区分可重试失败
    progress: int = 0               # 进度百分比
    batch_total: int = 0            # 批量生成总数
    batch_completed: int = 0        # 批量生成已完成数
    batch_results: Optional[List[Dict]] = None  # 批量生成结果列表
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return asdict(self)


class ModelState:
    """模型状态跟踪"""
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.is_busy = False
        self.current_task_id: Optional[str] = None
        self.task_start_time: Optional[str] = None
        self.completed_count = 0
        self.failed_count = 0
        self.total_execution_time = 0.0  # 秒

    def start_task(self, task_id: str):
        """开始执行任务"""
        self.is_busy = True
        self.current_task_id = task_id
        self.task_start_time = datetime.now().isoformat()

    def end_task(self, success: bool = True, execution_time: float = 0.0):
        """结束任务"""
        self.is_busy = False
        self.current_task_id = None
        self.task_start_time = None
        if success:
            self.completed_count += 1
        else:
            self.failed_count += 1
        self.total_execution_time += execution_time

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "model_name": self.model_name,
            "is_busy": self.is_busy,
            "current_task_id": self.current_task_id,
            "task_start_time": self.task_start_time,
            "completed_count": self.completed_count,
            "failed_count": self.failed_count,
            "total_execution_time": round(self.total_execution_time, 2)
        }


class TaskQueue:
    """任务队列管理器 - 支持模型级并发控制"""

    # 模型并发配置：每个模型允许的最大并发数
    DEFAULT_MODEL_CONCURRENCY = {
        "voxcpm": 1,      # 本地GPU模型，串行执行
        "qwen3tts": 1,    # 本地GPU模型，串行执行
        "omnivoice": 2,   # 独立HTTP服务，可有限并行
        "cosyvoice": 2,   # 独立HTTP服务，可有限并行
        "pilottts": 1,    # 本地GPU模型，串行执行
        "gptsovits": 1,   # 独立GPU服务，串行执行
    }

    def __init__(self, max_workers: int = 1, storage_dir: str = None):
        """
        初始化任务队列

        Args:
            max_workers: 最大并发工作线程数（TTS模型通常只能串行执行）
            storage_dir: 任务记录存储目录
        """
        self.max_workers = max_workers
        self.storage_dir = storage_dir or os.path.join(OUTPUTS_DIR, "task_records")
        os.makedirs(self.storage_dir, exist_ok=True)

        # 内存中的任务存储
        self.tasks: Dict[str, TaskRecord] = {}
        # 使用列表+锁实现真正的优先级队列（asyncio.Queue不支持优先级排序）
        self._pending_list: List[tuple] = []
        self._queue_event = asyncio.Event()
        self.processing_tasks: Dict[str, TaskRecord] = {}

        # 存储运行中的 asyncio.Task 句柄，用于取消执行中的任务
        self._active_tasks: Dict[str, asyncio.Task] = {}

        # 执行控制
        self._running = False
        self._worker_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None

        # 任务处理器映射
        self._handlers: Dict[str, Callable] = {}

        # 模型状态跟踪
        self._model_states: Dict[str, ModelState] = {}
        self._model_concurrency: Dict[str, int] = {}
        self._init_model_config()

        # 加载历史任务
        self._load_tasks()

        system_logger.info(f"【任务队列】初始化完成，存储目录: {self.storage_dir}")

    def _init_model_config(self):
        """初始化模型并发配置"""
        # 从环境变量读取配置，或使用默认值
        for model, default_concurrency in self.DEFAULT_MODEL_CONCURRENCY.items():
            env_key = f"MAX_CONCURRENT_{model.upper()}"
            concurrency = int(os.environ.get(env_key, default_concurrency))
            self._model_concurrency[model] = concurrency
            self._model_states[model] = ModelState(model)
            system_logger.info(f"【任务队列】模型 {model} 并发数: {concurrency}")

    def get_model_state(self, model: str) -> Optional[ModelState]:
        """获取模型状态"""
        return self._model_states.get(model)

    def get_all_model_states(self) -> Dict[str, Dict]:
        """获取所有模型状态"""
        return {name: state.to_dict() for name, state in self._model_states.items()}

    def _get_model_running_count(self, model: str) -> int:
        """获取指定模型正在运行的任务数"""
        count = 0
        for task in self.processing_tasks.values():
            if task.model == model:
                count += 1
        return count
    
    def _get_pending_count(self) -> int:
        """获取等待中的任务数量"""
        return len(self._pending_list)
    
    def _pop_next_task(self) -> Optional[TaskRecord]:
        """弹出优先级最高的任务"""
        if not self._pending_list:
            return None
        # 按优先级排序（数字越小优先级越高），相同优先级按创建时间先后
        self._pending_list.sort(key=lambda x: (x[0], x[1]))
        _, _, task = self._pending_list.pop(0)
        return task
    
    def _add_pending_task(self, priority: int, created_at: str, task: TaskRecord):
        """添加任务到等待列表"""
        self._pending_list.append((priority, created_at, task))
        # 重新排序确保优先级正确
        self._pending_list.sort(key=lambda x: (x[0], x[1]))
    
    def _load_tasks(self):
        """从磁盘加载历史任务"""
        try:
            for filename in os.listdir(self.storage_dir):
                if filename.endswith('.json'):
                    filepath = os.path.join(self.storage_dir, filename)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            task = TaskRecord(**data)
                            self.tasks[task.task_id] = task
                    except Exception as e:
                        system_logger.warning(f"【任务队列】加载任务失败 {filename}: {e}")
            
            system_logger.info(f"【任务队列】加载了 {len(self.tasks)} 个历史任务")
        except Exception as e:
            system_logger.error(f"【任务队列】加载任务出错: {e}")
    
    def _save_task(self, task: TaskRecord):
        """保存任务到磁盘（原子写入：先写临时文件，再 rename）"""
        try:
            filepath = os.path.join(self.storage_dir, f"{task.task_id}.json")
            tmp_path = filepath + ".tmp"
            with open(tmp_path, 'w', encoding='utf-8') as f:
                json.dump(task.to_dict(), f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, filepath)  # 原子重命名，避免进程崩溃导致文件损坏
        except Exception as e:
            system_logger.error(f"【任务队列】保存任务失败 {task.task_id}: {e}")
    
    def register_handler(self, model: str, handler: Callable):
        """
        注册任务处理器
        
        Args:
            model: 模型名称
            handler: 处理函数，接收 TaskRecord 参数，返回音频文件路径
        """
        self._handlers[model] = handler
        # 自动注册模型状态和并发配置
        if model not in self._model_states:
            self._model_states[model] = ModelState(model)
        if model not in self._model_concurrency:
            default = self.DEFAULT_MODEL_CONCURRENCY.get(model, 1)
            env_key = f"MAX_CONCURRENT_{model.upper()}"
            concurrency = int(os.environ.get(env_key, default))
            self._model_concurrency[model] = concurrency
        system_logger.info(f"【任务队列】注册处理器: {model}")
    
    async def submit_task(
        self,
        user_id: str,
        model: str,
        mode: str,
        text: str,
        params: Dict[str, Any] = None,
        priority: int = 0
    ) -> TaskRecord:
        """
        提交任务到队列
        
        Args:
            user_id: 用户标识
            model: TTS模型名称
            mode: 生成模式
            text: 合成文本
            params: 其他参数
            priority: 优先级（数字越小优先级越高，默认0）
            
        Returns:
            TaskRecord: 任务记录
        """
        task_id = f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        task = TaskRecord(
            task_id=task_id,
            user_id=user_id,
            model=model,
            mode=mode,
            text=text,
            params=params or {},
            status=TaskStatus.QUEUED.value,
            priority=priority
        )
        
        async with self._lock:
            self.tasks[task_id] = task
            self._save_task(task)
        
        # 添加到优先级队列
        self._add_pending_task(priority, task.created_at, task)
        self._queue_event.set()
        
        system_logger.info(f"【任务队列】任务已提交: {task_id} | 用户: {user_id} | 模型: {model} | 优先级: {priority}")
        
        # 确保工作线程在运行
        if not self._running:
            await self.start()
        
        return task
    
    async def start(self):
        """启动任务队列处理器"""
        if self._running:
            return

        self._running = True
        # 设置进度更新器的事件循环
        try:
            loop = asyncio.get_running_loop()
            progress_updater.set_loop(loop)
        except RuntimeError:
            pass

        self._worker_task = asyncio.create_task(self._worker_loop())
        # 启动定时清理
        await self.start_cleanup_scheduler(interval_hours=24)
        system_logger.info("【任务队列】处理器已启动")
    
    async def stop(self):
        """停止任务队列处理器"""
        self._running = False
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        system_logger.info("【任务队列】处理器已停止")
    
    def update_task_progress(self, task_id: str, progress: int):
        """
        更新任务进度
        
        Args:
            task_id: 任务ID
            progress: 进度百分比 (0-100)
        """
        task = self.tasks.get(task_id)
        if task and task.status == TaskStatus.PROCESSING.value:
            task.progress = max(0, min(100, progress))
            # 进度更新不频繁写入磁盘，仅在关键节点保存
            if progress in [25, 50, 75, 100]:
                self._save_task(task)

    def update_batch_progress(self, task_id: str, completed: int, total: int):
        """
        更新批量生成进度

        Args:
            task_id: 任务ID
            completed: 已完成数量
            total: 总数量
        """
        task = self.tasks.get(task_id)
        if task and task.status == TaskStatus.PROCESSING.value:
            task.batch_completed = completed
            task.batch_total = total
            if total > 0:
                task.progress = int((completed / total) * 100)
            system_logger.info(f"【进度更新】{task_id}: {completed}/{total} = {task.progress}%")
            # 每完成5个或全部完成时保存
            if completed % 5 == 0 or completed == total:
                self._save_task(task)
        else:
            system_logger.warning(f"【进度更新】跳过 {task_id}: task={'存在' if task else '不存在'} status={task.status if task else 'N/A'}")

    async def _worker_loop(self):
        """工作线程主循环 - 支持模型级并发"""
        system_logger.info("【任务队列】工作线程已启动（模型级并发模式）")
        while self._running:
            try:
                # 等待有任务或事件
                if not self._pending_list:
                    self._queue_event.clear()
                    try:
                        await asyncio.wait_for(self._queue_event.wait(), timeout=1.0)
                    except asyncio.TimeoutError:
                        continue

                # 获取当前各模型运行中的任务数
                running_models = {}
                for tid, t in list(self.processing_tasks.items()):
                    running_models[t.model] = running_models.get(t.model, 0) + 1

                # 调试日志
                if self._pending_list:
                    pending_models = [t.model for (_, _, t) in self._pending_list]
                    system_logger.info(f"【任务队列】调试: running_models={running_models}, pending={pending_models}")

                # 一次性收集所有可执行的任务（检查每个模型的并发限制）
                tasks_to_dispatch = []
                async with self._lock:
                    # 遍历 pending_list 找到所有可执行的任务
                    remaining = []
                    # 跟踪每个模型在本次派发中已分配的任务数
                    model_dispatch_count = {}

                    while self._pending_list:
                        item = heapq.heappop(self._pending_list)
                        _, _, candidate = item
                        model = candidate.model

                        # 获取该模型的并发限制
                        max_concurrent = self._model_concurrency.get(model, 1)
                        currently_running = running_models.get(model, 0)
                        already_dispatched = model_dispatch_count.get(model, 0)

                        # 检查是否还可以派发该模型的任务
                        if currently_running + already_dispatched < max_concurrent:
                            tasks_to_dispatch.append(candidate)
                            model_dispatch_count[model] = already_dispatched + 1
                            system_logger.info(f"【任务队列】调试: 选中任务 {candidate.task_id} [模型={model}, 并发={currently_running + already_dispatched + 1}/{max_concurrent}]")
                        else:
                            remaining.append(item)
                            system_logger.info(f"【任务队列】调试: 跳过任务 {candidate.task_id} [模型={model}] - 已达并发限制({max_concurrent})")

                    # 把剩下的放回去
                    for item in remaining:
                        heapq.heappush(self._pending_list, item)

                if not tasks_to_dispatch:
                    # 所有任务都被同模型任务阻塞，等待一下
                    system_logger.info(f"【任务队列】所有模型繁忙中，等待空闲: {list(running_models.keys())}")
                    await asyncio.sleep(0.3)
                    continue

                # 批量派发所有可并行执行的任务
                # 过滤掉已取消的任务（必须在锁内读取，防止与 cancel_task 产生 TOCTOU 竞态）
                valid_tasks = []
                async with self._lock:
                    for task in tasks_to_dispatch:
                        current = self.tasks.get(task.task_id)
                        if current and current.status == TaskStatus.CANCELLED.value:
                            system_logger.info(f"【任务队列】跳过已取消任务: {task.task_id}")
                            continue
                        valid_tasks.append(task)

                    # 将有效任务标记为执行中，防止同模型任务被重复派发
                    for task in valid_tasks:
                        task.status = TaskStatus.PROCESSING.value
                        task.started_at = datetime.now().isoformat()
                        self.processing_tasks[task.task_id] = task

                # 持久化在锁外执行（避免IO阻塞锁）
                for task in valid_tasks:
                    self._save_task(task)

                    # 更新模型状态
                    model_state = self._model_states.get(task.model)
                    if model_state:
                        model_state.start_task(task.task_id)

                for task in valid_tasks:
                    system_logger.info(f"【任务队列】派发任务: {task.task_id} [模型={task.model}]")
                    # 使用 asyncio.create_task 让不同模型的任务可以并行执行
                    async_task = asyncio.create_task(self._execute_task(task))
                    # 存储 asyncio.Task 句柄，用于取消运行中的任务
                    async with self._lock:
                        self._active_tasks[task.task_id] = async_task

            except Exception as e:
                system_logger.error(f"【任务队列】工作线程错误: {e}")
                import traceback
                traceback.print_exc()
                await asyncio.sleep(1)

    async def _execute_task(self, task: TaskRecord):
        """执行单个任务"""
        import time
        start_time = time.time()

        # 检查任务是否已被取消（在dispatch和execute之间可能被取消）
        current = self.tasks.get(task.task_id)
        if current and current.status == TaskStatus.CANCELLED.value:
            system_logger.info(f"【任务队列】任务已被取消，跳过执行: {task.task_id}")
            async with self._lock:
                self.processing_tasks.pop(task.task_id, None)
            # 恢复模型状态
            model_state = self._model_states.get(task.model)
            if model_state:
                model_state.end_task(success=False)
            return

        handler = self._handlers.get(task.model)
        if not handler:
            task.status = TaskStatus.FAILED.value
            task.error_message = f"未找到模型 {task.model} 的处理器"
            task.completed_at = datetime.now().isoformat()
            self._save_task(task)
            # 更新模型状态
            model_state = self._model_states.get(task.model)
            if model_state:
                model_state.end_task(success=False)
            system_logger.error(f"【任务队列】任务执行失败 {task.task_id}: {task.error_message}")
            return

        # 更新状态为执行中
        task.status = TaskStatus.PROCESSING.value
        task.started_at = datetime.now().isoformat()
        task.progress = 0
        async with self._lock:
            self.processing_tasks[task.task_id] = task
        self._save_task(task)

        system_logger.info(f"【任务队列】开始执行任务: {task.task_id} [模型={task.model}, 模式={task.mode}]")

        # 记录当前并发执行的任务数
        running_count = len(self.processing_tasks)
        running_models = set()
        for tid, t in list(self.processing_tasks.items()):
            running_models.add(t.model)
        if running_count > 1:
            system_logger.info(f"【任务队列】当前并发任务数: {running_count}, 运行的模型: {running_models}")

        success = False
        # 动态计算超时：进程内推理模型（含模型加载、串行批量）按 batch_count 放大；
        # 子服务模型维持固定 180s 上限。
        timeout = self._compute_task_timeout(task)
        try:
            # 执行处理函数，超时防止单个任务永久阻塞队列（批量任务按条数放大）
            result = await asyncio.wait_for(handler(task), timeout=timeout)

            # 更新成功状态
            task.status = TaskStatus.COMPLETED.value
            task.audio_file = result.get('audio_file')
            task.audio_url = result.get('audio_url')
            task.batch_results = result.get('batch_results')
            task.progress = 100
            task.completed_at = datetime.now().isoformat()
            success = True

            system_logger.info(f"【任务队列】任务完成: {task.task_id} | 文件: {task.audio_file}")

        except asyncio.TimeoutError:
            task.status = TaskStatus.FAILED.value
            batch_count = (task.params or {}).get('batch_count', 1)
            elapsed = time.time() - start_time
            task.error_message = f"任务执行超时（{timeout}秒）"
            task.completed_at = datetime.now().isoformat()
            # 超时诊断：打印已执行时长、进度、启动时间，定位时间黑洞
            batch_total = getattr(task, 'batch_total', 0) or 0
            batch_completed = getattr(task, 'batch_completed', 0) or 0
            system_logger.error(
                f"【任务队列】任务超时 {task.task_id} | 模型={task.model} 模式={task.mode} "
                f"批量={batch_count} 超时={timeout}s 已执行={elapsed:.1f}s "
                f"进度={batch_completed}/{batch_total} started_at={task.started_at}"
            )
        except asyncio.CancelledError:
            task.status = TaskStatus.CANCELLED.value
            task.error_message = "任务已被取消"
            task.completed_at = datetime.now().isoformat()
            elapsed = time.time() - start_time
            system_logger.info(
                f"【任务队列】任务被取消 {task.task_id} | 已执行={elapsed:.1f}s"
            )
        except Exception as e:
            task.status = TaskStatus.FAILED.value
            # 识别音频内容校验失败：写入 error_code 供前端区分可重试失败，
            # 并用面向用户的 user_message 替代技术异常文本。
            error_code = getattr(e, "error_code", None)
            if error_code == "AUDIO_VERIFY_FAILED":
                task.error_code = error_code
                task.error_message = getattr(e, "user_message", str(e))
            else:
                task.error_message = str(e)
            task.completed_at = datetime.now().isoformat()
            elapsed = time.time() - start_time
            batch_total = getattr(task, 'batch_total', 0) or 0
            batch_completed = getattr(task, 'batch_completed', 0) or 0
            # 失败诊断：异常类型 + 堆栈 + 已执行时长 + 进度
            import traceback
            system_logger.error(
                f"【任务队列】任务失败 {task.task_id} | 模型={task.model} 模式={task.mode} "
                f"异常={type(e).__name__}: {e} 已执行={elapsed:.1f}s "
                f"进度={batch_completed}/{batch_total}"
            )
            system_logger.error(f"【任务队列】失败堆栈:\n{traceback.format_exc()}")

        finally:
            execution_time = time.time() - start_time
            async with self._lock:
                if task.task_id in self.processing_tasks:
                    del self.processing_tasks[task.task_id]
                # 清理 asyncio.Task 句柄
                self._active_tasks.pop(task.task_id, None)
            self._save_task(task)
            # 更新模型状态
            model_state = self._model_states.get(task.model)
            if model_state:
                model_state.end_task(success=success, execution_time=execution_time)
            # 任务执行完后清理显存
            try:
                import torch
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except Exception as e:
                system_logger.warning(f"【任务队列】显存清理出错: {e}")
    
    # 批量任务超时配置：所有模型均按 batch_count 动态放大。
    # 公式: timeout = BASE + batch_count * PER_UNIT[model]
    # - BASE: 基础预算，覆盖模型冷启动加载 / 子服务唤醒 / 单条固定开销
    # - PER_UNIT: 每条批量生成的预算，按模型批量并发度折算
    #   * 进程内串行（qwen3tts/voxcpm）与子服务串行（pilottts/gptsovits）并发=1，按 30s/条
    #   * 子服务并发（omnivoice/cosyvoice）并发=2，按 15s/条（30s ÷ 并发度）
    # 单条任务（batch_count=1）退化为 BASE + 单条预算，足够覆盖推理本身。
    _TASK_TIMEOUT_BASE = 120
    _TASK_TIMEOUT_PER_UNIT = {
        "qwen3tts": 30,   # 进程内串行
        "voxcpm": 30,     # 进程内串行
        "pilottts": 30,   # 子服务串行（GPU，信号量并发=1）
        "gptsovits": 30,  # 子服务串行（GPU，信号量并发=1）
        "omnivoice": 15,  # 子服务并发=2 → 30/2
        "cosyvoice": 15,  # 子服务并发=2 → 30/2
    }
    _TASK_TIMEOUT_DEFAULT_PER_UNIT = 30  # 未知模型的兜底

    def _compute_task_timeout(self, task: TaskRecord) -> float:
        """根据模型与批量条数计算任务超时（秒）。

        所有模型统一公式 base + batch_count * per_unit，per_unit 按该模型
        批量并发度折算（并发越高单条均摊越少）。固定超时会在批量较大时
        误判失败，导致已生成的结果整批作废。
        """
        batch_count = int((task.params or {}).get('batch_count', 1) or 1)
        if batch_count < 1:
            batch_count = 1
        per_unit = self._TASK_TIMEOUT_PER_UNIT.get(
            task.model, self._TASK_TIMEOUT_DEFAULT_PER_UNIT)
        return float(self._TASK_TIMEOUT_BASE + batch_count * per_unit)

    def get_task(self, task_id: str) -> Optional[TaskRecord]:
        """获取任务记录"""
        return self.tasks.get(task_id)
    
    def get_user_tasks(
        self,
        user_id: str,
        status: Optional[str] = None,
        limit: int = 50
    ) -> List[TaskRecord]:
        """
        获取用户的任务列表
        
        Args:
            user_id: 用户标识
            status: 状态筛选（可选）
            limit: 返回数量限制
            
        Returns:
            List[TaskRecord]: 任务记录列表
        """
        user_tasks = [
            task for task in self.tasks.values()
            if task.user_id == user_id
        ]
        
        if status:
            user_tasks = [t for t in user_tasks if t.status == status]
        
        # 按创建时间倒序排列
        user_tasks.sort(key=lambda x: x.created_at, reverse=True)

        return user_tasks[:limit]

    def get_all_tasks(
        self,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
        model: Optional[str] = None
    ) -> List[TaskRecord]:
        """
        获取所有用户的任务列表（多用户共享视图）。

        用于任务列表页展示全局任务看板：所有用户都能看到彼此
        执行中/等待中/已完成的任务。取消/删除等操作仍由调用方
        按 user_id 校验权限（仅本人可操作自己的任务）。

        Args:
            status: 状态筛选（可选）
            limit: 返回数量限制
            offset: 偏移量（分页，从第 offset 条开始取 limit 条）
            model: 算法/模型筛选（可选）
        """
        all_tasks = list(self.tasks.values())

        if status:
            all_tasks = [t for t in all_tasks if t.status == status]
        if model:
            all_tasks = [t for t in all_tasks if t.model == model]

        # 按创建时间倒序排列
        all_tasks.sort(key=lambda x: x.created_at, reverse=True)

        return all_tasks[offset:offset + limit]

    def count_all_tasks(self, status: Optional[str] = None, model: Optional[str] = None) -> int:
        """返回符合筛选条件的任务总数（分页前），供前端翻页计算。"""
        tasks = self.tasks.values()
        if model:
            tasks = [t for t in tasks if t.model == model]
        if not status:
            return len(list(tasks)) if model else len(self.tasks)
        return sum(1 for t in tasks if t.status == status)

    def status_counts_all(self, model: Optional[str] = None) -> Dict[str, int]:
        """返回各状态的全量任务计数（忽略分页与状态筛选，仅受 model 筛选影响）。
        供任务页统计栏显示稳定的各状态总数。"""
        counts = {"queued": 0, "processing": 0, "completed": 0, "failed": 0, "cancelled": 0, "retried": 0}
        for t in self.tasks.values():
            if model and t.model != model:
                continue
            if t.status in counts:
                counts[t.status] += 1
        return counts

    def get_queue_status(self) -> Dict:
        """获取队列状态"""
        # 统计各模型排队中/等待中的任务数（不包含已完成/失败/取消）
        model_counts = {}
        for task in self.tasks.values():
            if task.status in (TaskStatus.PENDING.value, TaskStatus.QUEUED.value):
                model_counts[task.model] = model_counts.get(task.model, 0) + 1

        # 统计正在处理中的模型
        processing_models = {}
        for task in self.processing_tasks.values():
            processing_models[task.model] = processing_models.get(task.model, 0) + 1

        # 获取模型状态
        model_states = self.get_all_model_states()

        return {
            "pending_count": self._get_pending_count(),
            "processing_count": len(self.processing_tasks),
            "total_tasks": len(self.tasks),
            "is_running": self._running,
            "max_workers": self.max_workers,
            "model_counts": model_counts,
            "processing_models": processing_models,
            "concurrent_models": list(processing_models.keys()),
            "model_states": model_states,
            "model_concurrency": self._model_concurrency
        }
    
    async def cancel_task(self, task_id: str, user_id: str) -> bool:
        """
        取消任务
        
        Args:
            task_id: 任务ID
            user_id: 用户ID（用于权限验证）
            
        Returns:
            bool: 是否成功取消
        """
        task = self.tasks.get(task_id)
        if not task or task.user_id != user_id:
            return False
        
        if task.status in [TaskStatus.COMPLETED.value, TaskStatus.FAILED.value, TaskStatus.CANCELLED.value]:
            return False

        # 如果任务正在执行，发送 CancelledError 到对应的 asyncio.Task
        if task.status == TaskStatus.PROCESSING.value:
            async with self._lock:
                async_task = self._active_tasks.get(task_id)
            if async_task and not async_task.done():
                async_task.cancel()
                system_logger.info(f"【任务队列】已发送取消信号到运行中任务: {task_id}")
                return True
            return False
        
        # 从等待列表中移除
        async with self._lock:
            self._pending_list = [
                (p, c, t) for p, c, t in self._pending_list
                if t.task_id != task_id
            ]
        
        task.status = TaskStatus.CANCELLED.value
        task.completed_at = datetime.now().isoformat()
        self._save_task(task)
        
        system_logger.info(f"【任务队列】任务已取消: {task_id}")
        return True
    
    def _delete_task_files(self, task: "TaskRecord"):
        """删除任务关联的磁盘文件：任务记录 json + 音频文件（含批量结果的各段音频）。

        供 delete_task（手动删单条）与 cleanup_old_tasks（手动批量清理接口）
        共用，保证「删任务记录必连带删音频」的联动一致性，避免出现
        「记录没了、音频留着」或「记录留着、音频没了」的撕裂。
        """
        task_id = task.task_id

        # 1. 删除任务记录 json
        filepath = os.path.join(self.storage_dir, f"{task_id}.json")
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
            except Exception as e:
                system_logger.warning(f"【任务队列】删除任务记录失败 {filepath}: {e}")

        # 2. 收集要删除的音频文件
        files_to_delete = []
        if task.audio_file:
            files_to_delete.append(task.audio_file)
        # 批量任务：级联删除 batch_results 中的单个音频文件
        if getattr(task, 'batch_results', None):
            for item in task.batch_results:
                if isinstance(item, dict):
                    audio = item.get('audio_file') or item.get('audio_url')
                    if audio:
                        files_to_delete.append(audio)

        # 3. 逐个删除音频文件
        for fpath in files_to_delete:
            try:
                if os.path.exists(fpath):
                    os.remove(fpath)
                    system_logger.info(f"【任务队列】音频文件已删除: {fpath}")
                else:
                    system_logger.warning(f"【任务队列】音频文件不存在，跳过: {fpath}")
            except Exception as e:
                system_logger.warning(f"【任务队列】删除音频文件失败 {fpath}: {e}")

    async def delete_task(self, task_id: str, user_id: str) -> bool:
        """
        删除任务记录

        权限规则：
        - 失败类终态（failed/cancelled/retried）：任何人可删（避免跨用户残留任务占位）
        - 其他状态（queued/processing/completed）：仅本人可删

        Args:
            task_id: 任务ID
            user_id: 用户ID（用于权限验证）

        Returns:
            bool: 是否成功删除
        """
        task = self.tasks.get(task_id)
        if not task:
            return False

        # 失败类终态：放开 user_id 校验，任何人可删
        public_deletable = task.status in (
            TaskStatus.FAILED.value,
            TaskStatus.CANCELLED.value,
            TaskStatus.RETRIED.value,
        )
        # 其他状态：仅本人可删
        if not public_deletable and task.user_id != user_id:
            return False
        
        # 如果任务还在等待中，先从等待列表移除
        if task.status == TaskStatus.QUEUED.value:
            async with self._lock:
                self._pending_list = [
                    (p, c, t) for p, c, t in self._pending_list
                    if t.task_id != task_id
                ]
        
        # 如果任务正在执行，不能删除
        if task.status == TaskStatus.PROCESSING.value:
            return False

        # 删除任务记录 json + 级联删除音频文件（联动）
        self._delete_task_files(task)

        # 从内存中移除
        del self.tasks[task_id]

        system_logger.info(f"【任务队列】任务已删除: {task_id}")
        return True
    
    async def start_cleanup_scheduler(self, interval_hours: int = 24):
        """
        启动定时清理任务（默认关闭）。

        默认不自动清理任务记录/音频文件，避免「记录在、文件没了」的撕裂，
        也避免误删用户想保留的历史。如需恢复每 24h 自动清理 7 天前任务记录，
        将 ENABLE_TASK_CLEANUP 改为 True（注意：自动清理只删任务记录，
        不联动删音频，可能产生孤儿音频文件）。

        Args:
            interval_hours: 清理间隔（小时）
        """
        ENABLE_TASK_CLEANUP = False
        if not ENABLE_TASK_CLEANUP:
            system_logger.info("【任务队列】定时清理已禁用（保留所有任务记录与音频文件）")
            return

        async def _cleanup_loop():
            while self._running:
                try:
                    await asyncio.sleep(interval_hours * 3600)
                    if self._running:
                        self.cleanup_old_tasks(days=7)
                        system_logger.info("【任务队列】定时清理完成")
                except Exception as e:
                    system_logger.error(f"【任务队列】定时清理出错: {e}")

        self._cleanup_task = asyncio.create_task(_cleanup_loop())
        system_logger.info(f"【任务队列】定时清理已启动，间隔: {interval_hours}小时")
    
    def cleanup_old_tasks(self, days: int = 7):
        """
        清理旧任务记录（手动触发：管理接口 /tasks/cleanup 或并发管理接口）。

        删除指定天数前的任务记录，并联动删除其音频文件（含批量结果各段），
        保证「删记录必删音频」，避免撕裂。默认定时调度已关闭（见
        start_cleanup_scheduler 的 ENABLE_TASK_CLEANUP），此函数不再被自动调用。

        Args:
            days: 保留天数
        """
        cutoff = datetime.now().timestamp() - (days * 24 * 3600)
        removed_count = 0

        for task_id, task in list(self.tasks.items()):
            try:
                task_time = datetime.fromisoformat(task.created_at).timestamp()
                if task_time < cutoff:
                    # 联动删除任务记录 json + 音频文件
                    self._delete_task_files(task)
                    # 从内存中移除
                    del self.tasks[task_id]
                    removed_count += 1
            except Exception as e:
                system_logger.warning(f"【任务队列】清理任务失败 {task_id}: {e}")

        if removed_count > 0:
            system_logger.info(f"【任务队列】清理了 {removed_count} 个旧任务（含音频）")


# 全局任务队列实例
task_queue = TaskQueue(max_workers=1)
