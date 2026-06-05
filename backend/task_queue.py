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
        """保存任务到磁盘"""
        try:
            filepath = os.path.join(self.storage_dir, f"{task.task_id}.json")
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(task.to_dict(), f, ensure_ascii=False, indent=2)
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
            # 每完成5个或全部完成时保存
            if completed % 5 == 0 or completed == total:
                self._save_task(task)

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
                # 先过滤掉已取消的任务（从 self.tasks 重新读取状态，cancel_task 可能已修改）
                valid_tasks = []
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
                    async with self._lock:
                        self.processing_tasks[task.task_id] = task
                    self._save_task(task)

                    # 更新模型状态
                    model_state = self._model_states.get(task.model)
                    if model_state:
                        model_state.start_task(task.task_id)

                for task in valid_tasks:
                    system_logger.info(f"【任务队列】派发任务: {task.task_id} [模型={task.model}]")
                    # 使用 asyncio.create_task 让不同模型的任务可以并行执行
                    asyncio.create_task(self._execute_task(task))

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
        try:
            # 执行处理函数
            result = await handler(task)

            # 更新成功状态
            task.status = TaskStatus.COMPLETED.value
            task.audio_file = result.get('audio_file')
            task.audio_url = result.get('audio_url')
            task.batch_results = result.get('batch_results')
            task.progress = 100
            task.completed_at = datetime.now().isoformat()
            success = True

            system_logger.info(f"【任务队列】任务完成: {task.task_id} | 文件: {task.audio_file}")

        except Exception as e:
            task.status = TaskStatus.FAILED.value
            task.error_message = str(e)
            task.completed_at = datetime.now().isoformat()
            system_logger.error(f"【任务队列】任务失败 {task.task_id}: {e}")

        finally:
            execution_time = time.time() - start_time
            async with self._lock:
                if task.task_id in self.processing_tasks:
                    del self.processing_tasks[task.task_id]
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
        
        # 如果任务正在执行，无法取消（或者可以实现中断逻辑）
        if task.status == TaskStatus.PROCESSING.value:
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
    
    async def delete_task(self, task_id: str, user_id: str) -> bool:
        """
        删除任务记录
        
        Args:
            task_id: 任务ID
            user_id: 用户ID（用于权限验证）
            
        Returns:
            bool: 是否成功删除
        """
        task = self.tasks.get(task_id)
        if not task or task.user_id != user_id:
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
        
        # 删除存储文件
        filepath = os.path.join(self.storage_dir, f"{task_id}.json")
        if os.path.exists(filepath):
            os.remove(filepath)
        
        # 删除音频文件（如果存在且不是共享文件）
        system_logger.info(f"【任务队列】删除任务调试: task_id={task_id}, audio_file={task.audio_file}, exists={os.path.exists(task.audio_file) if task.audio_file else False}")
        if task.audio_file and os.path.exists(task.audio_file):
            try:
                os.remove(task.audio_file)
                system_logger.info(f"【任务队列】音频文件已删除: {task.audio_file}")
            except Exception as e:
                system_logger.warning(f"【任务队列】删除音频文件失败 {task.audio_file}: {e}")
        elif task.audio_file:
            system_logger.warning(f"【任务队列】音频文件不存在，无法删除: {task.audio_file}")
        
        # 从内存中移除
        del self.tasks[task_id]
        
        system_logger.info(f"【任务队列】任务已删除: {task_id}")
        return True
    
    async def start_cleanup_scheduler(self, interval_hours: int = 24):
        """
        启动定时清理任务
        
        Args:
            interval_hours: 清理间隔（小时）
        """
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
        清理旧任务记录
        
        Args:
            days: 保留天数
        """
        cutoff = datetime.now().timestamp() - (days * 24 * 3600)
        removed_count = 0
        
        for task_id, task in list(self.tasks.items()):
            try:
                task_time = datetime.fromisoformat(task.created_at).timestamp()
                if task_time < cutoff:
                    # 删除文件
                    filepath = os.path.join(self.storage_dir, f"{task_id}.json")
                    if os.path.exists(filepath):
                        os.remove(filepath)
                    # 从内存中移除
                    del self.tasks[task_id]
                    removed_count += 1
            except Exception as e:
                system_logger.warning(f"【任务队列】清理任务失败 {task_id}: {e}")
        
        if removed_count > 0:
            system_logger.info(f"【任务队列】清理了 {removed_count} 个旧任务")


# 全局任务队列实例
task_queue = TaskQueue(max_workers=1)
