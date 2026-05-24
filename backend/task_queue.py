#!/usr/bin/env python3
"""
任务队列模块
支持多用户后台排队、异步执行、任务记录管理
"""

import os
import json
import asyncio
import uuid
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
import threading
import time

from backend.logger_config import system_logger
from backend.config import OUTPUTS_DIR


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
    batch_results: Optional[List[Dict]] = None  # 批量生成结果列表
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return asdict(self)


class TaskQueue:
    """任务队列管理器"""
    
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
        
        # 加载历史任务
        self._load_tasks()
        
        system_logger.info(f"【任务队列】初始化完成，存储目录: {self.storage_dir}")
    
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
    
    async def _worker_loop(self):
        """工作线程主循环"""
        while self._running:
            try:
                # 等待有任务或事件
                if not self._pending_list:
                    self._queue_event.clear()
                    try:
                        await asyncio.wait_for(self._queue_event.wait(), timeout=1.0)
                    except asyncio.TimeoutError:
                        continue
                
                # 获取优先级最高的任务
                async with self._lock:
                    task = self._pop_next_task()
                
                if task is None:
                    continue
                
                # 检查任务是否已被取消
                if task.status == TaskStatus.CANCELLED.value:
                    system_logger.info(f"【任务队列】跳过已取消任务: {task.task_id}")
                    continue
                
                # 执行任务
                await self._execute_task(task)
                
            except Exception as e:
                system_logger.error(f"【任务队列】工作线程错误: {e}")
                await asyncio.sleep(1)
    
    async def _execute_task(self, task: TaskRecord):
        """执行单个任务"""
        handler = self._handlers.get(task.model)
        if not handler:
            task.status = TaskStatus.FAILED.value
            task.error_message = f"未找到模型 {task.model} 的处理器"
            task.completed_at = datetime.now().isoformat()
            self._save_task(task)
            system_logger.error(f"【任务队列】任务执行失败 {task.task_id}: {task.error_message}")
            return
        
        # 更新状态为执行中
        task.status = TaskStatus.PROCESSING.value
        task.started_at = datetime.now().isoformat()
        task.progress = 0
        async with self._lock:
            self.processing_tasks[task.task_id] = task
        self._save_task(task)
        
        system_logger.info(f"【任务队列】开始执行任务: {task.task_id}")
        
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
            
            system_logger.info(f"【任务队列】任务完成: {task.task_id} | 文件: {task.audio_file}")
            
        except Exception as e:
            task.status = TaskStatus.FAILED.value
            task.error_message = str(e)
            task.completed_at = datetime.now().isoformat()
            system_logger.error(f"【任务队列】任务失败 {task.task_id}: {e}")
        
        finally:
            async with self._lock:
                if task.task_id in self.processing_tasks:
                    del self.processing_tasks[task.task_id]
            self._save_task(task)
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
        return {
            "pending_count": self._get_pending_count(),
            "processing_count": len(self.processing_tasks),
            "total_tasks": len(self.tasks),
            "is_running": self._running,
            "max_workers": self.max_workers
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
        if task.audio_file and os.path.exists(task.audio_file):
            try:
                os.remove(task.audio_file)
            except Exception as e:
                system_logger.warning(f"【任务队列】删除音频文件失败 {task.audio_file}: {e}")
        
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
