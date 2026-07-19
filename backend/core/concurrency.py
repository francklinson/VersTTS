#!/usr/bin/env python3
"""
并发控制和资源管理模块
用于处理多用户并发TTS请求

功能：
1. 请求限流（Rate Limiting）
2. GPU资源锁（防止并发OOM）
3. 任务队列管理
4. 用户会话跟踪
"""

import asyncio
import time
import uuid
from typing import Dict, Optional, Set, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import wraps

import torch
from fastapi import HTTPException, Request

from backend.logger_config import system_logger


@dataclass
class UserSession:
    """用户会话信息"""
    session_id: str
    created_at: datetime
    last_active: datetime
    request_count: int = 0
    active_requests: Set[str] = field(default_factory=set)
    rate_limit_tokens: int = 10  # 令牌桶中的令牌数
    last_token_update: datetime = field(default_factory=datetime.now)


@dataclass
class GPULock:
    """GPU资源锁 - 确保同一时间只有一个TTS任务使用GPU"""
    
    def __init__(self):
        self._lock = asyncio.Lock()
        self.current_holder: Optional[str] = None
        self.holder_start_time: Optional[datetime] = None
        self.waiting_count: int = 0
    
    async def acquire(self, task_id: str, timeout: float = 300.0) -> bool:
        """
        获取GPU锁
        
        Args:
            task_id: 任务ID
            timeout: 超时时间（秒）
        
        Returns:
            是否成功获取锁
        """
        self.waiting_count += 1
        system_logger.info(f"【GPU锁】任务 {task_id[:8]}... 等待获取GPU锁，当前等待: {self.waiting_count}")
        
        try:
            await asyncio.wait_for(self._lock.acquire(), timeout=timeout)
            self.current_holder = task_id
            self.holder_start_time = datetime.now()
            self.waiting_count -= 1
            system_logger.info(f"【GPU锁】任务 {task_id[:8]}... 已获取GPU锁")
            return True
        except asyncio.TimeoutError:
            self.waiting_count -= 1
            system_logger.warning(f"【GPU锁】任务 {task_id[:8]}... 获取GPU锁超时")
            return False
    
    def release(self):
        """释放GPU锁"""
        if self._lock.locked():
            duration = None
            if self.holder_start_time:
                duration = (datetime.now() - self.holder_start_time).total_seconds()
            
            system_logger.info(
                f"【GPU锁】任务 {self.current_holder[:8] if self.current_holder else 'unknown'}... "
                f"释放GPU锁" + (f"，占用时长: {duration:.2f}s" if duration else "")
            )
            
            self.current_holder = None
            self.holder_start_time = None
            self._lock.release()
    
    @property
    def is_locked(self) -> bool:
        """检查GPU是否被锁定"""
        return self._lock.locked()
    
    def get_status(self) -> Dict:
        """获取锁状态"""
        return {
            "is_locked": self.is_locked,
            "current_holder": self.current_holder[:8] + "..." if self.current_holder else None,
            "holder_start_time": self.holder_start_time.isoformat() if self.holder_start_time else None,
            "waiting_count": self.waiting_count
        }


class RateLimiter:
    """速率限制器 - 使用令牌桶算法"""
    
    def __init__(self, requests_per_minute: int = 60, burst_size: int = 10):
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size
        self.sessions: Dict[str, UserSession] = {}
        self._cleanup_interval = 60  # 每60秒清理一次过期会话
        self._cleanup_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """启动清理任务"""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        system_logger.info(f"【限流器】已启动，限制: {self.requests_per_minute}/分钟，突发: {self.burst_size}")
    
    async def stop(self):
        """停止清理任务"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            system_logger.info("【限流器】已停止")
    
    async def _cleanup_loop(self):
        """定期清理过期会话"""
        while True:
            try:
                await asyncio.sleep(self._cleanup_interval)
                await self._cleanup_expired_sessions()
            except asyncio.CancelledError:
                break
            except Exception as e:
                system_logger.error(f"【限流器】清理会话时出错: {e}")
    
    async def _cleanup_expired_sessions(self):
        """清理过期会话（30分钟未活动）"""
        now = datetime.now()
        expired = [
            sid for sid, session in self.sessions.items()
            if (now - session.last_active) > timedelta(minutes=30)
        ]
        for sid in expired:
            del self.sessions[sid]
        if expired:
            system_logger.info(f"【限流器】清理 {len(expired)} 个过期会话")
    
    def _update_tokens(self, session: UserSession):
        """更新会话的令牌数"""
        now = datetime.now()
        time_passed = (now - session.last_token_update).total_seconds()
        tokens_to_add = int(time_passed * self.requests_per_minute / 60)
        session.rate_limit_tokens = min(
            session.rate_limit_tokens + tokens_to_add,
            self.burst_size
        )
        session.last_token_update = now
    
    async def acquire(self, session_id: str, user_id: Optional[str] = None) -> bool:
        """
        尝试获取请求许可
        
        Args:
            session_id: 会话ID
            user_id: 用户ID（可选）
        
        Returns:
            是否允许请求
        """
        now = datetime.now()
        
        # 获取或创建会话
        if session_id not in self.sessions:
            self.sessions[session_id] = UserSession(
                session_id=session_id,
                created_at=now,
                last_active=now,
                rate_limit_tokens=self.burst_size,
                last_token_update=now
            )
        
        session = self.sessions[session_id]
        session.last_active = now
        
        # 更新令牌
        self._update_tokens(session)
        
        # 检查是否有可用令牌
        if session.rate_limit_tokens > 0:
            session.rate_limit_tokens -= 1
            session.request_count += 1
            return True
        
        return False
    
    def get_session_info(self, session_id: str) -> Optional[Dict]:
        """获取会话信息"""
        session = self.sessions.get(session_id)
        if not session:
            return None
        
        self._update_tokens(session)
        
        return {
            "session_id": session.session_id,
            "created_at": session.created_at.isoformat(),
            "last_active": session.last_active.isoformat(),
            "request_count": session.request_count,
            "active_requests": len(session.active_requests),
            "rate_limit_tokens": session.rate_limit_tokens,
            "rate_limit_max": self.burst_size
        }


# ========== 全局实例 ==========
# GPU锁 - 确保同一时间只有一个TTS任务使用GPU
gpu_lock = GPULock()

# 速率限制器 - 限制每个会话的请求速率
rate_limiter = RateLimiter(requests_per_minute=60, burst_size=5)


# ========== 装饰器和中间件 ==========
def require_gpu_lock(timeout: float = 300.0):
    """装饰器：要求获取GPU锁才能执行"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            task_id = str(uuid.uuid4())
            
            # 获取GPU锁
            if not await gpu_lock.acquire(task_id, timeout=timeout):
                raise HTTPException(status_code=503, detail="GPU资源繁忙，请稍后重试")
            
            try:
                return await func(*args, **kwargs)
            finally:
                gpu_lock.release()
        
        return wrapper
    return decorator


async def concurrency_middleware(request: Request, call_next):
    """
    并发控制中间件
    
    为每个请求：
    1. 检查速率限制
    2. 记录会话信息
    3. 跟踪活跃请求
    """
    # 获取或创建会话ID
    session_id = request.headers.get("X-Session-ID")
    if not session_id:
        session_id = request.cookies.get("session_id")
    if not session_id:
        session_id = str(uuid.uuid4())
    
    # 检查速率限制
    if not await rate_limiter.acquire(session_id):
        raise HTTPException(
            status_code=429,
            detail="请求过于频繁，请稍后再试",
            headers={"Retry-After": "60"}
        )
    
    # 记录请求开始
    session = rate_limiter.sessions.get(session_id)
    if session:
        request_id = str(uuid.uuid4())
        session.active_requests.add(request_id)
    
    try:
        response = await call_next(request)
        
        # 设置会话ID到响应
        if session_id:
            response.headers["X-Session-ID"] = session_id
        
        return response
    
    finally:
        # 清理活跃请求记录
        if session and request_id in session.active_requests:
            session.active_requests.remove(request_id)


async def initialize_concurrency():
    """初始化并发控制系统"""
    await rate_limiter.start()
    system_logger.info("【并发控制】系统已初始化")


async def shutdown_concurrency():
    """关闭并发控制系统"""
    await rate_limiter.stop()
    system_logger.info("【并发控制】系统已关闭")
