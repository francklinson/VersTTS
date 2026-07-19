#!/usr/bin/env python3
"""
并发管理和队列状态路由
提供API接口查看和管理并发状态
"""

from typing import Optional, Dict, Any
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException, Header

from backend.logger_config import system_logger
from backend.core import (
    gpu_lock,
    rate_limiter,
    get_gpu_memory_info
)
from backend.task_queue import task_queue

router = APIRouter()


class ConcurrencyStatusResponse(BaseModel):
    """并发状态响应"""
    gpu_lock: Dict[str, Any]
    task_queue: Dict[str, Any]
    rate_limiter: Dict[str, Any]
    gpu_memory: Dict[str, float]


class SessionInfoResponse(BaseModel):
    """会话信息响应"""
    session_id: str
    created_at: str
    last_active: str
    request_count: int
    active_requests: int
    rate_limit_tokens: int
    rate_limit_max: int


@router.get("/status", response_model=ConcurrencyStatusResponse)
async def get_concurrency_status():
    """
    获取并发控制系统状态
    
    包括：
    - GPU锁状态（是否被占用、等待队列长度）
    - 任务队列状态（队列长度、活跃任务数）
    - 速率限制器状态（会话数）
    - GPU内存使用情况
    """
    try:
        gpu_memory = get_gpu_memory_info()
        queue_status = task_queue.get_queue_status()

        return ConcurrencyStatusResponse(
            gpu_lock=gpu_lock.get_status(),
            task_queue=queue_status,
            rate_limiter={
                "active_sessions": len(rate_limiter.sessions),
                "requests_per_minute": rate_limiter.requests_per_minute,
                "burst_size": rate_limiter.burst_size
            },
            gpu_memory=gpu_memory
        )
    except Exception as e:
        system_logger.error(f"【并发管理】获取状态失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取并发状态失败: {str(e)}")


@router.get("/session")
async def get_session_info(x_session_id: Optional[str] = Header(None)):
    """
    获取当前会话信息
    
    需要在请求头中提供 X-Session-ID
    """
    if not x_session_id:
        raise HTTPException(status_code=400, detail="缺少 X-Session-ID 请求头")
    
    session_info = rate_limiter.get_session_info(x_session_id)
    
    if not session_info:
        raise HTTPException(status_code=404, detail="会话不存在或已过期")
    
    return {
        "success": True,
        "session": session_info
    }


@router.get("/queue/wait-time")
async def estimate_wait_time():
    """
    预估等待时间

    根据当前队列长度估算新任务的预计等待时间
    """
    queue_status = task_queue.get_queue_status()
    queue_size = queue_status.get("pending_count", 0)

    # 估算每个任务平均处理时间（秒）
    avg_task_duration = 20.0

    # 考虑并发度
    concurrent_models = len(queue_status.get("concurrent_models", [])) or 1
    estimated_wait = queue_size * avg_task_duration / concurrent_models

    return {
        "success": True,
        "queue_size": queue_size,
        "estimated_wait_seconds": round(estimated_wait, 1),
        "estimated_wait_formatted": format_duration(estimated_wait),
        "gpu_busy": gpu_lock.is_locked
    }


def format_duration(seconds: float) -> str:
    """格式化持续时间"""
    if seconds < 60:
        return f"{int(seconds)}秒"
    elif seconds < 3600:
        minutes = int(seconds / 60)
        secs = int(seconds % 60)
        return f"{minutes}分{secs}秒"
    else:
        hours = int(seconds / 3600)
        minutes = int((seconds % 3600) / 60)
        return f"{hours}小时{minutes}分"


@router.post("/queue/clear-completed")
async def clear_completed_tasks():
    """
    清理已完成/失败/已取消/已重试的任务记录（保留最近完成的任务）
    """
    try:
        removable_statuses = [
            task_queue.TaskStatus.COMPLETED.value,
            task_queue.TaskStatus.FAILED.value,
            task_queue.TaskStatus.CANCELLED.value,
            task_queue.TaskStatus.RETRIED.value,
        ]
        old_time = None  # 使用内置的 cleanup_old_tasks
        removed_count = 0

        # 清理7天前的任务
        task_queue.cleanup_old_tasks(days=7)

        system_logger.info(f"【并发管理】已完成的任务记录")
        return {
            "success": True,
            "message": f"已触发清理",
            "before_count": len(task_queue.tasks),
        }
    except Exception as e:
        system_logger.error(f"【并发管理】清理任务记录失败: {e}")
        raise HTTPException(status_code=500, detail=f"清理失败: {str(e)}")


@router.get("/config")
async def get_concurrency_config():
    """
    获取并发控制配置
    
    查看当前限流和队列的配置参数
    """
    return {
        "success": True,
        "config": {
            "rate_limiter": {
                "requests_per_minute": rate_limiter.requests_per_minute,
                "burst_size": rate_limiter.burst_size,
                "session_timeout_minutes": 30
            },
            "task_queue": {
                "concurrent_models": task_queue._model_concurrency,
                "default_timeout_seconds": 180
            },
            "gpu_lock": {
                "acquire_timeout_seconds": 300,
                "enabled": True
            }
        }
    }
