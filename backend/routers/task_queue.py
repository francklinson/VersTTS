#!/usr/bin/env python3
"""
任务队列API路由
支持任务提交、状态查询、结果下载
"""

import os
from typing import Optional, List, Dict
from datetime import datetime

from fastapi import APIRouter, HTTPException, Request, Form, Query, Depends
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from backend.task_queue import task_queue, TaskRecord, TaskStatus
from backend.logger_config import system_logger
from backend.config import OUTPUTS_DIR
from backend.core.audio_utils import save_temp_audio

router = APIRouter(tags=["任务队列"])


# ============ 数据模型 ============

class TaskSubmitRequest(BaseModel):
    """任务提交请求"""
    model: str = Field(..., description="TTS模型名称: voxcpm, qwen3tts, omnivoice, cosyvoice")
    mode: str = Field(..., description="生成模式")
    text: str = Field(..., description="合成文本")
    speaker_id: Optional[str] = Field(None, description="说话人ID")
    speaker: Optional[str] = Field(None, description="预设音色名称（Qwen3-TTS CustomVoice）")
    voice_design_prompt: Optional[str] = Field(None, description="音色设计描述")
    control_prompt: Optional[str] = Field(None, description="控制指令")
    instruct_text: Optional[str] = Field(None, description="指令文本（Qwen3-TTS）")
    speed: float = Field(1.0, description="语速")
    priority: int = Field(0, description="优先级（数字越小优先级越高）")


class BatchTaskSubmitRequest(BaseModel):
    """批量任务提交请求"""
    tasks: List[TaskSubmitRequest] = Field(..., description="任务列表")


class TaskSubmitResponse(BaseModel):
    """任务提交响应"""
    success: bool
    task_id: str
    status: str
    message: str
    queue_position: Optional[int] = None


class BatchTaskSubmitResponse(BaseModel):
    """批量任务提交响应"""
    success: bool
    task_ids: List[str]
    message: str
    failed_count: int = 0


class TaskInfo(BaseModel):
    """任务信息"""
    task_id: str
    model: str
    mode: str
    text: str
    status: str
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    audio_url: Optional[str] = None
    error_message: Optional[str] = None
    progress: int
    batch_results: Optional[List[Dict]] = None


class TaskListResponse(BaseModel):
    """任务列表响应"""
    success: bool
    tasks: List[TaskInfo]
    total: int


class QueueStatusResponse(BaseModel):
    """队列状态响应"""
    pending_count: int
    processing_count: int
    total_tasks: int
    is_running: bool
    max_workers: int


# ============ 辅助函数 ============

def get_user_id(request: Request) -> str:
    """获取用户标识（从header或IP）"""
    # 从header获取
    user_id = request.headers.get("X-User-ID")
    if not user_id:
        # 使用客户端IP作为标识
        user_id = request.client.host if request.client else "anonymous"
    return user_id


# ============ API端点 ============

@router.post("/submit", response_model=TaskSubmitResponse)
async def submit_task(
    request: Request,
    model: str = Form(..., description="TTS模型名称"),
    mode: str = Form(..., description="生成模式"),
    text: str = Form(..., description="合成文本"),
    speaker_id: Optional[str] = Form(None, description="说话人ID"),
    speaker: Optional[str] = Form(None, description="预设音色名称"),
    voice_design_prompt: Optional[str] = Form(None, description="音色设计描述"),
    control_prompt: Optional[str] = Form(None, description="控制指令"),
    instruct_text: Optional[str] = Form(None, description="指令文本"),
    speed: float = Form(1.0, description="语速"),
    priority: int = Form(0, description="优先级")
):
    """
    提交TTS任务到队列

    任务将在后台异步执行，不会阻塞前端
    """
    try:
        user_id = get_user_id(request)

        # 参数验证
        if not text or not text.strip():
            raise HTTPException(status_code=400, detail="文本不能为空")

        if len(text) > 5000:
            raise HTTPException(status_code=400, detail="文本长度不能超过5000字符")

        # 构建参数
        params = {
            "speaker_id": speaker_id,
            "speaker": speaker,
            "voice_design_prompt": voice_design_prompt,
            "control_prompt": control_prompt,
            "instruct_text": instruct_text,
            "speed": speed
        }
        # 移除None值
        params = {k: v for k, v in params.items() if v is not None}

        # 提交任务
        task = await task_queue.submit_task(
            user_id=user_id,
            model=model,
            mode=mode,
            text=text.strip(),
            params=params,
            priority=priority
        )

        # 计算队列位置
        queue_position = None
        if task.status == TaskStatus.QUEUED.value:
            # 统计排在前面的任务数
            user_tasks = task_queue.get_user_tasks(user_id, status=TaskStatus.QUEUED.value)
            queue_position = sum(1 for t in user_tasks if t.created_at < task.created_at)

        system_logger.info(f"【任务API】任务提交成功: {task.task_id} | 用户: {user_id}")

        return TaskSubmitResponse(
            success=True,
            task_id=task.task_id,
            status=task.status,
            message="任务已提交到队列",
            queue_position=queue_position
        )

    except HTTPException:
        raise
    except Exception as e:
        system_logger.error(f"【任务API】提交任务失败: {e}")
        raise HTTPException(status_code=500, detail=f"提交任务失败: {str(e)}")


@router.get("/list", response_model=TaskListResponse)
async def list_tasks(
    request: Request,
    status: Optional[str] = Query(None, description="状态筛选: pending, processing, completed, failed, cancelled"),
    limit: int = Query(50, ge=1, le=100, description="返回数量限制")
):
    """
    获取当前用户的任务列表
    """
    try:
        user_id = get_user_id(request)
        tasks = task_queue.get_user_tasks(user_id, status=status, limit=limit)
        
        task_infos = []
        for task in tasks:
            task_infos.append(TaskInfo(
                task_id=task.task_id,
                model=task.model,
                mode=task.mode,
                text=task.text[:100] + "..." if len(task.text) > 100 else task.text,
                status=task.status,
                created_at=task.created_at,
                started_at=task.started_at,
                completed_at=task.completed_at,
                audio_url=task.audio_url,
                error_message=task.error_message,
                progress=task.progress,
                batch_results=task.batch_results
            ))
        
        return TaskListResponse(
            success=True,
            tasks=task_infos,
            total=len(task_infos)
        )
        
    except Exception as e:
        system_logger.error(f"【任务API】获取任务列表失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取任务列表失败: {str(e)}")


@router.get("/{task_id}/status")
async def get_task_status(task_id: str, request: Request):
    """
    获取任务状态
    """
    try:
        user_id = get_user_id(request)
        task = task_queue.get_task(task_id)
        
        if not task:
            raise HTTPException(status_code=404, detail="任务不存在")
        
        # 权限检查
        if task.user_id != user_id:
            raise HTTPException(status_code=403, detail="无权访问此任务")
        
        return {
            "success": True,
            "task_id": task.task_id,
            "model": task.model,
            "mode": task.mode,
            "status": task.status,
            "progress": task.progress,
            "created_at": task.created_at,
            "started_at": task.started_at,
            "completed_at": task.completed_at,
            "audio_url": task.audio_url,
            "error_message": task.error_message,
            "batch_results": task.batch_results
        }
        
    except HTTPException:
        raise
    except Exception as e:
        system_logger.error(f"【任务API】获取任务状态失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取任务状态失败: {str(e)}")


@router.get("/{task_id}/download")
async def download_task_result(task_id: str, request: Request):
    """
    下载任务生成的音频文件
    """
    try:
        user_id = get_user_id(request)
        task = task_queue.get_task(task_id)
        
        if not task:
            raise HTTPException(status_code=404, detail="任务不存在")
        
        # 权限检查
        if task.user_id != user_id:
            raise HTTPException(status_code=403, detail="无权访问此任务")
        
        # 检查任务状态
        if task.status != TaskStatus.COMPLETED.value:
            raise HTTPException(status_code=400, detail=f"任务未完成，当前状态: {task.status}")
        
        # 检查文件是否存在
        if not task.audio_file or not os.path.exists(task.audio_file):
            raise HTTPException(status_code=404, detail="音频文件不存在")
        
        # 返回文件
        filename = os.path.basename(task.audio_file)
        return FileResponse(
            task.audio_file,
            media_type="audio/wav",
            filename=filename
        )
        
    except HTTPException:
        raise
    except Exception as e:
        system_logger.error(f"【任务API】下载任务结果失败: {e}")
        raise HTTPException(status_code=500, detail=f"下载失败: {str(e)}")


@router.post("/{task_id}/cancel")
async def cancel_task(task_id: str, request: Request):
    """
    取消任务
    """
    try:
        user_id = get_user_id(request)
        
        success = await task_queue.cancel_task(task_id, user_id)
        
        if not success:
            raise HTTPException(status_code=400, detail="取消任务失败，任务可能已完成或正在执行中")
        
        return {
            "success": True,
            "message": "任务已取消",
            "task_id": task_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        system_logger.error(f"【任务API】取消任务失败: {e}")
        raise HTTPException(status_code=500, detail=f"取消任务失败: {str(e)}")


@router.post("/batch/submit", response_model=BatchTaskSubmitResponse)
async def batch_submit_tasks(
    request: Request,
    tasks_json: str = Form(..., description="任务列表JSON字符串")
):
    """
    批量提交TTS任务到队列
    
    支持一次提交多个任务，每个任务独立执行
    """
    try:
        import json
        tasks_data = json.loads(tasks_json)
        user_id = get_user_id(request)
        
        task_ids = []
        failed_count = 0
        
        for task_data in tasks_data:
            try:
                text = task_data.get('text', '').strip()
                if not text:
                    failed_count += 1
                    continue
                
                params = {
                    "speaker_id": task_data.get('speaker_id'),
                    "speaker": task_data.get('speaker'),
                    "voice_design_prompt": task_data.get('voice_design_prompt'),
                    "control_prompt": task_data.get('control_prompt'),
                    "instruct_text": task_data.get('instruct_text'),
                    "speed": task_data.get('speed', 1.0)
                }
                params = {k: v for k, v in params.items() if v is not None}
                
                task = await task_queue.submit_task(
                    user_id=user_id,
                    model=task_data.get('model', 'voxcpm'),
                    mode=task_data.get('mode', 'base'),
                    text=text,
                    params=params,
                    priority=task_data.get('priority', 0)
                )
                task_ids.append(task.task_id)
            except Exception as e:
                system_logger.error(f"【任务API】批量提交中单个任务失败: {e}")
                failed_count += 1
        
        system_logger.info(f"【任务API】批量提交完成: 成功{len(task_ids)}个, 失败{failed_count}个")
        
        return BatchTaskSubmitResponse(
            success=True,
            task_ids=task_ids,
            message=f"成功提交 {len(task_ids)} 个任务",
            failed_count=failed_count
        )
        
    except Exception as e:
        system_logger.error(f"【任务API】批量提交任务失败: {e}")
        raise HTTPException(status_code=500, detail=f"批量提交任务失败: {str(e)}")


@router.post("/{task_id}/retry")
async def retry_task(task_id: str, request: Request):
    """
    重新提交失败或已取消的任务
    """
    try:
        user_id = get_user_id(request)
        task = task_queue.get_task(task_id)
        
        if not task:
            raise HTTPException(status_code=404, detail="任务不存在")
        
        if task.user_id != user_id:
            raise HTTPException(status_code=403, detail="无权访问此任务")
        
        if task.status not in [TaskStatus.FAILED.value, TaskStatus.CANCELLED.value]:
            raise HTTPException(status_code=400, detail="只有失败或已取消的任务可以重试")
        
        # 重新提交任务
        new_task = await task_queue.submit_task(
            user_id=user_id,
            model=task.model,
            mode=task.mode,
            text=task.text,
            params=task.params,
            priority=task.priority
        )
        
        system_logger.info(f"【任务API】任务重试: {task_id} -> {new_task.task_id}")
        
        return {
            "success": True,
            "message": "任务已重新提交",
            "old_task_id": task_id,
            "new_task_id": new_task.task_id,
            "status": new_task.status
        }
        
    except HTTPException:
        raise
    except Exception as e:
        system_logger.error(f"【任务API】重试任务失败: {e}")
        raise HTTPException(status_code=500, detail=f"重试任务失败: {str(e)}")


@router.delete("/{task_id}")
async def delete_task(task_id: str, request: Request):
    """
    删除任务记录（仅限已完成、失败、已取消的任务）
    """
    try:
        user_id = get_user_id(request)
        
        success = await task_queue.delete_task(task_id, user_id)
        
        if not success:
            raise HTTPException(status_code=400, detail="删除任务失败，任务可能正在执行中或不存在")
        
        return {
            "success": True,
            "message": "任务已删除",
            "task_id": task_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        system_logger.error(f"【任务API】删除任务失败: {e}")
        raise HTTPException(status_code=500, detail=f"删除任务失败: {str(e)}")


@router.get("/queue/status", response_model=QueueStatusResponse)
async def get_queue_status():
    """
    获取队列状态
    """
    try:
        status = task_queue.get_queue_status()
        return QueueStatusResponse(**status)
    except Exception as e:
        system_logger.error(f"【任务API】获取队列状态失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取队列状态失败: {str(e)}")


@router.delete("/cleanup")
async def cleanup_old_tasks(days: int = Query(7, ge=1, le=30, description="保留天数")):
    """
    清理旧任务记录（管理接口）
    """
    try:
        task_queue.cleanup_old_tasks(days)
        return {
            "success": True,
            "message": f"已清理 {days} 天前的任务记录"
        }
    except Exception as e:
        system_logger.error(f"【任务API】清理任务失败: {e}")
        raise HTTPException(status_code=500, detail=f"清理任务失败: {str(e)}")
