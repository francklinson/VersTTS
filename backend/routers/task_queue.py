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
    model: str = Field(..., description="TTS模型名称: voxcpm, qwen3tts, omnivoice, cosyvoice, pilottts, gptsovits")
    mode: str = Field(..., description="生成模式")
    text: str = Field(..., description="合成文本")
    speaker_id: Optional[str] = Field(None, description="说话人ID")
    speaker: Optional[str] = Field(None, description="预设音色名称（Qwen3-TTS CustomVoice）")
    voice_design_prompt: Optional[str] = Field(None, description="音色设计描述")
    control_prompt: Optional[str] = Field(None, description="控制指令")
    instruct_text: Optional[str] = Field(None, description="指令文本（Qwen3-TTS）")
    version: Optional[str] = Field(None, description="模型版本（GPT-SoVITS: v1/v2/v2Pro/v2ProPlus/v3/v4）")
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
    error_code: Optional[str] = None  # 失败类型码（AUDIO_VERIFY_FAILED 表示可重试的内容校验失败）
    progress: int
    batch_total: int = 0
    batch_completed: int = 0
    batch_results: Optional[List[Dict]] = None
    speaker_name: Optional[str] = None
    batch_count: int = 1
    wait_time_seconds: int = 0  # 等待时长（秒）
    execution_time_seconds: Optional[int] = None  # 执行时长（秒）
    is_mine: bool = False  # 是否为当前用户提交的任务（共享视图下用于区分可操作任务）
    instruct_prompt: Optional[str] = None  # 指令文本（instruct_text/control_prompt/voice_design_prompt 取首个非空），供前端展示指令组合


class TaskListResponse(BaseModel):
    """任务列表响应"""
    success: bool
    tasks: List[TaskInfo]
    total: int
    status_counts: Dict[str, int] = Field(
        default_factory=dict,
        description="各状态的全量任务计数（忽略分页与状态筛选，仅受 model 筛选影响），供统计栏稳定显示"
    )


class QueueStatusResponse(BaseModel):
    """队列状态响应"""
    pending_count: int
    processing_count: int
    total_tasks: int
    is_running: bool
    max_workers: int
    model_counts: dict = {}
    processing_models: dict = {}
    concurrent_models: list = []
    model_states: dict = {}
    model_concurrency: dict = {}


# ============ 辅助函数 ============

def get_user_id(request: Request) -> str:
    """获取用户标识（从header或IP）"""
    # 从header获取
    user_id = request.headers.get("X-User-ID")
    if not user_id:
        # 使用客户端IP作为标识
        user_id = request.client.host if request.client else "anonymous"
    return user_id


def _extract_instruct_prompt(params: Optional[Dict]) -> Optional[str]:
    """从任务参数中提取指令文本，供前端展示指令组合。

    指令类模式会把具体指令词存入 params，但字段名因模型/模式而异：
    - Qwen3-TTS custom_voice: instruct_text
    - CosyVoice instruct / VoxCPM clone / ultimate_clone: control_prompt
    - VoxCPM / Qwen3-TTS / OmniVoice voice_design: voice_design_prompt
    按上述优先级取首个非空值。
    """
    if not params:
        return None
    for key in ('instruct_text', 'control_prompt', 'voice_design_prompt'):
        val = params.get(key)
        if val and str(val).strip():
            return str(val).strip()
    return None


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
    version: Optional[str] = Form(None, description="模型版本（GPT-SoVITS）"),
    batch_count: int = Form(1, description="批量生成数量"),
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
            "version": version,
            "batch_count": batch_count,
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
    model: Optional[str] = Query(None, description="算法/模型筛选: chattts, cosyvoice, f5tts, qwen3tts, openvoice, gptsovits, voxcpm, omnivoice, pilottts, indextts, fireredtts"),
    limit: int = Query(100, ge=1, le=500, description="返回数量限制"),
    offset: int = Query(0, ge=0, description="偏移量（分页）")
):
    """
    获取当前用户的任务列表
    """
    try:
        user_id = get_user_id(request)
        # 多用户共享视图：返回所有用户的任务，任务页展示全局看板。
        # 取消/删除等写操作仍按 user_id 校验（仅本人可操作自己的任务）。
        tasks = task_queue.get_all_tasks(status=status, limit=limit, offset=offset, model=model)
        # total 为符合筛选条件的任务总数（分页前），前端据此翻页
        total_count = task_queue.count_all_tasks(status=status, model=model)
        # 各状态全量计数（忽略分页与状态筛选，仅受 model 筛选影响），供统计栏稳定显示
        status_counts = task_queue.status_counts_all(model=model)

        task_infos = []
        from datetime import datetime as dt
        now = dt.now()

        for task in tasks:
            # 获取说话人名称
            speaker_name = None
            speaker_id = task.params.get('speaker_id') if task.params else None
            if speaker_id:
                try:
                    from backend.services import get_speaker_by_id
                    speaker_info = get_speaker_by_id(speaker_id)
                    if speaker_info:
                        speaker_name = speaker_info.get('name', speaker_id)
                except Exception:
                    pass

            # 获取批量数量
            batch_count = task.params.get('batch_count', 1) if task.params else 1

            # 提取指令文本（供前端展示指令组合）
            instruct_prompt = _extract_instruct_prompt(task.params)

            # 计算等待时长
            wait_time_seconds = 0
            if task.started_at:
                try:
                    created = dt.fromisoformat(task.created_at)
                    started = dt.fromisoformat(task.started_at)
                    wait_time_seconds = int((started - created).total_seconds())
                except Exception:
                    pass
            elif task.status in ['queued', 'pending']:
                try:
                    created = dt.fromisoformat(task.created_at)
                    wait_time_seconds = int((now - created).total_seconds())
                except Exception:
                    pass

            # 计算执行时长
            execution_time_seconds = None
            if task.started_at:
                try:
                    started = dt.fromisoformat(task.started_at)
                    if task.completed_at:
                        completed = dt.fromisoformat(task.completed_at)
                        execution_time_seconds = int((completed - started).total_seconds())
                    elif task.status == 'processing':
                        execution_time_seconds = int((now - started).total_seconds())
                except Exception:
                    pass

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
                error_code=task.error_code,
                progress=task.progress,
                batch_total=task.batch_total,
                batch_completed=task.batch_completed,
                batch_results=task.batch_results,
                speaker_name=speaker_name,
                batch_count=batch_count,
                wait_time_seconds=wait_time_seconds,
                execution_time_seconds=execution_time_seconds,
                is_mine=(task.user_id == user_id),
                instruct_prompt=instruct_prompt
            ))

        return TaskListResponse(
            success=True,
            tasks=task_infos,
            total=total_count,
            status_counts=status_counts
        )

    except Exception as e:
        system_logger.error(f"【任务API】获取任务列表失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取任务列表失败: {str(e)}")


@router.get("/queue/status", response_model=QueueStatusResponse)
async def get_queue_status():
    """
    获取队列状态
    注意: 必须定义在 /{task_id}/status 之前，否则 "queue" 会被 {task_id} 捕获
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
    手动清理旧任务记录（管理接口），并联动删除其音频文件。
    定时自动清理已关闭，此接口仅供手动调用。
    注意: 必须定义在 /{task_id} 路由之前
    """
    try:
        task_queue.cleanup_old_tasks(days)
        return {
            "success": True,
            "message": f"已清理 {days} 天前的任务记录（含音频文件）"
        }
    except Exception as e:
        system_logger.error(f"【任务API】清理任务失败: {e}")
        raise HTTPException(status_code=500, detail=f"清理任务失败: {str(e)}")


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
        
        # 获取说话人名称
        speaker_name = None
        speaker_id = task.params.get('speaker_id') if task.params else None
        if speaker_id:
            try:
                from backend.services import get_speaker_by_id
                speaker_info = get_speaker_by_id(speaker_id)
                if speaker_info:
                    speaker_name = speaker_info.get('name', speaker_id)
            except Exception:
                pass

        # 获取批量数量
        batch_count = task.params.get('batch_count', 1) if task.params else 1

        return {
            "success": True,
            "task_id": task.task_id,
            "model": task.model,
            "mode": task.mode,
            "status": task.status,
            "progress": task.progress,
            "batch_total": task.batch_total,
            "batch_completed": task.batch_completed,
            "batch_count": batch_count,
            "speaker_name": speaker_name,
            "created_at": task.created_at,
            "started_at": task.started_at,
            "completed_at": task.completed_at,
            "audio_url": task.audio_url,
            "error_message": task.error_message,
            "error_code": task.error_code,
            "batch_results": task.batch_results,
            "instruct_prompt": _extract_instruct_prompt(task.params)
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

        # 将旧任务标记为 RETRIED，关联到新任务，避免任务列表重复显示
        task.status = TaskStatus.RETRIED.value
        task.error_message = f"已重试 -> {new_task.task_id}"
        task.completed_at = datetime.now().isoformat()
        task_queue._save_task(task)

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
    删除任务记录。
    失败类终态（失败/已取消/已重试）任何人可删；其他状态（已完成/排队中/执行中）仅本人可删。
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
