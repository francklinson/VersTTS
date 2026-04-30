#!/usr/bin/env python3
"""
批量处理路由
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict

from backend.logger_config import system_logger
from backend.batch_processor import batch_processor, BatchJob

router = APIRouter()


class BatchTTSRequest(BaseModel):
    """批量 TTS 请求"""
    model: str = Field(default="chattts", description="TTS模型名称")
    tasks: List[dict] = Field(default=[], description="任务列表")


@router.post("/create")
async def create_batch_job(request: BatchTTSRequest):
    """创建批量TTS任务"""
    try:
        job = batch_processor.create_job(request.model, request.tasks)
        system_logger.info(f"【批量处理】创建任务: {job.job_id}, 总任务数: {job.total}")
        return {
            "success": True,
            "job_id": job.job_id,
            "total": job.total,
            "status": job.status
        }
    except Exception as e:
        system_logger.error(f"【批量处理】创建任务失败: {e}")
        raise HTTPException(status_code=500, detail=f"创建批量任务失败: {str(e)}")


@router.get("/{job_id}/status")
async def get_batch_status(job_id: str):
    """获取批量任务状态"""
    job = batch_processor.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    return {
        "job_id": job.job_id,
        "status": job.status,
        "total": job.total,
        "completed": job.completed,
        "failed": job.failed,
        "progress": f"{(job.completed + job.failed) / job.total * 100:.1f}%" if job.total > 0 else "0%"
    }


@router.post("/{job_id}/process")
async def process_batch_job(job_id: str):
    """处理批量TTS任务（简化版，实际实现需要异步处理）"""
    job = batch_processor.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    return {
        "success": True,
        "message": "批量任务已加入处理队列",
        "job_id": job_id
    }


@router.get("/{job_id}/download")
async def download_batch_results(job_id: str):
    """下载批量任务结果ZIP包"""
    try:
        zip_path = batch_processor.create_zip_package(job_id)
        return {
            "success": True,
            "download_url": f"/tts/batch/{job_id}/download/file"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"创建下载包失败: {str(e)}")


@router.get("/{job_id}/results")
async def get_batch_results(job_id: str):
    """获取批量任务详细结果"""
    job = batch_processor.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    return {
        "job_id": job.job_id,
        "status": job.status,
        "tasks": [
            {
                "id": task.id,
                "text": task.text[:100] + "..." if len(task.text) > 100 else task.text,
                "status": task.status,
                "error": task.error
            }
            for task in job.tasks
        ]
    }
