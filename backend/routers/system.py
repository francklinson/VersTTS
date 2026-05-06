#!/usr/bin/env python3
"""
系统管理路由 - 显存监控和内存管理
"""

import gc
import torch
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional

from backend.logger_config import system_logger
from backend.config import models
from backend.core import (
    cleanup_memory,
    get_gpu_memory_info,
    log_gpu_memory_usage,
    clear_model_cache
)

router = APIRouter()


class GPUMemoryResponse(BaseModel):
    """GPU内存信息响应"""
    allocated_gb: float
    reserved_gb: float
    total_gb: float
    free_gb: float
    device_name: Optional[str] = None
    cuda_available: bool


class ModelCacheResponse(BaseModel):
    """模型缓存信息响应"""
    loaded_models: List[str]
    count: int


class CleanupResponse(BaseModel):
    """清理操作响应"""
    success: bool
    message: str
    gpu_memory_before: Dict
    gpu_memory_after: Dict


@router.get("/gpu-memory", response_model=GPUMemoryResponse)
async def get_gpu_memory():
    """获取GPU内存使用情况"""
    mem_info = get_gpu_memory_info()

    device_name = None
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)

    return GPUMemoryResponse(
        allocated_gb=mem_info["allocated"],
        reserved_gb=mem_info["reserved"],
        total_gb=mem_info["total"],
        free_gb=mem_info["free"],
        device_name=device_name,
        cuda_available=torch.cuda.is_available()
    )


@router.get("/models", response_model=ModelCacheResponse)
async def get_loaded_models():
    """获取已加载的模型列表"""
    return ModelCacheResponse(
        loaded_models=list(models.keys()),
        count=len(models)
    )


@router.post("/cleanup", response_model=CleanupResponse)
async def cleanup_system_memory():
    """
    清理系统内存和显存

    此操作会：
    1. 强制运行Python垃圾回收
    2. 清空PyTorch CUDA缓存
    3. 记录清理前后的显存使用情况
    """
    mem_before = get_gpu_memory_info()

    try:
        # 执行清理
        cleanup_memory(force_gc=True, empty_cache=True, synchronize=True)

        mem_after = get_gpu_memory_info()

        system_logger.info(
            f"【系统管理】内存清理完成 | "
            f"显存释放: {mem_before['allocated'] - mem_after['allocated']:.2f}GB"
        )

        return CleanupResponse(
            success=True,
            message="内存清理成功",
            gpu_memory_before=mem_before,
            gpu_memory_after=mem_after
        )

    except Exception as e:
        system_logger.error(f"【系统管理】内存清理失败: {e}")
        raise HTTPException(status_code=500, detail=f"内存清理失败: {str(e)}")


@router.post("/clear-model/{model_key}")
async def clear_specific_model(model_key: str):
    """
    清理指定的模型缓存

    Args:
        model_key: 模型标识符，如 'chattts', 'f5tts', 'gpt_sovits_v2' 等
    """
    if model_key not in models:
        raise HTTPException(
            status_code=404,
            detail=f"模型不存在: {model_key}。已加载的模型: {list(models.keys())}"
        )

    mem_before = get_gpu_memory_info()

    try:
        clear_model_cache(model_key)

        mem_after = get_gpu_memory_info()
        released = mem_before["allocated"] - mem_after["allocated"]

        return {
            "success": True,
            "message": f"模型 {model_key} 已清理",
            "released_gb": released,
            "gpu_memory_before": mem_before,
            "gpu_memory_after": mem_after,
            "remaining_models": list(models.keys())
        }

    except Exception as e:
        system_logger.error(f"【系统管理】清理模型失败: {e}")
        raise HTTPException(status_code=500, detail=f"清理模型失败: {str(e)}")


@router.post("/clear-all-models")
async def clear_all_models():
    """清理所有模型缓存"""
    mem_before = get_gpu_memory_info()
    model_count = len(models)

    try:
        clear_model_cache(None)  # 清理所有模型

        mem_after = get_gpu_memory_info()
        released = mem_before["allocated"] - mem_after["allocated"]

        return {
            "success": True,
            "message": f"所有模型已清理，共 {model_count} 个",
            "released_gb": released,
            "gpu_memory_before": mem_before,
            "gpu_memory_after": mem_after
        }

    except Exception as e:
        system_logger.error(f"【系统管理】清理所有模型失败: {e}")
        raise HTTPException(status_code=500, detail=f"清理所有模型失败: {str(e)}")


@router.get("/status")
async def get_system_status():
    """获取系统状态概览"""
    mem_info = get_gpu_memory_info()

    return {
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_memory": mem_info,
        "loaded_models": list(models.keys()),
        "model_count": len(models),
        "python_gc_count": len(gc.get_objects())
    }
