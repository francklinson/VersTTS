#!/usr/bin/env python3
"""
健康检查路由
"""

import psutil
from fastapi import APIRouter, Request

import torch
from backend.logger_config import OperationLogger, system_logger
from backend.config import models

router = APIRouter()


@router.get("/health")
async def health(request: Request):
    """健康检查"""
    client_ip = request.client.host if request.client else "unknown"

    # 获取系统状态
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()

    gpu_info = ""
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.memory_allocated() / 1024 ** 3
        gpu_total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        gpu_info = f"{gpu_mem:.2f}GB / {gpu_total:.2f}GB"

    OperationLogger.log_api_request("/health", "GET", {}, client_ip, 0)
    OperationLogger.log_system_status(cpu_percent, memory.percent, gpu_info)

    return {
        "status": "healthy",
        "cuda_available": torch.cuda.is_available(),
        "models_loaded": list(models.keys()),
        "system": {
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "gpu_memory": gpu_info
        }
    }
