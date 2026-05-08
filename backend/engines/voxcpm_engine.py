#!/usr/bin/env python3
"""
VoxCPM 模型加载器
"""

import os
import sys
import time

import torch
from fastapi import HTTPException

from backend.logger_config import OperationLogger, system_logger
from backend.config import models, ALGORITHM_PATHS, MODEL_PATHS


def get_voxcpm_model():
    """获取或加载VoxCPM模型"""
    if "voxcpm" not in models:
        start_time = time.time()
        OperationLogger.log_model_load("VoxCPM", "开始加载")

        try:
            # 添加VoxCPM路径
            voxcpm_path = ALGORITHM_PATHS['voxcpm']
            if voxcpm_path not in sys.path:
                sys.path.insert(0, voxcpm_path)

            from voxcpm import VoxCPM

            model_path = os.path.join(MODEL_PATHS['voxcpm'], "VoxCPM2")
            
            # 检查是否离线模式
            is_offline = os.environ.get('TRANSFORMERS_OFFLINE') == '1' or os.environ.get('HF_HUB_OFFLINE') == '1'
            
            if not os.path.exists(model_path):
                if is_offline:
                    raise FileNotFoundError(f"离线模式下找不到本地模型: {model_path}")
                # 尝试HuggingFace模型ID
                model_path = "openbmb/VoxCPM2"
                system_logger.warning(f"【模型加载】本地模型不存在，尝试从HuggingFace加载: {model_path}")

            system_logger.info(f"【模型加载】VoxCPM 从路径: {model_path}")

            # 离线模式下强制使用本地文件
            models["voxcpm"] = VoxCPM.from_pretrained(
                model_path, 
                load_denoiser=False,
                local_files_only=is_offline
            )

            duration = time.time() - start_time
            gpu_mem = torch.cuda.memory_allocated() / 1024 ** 3 if torch.cuda.is_available() else 0
            OperationLogger.log_model_load("VoxCPM", "成功", duration, f"GPU内存: {gpu_mem:.2f}GB")
            OperationLogger.log_performance("VoxCPM加载", duration, 0, gpu_mem)
        except Exception as e:
            OperationLogger.log_model_load("VoxCPM", "失败", 0, str(e))
            system_logger.error(f"【模型加载】VoxCPM 失败: {e}")
            raise HTTPException(status_code=500, detail=f"VoxCPM模型加载失败: {str(e)}")

    return models["voxcpm"]
