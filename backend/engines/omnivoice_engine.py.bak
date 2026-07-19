#!/usr/bin/env python3
"""
OmniVoice 模型加载器
"""

import os
import sys
import time
import torch
from fastapi import HTTPException

from backend.logger_config import OperationLogger, system_logger
from backend.config import models, MODEL_PATHS, ALGORITHM_PATHS


def get_omnivoice_model():
    """获取或加载OmniVoice模型"""
    if "omnivoice" not in models:
        start_time = time.time()
        OperationLogger.log_model_load("OmniVoice", "开始加载")

        try:
            # 添加OmniVoice路径
            omnivoice_path = ALGORITHM_PATHS['omnivoice']
            if omnivoice_path not in sys.path:
                sys.path.insert(0, omnivoice_path)

            from omnivoice import OmniVoice

            # 使用本地模型路径
            model_path = MODEL_PATHS['omnivoice']
            
            # 检查是否离线模式
            is_offline = os.environ.get('TRANSFORMERS_OFFLINE') == '1' or os.environ.get('HF_HUB_OFFLINE') == '1'
            
            if not os.path.exists(model_path):
                if is_offline:
                    raise FileNotFoundError(f"离线模式下找不到本地模型: {model_path}")
                model_path = "k2-fsa/OmniVoice"
                system_logger.warning(f"【模型加载】本地模型不存在，尝试从HuggingFace加载: {model_path}")

            system_logger.info(f"【模型加载】OmniVoice 从路径: {model_path}")

            # 加载模型
            models["omnivoice"] = OmniVoice.from_pretrained(
                model_path,
                device_map="cuda:0" if torch.cuda.is_available() else "cpu",
                dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                local_files_only=is_offline
            )

            duration = time.time() - start_time
            gpu_mem = torch.cuda.memory_allocated() / 1024 ** 3 if torch.cuda.is_available() else 0
            OperationLogger.log_model_load("OmniVoice", "成功", duration, f"GPU内存: {gpu_mem:.2f}GB")
            OperationLogger.log_performance("OmniVoice加载", duration, 0, gpu_mem)
        except Exception as e:
            OperationLogger.log_model_load("OmniVoice", "失败", 0, str(e))
            system_logger.error(f"【模型加载】OmniVoice 失败: {e}")
            raise HTTPException(status_code=500, detail=f"OmniVoice模型加载失败: {str(e)}")

    return models["omnivoice"]
