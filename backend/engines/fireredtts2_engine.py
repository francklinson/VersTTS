#!/usr/bin/env python3
"""
FireRedTTS2 模型加载器
"""

import os
import sys
import time

import torch
from fastapi import HTTPException

from backend.logger_config import OperationLogger, system_logger
from backend.config import models, ALGORITHM_PATHS
from backend.core.model_manager import model_manager


def get_fireredtts2_model():
    """获取或加载FireRedTTS2模型"""
    if "fireredtts2" not in models:
        start_time = time.time()
        OperationLogger.log_model_load("FireRedTTS2", "开始加载")

        try:
            # 添加FireRedTTS2路径
            fireredtts2_path = ALGORITHM_PATHS['fireredtts2']
            if fireredtts2_path not in sys.path:
                sys.path.insert(0, fireredtts2_path)

            from fireredtts2.fireredtts2 import FireRedTTS2

            model_path = os.path.join(ALGORITHM_PATHS['fireredtts2'], "pretrained_models", "FireRedTTS2")

            system_logger.info(f"【模型加载】FireRedTTS2 从路径: {model_path}")

            device = "cuda" if torch.cuda.is_available() else "cpu"

            max_retries = 2
            for attempt in range(max_retries):
                try:
                    models["fireredtts2"] = FireRedTTS2(
                        pretrained_dir=model_path,
                        gen_type="monologue",
                        device=device
                    )
                    break
                except RuntimeError as e:
                    if "CUDA" in str(e) or "out of memory" in str(e).lower():
                        system_logger.warning("【模型加载】FireRedTTS2 OOM，尝试驱逐其他模型...")
                        model_manager.request_eviction(needed_mb=2000, exclude_key="fireredtts2")
                        time.sleep(3)
                        if attempt == max_retries - 1:
                            raise
                    else:
                        raise

            model_manager.touch("fireredtts2")
            duration = time.time() - start_time
            gpu_mem = torch.cuda.memory_allocated() / 1024 ** 3 if torch.cuda.is_available() else 0
            OperationLogger.log_model_load("FireRedTTS2", "成功", duration, f"GPU内存: {gpu_mem:.2f}GB")
            OperationLogger.log_performance("FireRedTTS2加载", duration, 0, gpu_mem)
        except Exception as e:
            OperationLogger.log_model_load("FireRedTTS2", "失败", 0, str(e))
            system_logger.error(f"【模型加载】FireRedTTS2 失败: {e}")
            raise HTTPException(status_code=500, detail=f"FireRedTTS2模型加载失败: {str(e)}")

    model_manager.touch("fireredtts2")
    return models["fireredtts2"]
