#!/usr/bin/env python3
"""
dots.tts 模型加载器
"""

import os
import sys
import time

import torch
from fastapi import HTTPException

from backend.logger_config import OperationLogger, system_logger
from backend.config import models, ALGORITHM_PATHS, MODEL_PATHS, MODELS_DIR, ALGORITHMS_DIR
from backend.core.model_manager import model_manager


def get_dotstts_model():
    """获取或加载dots.tts模型"""
    key = "dotstts"
    display_name = "dots.tts"
    estimated_vram_mb = 8000  # 约8GB显存

    if key not in models:
        start_time = time.time()
        OperationLogger.log_model_load(display_name, "开始加载")

        # 添加 dots.tts 源码路径
        dotstts_src = ALGORITHM_PATHS.get('dotstts_src') or os.path.join(ALGORITHMS_DIR, 'dotstts', 'src')
        if dotstts_src not in sys.path:
            sys.path.insert(0, dotstts_src)

        from dots_tts.runtime import DotsTtsRuntime

        # 确定模型权重路径
        model_path = os.environ.get("DOTSTTS_PATH")
        if not model_path:
            model_path = MODEL_PATHS.get('dotstts')
        if not model_path:
            model_path = os.path.join(MODELS_DIR, 'dotstts')

        # 检查是否离线模式
        is_offline = os.environ.get('TRANSFORMERS_OFFLINE') == '1' or os.environ.get('HF_HUB_OFFLINE') == '1'

        if not os.path.exists(model_path):
            if is_offline:
                raise FileNotFoundError(f"离线模式下找不到本地模型: {model_path}")
            system_logger.warning(f"【模型加载】本地模型不存在，尝试从HuggingFace加载: {model_path}")

        system_logger.info(f"【模型加载】dots.tts 模型路径: {model_path}")

        # OOM 驱逐重试
        oom_retried = False
        while True:
            try:
                models[key] = DotsTtsRuntime.from_pretrained(
                    model_path,
                    precision="bfloat16",
                    optimize=False,
                    max_generate_length=500,
                )
                break  # 加载成功，跳出循环
            except RuntimeError as e:
                if ("CUDA" in str(e) or "out of memory" in str(e).lower()) and not oom_retried:
                    system_logger.warning(f"【模型加载】{display_name} OOM，尝试驱逐其他模型...")
                    model_manager.request_eviction(needed_mb=estimated_vram_mb, exclude_key=key)
                    time.sleep(3)
                    models.pop(key, None)
                    oom_retried = True
                else:
                    raise
            except Exception as e:
                OperationLogger.log_model_load(display_name, "失败", 0, str(e))
                system_logger.error(f"【模型加载】dots.tts 失败: {e}")
                raise HTTPException(status_code=500, detail=f"dots.tts模型加载失败: {str(e)}")

        model_manager.touch(key)  # 加载后更新使用时间

        duration = time.time() - start_time
        gpu_mem = torch.cuda.memory_allocated() / 1024 ** 3 if torch.cuda.is_available() else 0
        OperationLogger.log_model_load(display_name, "成功", duration, f"GPU内存: {gpu_mem:.2f}GB")
        OperationLogger.log_performance(f"{display_name}加载", duration, 0, gpu_mem)

    model_manager.touch(key)  # 每次获取都更新使用时间
    return models[key]
