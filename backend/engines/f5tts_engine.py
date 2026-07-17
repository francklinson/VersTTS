#!/usr/bin/env python3
"""
F5-TTS 模型加载器
"""

import os
import time

import torch

from backend.logger_config import OperationLogger, system_logger
from backend.config import models, ALGORITHM_PATHS
from backend.core.model_manager import model_manager


def get_f5tts_model():
    """获取或加载F5-TTS模型"""
    if "f5tts" not in models:
        start_time = time.time()
        OperationLogger.log_model_load("F5-TTS", "开始加载")

        from f5_tts.api import F5TTS
        f5tts_root = os.path.dirname(ALGORITHM_PATHS['f5tts'])
        ckpt_path = os.path.join(f5tts_root, "models", "model_1200000.pt")
        vocoder_path = os.path.join(f5tts_root, "checkpoints", "vocos-mel-24khz")

        system_logger.info(f"【模型加载】F5-TTS 检查点: {ckpt_path}")
        system_logger.info(f"【模型加载】F5-TTS vocoder: {vocoder_path}")

        max_retries = 2
        for attempt in range(max_retries):
            try:
                # 使用本地 vocoder 路径，避免从 HuggingFace Hub 下载
                models["f5tts"] = F5TTS(
                    model="F5TTS_Base",
                    ckpt_file=ckpt_path,
                    vocoder_local_path=vocoder_path
                )
                break
            except RuntimeError as e:
                if "CUDA" in str(e) or "out of memory" in str(e).lower():
                    system_logger.warning("【模型加载】F5-TTS OOM，尝试驱逐其他模型...")
                    model_manager.request_eviction(needed_mb=1500, exclude_key="f5tts")
                    time.sleep(3)
                    if attempt == max_retries - 1:
                        raise
                else:
                    raise

        model_manager.touch("f5tts")
        duration = time.time() - start_time
        gpu_mem = torch.cuda.memory_allocated() / 1024 ** 3 if torch.cuda.is_available() else 0
        OperationLogger.log_model_load("F5-TTS", "成功", duration, f"GPU内存: {gpu_mem:.2f}GB")
        OperationLogger.log_performance("F5-TTS加载", duration, 0, gpu_mem)

    model_manager.touch("f5tts")
    return models["f5tts"]
