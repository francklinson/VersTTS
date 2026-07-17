#!/usr/bin/env python3
"""
Qwen3-TTS 模型加载器
"""

import os
import time

import torch
import transformers
from fastapi import HTTPException

from backend.logger_config import OperationLogger, system_logger
from backend.config import models, MODEL_PATHS
from backend.core.model_manager import model_manager

def get_qwen3tts_model(model_size: str = "1.7B", model_type: str = "Base"):
    """获取或加载Qwen3-TTS模型
    
    Args:
        model_size: 模型大小，仅支持 "1.7B"（0.6B已弃用）
        model_type: 模型类型 "Base", "CustomVoice", "VoiceDesign"
    """
    # 检查 transformers 版本
    tv = transformers.__version__.split('.')
    major, minor = int(tv[0]), int(tv[1])
    if major < 4 or (major == 4 and minor < 57):
        raise HTTPException(
            status_code=503,
            detail=f"Qwen3-TTS 需要 transformers >= 4.57.0，当前版本为 {transformers.__version__}。"
                   f"CosyVoice 需要 transformers 4.51.3，两个模型版本要求冲突。"
                   f"请使用 CosyVoice 或升级 transformers 到 4.57.3（但 CosyVoice 可能会产生杂音）。"
        )

    key = f"qwen3tts_{model_size}_{model_type}"

    # 根据 model_type 确定 display_name 和 estimated_vram_mb
    vram_map = {
        "Base": 3000,
        "CustomVoice": 3000,
        "VoiceDesign": 3500,
    }
    display_name = f"Qwen3-TTS-{model_size}-{model_type}"
    estimated_vram_mb = vram_map.get(model_type, 3000)

    if key not in models:
        start_time = time.time()
        OperationLogger.log_model_load(f"Qwen3-TTS-{model_size}-{model_type}", "开始加载")

        from qwen_tts import Qwen3TTSModel
        # 注：0.6B模型已弃用，仅保留1.7B支持
        size_map = {
            "1.7B": "1___7B"
        }
        # 如果传入0.6B，自动映射到1.7B（向后兼容）
        if model_size == "0.6B":
            system_logger.warning("0.6B模型已弃用，自动使用1.7B模型")
            model_size = "1.7B"
        size_str = size_map.get(model_size, model_size.replace('.', '___'))
        model_path = os.path.join(MODEL_PATHS['qwen3tts'], "Qwen",
                                  f"Qwen3-TTS-12Hz-{size_str}-{model_type}")

        # 如果指定类型模型不存在，尝试加载 Base 模型
        if not os.path.exists(model_path):
            if model_type != "Base":
                system_logger.warning(f"【模型加载】{model_type} 模型不存在，尝试加载 Base 模型")
                model_path = os.path.join(MODEL_PATHS['qwen3tts'], "Qwen",
                                          f"Qwen3-TTS-12Hz-{size_str}-Base")
            if not os.path.exists(model_path):
                raise HTTPException(status_code=500, detail=f"Qwen3-TTS 模型不存在: {model_path}")

        system_logger.info(f"【模型加载】Qwen3-TTS 路径: {model_path}")
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        attn_impl = "flash_attention_2" if torch.cuda.is_available() else "eager"

        # 检查是否离线模式
        is_offline = os.environ.get('TRANSFORMERS_OFFLINE') == '1' or os.environ.get('HF_HUB_OFFLINE') == '1'
        if is_offline:
            system_logger.info("【模型加载】Qwen3-TTS 离线模式，强制使用本地文件")

        # 兼容不同 transformers 版本，并添加 OOM 驱逐重试
        oom_retried = False
        while True:
            try:
                try:
                    models[key] = Qwen3TTSModel.from_pretrained(
                        model_path,
                        device_map=device,
                        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                        attn_implementation=attn_impl,
                        local_files_only=is_offline
                    )
                except TypeError:
                    # 旧版本 transformers 使用 dtype 参数
                    models[key] = Qwen3TTSModel.from_pretrained(
                        model_path,
                        device_map=device,
                        attn_implementation=attn_impl,
                        local_files_only=is_offline
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

        model_manager.touch(key)  # 加载后更新使用时间

        duration = time.time() - start_time
        gpu_mem = torch.cuda.memory_allocated() / 1024 ** 3 if torch.cuda.is_available() else 0
        OperationLogger.log_model_load(f"Qwen3-TTS-{model_size}-{model_type}", "成功", duration,
                                       f"GPU内存: {gpu_mem:.2f}GB")
        OperationLogger.log_performance("Qwen3-TTS加载", duration, 0, gpu_mem)

    model_manager.touch(key)  # 每次获取都更新使用时间
    return models[key]
