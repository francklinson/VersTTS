#!/usr/bin/env python3
"""
OpenVoice 模型加载器
"""

import os
import time

import torch

from backend.logger_config import OperationLogger, system_logger
from backend.config import models, ALGORITHM_PATHS, PROJECT_ROOT


def get_openvoice_models(use_v2=True):
    """获取或加载OpenVoice模型
    
    Args:
        use_v2: 是否使用V2版本（默认True）
    """
    model_key = "openvoice_v2" if use_v2 else "openvoice"
    if model_key not in models:
        start_time = time.time()
        OperationLogger.log_model_load("OpenVoice V2" if use_v2 else "OpenVoice V1", "开始加载")

        from openvoice.api import BaseSpeakerTTS, ToneColorConverter
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # V1版本路径（TTS模型）
        ckpt_base_en = os.path.join(ALGORITHM_PATHS['openvoice'], "checkpoints_v1", "checkpoints",
                                    "base_speakers", "EN")
        ckpt_base_zh = os.path.join(ALGORITHM_PATHS['openvoice'], "checkpoints_v1", "checkpoints",
                                    "base_speakers", "ZH")

        if use_v2:
            # V2版本：使用V1的TTS模型 + V2的Converter + V2的音色嵌入
            ckpt_converter = os.path.join(ALGORITHM_PATHS['openvoice'], "checkpoints_v2", "checkpoints_v2",
                                          "converter")
            ckpt_v2_speakers = os.path.join(ALGORITHM_PATHS['openvoice'], "checkpoints_v2", "checkpoints_v2",
                                            "base_speakers")
            system_logger.info(f"【模型加载】OpenVoice 使用V2版本（V1 TTS + V2 Converter）")
        else:
            # V1版本
            ckpt_converter = os.path.join(ALGORITHM_PATHS['openvoice'], "checkpoints_v1", "checkpoints",
                                          "converter")
            ckpt_v2_speakers = None
            system_logger.info(f"【模型加载】OpenVoice 使用V1版本")

        # 只加载中文TTS模型（本项目只合成中文）
        system_logger.info(f"【模型加载】加载中文TTS模型: {ckpt_base_zh}")
        tts_zh = BaseSpeakerTTS(f'{ckpt_base_zh}/config.json', device=device)
        tts_zh.load_ckpt(f'{ckpt_base_zh}/checkpoint.pth')

        # 加载音色转换器
        tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
        tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

        # 加载音色嵌入
        source_se = {}
        if use_v2 and ckpt_v2_speakers and os.path.exists(f'{ckpt_v2_speakers}/ses/zh.pth'):
            # V2版本音色嵌入
            source_se['en'] = torch.load(f'{ckpt_v2_speakers}/ses/en-default.pth').to(device)
            source_se['zh'] = torch.load(f'{ckpt_v2_speakers}/ses/zh.pth').to(device)
            system_logger.info(f"【模型加载】OpenVoice V2 音色嵌入加载成功")
        elif os.path.exists(f'{ckpt_base_en}/en_default_se.pth'):
            # V1版本音色嵌入
            source_se['en'] = torch.load(f'{ckpt_base_en}/en_default_se.pth').to(device)
            source_se['zh'] = torch.load(f'{ckpt_base_zh}/zh_default_se.pth').to(device)
        elif os.path.exists(f'{ckpt_base_en}/ses/en-default.pth'):
            source_se['en'] = torch.load(f'{ckpt_base_en}/ses/en-default.pth').to(device)
            source_se['zh'] = torch.load(f'{ckpt_base_zh}/ses/zh.pth').to(device)

        models[model_key] = {
            "tts": tts_zh,
            "converter": tone_color_converter,
            "source_se": source_se,
            "device": device,
            "ckpt_base_zh": ckpt_base_zh,
            "version": "v2" if use_v2 else "v1"
        }

        duration = time.time() - start_time
        gpu_mem = torch.cuda.memory_allocated() / 1024 ** 3 if torch.cuda.is_available() else 0
        OperationLogger.log_model_load("OpenVoice V2" if use_v2 else "OpenVoice V1", "成功", duration,
                                       f"GPU内存: {gpu_mem:.2f}GB")
        OperationLogger.log_performance("OpenVoice加载", duration, 0, gpu_mem)

    return models[model_key]
