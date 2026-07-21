#!/usr/bin/env python3
"""
独立 ASR 服务模块 — 基于 wenet (wenetspeech) 的语音识别。
不依赖 GPT-SoVITS 服务，可在 main backend 进程中直接调用。

用法:
    from backend.services.asr_service import transcribe
    text = transcribe("/path/to/audio.wav")
"""

import os
import sys
import logging

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WENET_PATH = os.path.join(PROJECT_ROOT, "lib")

_asr_model = None


def _get_asr_model():
    """懒加载 wenet ASR 模型（仅首次调用时加载，常驻内存）"""
    global _asr_model
    if _asr_model is None:
        if WENET_PATH not in sys.path:
            sys.path.insert(0, WENET_PATH)
        from wenet.cli.model import load_model
        logger.info("【ASR】加载 wenetspeech 模型（独立服务）...")
        _asr_model = load_model("wenetspeech", device="cpu")
        logger.info("【ASR】wenetspeech 模型加载完成")
    return _asr_model


def transcribe(audio_path: str) -> str:
    """
    对音频文件进行语音识别，返回识别文本。

    Args:
        audio_path: 音频文件绝对路径

    Returns:
        识别出的文本；失败返回空字符串
    """
    if not os.path.exists(audio_path):
        logger.warning(f"【ASR】音频文件不存在: {audio_path}")
        return ""

    try:
        model = _get_asr_model()
        result = model.transcribe(audio_path)
        text = result.text.strip() if result and result.text else ""
        if text:
            logger.info(f"【ASR】识别结果: '{text}'")
        else:
            logger.warning(f"【ASR】识别结果为空: {audio_path}")
        return text
    except Exception as e:
        logger.warning(f"【ASR】识别失败: {e}")
        return ""


def is_model_loaded() -> bool:
    """检查 ASR 模型是否已加载"""
    return _asr_model is not None
