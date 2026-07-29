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
import threading

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WENET_PATH = os.path.join(PROJECT_ROOT, "lib")

# 本地 wenet 模型目录（wenetspeech）。
# 优先用本地路径直接传给 load_model，可彻底跳过 wenet 默认的 ~/.wenet 缓存查找
# 与 modelscope 联网下载（生产环境离线/SSL 受限时不会卡在下载 0 字节）。
# 路径遵循项目 MODELS_DIR 约定，可被环境变量 MODELS_DIR 覆盖。
try:
    from backend.config import MODELS_DIR
except Exception:  # 兜底：config 不可用时退回 PROJECT_ROOT/models
    MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
WENET_MODEL_DIR = os.path.join(MODELS_DIR, "wenet", "wenetspeech")

# wenet load_model 要求目录内同时存在这三个文件，缺一即视为模型未就绪。
_WENET_REQUIRED_FILES = ("train.yaml", "final.pt", "units.txt")

_asr_model = None
# wenet 模型为全局单例，transcribe 可能非线程安全；
# 用锁串行化所有 ASR 推理调用，保证并发场景（如批量生成多协程并发校验）下不崩。
_asr_lock = threading.Lock()


def _resolve_wenet_model_source():
    """决定传给 load_model 的参数：本地目录优先，否则回退到模型名（走 ~/.wenet 缓存或联网下载）。"""
    if os.path.isdir(WENET_MODEL_DIR) and all(
        os.path.exists(os.path.join(WENET_MODEL_DIR, f)) for f in _WENET_REQUIRED_FILES
    ):
        logger.info(f"【ASR】使用本地模型目录: {WENET_MODEL_DIR}")
        return WENET_MODEL_DIR
    logger.warning(
        f"【ASR】本地模型目录不存在或不完整（期望 {WENET_MODEL_DIR} 含 "
        f"{list(_WENET_REQUIRED_FILES)}），回退到 wenetspeech 默认加载（可能联网下载）"
    )
    return "wenetspeech"


def _get_asr_model():
    """懒加载 wenet ASR 模型（仅首次调用时加载，常驻内存）"""
    global _asr_model
    if _asr_model is None:
        if WENET_PATH not in sys.path:
            sys.path.insert(0, WENET_PATH)
        from wenet.cli.model import load_model
        model_source = _resolve_wenet_model_source()
        logger.info(f"【ASR】加载 wenetspeech 模型（独立服务）... source={model_source}")
        _asr_model = load_model(model_source, device="cpu")
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
        # 串行化推理：wenet 单例可能非线程安全，并发 transcribe 会状态混乱
        with _asr_lock:
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
