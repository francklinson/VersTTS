#!/usr/bin/env python3
"""
数据模型模块
"""

from .schemas import (
    BaseTTSRequest,
    ChatTTSRequest,
    CosyVoiceRequest,
    F5TTSRequest,
    Qwen3TTSRequest,
    OpenVoiceRequest,
    GPTSoVITSRequest,
    VoxCPMRequest,
    IndexTTSRequest,
    FireRedTTS2Request,
    TTSResponse,
    Qwen3TTSModelStatus,
    BatchTTSRequest,
)

__all__ = [
    'BaseTTSRequest',
    'ChatTTSRequest',
    'CosyVoiceRequest',
    'F5TTSRequest',
    'Qwen3TTSRequest',
    'OpenVoiceRequest',
    'GPTSoVITSRequest',
    'VoxCPMRequest',
    'IndexTTSRequest',
    'FireRedTTS2Request',
    'TTSResponse',
    'Qwen3TTSModelStatus',
    'BatchTTSRequest',
]
