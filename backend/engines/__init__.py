#!/usr/bin/env python3
"""
TTS 引擎加载器模块
注：cosyvoice/gptsovits/omnivoice/pilottts 走独立 HTTP 服务，此处仅保留本地 GPU 模型引擎
"""

from .chattts_engine import get_chattts_model
from .f5tts_engine import get_f5tts_model
from .qwen3tts_engine import get_qwen3tts_model
from .openvoice_engine import get_openvoice_models
from .voxcpm_engine import get_voxcpm_model
from .indextts_engine import get_indextts_model
from .fireredtts2_engine import get_fireredtts2_model

__all__ = [
    'get_chattts_model',
    'get_f5tts_model',
    'get_qwen3tts_model',
    'get_openvoice_models',
    'get_voxcpm_model',
    'get_indextts_model',
    'get_fireredtts2_model',
]
