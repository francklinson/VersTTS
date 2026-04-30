#!/usr/bin/env python3
"""
TTS 引擎加载器模块
"""

from .chattts_engine import get_chattts_model
from .cosyvoice_engine import get_cosyvoice_model
from .f5tts_engine import get_f5tts_model
from .qwen3tts_engine import get_qwen3tts_model
from .openvoice_engine import get_openvoice_models
from .gptsovits_engine import get_gpt_sovits_model, init_gpt_sovits_pipeline
from .voxcpm_engine import get_voxcpm_model
from .indextts_engine import get_indextts_model
from .fireredtts2_engine import get_fireredtts2_model

__all__ = [
    'get_chattts_model',
    'get_cosyvoice_model',
    'get_f5tts_model',
    'get_qwen3tts_model',
    'get_openvoice_models',
    'get_gpt_sovits_model',
    'init_gpt_sovits_pipeline',
    'get_voxcpm_model',
    'get_indextts_model',
    'get_fireredtts2_model',
]
