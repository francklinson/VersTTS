#!/usr/bin/env python3
"""
transformers 兼容层
用于在 transformers 4.57.3 和 5.x 之间提供兼容性
"""

import sys
import transformers
from packaging import version

TRANSFORMERS_VERSION = version.parse(transformers.__version__)

if TRANSFORMERS_VERSION < version.parse("5.0.0"):
    # transformers 4.x 兼容性处理
    
    # 1. HiggsAudioV2TokenizerModel 占位类
    class HiggsAudioV2TokenizerModel:
        """
        占位类，用于兼容 transformers 4.x
        实际 tokenization 功能由 HiggsConfig 和模型自身处理
        """
        pass
    
    transformers.HiggsAudioV2TokenizerModel = HiggsAudioV2TokenizerModel
    
    # 2. 修复 PretrainedConfig 的问题
    # transformers 5.x 中 CONFIG_MAPPING 的行为可能不同
    original_auto_config = getattr(transformers.AutoConfig, 'from_pretrained', None)
    
    # 3. 确保 AutoConfig 能识别 qwen3_tts 类型
    if hasattr(transformers, 'CONFIG_MAPPING'):
        # 注册 qwen3_tts 配置（如果未注册）
        if 'qwen3_tts' not in transformers.CONFIG_MAPPING:
            # 尝试从已安装的 qwen_tts 包导入配置
            try:
                from qwen_tts.core.models.configuration_qwen3_tts import (
                    Qwen3TTSTalkerConfig, 
                    Qwen3TTSCodecConfig, 
                    Qwen3TTSConfig
                )
                transformers.CONFIG_MAPPING.register("qwen3_tts", Qwen3TTSConfig, exist_ok=True)
                print(f"[兼容层] 已注册 qwen3_tts 配置类")
            except ImportError:
                print(f"[兼容层] 无法导入 qwen_tts 配置，Qwen3-TTS 可能需要手动加载")
    
    print(f"[兼容层] transformers {transformers.__version__} 兼容层加载完成")
else:
    print(f"[兼容层] transformers {transformers.__version__} 已原生支持所需功能")
