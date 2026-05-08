#!/usr/bin/env python3
"""
transformers 版本隔离管理
用于 OmniVoice 使用独立的高版本 transformers (5.6.2)
"""

import sys
import os

# 独立的 transformers 路径
TRANSFORMERS5_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "lib", "transformers5")

def activate_transformers5():
    """激活 transformers 5.x 版本（用于 OmniVoice）"""
    # 移除旧版本 transformers 的路径
    sys.path = [p for p in sys.path if 'transformers' not in p or TRANSFORMERS5_PATH in p]
    
    # 将 transformers5 路径插入到最前面
    if TRANSFORMERS5_PATH not in sys.path:
        sys.path.insert(0, TRANSFORMERS5_PATH)
    
    # 清除已加载的 transformers 模块缓存
    modules_to_remove = [k for k in sys.modules.keys() if k.startswith('transformers')]
    for mod in modules_to_remove:
        del sys.modules[mod]
    
    # 验证版本
    import transformers
    print(f"[OmniVoice兼容] transformers 版本: {transformers.__version__} (隔离模式)")
    return transformers

def restore_transformers4():
    """恢复 transformers 4.57.3 版本（用于 Qwen3-TTS 等）"""
    # 移除 transformers5 路径
    sys.path = [p for p in sys.path if TRANSFORMERS5_PATH not in p]
    
    # 清除 transformers 模块缓存
    modules_to_remove = [k for k in sys.modules.keys() if k.startswith('transformers')]
    for mod in modules_to_remove:
        del sys.modules[mod]
    
    # 重新导入 transformers 4.57.3
    import transformers
    print(f"[恢复] transformers 版本: {transformers.__version__}")
    return transformers
