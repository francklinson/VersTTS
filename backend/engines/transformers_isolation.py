#!/usr/bin/env python3
"""
transformers 版本隔离管理

参照 CosyVoice 独立服务的做法：通过 sys.path.insert(0, ...) 优先加载指定版本的 transformers，
加载完成后恢复默认版本，保证不同 TTS 方案可以使用各自所需的 transformers 版本。

支持三种 transformers 环境：
  - default: 全局 pip 安装的 transformers（当前 4.57.3，供 Qwen3TTS/VoxCPM 等使用）
  - transformers4: lib/transformers4 (4.51.3，供 PilotTTS/CosyVoice 使用)
  - transformers5: lib/transformers5 (5.14.1，供 OmniVoice 使用)
"""

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
TRANSFORMERS4_PATH = os.path.join(PROJECT_ROOT, "lib", "transformers4")
TRANSFORMERS5_PATH = os.path.join(PROJECT_ROOT, "lib", "transformers5")

# 记录默认 transformers 是从哪个路径加载的（用于恢复）
_default_transformers_paths = []


def _clear_transformers_cache():
    """清除所有已缓存的 transformers 模块"""
    modules_to_remove = [k for k in sys.modules.keys() if k.startswith('transformers')]
    for mod in modules_to_remove:
        del sys.modules[mod]


def activate_transformers4():
    """
    激活 transformers 4.51.3（供 PilotTTS 使用）

    参照 CosyVoice 独立服务的做法：将 lib/transformers4 插入 sys.path 最前面，
    清除已缓存的 transformers 模块，确保后续 import 获取 4.51.3 版本。
    """
    if not os.path.isdir(TRANSFORMERS4_PATH):
        raise RuntimeError(f"transformers4 目录不存在: {TRANSFORMERS4_PATH}")

    # 移除其他 transformers 自定义路径，保留系统路径
    sys.path = [p for p in sys.path
                if TRANSFORMERS4_PATH not in p and TRANSFORMERS5_PATH not in p]

    # 插入 transformers4 到最前面（最高优先级）
    sys.path.insert(0, TRANSFORMERS4_PATH)

    # 清除已缓存的 transformers 模块
    _clear_transformers_cache()

    # 验证版本
    import transformers
    print(f"[PilotTTS兼容] transformers 版本: {transformers.__version__} (隔离模式)")
    return transformers


def activate_transformers5():
    """
    激活 transformers 5.x 版本（供 OmniVoice 使用）

    将 lib/transformers5 插入 sys.path 最前面，
    清除已缓存的 transformers 模块。
    """
    if not os.path.isdir(TRANSFORMERS5_PATH):
        raise RuntimeError(f"transformers5 目录不存在: {TRANSFORMERS5_PATH}")

    # 移除其他 transformers 自定义路径
    sys.path = [p for p in sys.path
                if TRANSFORMERS4_PATH not in p and TRANSFORMERS5_PATH not in p]

    # 插入 transformers5 到最前面
    sys.path.insert(0, TRANSFORMERS5_PATH)

    # 清除已缓存的 transformers 模块
    _clear_transformers_cache()

    # 验证版本
    import transformers
    print(f"[OmniVoice兼容] transformers 版本: {transformers.__version__} (隔离模式)")
    return transformers


def restore_default_transformers():
    """
    恢复默认的 transformers 版本（全局 pip 安装版本）

    移除所有自定义 transformers 路径，清除缓存，重新加载默认版本。
    """
    # 移除所有自定义 transformers 路径
    sys.path = [p for p in sys.path
                if TRANSFORMERS4_PATH not in p and TRANSFORMERS5_PATH not in p]

    # 清除已缓存的 transformers 模块
    _clear_transformers_cache()

    # 重新导入默认版本
    import transformers
    print(f"[恢复] transformers 版本: {transformers.__version__}")
    return transformers


# 向后兼容别名
restore_transformers4 = restore_default_transformers
