#!/usr/bin/env python3
"""
VersTTS 全局配置
"""

import os
import sys

# ========== 路径配置 ==========
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ========== 离线部署环境变量配置 ==========
# 设置项目内 HuggingFace 缓存目录（优先使用项目内缓存）
HF_CACHE_PATH = os.path.join(PROJECT_ROOT, "models", "hf_cache")
if os.path.exists(HF_CACHE_PATH):
    # 如果项目内有缓存目录，强制使用它
    os.environ['HF_HOME'] = HF_CACHE_PATH
    os.environ['HUGGINGFACE_HUB_CACHE'] = HF_CACHE_PATH
    os.environ['TRANSFORMERS_CACHE'] = os.path.join(PROJECT_ROOT, "models", "transformers_cache")
    print(f"[配置] 使用项目内 HF 缓存: {HF_CACHE_PATH}")

# 检查是否启用离线模式
if os.environ.get('TRANSFORMERS_OFFLINE') == '1' or os.environ.get('HF_HUB_OFFLINE') == '1':
    print("[离线部署] 检测到离线模式环境变量，禁用HuggingFace在线访问")
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['HF_DATASETS_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_HUB_DISABLE_DOWNLOADS'] = '1'

# 打印当前缓存配置
if 'HF_HOME' in os.environ:
    print(f"[配置] HF_HOME: {os.environ['HF_HOME']}")

# ========== 目录配置 ==========
SPEAKERS_DIR = os.path.join(PROJECT_ROOT, "speakers")
SPEAKERS_DB_FILE = os.path.join(SPEAKERS_DIR, "speakers_db.json")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
UPLOADS_DIR = os.path.join(PROJECT_ROOT, "uploads")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
ALGORITHMS_DIR = os.path.join(PROJECT_ROOT, "algorithms")

# ========== 算法路径配置 ==========
ALGORITHM_PATHS = {
    'chattts': os.path.join(ALGORITHMS_DIR, 'ChatTTS'),
    'cosyvoice': os.path.join(ALGORITHMS_DIR, 'CosyVoice'),
    'matchatts': os.path.join(ALGORITHMS_DIR, 'CosyVoice', 'third_party', 'Matcha-TTS'),
    'openvoice': os.path.join(ALGORITHMS_DIR, 'OpenVoice'),
    'qwen3tts': os.path.join(ALGORITHMS_DIR, 'Qwen3-TTS'),
    'gptsovits': os.path.join(ALGORITHMS_DIR, 'GPT-SoVITS'),
    'gptsovits_module': os.path.join(ALGORITHMS_DIR, 'GPT-SoVITS', 'GPT_SoVITS'),
    'f5tts': os.path.join(ALGORITHMS_DIR, 'F5-TTS', 'src'),
    'voxcpm': os.path.join(ALGORITHMS_DIR, 'VoxCPM', 'src'),
    'indextts': os.path.join(ALGORITHMS_DIR, 'IndexTTS'),
    'fireredtts2': os.path.join(ALGORITHMS_DIR, 'FireRedTTS2'),
}

# ========== F5-TTS 默认参考音频 ==========
DEFAULT_F5TTS_REF_ZH = os.path.join(ALGORITHMS_DIR, "F5-TTS", "refs", "default_zh.wav")
DEFAULT_F5TTS_REF_EN = os.path.join(ALGORITHMS_DIR, "F5-TTS", "refs", "default_en.wav")
DEFAULT_F5TTS_TEXT_ZH = "在一无所知的世界里，只有不停地走下去，才会知道出路在哪里。"
DEFAULT_F5TTS_TEXT_EN = "In a world where nothing is known, one must keep walking to find the way out."

# ========== 全局模型缓存 ==========
models = {}


def setup_algorithm_paths():
    """设置算法模块的系统路径"""
    for path in [
        ALGORITHM_PATHS['chattts'],
        ALGORITHM_PATHS['cosyvoice'],
        ALGORITHM_PATHS['matchatts'],
        ALGORITHM_PATHS['openvoice'],
        ALGORITHM_PATHS['qwen3tts'],
        ALGORITHM_PATHS['gptsovits'],
        ALGORITHM_PATHS['gptsovits_module'],
    ]:
        if path not in sys.path:
            sys.path.insert(0, path)
    
    # F5-TTS 路径必须最后插入，避免与 GPT-SoVITS 的 f5_tts 冲突
    f5tts_path = ALGORITHM_PATHS['f5tts']
    if f5tts_path not in sys.path:
        sys.path.insert(0, f5tts_path)


def ensure_directories():
    """确保必要的目录存在"""
    for dir_path in [SPEAKERS_DIR, OUTPUTS_DIR, UPLOADS_DIR, LOGS_DIR]:
        os.makedirs(dir_path, exist_ok=True)
