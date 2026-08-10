#!/usr/bin/env python3
"""
VersTTS 全局配置
所有配置项可通过环境变量覆盖（由 start_server.sh 导出）
"""

import os
import sys

# ========== 服务配置 ==========
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8000"))

# OmniVoice 独立服务配置
# 端口默认值需与 start_server.sh 中的子服务端口保持一致
OMNIVOICE_HOST = os.environ.get("OMNIVOICE_HOST", "127.0.0.1")
OMNIVOICE_PORT = int(os.environ.get("OMNIVOICE_PORT", "8007"))

# CosyVoice 独立服务配置
COSYVOICE_HOST = os.environ.get("COSYVOICE_HOST", "127.0.0.1")
COSYVOICE_PORT = int(os.environ.get("COSYVOICE_PORT", "8008"))

# PilotTTS 独立服务配置
PILOTTS_HOST = os.environ.get("PILOTTS_HOST", "127.0.0.1")
PILOTTS_PORT = int(os.environ.get("PILOTTS_PORT", "8009"))

# GPT-SoVITS 独立服务配置
GPTSOVITS_HOST = os.environ.get("GPTSOVITS_HOST", "127.0.0.1")
GPTSOVITS_PORT = int(os.environ.get("GPTSOVITS_PORT", "8010"))

# Fish-Speech 独立服务配置
FISHSPEECH_HOST = os.environ.get("FISHSPEECH_HOST", "127.0.0.1")
FISHSPEECH_PORT = int(os.environ.get("FISHSPEECH_PORT", "8005"))

# 日志配置
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
LOG_MAX_SIZE = int(os.environ.get("LOG_MAX_SIZE", "50"))
LOG_BACKUP_COUNT = int(os.environ.get("LOG_BACKUP_COUNT", "5"))

# ========== 路径配置 ==========
# 项目根目录（可通过环境变量覆盖）
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_env_path(key, default_relative):
    """获取路径配置：优先使用环境变量，否则使用相对于 PROJECT_ROOT 的路径"""
    env_val = os.environ.get(key)
    if env_val:
        return env_val if os.path.isabs(env_val) else os.path.join(PROJECT_ROOT, env_val)
    return os.path.join(PROJECT_ROOT, default_relative)

# ========== HuggingFace 缓存配置 ==========
# 可通过环境变量 HF_HOME / HUGGINGFACE_HUB_CACHE 覆盖
HF_CACHE_PATH = get_env_path("HF_HOME", "models/hf_cache")
if not os.environ.get("HF_HOME"):
    os.environ['HF_HOME'] = HF_CACHE_PATH
if not os.environ.get("HUGGINGFACE_HUB_CACHE"):
    os.environ['HUGGINGFACE_HUB_CACHE'] = HF_CACHE_PATH
if not os.environ.get("TRANSFORMERS_CACHE"):
    os.environ['TRANSFORMERS_CACHE'] = get_env_path("TRANSFORMERS_CACHE", "models/transformers_cache")

if os.path.exists(os.environ.get('HF_HOME', '')):
    print(f"[配置] 使用 HF 缓存: {os.environ['HF_HOME']}")

# 离线模式
if os.environ.get('TRANSFORMERS_OFFLINE') == '1' or os.environ.get('HF_HUB_OFFLINE') == '1':
    print("[离线部署] 检测到离线模式环境变量，禁用HuggingFace在线访问")
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['HF_DATASETS_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_HUB_DISABLE_DOWNLOADS'] = '1'

# ========== 目录配置 ==========
MODELS_DIR = get_env_path("MODELS_DIR", "models")
OUTPUTS_DIR = get_env_path("OUTPUTS_DIR", "outputs")
LOGS_DIR = get_env_path("LOGS_DIR", "logs")
SPEAKERS_DIR = get_env_path("SPEAKERS_DIR", "speakers")
UPLOADS_DIR = get_env_path("UPLOADS_DIR", "uploads")
ALGORITHMS_DIR = get_env_path("ALGORITHMS_DIR", "algorithms")

SPEAKERS_DB_FILE = os.path.join(SPEAKERS_DIR, "speakers_db.json")

# ========== 音频内容校验配置（需求2）==========
# 新生成音频写入后用 wenet ASR 识别并与输入文本比对相似度，低于阈值则删除。
# 通过环境变量 AUDIO_VERIFY_ENABLED / AUDIO_VERIFY_THRESHOLD 覆盖。
AUDIO_VERIFY_ENABLED = os.environ.get("AUDIO_VERIFY_ENABLED", "1") == "1"
AUDIO_VERIFY_THRESHOLD = float(os.environ.get("AUDIO_VERIFY_THRESHOLD", "0.6"))

# ========== 模型文件路径配置 ==========
MODEL_PATHS = {
    'qwen3tts': os.path.join(MODELS_DIR, 'Qwen3-TTS'),
    'voxcpm': os.path.join(MODELS_DIR, 'VoxCPM'),
    'omnivoice': os.path.join(MODELS_DIR, 'OmniVoice'),
    'fireredtts2': os.path.join(ALGORITHMS_DIR, 'FireRedTTS2', 'pretrained_models'),
    'pilottts': os.path.join(MODELS_DIR, 'PilotTTS'),
    'dotstts': os.path.join(MODELS_DIR, 'dotstts'),
    'fishspeech': os.path.join(MODELS_DIR, 'Fish-Speech'),
    'indextts': os.path.join(MODELS_DIR, 'IndexTTS'),
}

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
    'omnivoice': os.path.join(ALGORITHMS_DIR, 'OmniVoice'),
    'pilottts': os.path.join(ALGORITHMS_DIR, 'PilotTTS'),
    'pilottts_third_party': os.path.join(ALGORITHMS_DIR, 'PilotTTS', 'third_party'),
    'pilottts_matcha': os.path.join(ALGORITHMS_DIR, 'PilotTTS', 'third_party', 'Matcha-TTS'),
    'dotstts': os.path.join(ALGORITHMS_DIR, 'dotstts'),
    'dotstts_src': os.path.join(ALGORITHMS_DIR, 'dotstts', 'src'),
    'fishspeech': os.path.join(ALGORITHMS_DIR, 'Fish-Speech'),
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
    
    f5tts_path = ALGORITHM_PATHS['f5tts']
    if f5tts_path not in sys.path:
        sys.path.insert(0, f5tts_path)

    # PilotTTS 需要 third_party 和 Matcha-TTS 在 sys.path 最前面
    for pp in [ALGORITHM_PATHS.get('pilottts_third_party'), ALGORITHM_PATHS.get('pilottts_matcha')]:
        if pp and pp not in sys.path:
            sys.path.insert(0, pp)


def ensure_directories():
    """确保必要的目录存在"""
    for dir_path in [SPEAKERS_DIR, OUTPUTS_DIR, UPLOADS_DIR, LOGS_DIR, MODELS_DIR]:
        os.makedirs(dir_path, exist_ok=True)
