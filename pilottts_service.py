#!/usr/bin/env python3
"""
PilotTTS 独立服务
使用 transformers 4.51.3（lib/transformers4），运行在独立端口上

参照 CosyVoice 独立服务模式：
  - sys.path.insert(0, "lib/transformers4") 确保加载正确版本的 transformers
  - 独立进程运行，与主服务通过 HTTP 通信
  - 启动方式: nohup python pilottts_service.py > logs/pilottts_service.log 2>&1 &

日志文件: logs/pilottts_service.log
"""

import sys
import os

# === 关键：在任何模块导入之前，设置 transformers 4.51.3 路径 ===
# PilotTTS 依赖 transformers>=4.40.0,<=4.52.4，全局 4.57.3 不兼容
TRANSFORMERS4_PATH = os.path.join(os.path.dirname(__file__), "lib", "transformers4")
sys.path.insert(0, TRANSFORMERS4_PATH)

import time
import traceback
import torch
import torchaudio
import uvicorn
from fastapi import FastAPI, Form, HTTPException
from contextlib import asynccontextmanager
from typing import Optional
import logging
from logging.handlers import RotatingFileHandler

# ========== 日志配置 ==========
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(PROJECT_ROOT, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

PILOTTS_LOG = os.path.join(LOG_DIR, 'pilottts_service.log')

DETAILED_FORMATTER = logging.Formatter(
    '%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger("pilottts_service")
logger.setLevel(logging.INFO)
logger.handlers = []

file_handler = RotatingFileHandler(
    PILOTTS_LOG,
    maxBytes=10 * 1024 * 1024,
    backupCount=3,
    encoding='utf-8'
)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(DETAILED_FORMATTER)
logger.addHandler(file_handler)
logger.propagate = False

# ========== 路径初始化 ==========
PILOTTS_DIR = os.path.join(PROJECT_ROOT, "algorithms", "PilotTTS")
PRETRAINED_DIR = os.path.join(PROJECT_ROOT, "models", "PilotTTS")

# 添加 PilotTTS 及其依赖路径
for p in [
    PILOTTS_DIR,
    os.path.join(PILOTTS_DIR, "third_party"),
    os.path.join(PILOTTS_DIR, "third_party", "Matcha-TTS"),
]:
    if p not in sys.path:
        sys.path.insert(0, p)

# 修复 CUDA_VISIBLE_DEVICES 空字符串问题
if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip() == "":
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        del os.environ["CUDA_VISIBLE_DEVICES"]

# ========== 全局引擎缓存 ==========
_engines = {}  # {"base": (engine, config, device), "instruct": (engine, config, device)}

# 支持的情感标签
EMOTION_LABELS = [
    "happy", "sad", "angry", "surprise", "fear", "disgust",
    "serious", "concern", "blue", "disdain", "neutral", "psychology", "unknown"
]


def _load_engine(model_type: str):
    """加载 PilotTTS 引擎，带重试机制"""
    if model_type in _engines:
        return _engines[model_type]

    # 显存限制：base ↔ instruct 互斥加载
    other_type = "instruct" if model_type == "base" else "base"
    if other_type in _engines:
        logger.info(f"【引擎加载】显存限制，释放 {other_type} 模型以加载 {model_type}")
        del _engines[other_type]
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    import yaml
    from pilot_voice.engine import InferenceEngine

    if model_type == "instruct":
        config_path = os.path.join(PILOTTS_DIR, "configs", "infer_pilot_tts_instruct.yaml")
        checkpoint = os.path.join(PRETRAINED_DIR, "pilot_tts_instruct.pt")
    else:
        config_path = os.path.join(PILOTTS_DIR, "configs", "infer_pilot_tts.yaml")
        checkpoint = os.path.join(PRETRAINED_DIR, "pilot_tts.pt")

    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"PilotTTS 模型文件不存在: {checkpoint}")

    with open(config_path) as f:
        config = yaml.safe_load(f)
    config["checkpoint_path"] = checkpoint

    # 修正模型路径为绝对路径
    for key in ["model", "vocoder"]:
        if key in config and isinstance(config[key], dict):
            for subkey in config[key]:
                val = config[key][subkey]
                if isinstance(val, str) and val.startswith("pretrained_models/"):
                    abs_path = os.path.join(PRETRAINED_DIR, val.replace("pretrained_models/", ""))
                    if os.path.exists(abs_path):
                        config[key][subkey] = abs_path

    # 修正 tokenizer 路径
    tokenizer_path = config.get("tokenizer", {}).get("path", "")
    if tokenizer_path and not os.path.isabs(tokenizer_path):
        abs_tokenizer = os.path.join(PILOTTS_DIR, tokenizer_path)
        if os.path.exists(abs_tokenizer):
            config["tokenizer"]["path"] = abs_tokenizer

    # 修正 campplus 路径
    campplus_path = config.get("spk_embedding", {}).get("campplus_path", "")
    if campplus_path and not os.path.isabs(campplus_path):
        if campplus_path.startswith("pretrained_models/"):
            abs_campplus = os.path.join(PRETRAINED_DIR, campplus_path.replace("pretrained_models/", ""))
        else:
            abs_campplus = os.path.join(PILOTTS_DIR, campplus_path)
        if os.path.exists(abs_campplus):
            config["spk_embedding"]["campplus_path"] = abs_campplus

    # 加载模型（带重试）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_retries = 3
    for attempt in range(max_retries):
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                logger.info(f"【引擎加载】尝试 {attempt + 1}/{max_retries} | 类型: {model_type}")

            start_time = time.time()
            engine = InferenceEngine(config, device)
            duration = time.time() - start_time

            _engines[model_type] = (engine, config, device)
            gpu_mem = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
            logger.info(f"【引擎加载】成功 | 类型: {model_type} | 耗时: {duration:.2f}s | GPU: {gpu_mem:.2f}GB")
            return _engines[model_type]

        except RuntimeError as e:
            if "CUDA" in str(e) and attempt < max_retries - 1:
                logger.warning(f"【引擎加载】GPU 繁忙，等待 5 秒后重试... ({attempt + 1}/{max_retries})")
                time.sleep(5)
            else:
                logger.error(f"【引擎加载】失败: {str(e)}")
                raise

    raise RuntimeError(f"模型加载失败，已重试 {max_retries} 次")


def _get_engine_for_mode(mode: str):
    """根据合成模式确定使用的模型类型"""
    if mode in ["emotion", "dialect", "paralanguage"]:
        return "instruct"
    return "base"


# ========== FastAPI 应用 ==========

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    logger.info("=" * 60)
    logger.info("【PilotTTS服务】启动中...")
    logger.info(f"【日志文件】{PILOTTS_LOG}")
    logger.info(f"【transformers版本】{TRANSFORMERS4_PATH}")
    logger.info("=" * 60)

    preload_enabled = os.environ.get('PRELOAD_PILOTTS', '0') == '1'

    if preload_enabled:
        try:
            _load_engine("base")
            logger.info("【预加载】Base 模型加载完成")
        except Exception as e:
            logger.warning(f"【预加载】Base 模型加载失败: {e}")
    else:
        logger.info("【PilotTTS服务】预加载已禁用，模型将在首次请求时加载")

    yield

    # 清理
    for model_type in list(_engines.keys()):
        logger.info(f"【清理】卸载 {model_type} 模型...")
        del _engines[model_type]
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("【PilotTTS服务】已停止")


app = FastAPI(title="PilotTTS 独立服务", lifespan=lifespan)


@app.get("/health")
async def health():
    """健康检查"""
    return {
        "status": "ok",
        "base_loaded": "base" in _engines,
        "instruct_loaded": "instruct" in _engines,
    }


@app.post("/tts")
async def tts(
    text: str = Form(...),
    mode: str = Form("voice_clone"),
    ref_path: str = Form(...),
    emotion: Optional[str] = Form(None),
    language: str = Form("zh"),
):
    """
    PilotTTS 语音合成

    支持模式:
    - voice_clone: 零样本声音克隆（基础模型）
    - emotion: 情感合成（指令模型）
    - dialect: 方言合成（指令模型）
    - paralanguage: 副语言合成（指令模型）
    """
    start_time = time.time()
    logger.info(f"【TTS请求】模式: {mode} | 文本: {text[:80]}... | 参考音频: {ref_path}")

    try:
        # 参数验证
        if not text or not text.strip():
            raise HTTPException(status_code=400, detail="文本不能为空")

        if not ref_path or not os.path.exists(ref_path):
            raise HTTPException(status_code=400, detail=f"参考音频不存在: {ref_path}")

        if mode == "emotion" and emotion and emotion not in EMOTION_LABELS:
            raise HTTPException(status_code=400, detail=f"不支持的情感标签: {emotion}")

        # 确定模型类型并加载
        model_type = _get_engine_for_mode(mode)
        logger.info(f"【TTS】加载引擎: {model_type}")
        engine, config, device = _load_engine(model_type)

        # 构建合成文本
        synth_text = text
        if mode == "emotion" and emotion:
            synth_text = f"<|{emotion}|>{text}<|/{emotion}|>"
            logger.info(f"【TTS】情感标签: {emotion}")

        if mode == "dialect" and language == "zh":
            language = "zh-henan"
            logger.info(f"【TTS】方言默认为河南话")

        # 生成音频
        gen_start = time.time()
        codes, speech = engine.synthesize(ref_path, synth_text, language=language)
        gen_duration = time.time() - gen_start
        logger.info(f"【TTS】生成完成 | 耗时: {gen_duration:.3f}s")

        # 获取采样率
        sample_rate = config.get("vocoder", {}).get("sample_rate", 24000)

        # 保存临时文件
        timestamp = int(time.time() * 1000)
        output_path = f"/tmp/pilottts_{timestamp}.wav"
        torchaudio.save(output_path, speech.cpu(), sample_rate=sample_rate)

        # 清理显存
        del speech, codes
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        total_duration = time.time() - start_time
        logger.info(f"【TTS完成】音频: {output_path} | 总耗时: {total_duration:.3f}s")

        return {
            "success": True,
            "audio_path": output_path,
            "sample_rate": sample_rate,
        }

    except HTTPException:
        raise
    except Exception as e:
        total_duration = time.time() - start_time
        logger.error(f"【TTS错误】异常类型: {type(e).__name__} | 错误: {str(e)} | 耗时: {total_duration:.3f}s")
        logger.error(f"【错误堆栈】\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {str(e)}")


if __name__ == "__main__":
    port = int(os.environ.get("PILOTTS_PORT", 8003))
    host = os.environ.get("PILOTTS_HOST", "127.0.0.1")
    logger.info(f"【服务启动】地址: {host}:{port}")
    uvicorn.run(app, host=host, port=port)
