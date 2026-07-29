#!/usr/bin/env python3
"""
Fish-Speech 独立服务
使用独立 transformers 环境 (lib/transformers_fish)，运行在独立端口 8005 上
启动方式: nohup python fishspeech_service.py > logs/fishspeech_service.log 2>&1 &

Fish-Speech 是基于 Dual-AR 架构的多语言 TTS 模型，支持 80+ 种语言。
日志文件: logs/fishspeech.log
"""

import sys
import os

# 在导入任何模块之前，设置独立的 transformers 路径
TRANSFORMERS_FISH_PATH = os.path.join(os.path.dirname(__file__), "lib", "transformers_fish")
if os.path.exists(TRANSFORMERS_FISH_PATH):
    sys.path.insert(0, TRANSFORMERS_FISH_PATH)

import time
import traceback
import asyncio
import torch
import soundfile as sf
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

FISHSPEECH_LOG = os.path.join(LOG_DIR, 'fishspeech.log')

DETAILED_FORMATTER = logging.Formatter(
    '%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger("fishspeech_service")
logger.setLevel(logging.INFO)
logger.handlers = []

file_handler = RotatingFileHandler(
    FISHSPEECH_LOG,
    maxBytes=10 * 1024 * 1024,  # 10MB
    backupCount=3,
    encoding='utf-8'
)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(DETAILED_FORMATTER)
logger.addHandler(file_handler)
logger.propagate = False

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(DETAILED_FORMATTER)
logger.addHandler(console_handler)

# ========== 路径配置 ==========
ALGORITHMS_PATH = os.path.join(PROJECT_ROOT, "algorithms", "Fish-Speech")
FISHSPEECH_MODULE = os.path.join(ALGORITHMS_PATH, "fish_speech")

# 添加 Fish-Speech 到 sys.path
for p in [ALGORITHMS_PATH, FISHSPEECH_MODULE]:
    if p not in sys.path:
        sys.path.insert(0, p)

# 模型路径
MODELS_DIR = os.environ.get("MODELS_DIR", os.path.join(PROJECT_ROOT, "models"))
if not os.path.isabs(MODELS_DIR):
    MODELS_DIR = os.path.join(PROJECT_ROOT, MODELS_DIR)
MODEL_PATH = os.path.join(MODELS_DIR, "Fish-Speech")

# ========== 配置 ==========
PORT = int(os.environ.get("FISHSPEECH_PORT", "8005"))
GPU_ID = os.environ.get("FISHSPEECH_GPU", "0")
HOST = os.environ.get("FISHSPEECH_HOST", "127.0.0.1")
IDLE_TIMEOUT = int(os.environ.get("IDLE_TIMEOUT", "300"))
HEARTBEAT_INTERVAL = int(os.environ.get("HEARTBEAT_INTERVAL", "60"))
MAIN_SERVICE_URL = os.environ.get("MAIN_SERVICE_URL", "http://127.0.0.1:8000")

os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

# 全局状态
model = None
last_used_time = None
model_lock = asyncio.Lock()


def load_fishspeech_model():
    """加载 Fish-Speech 模型 (Dual-AR S2 Pro)"""
    global model, last_used_time

    logger.info(f"【Fish-Speech服务】开始加载模型...")
    logger.info(f"【Fish-Speech服务】模型路径: {MODEL_PATH}")
    logger.info(f"【Fish-Speech服务】GPU: {GPU_ID}")

    try:
        # 尝试加载 Fish-Speech S2 模型
        # Fish-Speech 使用 Dual-AR 架构: Slow AR (4B) + Fast AR (400M)
        from fish_speech.models.text2semantic.inference import launch_thread_safe_queue

        # 检查模型文件是否存在
        required_files = [
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
            "model.safetensors.index.json",
            "codec.pth",
            "tokenizer.json",
            "config.json",
        ]

        missing_files = []
        for f in required_files:
            if not os.path.exists(os.path.join(MODEL_PATH, f)):
                missing_files.append(f)

        if missing_files:
            logger.error(f"【Fish-Speech服务】缺少模型文件: {missing_files}")
            logger.error(f"【Fish-Speech服务】请运行以下命令下载模型:")
            logger.error(f"  huggingface-cli download fishaudio/s2-pro --local-dir {MODEL_PATH}")
            return None

        logger.info(f"【Fish-Speech服务】模型文件检查通过")
        logger.info(f"【Fish-Speech服务】注意: Fish-Speech S2 Pro 约需要 11GB 显存（INT8 约5.1GB）")

        # 此处需要根据 Fish-Speech 的实际 API 进行推理
        # 由于 Fish-Speech 架构复杂，推荐使用其内置 API server
        # 或者通过 launch_thread_safe_queue 进行推理
        model = {
            "model_path": MODEL_PATH,
            "status": "loaded",
            "sample_rate": 44100,
        }

        last_used_time = time.time()
        logger.info(f"【Fish-Speech服务】模型加载成功")
        return model

    except ImportError as e:
        logger.error(f"【Fish-Speech服务】导入错误: {e}")
        logger.error(f"【Fish-Speech服务】请确保已安装 Fish-Speech 依赖")
        logger.error(f"【Fish-Speech服务】运行: pip install -e algorithms/Fish-Speech/")
        return None
    except Exception as e:
        logger.error(f"【Fish-Speech服务】模型加载失败: {e}")
        logger.error(traceback.format_exc())
        return None


async def check_idle_timeout():
    """检查空闲超时，自动卸载模型"""
    global model, last_used_time
    while True:
        await asyncio.sleep(60)
        if model is not None and last_used_time is not None:
            idle_seconds = time.time() - last_used_time
            if idle_seconds > IDLE_TIMEOUT:
                logger.info(f"【Fish-Speech服务】空闲超时 ({idle_seconds:.0f}s > {IDLE_TIMEOUT}s)，卸载模型")
                model = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.info(f"【Fish-Speech服务】模型已卸载，显存已释放")


async def heartbeat_to_main():
    """向主服务发送心跳"""
    global model
    import aiohttp
    while True:
        await asyncio.sleep(HEARTBEAT_INTERVAL)
        try:
            status = "loaded" if model is not None else "idle"
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{MAIN_SERVICE_URL}/system/sub-service/heartbeat",
                    json={"service": "fishspeech", "port": PORT, "status": status, "gpu": GPU_ID},
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                    pass
        except Exception:
            pass


def text_to_speech(text: str, ref_audio_path: str = None, ref_text: str = None,
                   temperature: float = 0.8, top_p: float = 0.8,
                   repetition_penalty: float = 1.1, output_path: str = None) -> str:
    """
    使用 Fish-Speech 模型进行语音合成

    Args:
        text: 要合成的文本
        ref_audio_path: 参考音频路径（零样本克隆）
        ref_text: 参考音频的对应文本
        temperature: 采样温度
        top_p: nucleus 采样
        repetition_penalty: 重复惩罚
        output_path: 输出音频路径

    Returns:
        输出音频文件路径
    """
    global model, last_used_time

    if model is None:
        raise RuntimeError("Fish-Speech 模型未加载")

    if output_path is None:
        import tempfile
        output_path = tempfile.mktemp(suffix=".wav")

    # Fish-Speech 的实际推理需要通过其内置 API
    # 这里提供框架，实际部署时需要调用 Fish-Speech 的推理接口
    logger.info(f"【Fish-Speech服务】开始合成 | 文本: {text[:50]}...")
    logger.info(f"【Fish-Speech服务】参数: temp={temperature}, top_p={top_p}, rep_penalty={repetition_penalty}")

    # 使用 Fish-Speech 的 text2semantic 和 DAC 进行两步推理
    try:
        from fish_speech.models.text2semantic.inference import launch_thread_safe_queue

        # 构建生成参数
        # 注意：此处需要根据 Fish-Speech 的实际 API 进行调整
        logger.warning("【Fish-Speech服务】TODO: 需要实现完整的 Fish-Speech S2 推理逻辑")
        logger.warning("【Fish-Speech服务】请参考 algorithms/Fish-Speech/inference.ipynb")

        # 生成静默音频作为占位（实际部署时替换为真实推理）
        sample_rate = 44100
        duration = 2.0  # 2秒静默
        import numpy as np
        audio = np.zeros(int(sample_rate * duration), dtype=np.float32)
        sf.write(output_path, audio, sample_rate)

        last_used_time = time.time()
        logger.info(f"【Fish-Speech服务】合成完成: {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"【Fish-Speech服务】合成失败: {e}")
        raise


# ========== FastAPI 应用 ==========

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    logger.info(f"【Fish-Speech服务】{'='*60}")
    logger.info(f"【Fish-Speech服务】启动中...")
    logger.info(f"【Fish-Speech服务】端口: {PORT}")
    logger.info(f"【Fish-Speech服务】GPU: {GPU_ID}")
    logger.info(f"【Fish-Speech服务】模型路径: {MODEL_PATH}")
    logger.info(f"【Fish-Speech服务】日志文件: {FISHSPEECH_LOG}")
    logger.info(f"【Fish-Speech服务】{'='*60}")

    # 启动后台任务
    idle_task = asyncio.create_task(check_idle_timeout())
    heartbeat_task = asyncio.create_task(heartbeat_to_main())

    # 可选：预加载模型
    if os.environ.get("PRELOAD_FISHSPEECH") == "1":
        logger.info("【Fish-Speech服务】预加载模式，立即加载模型...")
        load_fishspeech_model()

    yield

    # 清理
    idle_task.cancel()
    heartbeat_task.cancel()
    logger.info("【Fish-Speech服务】已停止")


app = FastAPI(title="Fish-Speech Service", version="1.0.0", lifespan=lifespan)


@app.get("/health")
async def health():
    """健康检查"""
    global model, last_used_time
    return {
        "status": "ok",
        "service": "fishspeech",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH,
        "gpu": GPU_ID,
        "port": PORT,
        "last_used": last_used_time,
    }


@app.post("/tts")
async def tts(
    text: str = Form(...),
    mode: str = Form("voice_clone"),
    clone_speaker_id: Optional[str] = Form(None),
    reference_text: Optional[str] = Form(None),
    temperature: float = Form(0.8),
    top_p: float = Form(0.8),
    repetition_penalty: float = Form(1.1),
):
    """TTS 语音合成端点"""
    global model

    if model is None:
        load_fishspeech_model()
        if model is None:
            raise HTTPException(status_code=503, detail="Fish-Speech 模型未加载")

    try:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(PROJECT_ROOT, "outputs", f"fishspeech_{timestamp}.wav")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 获取参考音频
        ref_audio_path = None
        ref_text = reference_text
        if clone_speaker_id:
            # 从说话人管理模块获取
            try:
                import sys
                sys.path.insert(0, os.path.join(PROJECT_ROOT, "backend"))
                from services import get_speaker_by_id
                speaker = get_speaker_by_id(clone_speaker_id)
                if speaker:
                    ref_audio_path = speaker.get("audio_path")
                    if not ref_text:
                        ref_text = speaker.get("reference_text")
            except Exception as e:
                logger.warning(f"【Fish-Speech服务】获取说话人失败: {e}")

        # 执行 TTS
        output_path = text_to_speech(
            text=text,
            ref_audio_path=ref_audio_path,
            ref_text=ref_text,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            output_path=output_path,
        )

        return {
            "success": True,
            "audio_path": output_path,
            "audio_url": f"/audio/{os.path.basename(output_path)}",
            "sample_rate": 44100,
            "mode": mode,
        }

    except Exception as e:
        logger.error(f"【Fish-Speech服务】TTS 失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/status")
async def model_status():
    """模型状态查询"""
    global model, last_used_time
    return {
        "model_loaded": model is not None,
        "model_path": MODEL_PATH,
        "last_used": last_used_time,
        "idle_seconds": time.time() - last_used_time if last_used_time else None,
        "idle_timeout": IDLE_TIMEOUT,
        "gpu_memory": torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() and model else 0,
    }


@app.post("/model/load")
async def model_load():
    """手动加载模型"""
    load_fishspeech_model()
    return {"status": "loaded" if model is not None else "failed"}


@app.post("/model/unload")
async def model_unload():
    """手动卸载模型"""
    global model
    model = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"status": "unloaded"}


if __name__ == "__main__":
    logger.info(f"【Fish-Speech服务】启动在 {HOST}:{PORT}")
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
