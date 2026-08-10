#!/usr/bin/env python3
"""
Fish-Speech 独立服务
运行在独立端口 8005 上
启动方式: nohup python fishspeech_service.py > logs/fishspeech_service.log 2>&1 &

Fish-Speech 是基于 Dual-AR 架构的多语言 TTS 模型，支持 80+ 种语言。
日志文件: logs/fishspeech.log
"""

import sys
import os

import time
import traceback
import asyncio
import torch
import numpy as np
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
        from fish_speech.models.text2semantic.inference import (
            init_model,
            load_codec_model,
        )

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

        # 设置设备和精度
        device = "cuda" if torch.cuda.is_available() else "cpu"
        precision = torch.bfloat16

        logger.info(f"【Fish-Speech服务】加载 Dual-AR 文本→语义模型...")
        t0 = time.time()
        fs_model, decode_one_token = init_model(MODEL_PATH, device, precision)
        logger.info(f"【Fish-Speech服务】Dual-AR 模型加载耗时: {time.time() - t0:.02f}s")

        # 初始化 KV 缓存
        logger.info(f"【Fish-Speech服务】初始化 KV 缓存...")
        with torch.device(device):
            fs_model.setup_caches(
                max_batch_size=1,
                max_seq_len=fs_model.config.max_seq_len,
                dtype=next(fs_model.parameters()).dtype,
            )

        # 加载 DAC 编解码器
        logger.info(f"【Fish-Speech服务】加载 DAC 编解码器...")
        codec_path = os.path.join(MODEL_PATH, "codec.pth")
        codec = load_codec_model(codec_path, device, precision)
        logger.info(f"【Fish-Speech服务】DAC 编解码器加载完成, 采样率: {codec.sample_rate}")

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            mem_gb = torch.cuda.memory_allocated() / 1024**3
            logger.info(f"【Fish-Speech服务】GPU 显存占用: {mem_gb:.02f} GB")

        model = {
            "model": fs_model,
            "decode_one_token": decode_one_token,
            "codec": codec,
            "device": device,
            "precision": precision,
            "sample_rate": codec.sample_rate,
            "status": "loaded",
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
                   top_k: int = 30,
                   repetition_penalty: float = 1.1, output_path: str = None) -> str:
    """
    使用 Fish-Speech 模型进行语音合成

    Args:
        text: 要合成的文本
        ref_audio_path: 参考音频路径（零样本克隆）
        ref_text: 参考音频的对应文本
        temperature: 采样温度
        top_p: nucleus 采样
        top_k: top-k 采样
        repetition_penalty: 重复惩罚
        output_path: 输出音频路径

    Returns:
        输出音频文件路径
    """
    global model, last_used_time

    if model is None:
        raise RuntimeError("Fish-Speech 模型未加载")

    fs_model = model["model"]
    decode_one_token = model["decode_one_token"]
    codec = model["codec"]
    device = model["device"]
    sample_rate = model["sample_rate"]

    if output_path is None:
        import tempfile
        output_path = tempfile.mktemp(suffix=".wav")

    logger.info(f"【Fish-Speech服务】开始合成 | 文本: {text[:50]}...")
    logger.info(f"【Fish-Speech服务】参数: temp={temperature}, top_p={top_p}, top_k={top_k}, rep_penalty={repetition_penalty}")

    try:
        from fish_speech.models.text2semantic.inference import (
            generate_long,
            encode_audio,
            decode_to_audio,
        )

        # 如果提供了参考音频，编码为 prompt_tokens
        prompt_tokens = None
        prompt_text = None
        if ref_audio_path and os.path.exists(ref_audio_path):
            logger.info(f"【Fish-Speech服务】编码参考音频: {ref_audio_path}")
            prompt_tokens = encode_audio(ref_audio_path, codec, device)
            prompt_text = ref_text if ref_text else ""
            logger.info(f"【Fish-Speech服务】参考音频编码完成, shape: {prompt_tokens.shape}")

        # 调用 generate_long 进行 Dual-AR 生成
        logger.info(f"【Fish-Speech服务】开始 Dual-AR 生成...")
        t0 = time.time()

        generator = generate_long(
            model=fs_model,
            device=device,
            decode_one_token=decode_one_token,
            text=text,
            num_samples=1,
            max_new_tokens=0,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            iterative_prompt=True,
            chunk_length=512,
            prompt_text=prompt_text,
            prompt_tokens=prompt_tokens.cpu() if prompt_tokens is not None else None,
        )

        # 收集所有 sample 响应的 codes
        all_codes = []
        for response in generator:
            if response.action == "sample" and response.codes is not None:
                all_codes.append(response.codes)
                logger.info(f"【Fish-Speech服务】收到片段, codes shape: {response.codes.shape}, 文本: {response.text[:30] if response.text else 'N/A'}...")
            elif response.action == "next":
                break

        if not all_codes:
            raise RuntimeError("Fish-Speech 生成失败：未产生任何音频 tokens")

        # 合并所有片段的 codes
        merged_codes = torch.cat(all_codes, dim=1)
        logger.info(f"【Fish-Speech服务】生成完成, 总 codes shape: {merged_codes.shape}, 耗时: {time.time() - t0:.02f}s")

        # 通过 DAC 解码为音频波形
        logger.info(f"【Fish-Speech服务】DAC 解码音频...")
        audio = decode_to_audio(merged_codes.to(device), codec)
        audio_np = audio.cpu().float().numpy()

        # 保存音频文件
        sf.write(output_path, audio_np, sample_rate)

        last_used_time = time.time()
        duration = len(audio_np) / sample_rate
        logger.info(f"【Fish-Speech服务】合成完成: {output_path}, 时长: {duration:.2f}s")
        return output_path

    except Exception as e:
        logger.error(f"【Fish-Speech服务】合成失败: {e}")
        logger.error(traceback.format_exc())
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
            "sample_rate": model["sample_rate"] if model else 44100,
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
