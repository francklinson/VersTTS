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
import asyncio
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

last_used_time = None  # 最后使用时间
_gpu_id = None  # 当前服务使用的 GPU ID
_main_service_url = None  # 主服务地址，用于 OOM 驱逐

# ========== 空闲超时配置 ==========
IDLE_TIMEOUT = int(os.environ.get("IDLE_TIMEOUT", "300"))  # 默认 5 分钟
HEARTBEAT_INTERVAL = int(os.environ.get("HEARTBEAT_INTERVAL", "60"))  # 心跳间隔秒
_idle_check_task = None

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
            _touch_last_used()
            gpu_mem = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
            logger.info(f"【引擎加载】成功 | 类型: {model_type} | 耗时: {duration:.2f}s | GPU: {gpu_mem:.2f}GB")
            return _engines[model_type]

        except RuntimeError as e:
            if ("CUDA" in str(e) or "out of memory" in str(e).lower()) and attempt < max_retries - 1:
                logger.warning(f"【引擎加载】GPU OOM，尝试驱逐其他模型... ({attempt + 1}/{max_retries})")
                _request_eviction_from_main_service(needed_mb=3000)
                time.sleep(3)
            else:
                logger.error(f"【引擎加载】失败: {str(e)}")
                raise

    raise RuntimeError(f"模型加载失败，已重试 {max_retries} 次")


def _get_engine_for_mode(mode: str):
    """根据合成模式确定使用的模型类型"""
    if mode in ["emotion", "dialect", "paralanguage"]:
        return "instruct"
    return "base"


def unload_all_models():
    """卸载所有模型，释放显存"""
    global last_used_time
    if not _engines:
        return
    logger.info(f"【模型卸载】正在卸载所有模型 (当前: {list(_engines.keys())})...")
    _engines.clear()
    last_used_time = None
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("【模型卸载】GPU缓存已清理")


def _touch_last_used():
    """更新最后使用时间"""
    global last_used_time
    last_used_time = time.time()


async def _idle_check_loop():
    """后台定时检查空闲超时"""
    while True:
        await asyncio.sleep(HEARTBEAT_INTERVAL)
        if _engines and last_used_time is not None:
            idle_seconds = time.time() - last_used_time
            if idle_seconds > IDLE_TIMEOUT:
                logger.info(f"【空闲超时】模型已空闲 {idle_seconds:.0f}s > {IDLE_TIMEOUT}s，自动卸载")
                unload_all_models()
        # 心跳上报
        _heartbeat()


def _register_to_main_service():
    """向主服务注册"""
    try:
        import requests
        port = int(os.environ.get("PILOTTS_PORT", "8003"))
        host = os.environ.get("PILOTTS_HOST", "127.0.0.1")
        requests.post(
            f"{_main_service_url}/services/register",
            json={
                "service_id": "pilottts",
                "port": port,
                "host": host,
                "gpu_id": _gpu_id,
            },
            timeout=5,
            verify=False,
        )
        logger.info(f"【服务注册】已注册到主服务 (GPU: {_gpu_id})")
        return True
    except Exception as e:
        logger.warning(f"【服务注册】注册失败: {e}")
        return False


def _unregister_from_main_service():
    """从主服务注销"""
    try:
        import requests
        requests.post(
            f"{_main_service_url}/services/unregister",
            json={"service_id": "pilottts"},
            timeout=5,
            verify=False,
        )
        logger.info("【服务注销】已从主服务注销")
    except Exception as e:
        logger.warning(f"【服务注销】注销失败: {e}")


def _heartbeat():
    """向主服务上报心跳"""
    try:
        import requests
        vram_mb = 0
        if _engines and torch.cuda.is_available():
            vram_mb = torch.cuda.memory_allocated() // (1024 * 1024)
        requests.post(
            f"{_main_service_url}/services/heartbeat",
            json={
                "service_id": "pilottts",
                "model_loaded": "base" in _engines or "instruct" in _engines,
                "vram_used_mb": vram_mb,
                "last_used_time": last_used_time,
                "gpu_id": _gpu_id,
                "host": os.environ.get("PILOTTS_HOST", "127.0.0.1"),
                "port": int(os.environ.get("PILOTTS_PORT", "8003")),
            },
            timeout=5,
            verify=False,
        )
    except Exception:
        pass  # 心跳失败不影响服务


def _request_eviction_from_main_service(needed_mb: int):
    """OOM 时请求主服务驱逐同 GPU 上其他模型"""
    try:
        import requests
        logger.info(f"【OOM驱逐】请求主服务释放 {needed_mb}MB 显存 (GPU: {_gpu_id})")
        resp = requests.post(
            f"{_main_service_url}/services/evict",
            json={"gpu_id": _gpu_id, "exclude_service": "pilottts", "needed_mb": needed_mb},
            timeout=30,
            verify=False,
        )
        if resp.status_code == 200:
            result = resp.json()
            evicted = result.get("evicted_services", [])
            logger.info(f"【OOM驱逐】主服务已驱逐: {evicted}")
            return True
        else:
            logger.warning(f"【OOM驱逐】主服务返回: {resp.status_code}")
            return False
    except Exception as e:
        logger.warning(f"【OOM驱逐】请求失败: {e}")
        return False


# ========== FastAPI 应用 ==========

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global _idle_check_task, _gpu_id, _main_service_url

    logger.info("=" * 60)
    logger.info("【PilotTTS服务】启动中...")
    logger.info(f"【日志文件】{PILOTTS_LOG}")
    logger.info(f"【transformers版本】{TRANSFORMERS4_PATH}")
    logger.info(f"【空闲超时】{IDLE_TIMEOUT}s | 【心跳间隔】{HEARTBEAT_INTERVAL}s")
    logger.info("=" * 60)

    # 获取 GPU ID
    _gpu_id = os.environ.get("GPU_ID", "0")
    # 主服务地址
    main_host = os.environ.get("MAIN_HOST", "127.0.0.1")
    main_port = os.environ.get("MAIN_PORT", "8000")
    main_scheme = os.environ.get("MAIN_SCHEME", "https")
    _main_service_url = f"{main_scheme}://{main_host}:{main_port}"

    # 注册到主服务
    for attempt in range(3):
        if _register_to_main_service():
            break
        await asyncio.sleep(2)

    # 启动空闲检查定时器
    _idle_check_task = asyncio.create_task(_idle_check_loop())

    yield

    # 停止定时器
    if _idle_check_task:
        _idle_check_task.cancel()

    # 卸载模型
    unload_all_models()

    # 从主服务注销
    _unregister_from_main_service()

    logger.info("【PilotTTS服务】已停止")


app = FastAPI(title="PilotTTS 独立服务", lifespan=lifespan)


@app.get("/health")
async def health():
    """健康检查"""
    return {
        "status": "ok",
        "base_loaded": "base" in _engines,
        "instruct_loaded": "instruct" in _engines,
        "last_used_time": last_used_time,
        "idle_timeout": IDLE_TIMEOUT,
        "gpu_id": _gpu_id,
    }


@app.post("/model/load")
async def model_load(model_type: str = Form(...)):
    """手动加载模型"""
    if model_type not in ("base", "instruct"):
        raise HTTPException(status_code=400, detail=f"不支持的模型类型: {model_type}，支持: base, instruct")
    try:
        _load_engine(model_type)
        return {"success": True, "model_type": model_type}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/model/unload")
async def model_unload():
    """手动卸载模型，释放显存"""
    if not _engines:
        return {"success": True, "message": "模型未加载"}
    _engines.clear()
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"success": True, "message": "模型已卸载"}


@app.get("/model/status")
async def model_status():
    """获取模型状态"""
    vram_mb = 0
    if _engines and torch.cuda.is_available():
        vram_mb = torch.cuda.memory_allocated() // (1024 * 1024)
    idle_seconds = time.time() - last_used_time if last_used_time else None
    return {
        "service_id": "pilottts",
        "base_loaded": "base" in _engines,
        "instruct_loaded": "instruct" in _engines,
        "vram_used_mb": vram_mb,
        "last_used_time": last_used_time,
        "idle_seconds": int(idle_seconds) if idle_seconds is not None else None,
        "idle_timeout": IDLE_TIMEOUT,
        "gpu_id": _gpu_id,
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
        _touch_last_used()

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
