#!/usr/bin/env python3
"""
CosyVoice 独立服务
使用 transformers 4.51.3，运行在独立端口上
启动方式: nohup python cosyvoice_service.py > logs/cosyvoice_service.log 2>&1 &

日志文件: logs/cosyvoice_service.log
"""

import sys
import os

# 在导入任何模块之前，设置 transformers 4.x 路径
TRANSFORMERS4_PATH = os.path.join(os.path.dirname(__file__), "lib", "transformers4")
sys.path.insert(0, TRANSFORMERS4_PATH)

import time
import traceback
import tempfile
import asyncio
import torch
import soundfile as sf
import uvicorn
from fastapi import FastAPI, Form, HTTPException, UploadFile, File
from contextlib import asynccontextmanager
from typing import Optional
import logging
from logging.handlers import RotatingFileHandler

# ========== 日志配置 ==========
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(PROJECT_ROOT, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

# 日志文件路径
COSYVOICE_LOG = os.path.join(LOG_DIR, 'cosyvoice_service.log')

# 日志格式 - 与主项目保持一致
DETAILED_FORMATTER = logging.Formatter(
    '%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# 设置日志记录器 - 使用独立名称避免冲突
logger = logging.getLogger("cosyvoice_service")
logger.setLevel(logging.INFO)

# 清除已有处理器（避免重复）
logger.handlers = []

# 只添加文件处理器（控制台输出由启动脚本重定向处理）
file_handler = RotatingFileHandler(
    COSYVOICE_LOG,
    maxBytes=10 * 1024 * 1024,  # 10MB
    backupCount=3,
    encoding='utf-8'
)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(DETAILED_FORMATTER)
logger.addHandler(file_handler)

# 阻止日志向父记录器传播（避免重复）
logger.propagate = False

# 添加 CosyVoice 算法路径
ALGORITHMS_PATH = os.path.join(os.path.dirname(__file__), "algorithms", "CosyVoice")
if ALGORITHMS_PATH not in sys.path:
    sys.path.insert(0, ALGORITHMS_PATH)

# 添加 CosyVoice 依赖路径
COSYVOICE_DEPS = [
    os.path.join(ALGORITHMS_PATH, "third_party", "Matcha-TTS"),
]
for dep_path in COSYVOICE_DEPS:
    if dep_path not in sys.path:
        sys.path.insert(0, dep_path)

# 延迟导入 CosyVoice 相关模块
cosyvoice = None
CosyVoice = None

# 全局变量
last_used_time = None  # 最后使用时间
_gpu_id = None  # 当前服务使用的 GPU ID
_main_service_url = None  # 主服务地址，用于 OOM 驱逐
_idle_check_task = None

# ========== 空闲超时配置 ==========
IDLE_TIMEOUT = int(os.environ.get("IDLE_TIMEOUT", "300"))  # 默认 5 分钟
HEARTBEAT_INTERVAL = int(os.environ.get("HEARTBEAT_INTERVAL", "60"))  # 心跳间隔秒


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global _idle_check_task, _gpu_id, _main_service_url

    logger.info("=" * 60)
    logger.info("【CosyVoice服务】启动中...")
    logger.info(f"【日志文件】{COSYVOICE_LOG}")
    logger.info(f"【空闲超时】{IDLE_TIMEOUT}s | 【心跳间隔】{HEARTBEAT_INTERVAL}s")
    logger.info("=" * 60)

    # 获取 GPU ID
    _gpu_id = os.environ.get("GPU_ID", "0")
    # 主服务地址
    main_host = os.environ.get("MAIN_HOST", "127.0.0.1")
    main_port = os.environ.get("MAIN_PORT", "8000")
    _main_service_url = f"http://{main_host}:{main_port}"

    # 注册到主服务
    _register_to_main_service()

    # 启动空闲检查定时器
    _idle_check_task = asyncio.create_task(_idle_check_loop())

    yield

    # 停止定时器
    if _idle_check_task:
        _idle_check_task.cancel()

    # 卸载模型
    unload_model()

    # 从主服务注销
    _unregister_from_main_service()

    logger.info("【CosyVoice服务】已停止")


app = FastAPI(title="CosyVoice 独立服务", lifespan=lifespan)

# 全局模型
model = None

# 模型路径：优先使用环境变量 MODELS_DIR，否则使用相对于脚本的路径
MODELS_DIR = os.environ.get("MODELS_DIR", os.path.join(PROJECT_ROOT, "models"))
MODEL_PATH = os.path.join(MODELS_DIR, "CosyVoice")


def unload_model():
    """卸载模型，释放显存"""
    global model, last_used_time
    if model is None:
        return
    logger.info("【模型卸载】正在卸载模型...")
    model = None
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
        if model is not None and last_used_time is not None:
            idle_seconds = time.time() - last_used_time
            if idle_seconds > IDLE_TIMEOUT:
                logger.info(f"【空闲超时】模型已空闲 {idle_seconds:.0f}s > {IDLE_TIMEOUT}s，自动卸载")
                unload_model()
        # 心跳上报
        _heartbeat()


def _register_to_main_service():
    """向主服务注册"""
    try:
        import requests
        port = int(os.environ.get("COSYVOICE_PORT", "8002"))
        host = os.environ.get("COSYVOICE_HOST", "127.0.0.1")
        requests.post(
            f"{_main_service_url}/api/services/register",
            json={
                "service_id": "cosyvoice",
                "port": port,
                "host": host,
                "gpu_id": _gpu_id,
            },
            timeout=5,
        )
        logger.info(f"【服务注册】已注册到主服务 (GPU: {_gpu_id})")
    except Exception as e:
        logger.warning(f"【服务注册】注册失败: {e}")


def _unregister_from_main_service():
    """从主服务注销"""
    try:
        import requests
        requests.post(
            f"{_main_service_url}/api/services/unregister",
            json={"service_id": "cosyvoice"},
            timeout=5,
        )
        logger.info("【服务注销】已从主服务注销")
    except Exception as e:
        logger.warning(f"【服务注销】注销失败: {e}")


def _heartbeat():
    """向主服务上报心跳"""
    try:
        import requests
        vram_mb = 0
        if model is not None and torch.cuda.is_available():
            vram_mb = torch.cuda.memory_allocated() // (1024 * 1024)
        requests.post(
            f"{_main_service_url}/api/services/heartbeat",
            json={
                "service_id": "cosyvoice",
                "model_loaded": model is not None,
                "vram_used_mb": vram_mb,
                "last_used_time": last_used_time,
                "gpu_id": _gpu_id,
            },
            timeout=5,
        )
    except Exception:
        pass  # 心跳失败不影响服务


def _request_eviction_from_main_service(needed_mb: int):
    """OOM 时请求主服务驱逐同 GPU 上其他模型"""
    try:
        import requests
        logger.info(f"【OOM驱逐】请求主服务释放 {needed_mb}MB 显存 (GPU: {_gpu_id})")
        resp = requests.post(
            f"{_main_service_url}/api/services/evict",
            json={"gpu_id": _gpu_id, "exclude_service": "cosyvoice", "needed_mb": needed_mb},
            timeout=30,
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


def load_model():
    """加载 CosyVoice 模型，带重试机制，OOM 时请求主服务驱逐"""
    global model
    if model is not None:
        _touch_last_used()
        return

    # 延迟导入 - 使用 AutoModel 自动检测模型类型
    from cosyvoice.cli.cosyvoice import AutoModel

    logger.info(f"【模型加载】路径: {MODEL_PATH}")
    start_time = time.time()

    # 模型名称
    model_name = "Fun-CosyVoice3-0.5B"
    local_model_path = os.path.join(MODEL_PATH, model_name)

    # 判断使用本地路径还是 ModelScope repo_id
    if os.path.exists(local_model_path):
        model_path = local_model_path
        logger.info(f"【模型加载】使用本地模型: {model_path}")
    else:
        # 使用 ModelScope repo_id 格式
        model_path = f"FunAudioLLM/{model_name}-2512"
        logger.info(f"【模型加载】本地模型不存在，使用 ModelScope: {model_path}")

    # 尝试加载模型，OOM 时请求驱逐后重试
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # 在加载前清理显存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                logger.info(f"【模型加载】已清理GPU缓存，尝试 {attempt + 1}/{max_retries}")

            # 使用 AutoModel 自动检测模型类型 (CosyVoice/CosyVoice2/CosyVoice3)
            model = AutoModel(model_dir=model_path)
            _touch_last_used()

            duration = time.time() - start_time
            logger.info(f"【模型加载】完成 | 模型类型: {model.__class__.__name__} | 耗时: {duration:.2f}s")
            return

        except RuntimeError as e:
            if ("CUDA" in str(e) or "out of memory" in str(e).lower()) and attempt < max_retries - 1:
                logger.warning(f"【模型加载】GPU OOM，尝试驱逐其他模型... ({attempt + 1}/{max_retries})")
                # 请求主服务驱逐同 GPU 上其他模型
                _request_eviction_from_main_service(needed_mb=3000)
                time.sleep(3)
            else:
                logger.error(f"【模型加载】失败: {str(e)}")
                raise

    raise RuntimeError(f"模型加载失败，已重试 {max_retries} 次")


@app.get("/health")
async def health():
    """健康检查"""
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "last_used_time": last_used_time,
        "idle_timeout": IDLE_TIMEOUT,
        "gpu_id": _gpu_id,
    }


@app.post("/model/load")
async def model_load():
    """手动加载模型"""
    try:
        load_model()
        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/model/unload")
async def model_unload():
    """手动卸载模型，释放显存"""
    if model is None:
        return {"success": True, "message": "模型未加载"}
    unload_model()
    return {"success": True, "message": "模型已卸载"}


@app.get("/model/status")
async def model_status():
    """获取模型状态"""
    vram_mb = 0
    if model is not None and torch.cuda.is_available():
        vram_mb = torch.cuda.memory_allocated() // (1024 * 1024)
    idle_seconds = time.time() - last_used_time if last_used_time else None
    return {
        "service_id": "cosyvoice",
        "model_loaded": model is not None,
        "vram_used_mb": vram_mb,
        "last_used_time": last_used_time,
        "idle_seconds": int(idle_seconds) if idle_seconds is not None else None,
        "idle_timeout": IDLE_TIMEOUT,
        "gpu_id": _gpu_id,
    }


@app.post("/tts")
async def tts(
    text: str = Form(...),
    mode: str = Form("zero_shot"),
    prompt_text: Optional[str] = Form(None),
    instruct_text: Optional[str] = Form(None),
    prompt_wav_path: Optional[str] = Form(None),
    output_format: str = Form("url")
):
    """CosyVoice TTS 合成"""
    load_model()
    _touch_last_used()
    
    start_time = time.time()
    logger.info(f"【TTS请求】模式: {mode} | 文本: {text[:50]}...")

    try:
        if mode == "zero_shot":
            if not prompt_wav_path or not os.path.exists(prompt_wav_path):
                logger.error(f"【TTS错误】参考音频不存在: {prompt_wav_path}")
                raise HTTPException(status_code=400, detail="zero_shot模式需要提供有效的参考音频路径")

            logger.info(f"【Zero-shot】参考音频: {prompt_wav_path} | 参考文本: {prompt_text or '无'}")
            
            if prompt_text:
                formatted_prompt = f"You are a helpful assistant.<|endofprompt|>{prompt_text}"
                model_output = model.inference_zero_shot(text, formatted_prompt, prompt_wav_path, stream=False)
            else:
                formatted_text = f"You are a helpful assistant.<|endofprompt|>{text}"
                model_output = model.inference_cross_lingual(formatted_text, prompt_wav_path, stream=False)

        elif mode == "instruct":
            if not instruct_text:
                raise HTTPException(status_code=400, detail="instruct模式需要提供instruct_text指令文本")

            if not prompt_wav_path or not os.path.exists(prompt_wav_path):
                logger.error(f"【TTS错误】参考音频不存在: {prompt_wav_path}")
                raise HTTPException(status_code=400, detail="instruct模式需要提供有效的参考音频路径")

            formatted_instruct = f"You are a helpful assistant.{instruct_text}<|endofprompt|>"
            logger.info(f"【Instruct模式】指令: {formatted_instruct}")
            model_output = model.inference_instruct2(text, formatted_instruct, prompt_wav_path, stream=False)

        else:
            raise HTTPException(status_code=400, detail=f"不支持的模式: {mode}")

        # 处理 generator 输出
        output_list = list(model_output)
        if not output_list:
            raise HTTPException(status_code=500, detail="模型未返回音频数据")

        # 获取第一个输出结果
        first_output = output_list[0]
        sr = 22050
        audio_data = first_output['tts_speech'].numpy().squeeze()

        # 保存临时文件
        timestamp = int(time.time() * 1000)
        output_path = f"/tmp/cosyvoice_{timestamp}.wav"
        sf.write(output_path, audio_data, samplerate=sr)

        duration = time.time() - start_time
        logger.info(f"【TTS完成】音频路径: {output_path} | 耗时: {duration:.2f}s")

        # 清理显存
        if torch.cuda.is_available():
            del first_output
            del output_list
            torch.cuda.empty_cache()

        # 返回音频文件路径
        return {"success": True, "audio_path": output_path, "sample_rate": sr}

    except HTTPException:
        raise
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"【TTS错误】异常类型: {type(e).__name__} | 错误: {str(e)} | 耗时: {duration:.2f}s")
        logger.error(f"【错误堆栈】\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {str(e)}")


if __name__ == "__main__":
    port = int(os.environ.get("COSYVOICE_PORT", 8002))
    host = os.environ.get("COSYVOICE_HOST", "127.0.0.1")
    logger.info(f"【服务启动】地址: {host}:{port}")
    uvicorn.run(app, host=host, port=port)
