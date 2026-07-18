#!/usr/bin/env python3
"""
OmniVoice 独立服务
使用 transformers 5.x，运行在独立端口上
启动方式: nohup python omnivoice_service.py > logs/omnivoice_service.log 2>&1 &
"""

import sys
import os

# 在导入任何模块之前，设置 transformers 5.x 路径
TRANSFORMERS5_PATH = os.path.join(os.path.dirname(__file__), "lib", "transformers5")
sys.path.insert(0, TRANSFORMERS5_PATH)

import time
import traceback
import asyncio
import gc
import torch
import soundfile as sf
import uvicorn
from fastapi import FastAPI, Form, HTTPException
from contextlib import asynccontextmanager
from typing import Optional
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 添加 OmniVoice 算法路径
ALGORITHMS_PATH = os.path.join(os.path.dirname(__file__), "algorithms", "OmniVoice")
if ALGORITHMS_PATH not in sys.path:
    sys.path.insert(0, ALGORITHMS_PATH)

from omnivoice import OmniVoice

# 全局模型
model = None
last_used_time = None
_gpu_id = None
_main_service_url = None
_idle_check_task = None

# 模型路径：优先使用环境变量 MODELS_DIR，否则使用相对于脚本的路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.environ.get("MODELS_DIR", os.path.join(PROJECT_ROOT, "models"))
if not os.path.isabs(MODELS_DIR):
    MODELS_DIR = os.path.join(PROJECT_ROOT, MODELS_DIR)
MODEL_PATH = os.path.join(MODELS_DIR, "OmniVoice")

# ========== 空闲超时配置 ==========
IDLE_TIMEOUT = int(os.environ.get("IDLE_TIMEOUT", "300"))  # 默认 5 分钟
HEARTBEAT_INTERVAL = int(os.environ.get("HEARTBEAT_INTERVAL", "60"))  # 心跳间隔秒


def unload_model():
    """卸载模型，释放显存"""
    global model, last_used_time
    if model is None:
        return
    print("【OmniVoice服务】正在卸载模型...")
    model = None
    last_used_time = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("【OmniVoice服务】GPU缓存已清理")


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
                print(f"【OmniVoice服务】模型已空闲 {idle_seconds:.0f}s > {IDLE_TIMEOUT}s，自动卸载")
                unload_model()
        # 心跳上报
        _heartbeat()


def _register_to_main_service():
    """向主服务注册"""
    try:
        import requests
        port = int(os.environ.get("OMNIVOICE_PORT", "8001"))
        host = os.environ.get("OMNIVOICE_HOST", "127.0.0.1")
        requests.post(
            f"{_main_service_url}/services/register",
            json={
                "service_id": "omnivoice",
                "port": port,
                "host": host,
                "gpu_id": _gpu_id,
            },
            timeout=5,
            verify=False,
        )
        print(f"【OmniVoice服务】已注册到主服务 (GPU: {_gpu_id})")
        return True
    except Exception as e:
        print(f"【OmniVoice服务】注册到主服务失败: {e}")
        return False


def _unregister_from_main_service():
    """从主服务注销"""
    try:
        import requests
        requests.post(
            f"{_main_service_url}/services/unregister",
            json={"service_id": "omnivoice"},
            timeout=5,
            verify=False,
        )
        print("【OmniVoice服务】已从主服务注销")
    except Exception as e:
        print(f"【OmniVoice服务】从主服务注销失败: {e}")


def _heartbeat():
    """向主服务上报心跳"""
    try:
        import requests
        vram_mb = 0
        if model is not None and torch.cuda.is_available():
            vram_mb = torch.cuda.memory_allocated() // (1024 * 1024)
        requests.post(
            f"{_main_service_url}/services/heartbeat",
            json={
                "service_id": "omnivoice",
                "model_loaded": model is not None,
                "vram_used_mb": vram_mb,
                "last_used_time": last_used_time,
                "gpu_id": _gpu_id,
                "host": os.environ.get("OMNIVOICE_HOST", "127.0.0.1"),
                "port": int(os.environ.get("OMNIVOICE_PORT", "8001")),
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
        print(f"【OmniVoice服务】请求主服务释放 {needed_mb}MB 显存 (GPU: {_gpu_id})")
        resp = requests.post(
            f"{_main_service_url}/services/evict",
            json={"gpu_id": _gpu_id, "exclude_service": "omnivoice", "needed_mb": needed_mb},
            timeout=30,
            verify=False,
        )
        if resp.status_code == 200:
            result = resp.json()
            evicted = result.get("evicted_services", [])
            print(f"【OmniVoice服务】主服务已驱逐: {evicted}")
            return True
        else:
            print(f"【OmniVoice服务】主服务返回: {resp.status_code}")
            return False
    except Exception as e:
        print(f"【OmniVoice服务】OOM驱逐请求失败: {e}")
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global _idle_check_task, _gpu_id, _main_service_url

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
    unload_model()

    # 从主服务注销
    _unregister_from_main_service()

    print("【OmniVoice服务】已停止")


app = FastAPI(title="OmniVoice 独立服务", lifespan=lifespan)


def load_model():
    """加载 OmniVoice 模型，带重试机制，OOM 时请求主服务驱逐"""
    global model
    if model is not None:
        _touch_last_used()
        return

    print(f"【OmniVoice服务】正在加载模型: {MODEL_PATH}")
    is_offline = os.environ.get('TRANSFORMERS_OFFLINE') == '1'

    # 尝试加载模型，OOM 时请求驱逐后重试
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # 在加载前清理显存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                print(f"【OmniVoice服务】已清理GPU缓存，尝试 {attempt + 1}/{max_retries}")

            model = OmniVoice.from_pretrained(
                MODEL_PATH,
                device_map="cuda:0" if torch.cuda.is_available() else "cpu",
                dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                local_files_only=is_offline
            )
            _touch_last_used()
            print("【OmniVoice服务】模型加载完成")
            return

        except RuntimeError as e:
            if ("CUDA" in str(e) or "out of memory" in str(e).lower()) and attempt < max_retries - 1:
                print(f"【OmniVoice服务】GPU OOM，尝试驱逐其他模型... ({attempt + 1}/{max_retries})")
                _request_eviction_from_main_service(needed_mb=4000)
                time.sleep(3)
            else:
                raise

    raise RuntimeError(f"模型加载失败，已重试 {max_retries} 次")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "last_used_time": last_used_time,
        "idle_timeout": IDLE_TIMEOUT,
        "gpu_id": _gpu_id,
    }


@app.post("/tts")
async def tts(
    text: str = Form(...),
    mode: str = Form("auto_voice"),
    ref_audio: Optional[str] = Form(None),
    ref_text: Optional[str] = Form(None),
    voice_design_prompt: Optional[str] = Form(None),
    num_steps: int = Form(32),
    speed: float = Form(1.0)
):
    """OmniVoice TTS 合成"""
    load_model()
    _touch_last_used()

    try:
        logger.info(f"【OmniVoice TTS请求】模式: {mode}, 文本: {text[:50]}..., 步数: {num_steps}, 语速: {speed}")

        kwargs = {"text": text, "num_step": num_steps, "speed": speed}

        if mode == "voice_clone" and ref_audio:
            kwargs["ref_audio"] = ref_audio
            if ref_text:
                kwargs["ref_text"] = ref_text
            logger.info(f"【声音克隆】参考音频: {ref_audio}, 参考文本: {ref_text}")
        elif mode == "voice_design" and voice_design_prompt:
            kwargs["instruct"] = voice_design_prompt
            logger.info(f"【声音设计】提示: {voice_design_prompt}")
        else:
            logger.info(f"【自动音色】模式")

        logger.info(f"【生成参数】{kwargs}")
        audio_list = model.generate(**kwargs)
        audio_data = audio_list[0] if isinstance(audio_list, list) else audio_list

        logger.info(f"【音频生成】数据类型: {type(audio_data)}, 形状: {audio_data.shape if hasattr(audio_data, 'shape') else 'N/A'}")

        # 保存临时文件
        timestamp = int(time.time() * 1000)
        output_path = f"/tmp/omnivoice_{timestamp}.wav"
        sf.write(output_path, audio_data, samplerate=24000)

        logger.info(f"【音频保存】路径: {output_path}")

        # 返回音频文件路径
        return {"success": True, "audio_path": output_path, "sample_rate": 24000}

    except ValueError as e:
        error_msg = str(e)
        if "Conflicting instruct" in error_msg:
            logger.error(f"【OmniVoice TTS错误】音色描述参数冲突: {error_msg}")
            raise HTTPException(
                status_code=400,
                detail=f"音色描述参数冲突：每个类别（性别、年龄、音调、风格、口音、方言）只能指定一个值。例如：'男，儿童，低音调'，不要在同一类别中指定多个值。"
            )
        else:
            logger.error(f"【OmniVoice TTS错误】参数错误: {error_msg}")
            logger.error(f"【错误堆栈】\n{traceback.format_exc()}")
            raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"【OmniVoice TTS错误】异常类型: {type(e).__name__}, 错误信息: {str(e)}")
        logger.error(f"【错误堆栈】\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {str(e)}")


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
        "service_id": "omnivoice",
        "model_loaded": model is not None,
        "vram_used_mb": vram_mb,
        "last_used_time": last_used_time,
        "idle_seconds": int(idle_seconds) if idle_seconds is not None else None,
        "idle_timeout": IDLE_TIMEOUT,
        "gpu_id": _gpu_id,
    }


if __name__ == "__main__":
    port = int(os.environ.get("OMNIVOICE_PORT", "8001"))
    host = os.environ.get("OMNIVOICE_HOST", "127.0.0.1")
    print(f"【OmniVoice服务】启动服务，地址: {host}:{port}")
    uvicorn.run(app, host=host, port=port)
