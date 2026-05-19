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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 检查是否启用预加载
    preload_enabled = os.environ.get('PRELOAD_OMNIVOICE', '1') == '1'
    
    if preload_enabled:
        # 启动时加载模型
        load_model()
    else:
        print("【OmniVoice服务】预加载已禁用，模型将在首次请求时加载")
    
    yield
    
    # 关闭时清理资源
    global model
    if model is not None:
        print("【OmniVoice服务】正在卸载模型...")
        model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("【OmniVoice服务】GPU缓存已清理")


app = FastAPI(title="OmniVoice 独立服务", lifespan=lifespan)

# 全局模型
model = None

# 模型路径：优先使用环境变量 MODELS_DIR，否则使用相对于脚本的路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.environ.get("MODELS_DIR", os.path.join(PROJECT_ROOT, "models"))
MODEL_PATH = os.path.join(MODELS_DIR, "OmniVoice")


def load_model():
    """加载 OmniVoice 模型，带重试机制"""
    global model
    if model is not None:
        return
    
    print(f"【OmniVoice服务】正在加载模型: {MODEL_PATH}")
    is_offline = os.environ.get('TRANSFORMERS_OFFLINE') == '1'
    
    # 尝试加载模型，如果GPU被占用则等待重试
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
            print("【OmniVoice服务】模型加载完成")
            return
            
        except RuntimeError as e:
            if "CUDA" in str(e) and attempt < max_retries - 1:
                print(f"【OmniVoice服务】GPU繁忙，等待5秒后重试... ({attempt + 1}/{max_retries})")
                time.sleep(5)
            else:
                raise
    
    raise RuntimeError(f"模型加载失败，已重试 {max_retries} 次")


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}


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


if __name__ == "__main__":
    port = int(os.environ.get("OMNIVOICE_PORT", 8001))
    host = os.environ.get("OMNIVOICE_HOST", "127.0.0.1")
    print(f"【OmniVoice服务】启动服务，地址: {host}:{port}")
    uvicorn.run(app, host=host, port=port)
