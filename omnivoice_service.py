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
import torch
import soundfile as sf
import uvicorn
from fastapi import FastAPI, Form, HTTPException
from typing import Optional

# 添加 OmniVoice 算法路径
ALGORITHMS_PATH = os.path.join(os.path.dirname(__file__), "algorithms", "OmniVoice")
if ALGORITHMS_PATH not in sys.path:
    sys.path.insert(0, ALGORITHMS_PATH)

from omnivoice import OmniVoice

app = FastAPI(title="OmniVoice 独立服务")

# 全局模型
model = None

MODEL_PATH = "/home/zhouchenghao/PycharmProjects/VersTTS/models/OmniVoice"


def load_model():
    """加载 OmniVoice 模型"""
    global model
    if model is None:
        print(f"【OmniVoice服务】正在加载模型: {MODEL_PATH}")
        is_offline = os.environ.get('TRANSFORMERS_OFFLINE') == '1'
        model = OmniVoice.from_pretrained(
            MODEL_PATH,
            device_map="cuda:0" if torch.cuda.is_available() else "cpu",
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            local_files_only=is_offline
        )
        print("【OmniVoice服务】模型加载完成")


@app.on_event("startup")
async def startup():
    load_model()


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
        kwargs = {"text": text, "num_step": num_steps, "speed": speed}
        
        if mode == "voice_clone" and ref_audio:
            kwargs["ref_audio"] = ref_audio
            if ref_text:
                kwargs["ref_text"] = ref_text
        elif mode == "voice_design" and voice_design_prompt:
            kwargs["instruct"] = voice_design_prompt
        
        audio_list = model.generate(**kwargs)
        audio_data = audio_list[0] if isinstance(audio_list, list) else audio_list
        
        # 保存临时文件
        timestamp = int(time.time() * 1000)
        output_path = f"/tmp/omnivoice_{timestamp}.wav"
        sf.write(output_path, audio_data, samplerate=24000)
        
        # 返回音频文件路径
        return {"success": True, "audio_path": output_path, "sample_rate": 24000}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    port = int(os.environ.get("OMNIVOICE_PORT", 8001))
    print(f"【OmniVoice服务】启动服务，端口: {port}")
    uvicorn.run(app, host="127.0.0.1", port=port)
