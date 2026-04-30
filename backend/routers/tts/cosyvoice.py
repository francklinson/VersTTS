#!/usr/bin/env python3
"""
CosyVoice 路由
"""

import os
import tempfile
from typing import Optional

from fastapi import APIRouter, Form, UploadFile, File, HTTPException

from backend.logger_config import system_logger
from backend.config import models
from backend.core import save_temp_audio, audio_to_base64
from backend.engines import get_cosyvoice_model
from backend.services import load_speakers_db
from backend.models import TTSResponse

router = APIRouter()


@router.post("/")
async def tts_cosyvoice(
        text: str = Form(...),
        mode: str = Form("sft"),
        speaker_id: str = Form("中文女"),
        prompt_text: Optional[str] = Form(None),
        instruct_text: Optional[str] = Form(None),
        prompt_wav: Optional[UploadFile] = File(None),
        clone_speaker_id: Optional[str] = Form(None),
        output_format: str = Form("url")
):
    """CosyVoice语音合成"""
    try:
        system_logger.info(f"CosyVoice请求: {text[:50]}... 模式: {mode}")

        # 使用 CosyVoice 3.0
        model_dir = "Fun-CosyVoice3-0.5B"
        cosyvoice = get_cosyvoice_model(model_dir)

        if mode == "sft":
            raise HTTPException(status_code=400, detail="CosyVoice 3.0 不支持SFT预训练音色模式，请使用Zero-shot克隆模式")
            
        elif mode == "zero_shot":
            from cosyvoice.utils.file_utils import load_wav

            if clone_speaker_id:
                db = load_speakers_db()
                speaker = None
                for s in db.get("speakers", []):
                    if s["id"] == clone_speaker_id:
                        speaker = s
                        break

                if not speaker:
                    raise HTTPException(status_code=404, detail=f"说话人不存在: {clone_speaker_id}")

                audio_path = speaker.get("audio_path")
                if not audio_path or not os.path.exists(audio_path):
                    raise HTTPException(status_code=404, detail=f"说话人音频文件不存在")

                ref_text = speaker.get("reference_text", "")
                
                if ref_text:
                    prompt_text = f"You are a helpful assistant.<|endofprompt|>{ref_text}"
                    model_output = cosyvoice.inference_zero_shot(text, prompt_text, audio_path, stream=False)
                else:
                    formatted_text = f"You are a helpful assistant.<|endofprompt|>{text}"
                    model_output = cosyvoice.inference_cross_lingual(formatted_text, audio_path, stream=False)

            elif prompt_wav:
                file_content = prompt_wav.file.read()
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                    tmp_file.write(file_content)
                    tmp_path = tmp_file.name
                try:
                    formatted_text = f"You are a helpful assistant.<|endofprompt|>{text}"
                    model_output = cosyvoice.inference_cross_lingual(formatted_text, tmp_path, stream=False)
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
            else:
                raise HTTPException(status_code=400, detail="zero_shot模式需要提供clone_speaker_id或上传参考音频")
        else:
            raise HTTPException(status_code=400, detail=f"不支持的模式: {mode}")

        # 保存音频
        sr = 22050
        audio_data = model_output['tts_speech'].numpy().squeeze()
        audio_path = save_temp_audio(audio_data, sr)

        if output_format == "base64":
            audio_b64 = audio_to_base64(audio_path)
            return TTSResponse(
                success=True,
                message="合成成功",
                audio_base64=audio_b64,
                sample_rate=sr
            )
        else:
            return TTSResponse(
                success=True,
                message="合成成功",
                audio_url=f"/audio/{audio_path.split('/')[-1]}",
                sample_rate=sr
            )

    except Exception as e:
        system_logger.error(f"CosyVoice错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/speakers")
async def get_cosyvoice_speakers():
    """获取CosyVoice可用的说话人列表"""
    return {
        "success": True,
        "message": "CosyVoice 3.0 不支持预设音色，请使用Zero-shot克隆模式上传参考音频",
        "speakers": []
    }
