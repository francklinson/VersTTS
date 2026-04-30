#!/usr/bin/env python3
"""
F5-TTS 路由
"""

import os
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Form, UploadFile, File, HTTPException

from backend.logger_config import system_logger
from backend.config import models, DEFAULT_F5TTS_REF_ZH, DEFAULT_F5TTS_REF_EN, DEFAULT_F5TTS_TEXT_ZH, DEFAULT_F5TTS_TEXT_EN
from backend.core import save_temp_audio, audio_to_base64
from backend.engines import get_f5tts_model
from backend.services import load_speakers_db
from backend.models import TTSResponse

router = APIRouter()


@router.post("/")
async def tts_f5tts(
        text: Optional[str] = Form(None),
        gen_text: Optional[str] = Form(None),
        ref_text: Optional[str] = Form(None),
        ref_audio: Optional[UploadFile] = File(None),
        clone_speaker_id: Optional[str] = Form(None),
        nfe_step: int = Form(32),
        cfg_strength: float = Form(2.0),
        speed: float = Form(1.0),
        output_format: str = Form("url")
):
    """F5-TTS语音合成"""
    try:
        # 兼容前端参数名
        use_gen_text = gen_text or text
        if not use_gen_text:
            raise HTTPException(status_code=400, detail="缺少生成文本参数")

        system_logger.info(f"F5-TTS请求: {use_gen_text[:50]}...")

        ref_path = None
        use_ref_text = None
        is_temp = False

        # 优先使用clone_speaker_id
        if clone_speaker_id:
            db = load_speakers_db()
            speaker = None
            for s in db.get("speakers", []):
                if s["id"] == clone_speaker_id:
                    speaker = s
                    break

            if not speaker:
                raise HTTPException(status_code=404, detail=f"说话人不存在: {clone_speaker_id}")

            ref_path = speaker.get("audio_path")
            if not ref_path or not os.path.exists(ref_path):
                raise HTTPException(status_code=404, detail=f"说话人音频文件不存在")

            use_ref_text = ref_text or speaker.get("reference_text", "")
            if not use_ref_text:
                raise HTTPException(status_code=400, detail=f"说话人没有参考文本")

        elif ref_audio:
            ref_path = f"uploads/ref_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.wav"
            with open(ref_path, "wb") as f:
                f.write(await ref_audio.read())
            use_ref_text = ref_text or "参考音频文本"
            is_temp = True
        else:
            # 使用默认参考音频
            has_chinese = any('\u4e00' <= char <= '\u9fff' for char in use_gen_text)
            if has_chinese and os.path.exists(DEFAULT_F5TTS_REF_ZH):
                ref_path = DEFAULT_F5TTS_REF_ZH
                use_ref_text = ref_text or DEFAULT_F5TTS_TEXT_ZH
            else:
                ref_path = DEFAULT_F5TTS_REF_EN
                use_ref_text = ref_text or DEFAULT_F5TTS_TEXT_EN

        # 加载模型并推理
        f5tts = get_f5tts_model()
        wav, sr, _ = f5tts.infer(
            ref_file=ref_path,
            ref_text=use_ref_text,
            gen_text=use_gen_text,
            nfe_step=nfe_step,
            cfg_strength=cfg_strength,
            speed=speed,
        )

        # 保存音频
        audio_path = save_temp_audio(wav, sr)

        # 清理临时参考音频
        if is_temp and os.path.exists(ref_path):
            os.remove(ref_path)

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
        system_logger.error(f"F5-TTS错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))
