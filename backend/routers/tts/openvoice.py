#!/usr/bin/env python3
"""
OpenVoice 路由
"""

import time
import os
import torch
import numpy as np
import soundfile as sf
from fastapi import APIRouter, Form, Request, HTTPException, UploadFile, File
from typing import Optional

from backend.config import OUTPUTS_DIR, UPLOADS_DIR
from backend.logger_config import OperationLogger, system_logger
from backend.models import TTSResponse
from backend.engines import get_openvoice_models
from backend.core import save_temp_audio, audio_to_base64, cleanup_memory, log_gpu_memory_usage

router = APIRouter()


@router.post("/")
async def tts_openvoice(
        request: Request,
        text: str = Form(...),
        style: str = Form("default"),
        speed: float = Form(1.0),
        speaker: str = Form("default"),
        prompt_wav: Optional[UploadFile] = File(None),
        clone_speaker_id: Optional[str] = Form(None),
        output_format: str = Form("url")
):
    """OpenVoice语音合成"""
    start_time = time.time()
    client_ip = request.client.host if request.client else "unknown"

    try:
        OperationLogger.log_api_request("/tts/openvoice", "POST", {
            "text_preview": text[:50],
            "style": style,
            "speed": speed,
            "speaker": speaker
        }, client_ip)

        system_logger.info(f"【OpenVoice】开始合成 | 文本: {text[:50]}... | 风格: {style}")

        # 获取模型
        model_info = get_openvoice_models(use_v2=True)
        tts_model = model_info["tts"]
        converter = model_info["converter"]
        source_se = model_info["source_se"]
        device = model_info["device"]

        # 处理参考音频
        ref_path = None
        is_temp = False

        if clone_speaker_id:
            from backend.services import get_speaker_by_id
            speaker_data = get_speaker_by_id(clone_speaker_id)
            if not speaker_data:
                raise HTTPException(status_code=404, detail=f"说话人不存在: {clone_speaker_id}")
            ref_path = speaker_data.get("audio_path")
            if not ref_path or not os.path.exists(ref_path):
                raise HTTPException(status_code=404, detail="说话人音频文件不存在")
        elif prompt_wav:
            from datetime import datetime
            ref_path = os.path.join(UPLOADS_DIR, f"openvoice_ref_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.wav")
            with open(ref_path, "wb") as f:
                f.write(await prompt_wav.read())
            is_temp = True
        else:
            # 使用默认音色
            ref_path = None

        # TTS生成
        infer_start = time.time()

        # 生成目标音色编码
        if ref_path:
            target_se = converter.extract_se([ref_path])
        else:
            # 使用默认音色
            target_se = source_se.get("zh", source_se.get("en"))

        # 生成音频
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        src_path = os.path.join(OUTPUTS_DIR, f"openvoice_tmp_{timestamp}.wav")
        tts_model.tts(text, src_path, speaker=speaker, language='Chinese', speed=speed)

        # 音色转换
        save_path = os.path.join(OUTPUTS_DIR, f"openvoice_{timestamp}.wav")
        encode_message = "@MyShell"
        converter.convert(
            audio_src_path=src_path,
            src_se=source_se.get("zh", source_se.get("en")),
            tgt_se=target_se,
            output_path=save_path,
            message=encode_message
        )

        # 清理临时文件
        if os.path.exists(src_path):
            os.remove(src_path)
        if is_temp and ref_path and os.path.exists(ref_path):
            os.remove(ref_path)

        # 清理显存 - 防止内存泄漏
        if torch.cuda.is_available():
            cleanup_memory()
            log_gpu_memory_usage("OpenVoice")

        infer_duration = time.time() - infer_start

        # 读取生成的音频
        audio_data, sr = sf.read(save_path)
        audio_path = save_temp_audio(audio_data, sr)

        total_duration = time.time() - start_time
        OperationLogger.log_tts_request("OpenVoice", text, {
            "style": style,
            "speed": speed
        }, total_duration, "成功")

        system_logger.info(f"【OpenVoice】合成完成 | 耗时: {total_duration:.3f}s")

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

    except HTTPException:
        raise
    except Exception as e:
        total_duration = time.time() - start_time
        OperationLogger.log_error("OpenVoice合成错误", str(e))
        OperationLogger.log_tts_request("OpenVoice", text, {}, total_duration, f"失败: {str(e)}")
        system_logger.error(f"【OpenVoice】错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))
