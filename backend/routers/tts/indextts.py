#!/usr/bin/env python3
"""
IndexTTS 路由
"""

import time
import os
import torch
from datetime import datetime
from fastapi import APIRouter, Form, Request, HTTPException, UploadFile, File
from typing import Optional

from backend.logger_config import OperationLogger, system_logger
from backend.models import TTSResponse
from backend.engines import get_indextts_model
from backend.services import get_speaker_by_id
from backend.core import audio_to_base64

router = APIRouter()


@router.post("/")
async def tts_indextts(
        request: Request,
        text: str = Form(...),
        mode: str = Form("free"),
        prompt_wav: Optional[UploadFile] = File(None),
        emotion_text: Optional[str] = Form(None),
        duration_tokens: Optional[int] = Form(None),
        clone_speaker_id: Optional[str] = Form(None),
        output_format: str = Form("url")
):
    """IndexTTS2语音合成 - 支持说话人管理模块
    
    参数:
    - clone_speaker_id: 说话人管理中的说话人ID，优先使用
    - prompt_wav: 直接上传参考音频（当clone_speaker_id为空时使用）
    - emotion_text: 情感描述文本（可选）
    - duration_tokens: 时长控制token数（可选，当前版本暂不支持）
    
    使用方法与官方一致:
    tts.infer(spk_audio_prompt='voice.wav', text=text, output_path="gen.wav")
    """
    ref_path = None
    is_temp = False
    start_time = time.time()
    client_ip = request.client.host if request.client else "unknown"

    try:
        OperationLogger.log_api_request("/tts/indextts", "POST", {
            "text_preview": text[:50],
            "mode": mode,
            "clone_speaker_id": clone_speaker_id
        }, client_ip)

        system_logger.info(f"【IndexTTS2】请求 | 模式: {mode} | 文本: {text[:50]}...")

        # 方式1: 优先使用clone_speaker_id从说话人管理模块获取音频
        if clone_speaker_id:
            speaker = get_speaker_by_id(clone_speaker_id)
            if not speaker:
                raise HTTPException(status_code=404, detail=f"说话人不存在: {clone_speaker_id}")

            ref_path = speaker.get("audio_path")
            if not ref_path or not os.path.exists(ref_path):
                raise HTTPException(status_code=404, detail=f"说话人音频文件不存在: {ref_path}")

            system_logger.info(f"【IndexTTS2】使用说话人 {speaker['name']} 的音频: {ref_path}")

        # 方式2: 兼容旧版本，使用上传的参考音频
        elif prompt_wav:
            ref_path = f"uploads/indextts_ref_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.wav"
            with open(ref_path, "wb") as f:
                f.write(await prompt_wav.read())
            is_temp = True
            system_logger.info(f"【IndexTTS2】使用上传的参考音频: {ref_path}")

        else:
            raise HTTPException(status_code=400, detail="需要提供clone_speaker_id（说话人ID）或prompt_wav（参考音频）")

        # 加载模型
        model = get_indextts_model()

        # 生成音频 - 按照GitHub示例使用infer方法
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        audio_path = f"outputs/indextts_{timestamp}.wav"

        # 准备infer参数
        infer_kwargs = {
            "spk_audio_prompt": ref_path,
            "text": text,
            "output_path": audio_path,
            "verbose": True
        }

        # 添加情感描述参数（如果提供）
        if emotion_text and emotion_text.strip():
            infer_kwargs["use_emo_text"] = True
            infer_kwargs["emo_text"] = emotion_text.strip()
            infer_kwargs["emo_alpha"] = 0.6  # 使用推荐的emo_alpha值
            system_logger.info(f"【IndexTTS2】使用情感描述: {emotion_text}")

        # 调用模型推理
        system_logger.info(f"【IndexTTS2】开始推理...")
        model.infer(**infer_kwargs)
        system_logger.info(f"【IndexTTS2】推理完成: {audio_path}")

        # 清理临时文件
        if is_temp and ref_path and os.path.exists(ref_path):
            os.remove(ref_path)
            system_logger.info(f"【IndexTTS2】清理临时文件: {ref_path}")

        total_duration = time.time() - start_time
        OperationLogger.log_tts_request("IndexTTS", text, {
            "mode": mode,
            "emotion_text": emotion_text
        }, total_duration, "成功")

        if output_format == "base64":
            audio_b64 = audio_to_base64(audio_path)
            return TTSResponse(
                success=True,
                message="合成成功",
                audio_base64=audio_b64,
                sample_rate=22050
            )
        else:
            return TTSResponse(
                success=True,
                message="合成成功",
                audio_url=f"/audio/{os.path.basename(audio_path)}",
                sample_rate=22050
            )

    except HTTPException:
        raise
    except Exception as e:
        total_duration = time.time() - start_time
        OperationLogger.log_error("IndexTTS合成错误", str(e))
        OperationLogger.log_tts_request("IndexTTS", text, {}, total_duration, f"失败: {str(e)}")
        system_logger.error(f"【IndexTTS2】错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))
