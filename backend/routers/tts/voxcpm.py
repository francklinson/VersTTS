#!/usr/bin/env python3
"""
VoxCPM 路由
"""

import time
import os
import sys
import torch
import numpy as np
import soundfile as sf
from datetime import datetime
from fastapi import APIRouter, Form, Request, HTTPException, UploadFile, File
from typing import Optional

from backend.logger_config import OperationLogger, system_logger
from backend.models import TTSResponse
from backend.engines import get_voxcpm_model
from backend.services import get_speaker_by_id
from backend.core import audio_to_base64

router = APIRouter()


@router.post("/")
async def tts_voxcpm(
        request: Request,
        text: str = Form(...),
        mode: str = Form("base"),
        ref_audio: Optional[UploadFile] = File(None),
        ref_text: Optional[str] = Form(None),
        voice_design_prompt: Optional[str] = Form(None),
        clone_speaker_id: Optional[str] = Form(None),
        cfg_value: float = Form(2.0),
        inference_timesteps: int = Form(10),
        output_format: str = Form("url")
):
    """VoxCPM语音合成 - 支持30种语言的无Tokenizer TTS"""
    ref_path = None
    temp_upload_path = None
    start_time = time.time()
    client_ip = request.client.host if request.client else "unknown"

    try:
        OperationLogger.log_api_request("/tts/voxcpm", "POST", {
            "text_preview": text[:50],
            "mode": mode,
            "cfg_value": cfg_value
        }, client_ip)

        system_logger.info(f"【VoxCPM】新请求 | 模式: {mode} | 文本: {text[:50]}...")
        system_logger.info(f"【VoxCPM】参数 | cfg_value={cfg_value} | inference_timesteps={inference_timesteps}")

        # 获取说话人信息（如果提供了clone_speaker_id）
        speaker = None
        speaker_ref_text = None
        if clone_speaker_id:
            speaker = get_speaker_by_id(clone_speaker_id)
            if not speaker:
                raise HTTPException(status_code=404, detail=f"说话人不存在: {clone_speaker_id}")
            ref_path = speaker.get("audio_path")
            speaker_ref_text = speaker.get("reference_text")
            system_logger.info(f"【VoxCPM】找到说话人 | 名称: {speaker.get('name')}")

            # 检查音频文件是否存在
            if ref_path and os.path.exists(ref_path):
                file_size = os.path.getsize(ref_path)
                system_logger.info(f"【VoxCPM】音频文件存在 | 大小: {file_size / 1024:.2f} KB")
            else:
                raise HTTPException(status_code=404, detail=f"参考音频文件不存在: {ref_path}")

        # 兼容旧版：如果直接上传了参考音频
        if ref_audio and mode in ["clone", "ultimate_clone"]:
            temp_upload_path = f"uploads/voxcpm_ref_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.wav"
            with open(temp_upload_path, "wb") as f:
                f.write(await ref_audio.read())
            ref_path = temp_upload_path
            system_logger.info(f"【VoxCPM】使用上传的参考音频: {ref_path}")

        # 加载模型
        system_logger.info(f"【VoxCPM】正在加载模型...")
        model = get_voxcpm_model()
        system_logger.info(f"【VoxCPM】模型加载完成")

        # 检查模型类型
        voxcpm_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "algorithms", "VoxCPM", "src")
        if voxcpm_path not in sys.path:
            sys.path.insert(0, voxcpm_path)
        from voxcpm.model.voxcpm2 import VoxCPM2Model
        is_v2 = isinstance(model.tts_model, VoxCPM2Model)
        system_logger.info(f"【VoxCPM】模型类型: {'VoxCPM2' if is_v2 else 'VoxCPM v1'}")

        # 构建生成参数
        generate_kwargs = {
            "cfg_value": cfg_value,
            "inference_timesteps": inference_timesteps
        }

        # 根据模式处理text和参考音频
        if mode == "voice_design":
            # voice_design模式: 在text前添加(voice description)
            if voice_design_prompt:
                generate_kwargs["text"] = f"({voice_design_prompt}){text}"
                system_logger.info(f"【VoxCPM】音色设计模式 | 描述: {voice_design_prompt}")
            else:
                generate_kwargs["text"] = f"(A natural speaking voice){text}"
                system_logger.info("【VoxCPM】音色设计模式 | 使用默认描述")

        elif mode == "clone" and ref_path:
            # clone模式: 使用Reference-only mode进行声音克隆
            generate_kwargs["text"] = text
            if is_v2:
                # VoxCPM2: 使用 reference_wav_path (Reference-only mode)
                generate_kwargs["reference_wav_path"] = ref_path
                system_logger.info(f"【VoxCPM】声音克隆模式 (Reference-only)")
            else:
                # VoxCPM v1: 只能使用 prompt_wav_path (Continuation mode)
                generate_kwargs["prompt_wav_path"] = ref_path
                if speaker_ref_text:
                    generate_kwargs["prompt_text"] = speaker_ref_text
                system_logger.info(f"【VoxCPM】声音克隆模式 (Continuation - VoxCPM v1)")

        elif mode == "ultimate_clone" and ref_path:
            # ultimate_clone模式: 使用Combined mode进行极致克隆
            generate_kwargs["text"] = text
            prompt_text = speaker_ref_text if speaker else ref_text

            if is_v2 and prompt_text:
                # VoxCPM2 Combined mode
                generate_kwargs["reference_wav_path"] = ref_path
                generate_kwargs["prompt_wav_path"] = ref_path
                generate_kwargs["prompt_text"] = prompt_text
                system_logger.info(f"【VoxCPM】极致克隆模式 (Combined mode)")
            elif prompt_text:
                # VoxCPM v1 或没有 reference_wav_path
                generate_kwargs["prompt_wav_path"] = ref_path
                generate_kwargs["prompt_text"] = prompt_text
                system_logger.info(f"【VoxCPM】极致克隆模式 (Continuation mode)")
            else:
                # 没有参考文本，退化为clone模式
                generate_kwargs["reference_wav_path"] = ref_path
                system_logger.warning(f"【VoxCPM】极致克隆模式缺少参考文本，退化为声音克隆模式")
        else:
            # base模式: 基础生成
            generate_kwargs["text"] = text
            system_logger.info("【VoxCPM】基础生成模式")

        # 生成音频
        system_logger.info(f"【VoxCPM】开始生成音频...")
        audio_data = model.generate(**generate_kwargs)
        system_logger.info(f"【VoxCPM】音频生成完成 | 数据类型: {type(audio_data)}")

        # 保存音频
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        audio_path = f"outputs/voxcpm_{timestamp}.wav"
        sf.write(audio_path, audio_data, samplerate=48000)
        file_size = os.path.getsize(audio_path)
        system_logger.info(f"【VoxCPM】音频保存完成: {audio_path} | 大小: {file_size / 1024:.2f} KB")

        total_duration = time.time() - start_time
        OperationLogger.log_tts_request("VoxCPM", text, {
            "mode": mode,
            "cfg_value": cfg_value
        }, total_duration, "成功")

        if output_format == "base64":
            audio_b64 = audio_to_base64(audio_path)
            return TTSResponse(
                success=True,
                message="合成成功",
                audio_base64=audio_b64,
                sample_rate=48000
            )
        else:
            return TTSResponse(
                success=True,
                message="合成成功",
                audio_url=f"/audio/{os.path.basename(audio_path)}",
                sample_rate=48000
            )

    except HTTPException:
        raise
    except Exception as e:
        total_duration = time.time() - start_time
        OperationLogger.log_error("VoxCPM合成错误", str(e))
        OperationLogger.log_tts_request("VoxCPM", text, {}, total_duration, f"失败: {str(e)}")
        system_logger.error(f"【VoxCPM】合成错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # 清理临时上传的文件
        if temp_upload_path and os.path.exists(temp_upload_path):
            os.remove(temp_upload_path)
            system_logger.info(f"【VoxCPM】清理临时文件: {temp_upload_path}")
