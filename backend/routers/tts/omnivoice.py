#!/usr/bin/env python3
"""
OmniVoice 路由
通过独立服务调用 OmniVoice（需要 transformers 5.x）
"""

import time
import os
import re
import requests
import soundfile as sf
from datetime import datetime
from fastapi import APIRouter, Form, Request, HTTPException
from typing import Optional

from backend.logger_config import OperationLogger, system_logger
from backend.models import TTSResponse
from backend.services import get_speaker_by_id
from backend.core import audio_to_base64
from backend.config import OMNIVOICE_HOST, OMNIVOICE_PORT

router = APIRouter()

OMNIVOICE_SERVICE_URL = f"http://{OMNIVOICE_HOST}:{OMNIVOICE_PORT}/tts"
OMNIVOICE_HEALTH_URL = f"http://{OMNIVOICE_HOST}:{OMNIVOICE_PORT}/health"


def _check_omnivoice_service():
    """检查 OmniVoice 独立服务是否运行"""
    try:
        response = requests.get(OMNIVOICE_HEALTH_URL, timeout=3)
        return response.status_code == 200
    except:
        return False


@router.post("/")
async def tts_omnivoice(
        request: Request,
        text: str = Form(...),
        mode: str = Form("auto_voice"),
        clone_speaker_id: Optional[str] = Form(None),
        voice_design_prompt: Optional[str] = Form(None),
        num_steps: int = Form(32),
        speed: float = Form(1.0),
        output_format: str = Form("url")
):
    """OmniVoice语音合成 - 通过独立服务"""
    ref_path = None
    speaker_ref_text = None
    start_time = time.time()
    client_ip = request.client.host if request.client else "unknown"

    try:
        OperationLogger.log_api_request("/tts/omnivoice", "POST", {
            "text_preview": text[:50],
            "mode": mode,
            "num_steps": num_steps,
            "speed": speed
        }, client_ip)

        system_logger.info(f"【OmniVoice】新请求 | 模式: {mode} | 文本: {text[:50]}...")

        # 检查独立服务状态
        if not _check_omnivoice_service():
            raise HTTPException(
                status_code=503,
                detail="OmniVoice 独立服务未运行。请执行: nohup python omnivoice_service.py > logs/omnivoice_service.log 2>&1 &"
            )

        # 获取说话人信息
        if mode == "voice_clone" and clone_speaker_id:
            speaker = get_speaker_by_id(clone_speaker_id)
            if not speaker:
                raise HTTPException(status_code=404, detail=f"说话人不存在: {clone_speaker_id}")
            ref_path = speaker.get("audio_path")
            speaker_ref_text = speaker.get("reference_text")
            system_logger.info(f"【OmniVoice】找到说话人 | 名称: {speaker.get('name')}")

        # 声音设计模式校验
        if mode == "voice_design" and voice_design_prompt:
            has_chinese = bool(re.search(r'[\u4e00-\u9fff]', voice_design_prompt))
            has_english = bool(re.search(r'[a-zA-Z]', voice_design_prompt))
            if has_chinese and has_english:
                raise HTTPException(
                    status_code=400,
                    detail="声音设计属性不能中英文混用！请使用纯中文（如'男，四川话'）或纯英文（如'male, high pitch'）"
                )

        # 调用独立服务
        system_logger.info(f"【OmniVoice】调用独立服务...")
        data = {
            "text": text,
            "mode": mode,
            "num_steps": num_steps,
            "speed": speed
        }
        if mode == "voice_clone" and ref_path:
            data["ref_audio"] = ref_path
            if speaker_ref_text:
                data["ref_text"] = speaker_ref_text
        elif mode == "voice_design" and voice_design_prompt:
            data["voice_design_prompt"] = voice_design_prompt

        response = requests.post(OMNIVOICE_SERVICE_URL, data=data, timeout=120)

        if response.status_code != 200:
            error_detail = response.json().get("detail", "未知错误")
            raise HTTPException(status_code=response.status_code, detail=error_detail)

        result = response.json()
        if not result.get("success"):
            raise HTTPException(status_code=500, detail="OmniVoice 合成失败")

        audio_path = result.get("audio_path")
        if not audio_path or not os.path.exists(audio_path):
            raise HTTPException(status_code=500, detail="OmniVoice 音频文件未生成")

        # 保存音频到 outputs 目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"outputs/omnivoice_{timestamp}.wav"
        audio_data, sample_rate = sf.read(audio_path)
        sf.write(output_path, audio_data, samplerate=sample_rate)
        file_size = os.path.getsize(output_path)
        system_logger.info(f"【OmniVoice】音频保存完成: {output_path} | 大小: {file_size / 1024:.2f} KB")

        # 清理临时文件
        try:
            os.remove(audio_path)
        except:
            pass

        total_duration = time.time() - start_time
        OperationLogger.log_tts_request("OmniVoice", text, {
            "mode": mode,
            "num_steps": num_steps,
            "speed": speed
        }, total_duration, "成功")

        if output_format == "base64":
            audio_b64 = audio_to_base64(output_path)
            return TTSResponse(
                success=True,
                message="合成成功",
                audio_base64=audio_b64,
                sample_rate=sample_rate
            )
        else:
            return TTSResponse(
                success=True,
                message="合成成功",
                audio_url=f"/audio/{os.path.basename(output_path)}",
                sample_rate=sample_rate
            )

    except HTTPException:
        raise
    except Exception as e:
        total_duration = time.time() - start_time
        OperationLogger.log_error("OmniVoice合成错误", str(e))
        OperationLogger.log_tts_request("OmniVoice", text, {}, total_duration, f"失败: {str(e)}")
        system_logger.error(f"【OmniVoice】合成错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))
