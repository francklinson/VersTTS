#!/usr/bin/env python3
"""
Fish-Speech 路由
通过独立服务调用 Fish-Speech (Dual-AR 架构, 80+语言)
"""

import os
import time
from typing import Optional

import requests
from fastapi import APIRouter, Form, Request, HTTPException

from backend.logger_config import OperationLogger, system_logger
from backend.models import TTSResponse
from backend.services import get_speaker_by_id
from backend.core import save_temp_audio, audio_to_base64, cleanup_memory
from backend.config import FISHSPEECH_HOST, FISHSPEECH_PORT

router = APIRouter()

FISHSPEECH_SERVICE_URL = f"http://{FISHSPEECH_HOST}:{FISHSPEECH_PORT}/tts"
FISHSPEECH_HEALTH_URL = f"http://{FISHSPEECH_HOST}:{FISHSPEECH_PORT}/health"


def _check_fishspeech_service():
    """检查 Fish-Speech 独立服务是否运行"""
    try:
        response = requests.get(FISHSPEECH_HEALTH_URL, timeout=3)
        return response.status_code == 200
    except Exception:
        return False


@router.post("/")
async def tts_fishspeech(
        request: Request,
        text: str = Form(...),
        mode: str = Form("voice_clone"),
        clone_speaker_id: Optional[str] = Form(None),
        temperature: float = Form(0.8),
        top_p: float = Form(0.8),
        repetition_penalty: float = Form(1.1),
        emotion_tags: Optional[str] = Form(None),
        output_format: str = Form("url"),
):
    """Fish-Speech语音合成 - 通过独立服务调用

    模式:
    - voice_clone: 零样本声音克隆（需要选择说话人）
    - voice_design: 声音设计（自然语言描述音色）

    参数:
    - temperature: 采样温度 (0.1-1.5)
    - top_p: nucleus采样 (0.5-1.0)
    - repetition_penalty: 重复惩罚 (1.0-1.5)
    - emotion_tags: 内联情感标签，如 [laugh], [sad]
    """
    start_time = time.time()
    client_ip = request.client.host if request.client else "unknown"

    system_logger.info(f"【Fish-Speech】{'='*60}")
    system_logger.info(f"【Fish-Speech】请求开始 | 模式: {mode} | 客户端: {client_ip}")
    system_logger.info(f"【Fish-Speech】输入文本: {text[:100]}...")

    try:
        # 检查独立服务状态
        if not _check_fishspeech_service():
            system_logger.error("【Fish-Speech】独立服务不可用")
            raise HTTPException(
                status_code=503,
                detail="Fish-Speech独立服务不可用，请先启动: ./start_server.sh start-fishspeech"
            )

        request_params = {
            "text_preview": text[:50],
            "mode": mode,
            "clone_speaker_id": clone_speaker_id,
            "temperature": temperature,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
        }
        OperationLogger.log_api_request("/tts/fishspeech", "POST", request_params, client_ip)

        # 获取说话人信息
        speaker_name = None
        speaker_ref_text = None
        if clone_speaker_id:
            system_logger.info(f"【Fish-Speech】查找说话人: {clone_speaker_id}")
            speaker = get_speaker_by_id(clone_speaker_id)
            if not speaker:
                system_logger.error(f"【Fish-Speech】说话人不存在: {clone_speaker_id}")
                raise HTTPException(status_code=404, detail=f"说话人不存在: {clone_speaker_id}")
            speaker_name = speaker.get("name")
            speaker_ref_text = speaker.get("reference_text")
            request_params["speaker_name"] = speaker_name

        # 处理情感标签
        final_text = text
        if emotion_tags and emotion_tags.strip():
            final_text = f"{emotion_tags.strip()} {text}"
            system_logger.info(f"【Fish-Speech】应用情感标签: {emotion_tags}")

        # 构建请求数据
        form_data = {
            "text": final_text,
            "mode": mode,
            "temperature": str(temperature),
            "top_p": str(top_p),
            "repetition_penalty": str(repetition_penalty),
        }
        if clone_speaker_id:
            form_data["clone_speaker_id"] = clone_speaker_id
        if speaker_ref_text:
            form_data["reference_text"] = speaker_ref_text

        # 转发到独立服务
        system_logger.info(f"【Fish-Speech】转发请求到: {FISHSPEECH_SERVICE_URL}")
        gen_start = time.time()
        response = requests.post(FISHSPEECH_SERVICE_URL, data=form_data, timeout=180)
        gen_duration = time.time() - gen_start

        if response.status_code not in (200, 201):
            system_logger.error(f"【Fish-Speech】服务返回错误: {response.status_code} - {response.text}")
            raise HTTPException(
                status_code=502,
                detail=f"Fish-Speech服务错误: {response.text[:200]}"
            )

        result = response.json()
        audio_path = result.get("audio_path")
        audio_url = result.get("audio_url", "")

        if not audio_path or not os.path.exists(audio_path):
            raise HTTPException(status_code=500, detail="Fish-Speech服务未返回有效音频")

        sample_rate = result.get("sample_rate", 44100)
        system_logger.info(f"【Fish-Speech】音频生成完成 | 耗时: {gen_duration:.3f}s | 路径: {audio_path}")

        # 清理显存
        cleanup_memory()

        total_duration = time.time() - start_time
        OperationLogger.log_tts_request("Fish-Speech", text, request_params, total_duration, "成功")
        system_logger.info(f"【Fish-Speech】请求完成 | 总耗时: {total_duration:.3f}s")
        system_logger.info(f"【Fish-Speech】{'='*60}")

        if output_format == "base64":
            audio_b64 = audio_to_base64(audio_path)
            return TTSResponse(success=True, message="合成成功", audio_base64=audio_b64, sample_rate=sample_rate)
        else:
            return TTSResponse(success=True, message="合成成功", audio_url=audio_url or f"/audio/{os.path.basename(audio_path)}", sample_rate=sample_rate)

    except HTTPException:
        raise
    except Exception as e:
        total_duration = time.time() - start_time
        OperationLogger.log_error("Fish-Speech合成错误", str(e))
        OperationLogger.log_tts_request("Fish-Speech", text, {"mode": mode}, total_duration, f"失败: {str(e)}")
        system_logger.error(f"【Fish-Speech】合成错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))
