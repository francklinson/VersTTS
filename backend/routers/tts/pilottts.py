#!/usr/bin/env python3
"""
PilotTTS 路由
通过独立服务调用 PilotTTS（需要 transformers 4.51.3）

支持: 声音克隆 / 情感合成 / 方言合成 / 副语言合成
"""

import os
import time
import tempfile
from typing import Optional

import requests
import soundfile as sf
from fastapi import APIRouter, Form, Request, HTTPException, UploadFile, File

from backend.logger_config import OperationLogger, system_logger
from backend.models import TTSResponse
from backend.services import get_speaker_by_id
from backend.core import save_temp_audio, audio_to_base64, cleanup_memory, log_gpu_memory_usage
from backend.config import PILOTTS_HOST, PILOTTS_PORT

router = APIRouter()

PILOTTS_SERVICE_URL = f"http://{PILOTTS_HOST}:{PILOTTS_PORT}/tts"
PILOTTS_HEALTH_URL = f"http://{PILOTTS_HOST}:{PILOTTS_PORT}/health"

# 支持的情感标签
EMOTION_LABELS = [
    "happy", "sad", "angry", "surprise", "fear", "disgust",
    "serious", "concern", "blue", "disdain", "neutral", "psychology", "unknown"
]

# 支持的方言
DIALECT_MAP = {
    "zh-dongbei": "东北话", "zh-shandong": "山东话", "zh-henan": "河南话",
    "zh-shan1xi": "山西话", "zh-minnan": "闽南语", "zh-gansu": "甘肃话",
    "zh-ningxia": "宁夏话", "zh-shanghai": "上海话", "zh-chongqing": "重庆话",
    "zh-hubei": "湖北话", "zh-hunan": "湖南话", "zh-jiangxi": "江西话",
    "zh-guizhou": "贵州话", "zh-yunnan": "云南话",
}


def _check_pilottts_service():
    """检查 PilotTTS 独立服务是否运行"""
    try:
        response = requests.get(PILOTTS_HEALTH_URL, timeout=3)
        return response.status_code == 200
    except Exception:
        return False


@router.post("/")
async def tts_pilottts(
        request: Request,
        text: str = Form(...),
        mode: str = Form("voice_clone"),
        clone_speaker_id: Optional[str] = Form(None),
        ref_audio: Optional[UploadFile] = File(None),
        emotion: Optional[str] = Form(None),
        language: str = Form("zh"),
        output_format: str = Form("url")
):
    """
    PilotTTS语音合成 — 通过独立服务调用

    支持模式:
    - voice_clone: 零样本声音克隆（基础模型）
    - emotion: 情感合成（指令模型）- 11种情感
    - dialect: 方言合成（指令模型）- 14种方言
    - paralanguage: 副语言合成（指令模型）
    """
    ref_path = None
    temp_upload_path = None
    start_time = time.time()
    client_ip = request.client.host if request.client else "unknown"

    system_logger.info(f"【PilotTTS】{'='*60}")
    system_logger.info(f"【PilotTTS】请求开始 | 模式: {mode} | 客户端: {client_ip}")
    system_logger.info(f"【PilotTTS】输入文本: {text[:100]}...")

    try:
        # 参数验证
        if not text or not text.strip():
            raise HTTPException(status_code=400, detail="文本不能为空")

        request_params = {
            "text_preview": text[:50],
            "mode": mode,
            "clone_speaker_id": clone_speaker_id,
            "emotion": emotion,
            "language": language,
            "has_upload": ref_audio is not None
        }
        OperationLogger.log_api_request("/tts/pilottts", "POST", request_params, client_ip)
        system_logger.info(f"【PilotTTS】请求参数: {request_params}")

        # 检查独立服务状态
        system_logger.info(f"【PilotTTS】检查独立服务状态...")
        if not _check_pilottts_service():
            system_logger.error(f"【PilotTTS】独立服务未运行")
            raise HTTPException(
                status_code=503,
                detail="PilotTTS 独立服务未运行。请执行: nohup python pilottts_service.py > logs/pilottts_service.log 2>&1 &"
            )
        system_logger.info(f"【PilotTTS】独立服务运行正常")

        # 获取参考音频
        speaker = None
        speaker_name = None
        if clone_speaker_id:
            system_logger.info(f"【PilotTTS】查找说话人: {clone_speaker_id}")
            speaker = get_speaker_by_id(clone_speaker_id)
            if not speaker:
                system_logger.error(f"【PilotTTS】说话人不存在: {clone_speaker_id}")
                raise HTTPException(status_code=404, detail=f"说话人不存在: {clone_speaker_id}")
            ref_path = speaker.get("audio_path")
            speaker_name = speaker.get("name")

            if not ref_path or not os.path.exists(ref_path):
                raise HTTPException(status_code=404, detail=f"参考音频文件不存在: {ref_path}")
            system_logger.info(f"【PilotTTS】使用说话人音频: {ref_path}")

        elif ref_audio:
            import datetime
            temp_upload_path = f"uploads/pilottts_ref_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.wav"
            os.makedirs("uploads", exist_ok=True)
            with open(temp_upload_path, "wb") as f:
                f.write(await ref_audio.read())
            ref_path = temp_upload_path
            system_logger.info(f"【PilotTTS】上传音频已保存: {ref_path}")

        if not ref_path:
            raise HTTPException(status_code=400, detail="需要提供参考音频（clone_speaker_id或ref_audio）")

        # 验证情感标签
        if mode == "emotion" and emotion and emotion not in EMOTION_LABELS:
            raise HTTPException(status_code=400, detail=f"不支持的情感标签: {emotion}")

        # 调用独立服务
        data = {
            "text": text,
            "mode": mode,
            "ref_path": ref_path,
            "language": language,
        }
        if emotion:
            data["emotion"] = emotion

        system_logger.info(f"【PilotTTS】调用独立服务: {PILOTTS_SERVICE_URL}")
        service_start = time.time()
        response = requests.post(PILOTTS_SERVICE_URL, data=data, timeout=300)
        service_duration = time.time() - service_start
        system_logger.info(f"【PilotTTS】独立服务响应 | 状态码: {response.status_code} | 耗时: {service_duration:.3f}s")

        if response.status_code != 200:
            error_detail = response.json().get("detail", "未知错误")
            system_logger.error(f"【PilotTTS】独立服务错误: {error_detail}")
            raise HTTPException(status_code=response.status_code, detail=error_detail)

        result = response.json()
        if not result.get("success"):
            system_logger.error(f"【PilotTTS】合成失败: {result}")
            raise HTTPException(status_code=500, detail="PilotTTS 合成失败")

        audio_path = result.get("audio_path")
        sample_rate = result.get("sample_rate", 24000)
        system_logger.info(f"【PilotTTS】独立服务返回音频: {audio_path} | 采样率: {sample_rate}")

        if not audio_path or not os.path.exists(audio_path):
            system_logger.error(f"【PilotTTS】音频文件未生成: {audio_path}")
            raise HTTPException(status_code=500, detail="PilotTTS 音频文件未生成")

        # 读取音频并保存到 outputs 目录
        system_logger.info(f"【PilotTTS】读取音频文件: {audio_path}")
        audio_data, sr = sf.read(audio_path)
        output_path = save_temp_audio(audio_data, sr, prefix="pilottts", mode=mode,
                                      text=text, speaker_name=speaker_name)
        system_logger.info(f"【PilotTTS】音频保存完成: {output_path}")

        # 清理临时文件
        if temp_upload_path and os.path.exists(temp_upload_path):
            os.remove(temp_upload_path)
        try:
            os.remove(audio_path)
            system_logger.info(f"【PilotTTS】清理服务临时文件: {audio_path}")
        except Exception as e:
            system_logger.warning(f"【PilotTTS】清理服务临时文件失败: {e}")

        total_duration = time.time() - start_time
        actual_params = {
            "mode": mode,
            "emotion": emotion,
            "language": language,
            "speaker_name": speaker_name,
            "output_path": output_path,
            "sample_rate": sample_rate,
            "service_duration": round(service_duration, 3),
            "total_duration": round(total_duration, 3),
        }

        OperationLogger.log_tts_request("PilotTTS", text, actual_params, total_duration, "成功")
        system_logger.info(f"【PilotTTS】请求完成 | 总耗时: {total_duration:.3f}s")
        system_logger.info(f"【PilotTTS】{'='*60}")

        if output_format == "base64":
            audio_b64 = audio_to_base64(output_path)
            return TTSResponse(success=True, message="合成成功", audio_base64=audio_b64, sample_rate=sample_rate)
        else:
            return TTSResponse(success=True, message="合成成功",
                               audio_url=f"/audio/{os.path.basename(output_path)}", sample_rate=sample_rate)

    except HTTPException:
        raise
    except Exception as e:
        total_duration = time.time() - start_time
        OperationLogger.log_error("PilotTTS合成错误", str(e))
        OperationLogger.log_tts_request("PilotTTS", text, {"mode": mode}, total_duration, f"失败: {str(e)}")
        system_logger.error(f"【PilotTTS】合成错误: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_upload_path and os.path.exists(temp_upload_path):
            os.remove(temp_upload_path)
