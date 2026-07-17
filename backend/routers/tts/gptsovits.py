#!/usr/bin/env python3
"""
GPT-SoVITS 路由
通过独立服务调用 GPT-SoVITS（需要 transformers <=4.50，使用 lib/transformers4 的 4.51.3）
"""

import os
import time
import tempfile
from typing import Optional

import requests
import soundfile as sf
from fastapi import APIRouter, Form, UploadFile, File, Request, HTTPException

from backend.logger_config import OperationLogger, system_logger
from backend.models import TTSResponse
from backend.services import get_speaker_by_id
from backend.core import save_temp_audio, audio_to_base64, cleanup_memory
from backend.config import GPTSOVITS_HOST, GPTSOVITS_PORT

router = APIRouter()

GPTSOVITS_SERVICE_URL = f"http://{GPTSOVITS_HOST}:{GPTSOVITS_PORT}/tts"
GPTSOVITS_HEALTH_URL = f"http://{GPTSOVITS_HOST}:{GPTSOVITS_PORT}/health"


def _check_gptsovits_service():
    """检查 GPT-SoVITS 独立服务是否运行"""
    try:
        response = requests.get(GPTSOVITS_HEALTH_URL, timeout=3)
        return response.status_code == 200
    except:
        return False


@router.post("/")
async def tts_gptsovits(
        request: Request,
        text: str = Form(...),
        text_lang: str = Form("zh"),
        prompt_lang: str = Form("zh"),
        clone_speaker_id: Optional[str] = Form(None),
        prompt_text: Optional[str] = Form(None),
        top_k: int = Form(15),
        top_p: float = Form(1.0),
        temperature: float = Form(1.0),
        text_split_method: str = Form("cut5"),
        batch_size: int = Form(1),
        speed_factor: float = Form(1.0),
        version: str = Form("v2"),
        output_format: str = Form("url"),
        prompt_wav: Optional[UploadFile] = File(None),
):
    """GPT-SoVITS语音合成 - 通过独立服务"""
    ref_path = None
    start_time = time.time()
    client_ip = request.client.host if request.client else "unknown"

    system_logger.info(f"【GPT-SoVITS】{'='*60}")
    system_logger.info(f"【GPT-SoVITS】请求开始 | 版本: {version} | 客户端: {client_ip}")
    system_logger.info(f"【GPT-SoVITS】输入文本: {text[:100]}...")

    try:
        # 记录API请求
        request_params = {
            "text_preview": text[:50],
            "text_lang": text_lang,
            "prompt_lang": prompt_lang,
            "version": version,
            "clone_speaker_id": clone_speaker_id
        }
        OperationLogger.log_api_request("/tts/gptsovits", "POST", request_params, client_ip)

        # 检查独立服务状态
        system_logger.info(f"【GPT-SoVITS】检查独立服务状态...")
        if not _check_gptsovits_service():
            system_logger.error(f"【GPT-SoVITS】独立服务未运行")
            raise HTTPException(
                status_code=503,
                detail="GPT-SoVITS 独立服务未运行。请执行: nohup python gptsovits_service.py > logs/gptsovits_service.log 2>&1 &"
            )
        system_logger.info(f"【GPT-SoVITS】独立服务运行正常")

        # 处理参考音频
        if clone_speaker_id:
            system_logger.info(f"【GPT-SoVITS】查找说话人: {clone_speaker_id}")
            speaker = get_speaker_by_id(clone_speaker_id)
            if not speaker:
                system_logger.error(f"【GPT-SoVITS】说话人不存在: {clone_speaker_id}")
                raise HTTPException(status_code=404, detail=f"说话人不存在: {clone_speaker_id}")

            ref_path = speaker.get("audio_path")
            if not ref_path or not os.path.exists(ref_path):
                system_logger.error(f"【GPT-SoVITS】参考音频文件不存在: {ref_path}")
                raise HTTPException(status_code=404, detail="说话人音频文件不存在")

            if not prompt_text and speaker.get("reference_text"):
                prompt_text = speaker.get("reference_text")

            system_logger.info(f"【GPT-SoVITS】使用说话人 | 名称: {speaker.get('name')} | 音频: {ref_path} | 参考文本: {prompt_text}")

        elif prompt_wav:
            system_logger.info(f"【GPT-SoVITS】处理上传的参考音频...")
            file_content = await prompt_wav.read()
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                tmp_file.write(file_content)
                ref_path = tmp_file.name
            system_logger.info(f"【GPT-SoVITS】上传音频已保存: {ref_path}")
        else:
            system_logger.error(f"【GPT-SoVITS】缺少参考音频")
            raise HTTPException(status_code=400, detail="请提供参考音频或选择说话人")

        if not prompt_text:
            raise HTTPException(status_code=400, detail="请提供参考音频文本")

        # 调用独立服务
        data = {
            "text": text,
            "text_lang": text_lang,
            "prompt_text": prompt_text,
            "prompt_lang": prompt_lang,
            "ref_audio_path": ref_path,
            "top_k": top_k,
            "top_p": top_p,
            "temperature": temperature,
            "text_split_method": text_split_method,
            "batch_size": batch_size,
            "speed_factor": speed_factor,
            "version": version,
            "output_format": output_format,
        }

        system_logger.info(f"【GPT-SoVITS】调用独立服务: {GPTSOVITS_SERVICE_URL}")
        service_start = time.time()
        response = requests.post(GPTSOVITS_SERVICE_URL, data=data, timeout=120)
        service_duration = time.time() - service_start
        system_logger.info(f"【GPT-SoVITS】独立服务响应 | 状态码: {response.status_code} | 耗时: {service_duration:.3f}s")

        if response.status_code != 200:
            error_detail = response.json().get("detail", "未知错误")
            system_logger.error(f"【GPT-SoVITS】独立服务错误: {error_detail}")
            raise HTTPException(status_code=response.status_code, detail=error_detail)

        result = response.json()
        if not result.get("success"):
            system_logger.error(f"【GPT-SoVITS】合成失败: {result}")
            raise HTTPException(status_code=500, detail="GPT-SoVITS 合成失败")

        audio_path = result.get("audio_path")
        sr = result.get("sample_rate", 32000)
        system_logger.info(f"【GPT-SoVITS】独立服务返回音频: {audio_path} | 采样率: {sr}")

        if not audio_path or not os.path.exists(audio_path):
            system_logger.error(f"【GPT-SoVITS】音频文件未生成: {audio_path}")
            raise HTTPException(status_code=500, detail="GPT-SoVITS 音频文件未生成")

        # 读取并保存音频到 outputs 目录
        audio_data, sample_rate = sf.read(audio_path)
        output_path = save_temp_audio(audio_data, sample_rate, prefix="gptsovits")
        system_logger.info(f"【GPT-SoVITS】音频保存完成: {output_path}")

        # 清理临时文件
        if prompt_wav and ref_path and "/tmp" in ref_path:
            try:
                os.remove(ref_path)
            except:
                pass
        try:
            os.remove(audio_path)
        except:
            pass

        total_duration = time.time() - start_time

        actual_params = {
            "version": version,
            "text_lang": text_lang,
            "clone_speaker_id": clone_speaker_id,
            "speaker_name": speaker.get('name') if clone_speaker_id and 'speaker' in locals() else None,
            "service_duration": round(service_duration, 3),
            "total_duration": round(total_duration, 3)
        }
        OperationLogger.log_tts_request("GPT-SoVITS", text, actual_params, total_duration, "成功")
        system_logger.info(f"【GPT-SoVITS】请求完成 | 总耗时: {total_duration:.3f}s | 输出: {output_path}")
        system_logger.info(f"【GPT-SoVITS】{'='*60}")

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
                audio_url=f"/audio/{output_path.split('/')[-1]}",
                sample_rate=sample_rate
            )

    except HTTPException:
        raise
    except Exception as e:
        total_duration = time.time() - start_time
        OperationLogger.log_error("GPT-SoVITS合成错误", str(e))
        system_logger.error(f"【GPT-SoVITS】合成错误: {e}")
        system_logger.error(f"【GPT-SoVITS】{'='*60}")
        raise HTTPException(status_code=500, detail=str(e))
