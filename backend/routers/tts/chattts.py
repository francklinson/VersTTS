#!/usr/bin/env python3
"""
ChatTTS 路由
"""

import os
import time

import torch
from fastapi import APIRouter, Form, Request, HTTPException

from backend.logger_config import OperationLogger, system_logger
from backend.config import models
from backend.core import preprocess_text_for_chattts, save_temp_audio, audio_to_base64, cleanup_memory, log_gpu_memory_usage
from backend.engines import get_chattts_model
from backend.models import TTSResponse

router = APIRouter()


@router.post("/")
async def tts_chattts(
        request: Request,
        text: str = Form(...),
        temperature: float = Form(0.3),
        top_P: float = Form(0.7),
        top_K: float = Form(20),
        output_format: str = Form("url")
):
    """ChatTTS语音合成 - 使用随机说话人"""
    start_time = time.time()
    client_ip = request.client.host if request.client else "unknown"

    try:
        # 预处理文本
        original_text = text
        text = preprocess_text_for_chattts(text)
        if text != original_text:
            system_logger.info(f"【ChatTTS】文本预处理 | 原始: {original_text[:50]}... | 处理后: {text[:50]}...")

        # 记录API请求
        OperationLogger.log_api_request("/tts/chattts", "POST", {
            "text_preview": text[:50],
            "temperature": temperature,
            "top_P": top_P,
            "top_K": top_K,
        }, client_ip)

        system_logger.info(f"【ChatTTS】开始合成 | 文本: {text[:50]}... | 客户端: {client_ip}")
        chat = get_chattts_model()

        # 使用随机说话人
        spk_emb = chat.sample_random_speaker()
        system_logger.info(f"【ChatTTS】使用随机说话人")

        # 记录推理参数
        system_logger.info(f"【ChatTTS】推理参数 - temperature={temperature}, top_P={top_P}, top_K={top_K}")

        # 清理 GPU 缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gpu_mem_before = torch.cuda.memory_allocated() / 1024 ** 3
            system_logger.info(f"【ChatTTS】GPU内存清理完成，当前使用: {gpu_mem_before:.2f}GB")

        # 限制 temperature 最小值为 0.1
        safe_temperature = max(float(temperature), 0.1)
        if safe_temperature != float(temperature):
            system_logger.info(f"【ChatTTS】temperature 从 {temperature} 调整为 {safe_temperature}")

        params = chat.InferCodeParams(
            spk_emb=spk_emb,
            temperature=safe_temperature,
            top_P=float(top_P),
            top_K=int(top_K),
        )

        system_logger.info(f"【ChatTTS】开始推理...")
        infer_start = time.time()
        
        wavs = chat.infer(
            [text],
            stream=False,
            params_infer_code=params,
        )
        
        infer_duration = time.time() - infer_start
        system_logger.info(f"【ChatTTS】推理完成，耗时: {infer_duration:.3f}s")

        # 处理音频
        audio_data = wavs[0]
        if isinstance(audio_data, torch.Tensor):
            audio_data = audio_data.cpu().numpy()
        if audio_data.ndim > 1:
            audio_data = audio_data.squeeze()

        # 保存音频
        sr = 24000
        audio_path = save_temp_audio(audio_data, sr, prefix="chattts", text=original_text)

        # 清理显存 - 防止内存泄漏
        if torch.cuda.is_available():
            del wavs, audio_data
            cleanup_memory()
            log_gpu_memory_usage("ChatTTS")

        total_duration = time.time() - start_time
        OperationLogger.log_tts_request("ChatTTS", text, {
            "temperature": temperature,
            "output_format": output_format
        }, total_duration, "成功")

        system_logger.info(f"【ChatTTS】合成完成 | 总耗时: {total_duration:.3f}s | 音频: {audio_path}")

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
        total_duration = time.time() - start_time
        OperationLogger.log_error("ChatTTS合成错误", str(e))
        OperationLogger.log_tts_request("ChatTTS", text, {}, total_duration, f"失败: {str(e)}")
        system_logger.error(f"【ChatTTS】错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/speakers")
async def get_chattts_speakers():
    """获取ChatTTS可用的说话人列表（随机生成示例）"""
    return {
        "success": True,
        "message": "ChatTTS使用随机说话人，每次合成会自动生成新的音色",
        "speakers": []
    }
