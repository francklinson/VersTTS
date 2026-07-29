#!/usr/bin/env python3
"""
dots.tts 路由
"""

import time
import os

import torch
import numpy as np
from datetime import datetime
from fastapi import APIRouter, Form, Request, HTTPException
from typing import Optional

from backend.logger_config import OperationLogger, system_logger
from backend.models import TTSResponse
from backend.engines import get_dotstts_model
from backend.services import get_speaker_by_id
from backend.core import audio_to_base64, cleanup_memory, log_gpu_memory_usage, save_temp_audio

router = APIRouter()


@router.post("/")
async def tts_dotstts(
        request: Request,
        text: str = Form(...),
        mode: str = Form("voice_clone"),
        clone_speaker_id: Optional[str] = Form(None),
        language: Optional[str] = Form(None),
        num_steps: int = Form(16),
        guidance_scale: float = Form(1.2),
        speaker_scale: float = Form(1.5),
        output_format: str = Form("url")
):
    """dots.tts 语音合成 - 支持连续自回归+流匹配的TTS系统"""
    start_time = time.time()
    client_ip = request.client.host if request.client else "unknown"

    # 详细日志：请求开始
    system_logger.info(f"【dots.tts】{'='*60}")
    system_logger.info(f"【dots.tts】请求开始 | 模式: {mode} | 客户端: {client_ip}")
    system_logger.info(f"【dots.tts】输入文本: {text[:100]}...")

    try:
        # 记录API请求参数
        request_params = {
            "text_preview": text[:50],
            "mode": mode,
            "clone_speaker_id": clone_speaker_id,
            "num_steps": num_steps,
            "guidance_scale": guidance_scale,
            "speaker_scale": speaker_scale,
        }
        OperationLogger.log_api_request("/tts/dotstts", "POST", request_params, client_ip)
        system_logger.info(f"【dots.tts】请求参数: {request_params}")

        # 获取说话人信息（如果提供了clone_speaker_id）
        ref_path = None
        ref_text = None
        speaker_name = None
        if clone_speaker_id:
            system_logger.info(f"【dots.tts】查找说话人: {clone_speaker_id}")
            speaker = get_speaker_by_id(clone_speaker_id)
            if not speaker:
                system_logger.error(f"【dots.tts】说话人不存在: {clone_speaker_id}")
                raise HTTPException(status_code=404, detail=f"说话人不存在: {clone_speaker_id}")
            ref_path = speaker.get("audio_path")
            ref_text = speaker.get("reference_text")
            speaker_name = speaker.get("name")
            system_logger.info(f"【dots.tts】找到说话人 | 名称: {speaker_name} | 音频: {ref_path}")

            # 检查音频文件是否存在
            if ref_path and os.path.exists(ref_path):
                file_size = os.path.getsize(ref_path)
                system_logger.info(f"【dots.tts】音频文件存在 | 大小: {file_size / 1024:.2f} KB")
            else:
                system_logger.error(f"【dots.tts】参考音频文件不存在: {ref_path}")
                raise HTTPException(status_code=404, detail=f"参考音频文件不存在: {ref_path}")

        # 加载模型
        system_logger.info(f"【dots.tts】加载模型...")
        model_load_start = time.time()
        model = get_dotstts_model()
        model_load_duration = time.time() - model_load_start
        system_logger.info(f"【dots.tts】模型加载完成 | 耗时: {model_load_duration:.3f}s")

        # 记录模型信息
        system_logger.info(f"【dots.tts】模型信息:")
        system_logger.info(f"【dots.tts】   - 模型名称: dots.tts")
        system_logger.info(f"【dots.tts】   - 采样率: 48000 Hz")
        system_logger.info(f"【dots.tts】   - 支持语言: 24种+粤语")
        system_logger.info(f"【dots.tts】   - 生成模式: {mode}")
        system_logger.info(f"【dots.tts】   - 推理步数: {num_steps}")
        system_logger.info(f"【dots.tts】   - guidance_scale: {guidance_scale}")
        system_logger.info(f"【dots.tts】   - speaker_scale: {speaker_scale}")

        # 模板映射
        template_map = {
            "voice_clone": "tts",
            "instruct": "instruction_tts",
        }
        template_name = template_map.get(mode, "tts")

        # 构建生成参数
        generate_kwargs = {
            "text": text,
            "num_steps": num_steps,
            "guidance_scale": guidance_scale,
            "speaker_scale": speaker_scale,
            "template_name": template_name,
        }

        actual_params = {
            "mode": mode,
            "num_steps": num_steps,
            "guidance_scale": guidance_scale,
            "speaker_scale": speaker_scale,
            "template_name": template_name,
        }

        if language:
            generate_kwargs["language"] = language
            actual_params["language"] = language

        if mode == "voice_clone" and ref_path:
            # 声音克隆模式：使用参考音频和文本
            generate_kwargs["prompt_audio_path"] = ref_path
            if ref_text:
                generate_kwargs["prompt_text"] = ref_text
                actual_params["has_ref_text"] = True
            actual_params["clone_speaker_id"] = clone_speaker_id
            actual_params["speaker_name"] = speaker_name
            actual_params["ref_audio"] = ref_path
            system_logger.info(f"【dots.tts】声音克隆模式 | 参考音频: {ref_path}")
        elif mode == "voice_clone" and not ref_path:
            # 无参考音频的voice_clone模式，退化为普通TTS
            system_logger.warning(f"【dots.tts】声音克隆模式未提供说话人ID或参考音频，使用基础TTS模式")
            generate_kwargs["template_name"] = "tts"
            actual_params["generation_mode"] = "base_tts_fallback"
        elif mode == "instruct":
            # 指令TTS模式
            system_logger.info(f"【dots.tts】指令TTS模式 | 使用 instruction_tts 模板")
            actual_params["generation_mode"] = "instruction_tts"

        # 生成音频
        system_logger.info(f"【dots.tts】开始生成音频...")
        system_logger.info(f"【dots.tts】生成参数: {generate_kwargs}")
        gen_start = time.time()
        result = model.generate(**generate_kwargs)
        gen_duration = time.time() - gen_start
        system_logger.info(f"【dots.tts】音频生成完成 | 耗时: {gen_duration:.3f}s")

        actual_params["generation_duration"] = round(gen_duration, 3)

        # 提取音频数据
        audio = result["audio"]  # torch.Tensor, shape (1, samples)
        sample_rate = result["sample_rate"]

        # 转换numpy
        if torch.is_tensor(audio):
            audio_np = audio.cpu().numpy().squeeze()
        elif isinstance(audio, np.ndarray):
            audio_np = audio.squeeze()
        else:
            raise TypeError(f"不支持的音频类型: {type(audio)}")

        # 保存音频
        save_start = time.time()
        audio_path = save_temp_audio(
            audio_np, sample_rate, prefix="dotstts", mode=mode,
            text=text, speaker_name=speaker_name
        )
        save_duration = time.time() - save_start
        file_size = os.path.getsize(audio_path)
        system_logger.info(f"【dots.tts】音频保存完成: {audio_path} | 大小: {file_size / 1024:.2f} KB | 耗时: {save_duration:.3f}s")

        actual_params["output_path"] = audio_path
        actual_params["file_size_kb"] = round(file_size / 1024, 2)

        # 清理显存
        if torch.cuda.is_available():
            system_logger.info(f"【dots.tts】清理GPU显存...")
            cleanup_memory()
            log_gpu_memory_usage("dots.tts")

        total_duration = time.time() - start_time
        actual_params["total_duration"] = round(total_duration, 3)

        OperationLogger.log_tts_request("dots.tts", text, actual_params, total_duration, "成功")
        system_logger.info(f"【dots.tts】请求完成 | 总耗时: {total_duration:.3f}s | 输出: {audio_path}")
        system_logger.info(f"【dots.tts】{'='*60}")

        if output_format == "base64":
            audio_b64 = audio_to_base64(audio_path)
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
                audio_url=f"/audio/{os.path.basename(audio_path)}",
                sample_rate=sample_rate
            )

    except HTTPException:
        raise
    except Exception as e:
        total_duration = time.time() - start_time
        OperationLogger.log_error("dots.tts合成错误", str(e))
        OperationLogger.log_tts_request("dots.tts", text, {"mode": mode}, total_duration, f"失败: {str(e)}")
        system_logger.error(f"【dots.tts】合成错误: {e}")
        system_logger.error(f"【dots.tts】{'='*60}")
        raise HTTPException(status_code=500, detail=str(e))
