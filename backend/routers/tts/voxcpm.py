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
from backend.core import audio_to_base64, cleanup_memory, log_gpu_memory_usage, save_temp_audio

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
        control_prompt: Optional[str] = Form(None),  # clone模式的控制指令
        cfg_value: float = Form(2.0),
        inference_timesteps: int = Form(10),
        output_format: str = Form("url")
):
    """VoxCPM语音合成 - 支持30种语言的无Tokenizer TTS"""
    ref_path = None
    temp_upload_path = None
    start_time = time.time()
    client_ip = request.client.host if request.client else "unknown"
    
    # 详细日志：请求开始
    system_logger.info(f"【VoxCPM】{'='*60}")
    system_logger.info(f"【VoxCPM】请求开始 | 模式: {mode} | 客户端: {client_ip}")
    system_logger.info(f"【VoxCPM】输入文本: {text[:100]}...")

    try:
        # 记录API请求参数
        request_params = {
            "text_preview": text[:50],
            "mode": mode,
            "clone_speaker_id": clone_speaker_id,
            "voice_design_prompt": voice_design_prompt[:50] if voice_design_prompt else None,
            "control_prompt": control_prompt[:50] if control_prompt else None,
            "cfg_value": cfg_value,
            "inference_timesteps": inference_timesteps,
            "has_upload": ref_audio is not None
        }
        OperationLogger.log_api_request("/tts/voxcpm", "POST", request_params, client_ip)
        system_logger.info(f"【VoxCPM】请求参数: {request_params}")

        # 获取说话人信息（如果提供了clone_speaker_id）
        speaker = None
        speaker_ref_text = None
        if clone_speaker_id:
            system_logger.info(f"【VoxCPM】查找说话人: {clone_speaker_id}")
            speaker = get_speaker_by_id(clone_speaker_id)
            if not speaker:
                system_logger.error(f"【VoxCPM】说话人不存在: {clone_speaker_id}")
                raise HTTPException(status_code=404, detail=f"说话人不存在: {clone_speaker_id}")
            ref_path = speaker.get("audio_path")
            speaker_ref_text = speaker.get("reference_text")
            system_logger.info(f"【VoxCPM】找到说话人 | 名称: {speaker.get('name')} | 音频: {ref_path}")
            system_logger.info(f"【VoxCPM】参考文本: {speaker_ref_text[:50] if speaker_ref_text else '无'}...")

            # 检查音频文件是否存在
            if ref_path and os.path.exists(ref_path):
                file_size = os.path.getsize(ref_path)
                system_logger.info(f"【VoxCPM】音频文件存在 | 大小: {file_size / 1024:.2f} KB")
            else:
                system_logger.error(f"【VoxCPM】参考音频文件不存在: {ref_path}")
                raise HTTPException(status_code=404, detail=f"参考音频文件不存在: {ref_path}")

        # 兼容旧版：如果直接上传了参考音频
        if ref_audio and mode in ["clone", "ultimate_clone"]:
            system_logger.info(f"【VoxCPM】处理上传的参考音频...")
            temp_upload_path = f"uploads/voxcpm_ref_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.wav"
            with open(temp_upload_path, "wb") as f:
                f.write(await ref_audio.read())
            ref_path = temp_upload_path
            system_logger.info(f"【VoxCPM】上传音频已保存: {ref_path}")

        # 加载模型
        system_logger.info(f"【VoxCPM】加载模型...")
        model_load_start = time.time()
        model = get_voxcpm_model()
        model_load_duration = time.time() - model_load_start
        system_logger.info(f"【VoxCPM】模型加载完成 | 耗时: {model_load_duration:.3f}s")

        # 检查模型类型
        voxcpm_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "algorithms", "VoxCPM", "src")
        if voxcpm_path not in sys.path:
            sys.path.insert(0, voxcpm_path)
        from voxcpm.model.voxcpm2 import VoxCPM2Model
        is_v2 = isinstance(model.tts_model, VoxCPM2Model)
        system_logger.info(f"【VoxCPM】模型类型: {'VoxCPM2' if is_v2 else 'VoxCPM v1'}")
        
        # 记录模型信息
        system_logger.info(f"【VoxCPM】模型信息:")
        system_logger.info(f"【VoxCPM】   - 模型名称: VoxCPM{'2' if is_v2 else ''}")
        system_logger.info(f"【VoxCPM】   - 模型类型: {'VoxCPM2 (双版本模型)' if is_v2 else 'VoxCPM v1 (单版本模型)'}")
        system_logger.info(f"【VoxCPM】   - 模型来源: 离线部署")
        system_logger.info(f"【VoxCPM】   - 模型路径: /home/zhouchenghao/PycharmProjects/VersTTS/algorithms/VoxCPM")
        system_logger.info(f"【VoxCPM】   - 支持模式: base, clone, voice_design, ultimate_clone")
        system_logger.info(f"【VoxCPM】   - 生成模式: {'Reference-only / Continuation / Combined' if is_v2 else 'Continuation'}")
        system_logger.info(f"【VoxCPM】   - 采样率: 48000 Hz")
        system_logger.info(f"【VoxCPM】   - 支持语言: 30+种语言（无Tokenizer设计）")
        system_logger.info(f"【VoxCPM】   - CFG参数: {cfg_value}")
        system_logger.info(f"【VoxCPM】   - 推理步数: {inference_timesteps}")

        # 构建生成参数
        generate_kwargs = {
            "cfg_value": cfg_value,
            "inference_timesteps": inference_timesteps
        }
        
        actual_params = {
            "mode": mode,
            "cfg_value": cfg_value,
            "inference_timesteps": inference_timesteps,
            "is_v2": is_v2
        }

        # 根据模式处理text和参考音频
        if mode == "voice_design":
            # voice_design模式: 在text前添加(voice description)
            system_logger.info(f"【VoxCPM】进入音色设计模式")
            if voice_design_prompt:
                generate_kwargs["text"] = f"({voice_design_prompt}){text}"
                actual_params["voice_design_prompt"] = voice_design_prompt[:100]
                system_logger.info(f"【VoxCPM】音色描述: {voice_design_prompt}")
            else:
                generate_kwargs["text"] = f"(A natural speaking voice){text}"
                actual_params["voice_design_prompt"] = "A natural speaking voice"
                system_logger.info("【VoxCPM】使用默认音色描述")

        elif mode == "clone" and ref_path:
            # clone模式: 使用Reference-only mode进行声音克隆
            system_logger.info(f"【VoxCPM】进入声音克隆模式")
            actual_params["clone_speaker_id"] = clone_speaker_id
            actual_params["speaker_name"] = speaker.get('name') if speaker else None
            actual_params["ref_audio"] = ref_path
            
            # 如果有控制指令，将控制指令添加到文本前面
            if control_prompt:
                generate_kwargs["text"] = f"({control_prompt}){text}"
                actual_params["control_prompt"] = control_prompt[:50]
                system_logger.info(f"【VoxCPM】控制指令: {control_prompt}")
            else:
                generate_kwargs["text"] = text
                system_logger.info(f"【VoxCPM】无控制指令")

            if is_v2:
                # VoxCPM2: 使用 reference_wav_path (Reference-only mode)
                generate_kwargs["reference_wav_path"] = ref_path
                actual_params["generation_mode"] = "Reference-only"
                system_logger.info(f"【VoxCPM】使用Reference-only mode")
            else:
                # VoxCPM v1: 只能使用 prompt_wav_path (Continuation mode)
                generate_kwargs["prompt_wav_path"] = ref_path
                if speaker_ref_text:
                    generate_kwargs["prompt_text"] = speaker_ref_text
                actual_params["generation_mode"] = "Continuation"
                system_logger.info(f"【VoxCPM】使用Continuation mode (VoxCPM v1)")

        elif mode == "ultimate_clone" and ref_path:
            # ultimate_clone模式: 使用Combined mode进行极致克隆
            system_logger.info(f"【VoxCPM】进入极致克隆模式")
            actual_params["clone_speaker_id"] = clone_speaker_id
            actual_params["speaker_name"] = speaker.get('name') if speaker else None
            actual_params["ref_audio"] = ref_path
            generate_kwargs["text"] = text
            prompt_text = speaker_ref_text if speaker else ref_text

            if is_v2 and prompt_text:
                # VoxCPM2 Combined mode
                generate_kwargs["reference_wav_path"] = ref_path
                generate_kwargs["prompt_wav_path"] = ref_path
                generate_kwargs["prompt_text"] = prompt_text
                actual_params["generation_mode"] = "Combined"
                actual_params["prompt_text"] = prompt_text[:50] if prompt_text else None
                system_logger.info(f"【VoxCPM】使用Combined mode")
            elif prompt_text:
                # VoxCPM v1 或没有 reference_wav_path
                generate_kwargs["prompt_wav_path"] = ref_path
                generate_kwargs["prompt_text"] = prompt_text
                actual_params["generation_mode"] = "Continuation"
                actual_params["prompt_text"] = prompt_text[:50] if prompt_text else None
                system_logger.info(f"【VoxCPM】使用Continuation mode")
            else:
                # 没有参考文本，退化为clone模式
                generate_kwargs["reference_wav_path"] = ref_path
                actual_params["generation_mode"] = "Reference-only (fallback)"
                system_logger.warning(f"【VoxCPM】极致克隆缺少参考文本，退化为声音克隆模式")
        else:
            # base模式: 基础生成
            system_logger.info(f"【VoxCPM】进入基础生成模式")
            generate_kwargs["text"] = text
            actual_params["generation_mode"] = "Base"

        # 生成音频
        system_logger.info(f"【VoxCPM】开始生成音频...")
        system_logger.info(f"【VoxCPM】生成参数: {generate_kwargs}")
        gen_start = time.time()
        audio_data = model.generate(**generate_kwargs)
        gen_duration = time.time() - gen_start
        system_logger.info(f"【VoxCPM】音频生成完成 | 耗时: {gen_duration:.3f}s | 数据类型: {type(audio_data)}")
        
        actual_params["generation_duration"] = round(gen_duration, 3)

        # 保存音频（使用有意义的文件名）
        save_start = time.time()
        speaker_name_val = speaker.get('name') if speaker else None
        audio_path = save_temp_audio(
            audio_data, 48000, prefix="voxcpm", mode=mode,
            text=text, speaker_name=speaker_name_val,
            instruct_prompt=(control_prompt or voice_design_prompt)
        )
        save_duration = time.time() - save_start
        file_size = os.path.getsize(audio_path)
        system_logger.info(f"【VoxCPM】音频保存完成: {audio_path} | 大小: {file_size / 1024:.2f} KB | 耗时: {save_duration:.3f}s")
        
        actual_params["output_path"] = audio_path
        actual_params["file_size_kb"] = round(file_size / 1024, 2)

        # 清理显存 - 防止内存泄漏
        if torch.cuda.is_available():
            system_logger.info(f"【VoxCPM】清理GPU显存...")
            cleanup_memory()
            log_gpu_memory_usage("VoxCPM")

        total_duration = time.time() - start_time
        actual_params["total_duration"] = round(total_duration, 3)
        
        OperationLogger.log_tts_request("VoxCPM", text, actual_params, total_duration, "成功")
        system_logger.info(f"【VoxCPM】请求完成 | 总耗时: {total_duration:.3f}s | 输出: {audio_path}")
        system_logger.info(f"【VoxCPM】{'='*60}")

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
        OperationLogger.log_tts_request("VoxCPM", text, {"mode": mode}, total_duration, f"失败: {str(e)}")
        system_logger.error(f"【VoxCPM】合成错误: {e}")
        system_logger.error(f"【VoxCPM】{'='*60}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # 清理临时上传的文件
        if temp_upload_path and os.path.exists(temp_upload_path):
            os.remove(temp_upload_path)
            system_logger.info(f"【VoxCPM】清理临时文件: {temp_upload_path}")
