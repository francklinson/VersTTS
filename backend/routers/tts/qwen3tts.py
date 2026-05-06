#!/usr/bin/env python3
"""
Qwen3-TTS 路由
"""

import os
import time
import torch
import transformers
from fastapi import APIRouter, Form, Request, HTTPException, UploadFile, File
from typing import Optional

from backend.logger_config import OperationLogger, system_logger
from backend.models import TTSResponse, Qwen3TTSModelStatus
from backend.engines import get_qwen3tts_model
from backend.services import load_speakers_db
from backend.core import save_temp_audio, audio_to_base64, cleanup_memory, log_gpu_memory_usage

router = APIRouter()


@router.post("/")
async def tts_qwen3tts(
        request: Request,
        text: str = Form(...),
        language: str = Form("zh"),
        model_size: str = Form("1.7B"),
        mode: str = Form("voice_clone"),
        # 注意：0.6B模型已弃用，传入任意值都将使用1.7B模型
        speaker: Optional[str] = Form(None),
        clone_speaker_id: Optional[str] = Form(None),
        ref_audio: Optional[str] = Form(None),
        ref_text: Optional[str] = Form(None),
        voice_design_prompt: Optional[str] = Form(None),
        instruct_text: Optional[str] = Form(None),
        x_vector_only_mode: bool = Form(False),
        streaming: bool = Form(False),
        output_format: str = Form("url")
):
    """Qwen3-TTS语音合成"""
    start_time = time.time()
    client_ip = request.client.host if request.client else "unknown"

    try:
        OperationLogger.log_api_request("/tts/qwen3tts", "POST", {
            "text_preview": text[:50],
            "model_size": model_size,
            "mode": mode,
            "speaker": speaker
        }, client_ip)

        system_logger.info(f"【Qwen3-TTS】开始合成 | 文本: {text[:50]}... | 模式: {mode}")

        # 检查 transformers 版本
        tv = transformers.__version__.split('.')
        major, minor = int(tv[0]), int(tv[1])
        if major < 4 or (major == 4 and minor < 57):
            raise HTTPException(
                status_code=503,
                detail=f"Qwen3-TTS 需要 transformers >= 4.57.0，当前版本为 {transformers.__version__}"
            )

        # 根据模式确定模型类型
        model_type_map = {
            "voice_clone": "VoiceClone",
            "custom_voice": "CustomVoice",
            "voice_design": "VoiceDesign",
            "base": "Base"
        }
        model_type = model_type_map.get(mode, "Base")
        system_logger.info(f"模式: {mode}, 选择模型类型: {model_type}")

        # 获取模型
        tts = get_qwen3tts_model(model_size, model_type)

        # 验证模型类型
        actual_model_type = getattr(tts.model, 'tts_model_type', 'unknown')
        system_logger.info(f"实际加载的模型类型: {actual_model_type}")

        # 根据模式调用不同的生成方法
        if mode == "voice_clone":
            # 声音克隆模式
            if not clone_speaker_id:
                raise HTTPException(status_code=400, detail="voice_clone 模式需要选择说话人")

            db = load_speakers_db()
            speaker_data = None
            for s in db.get("speakers", []):
                if s["id"] == clone_speaker_id:
                    speaker_data = s
                    break

            if not speaker_data:
                raise HTTPException(status_code=404, detail=f"说话人不存在: {clone_speaker_id}")

            audio_path = speaker_data.get("audio_path")
            if not audio_path or not os.path.exists(audio_path):
                raise HTTPException(status_code=404, detail=f"说话人音频文件不存在: {audio_path}")

            # 使用说话人保存的参考文本
            ref_text_to_use = speaker_data.get("reference_text", "")
            
            # 如果没有参考文本，强制使用 x_vector_only_mode
            effective_x_vector_mode = x_vector_only_mode
            if not ref_text_to_use and not x_vector_only_mode:
                system_logger.info(f"说话人 {speaker_data['name']} 没有参考文本，自动切换到 x_vector_only_mode=True")
                effective_x_vector_mode = True

            system_logger.info(f"voice_clone模式：使用说话人 {speaker_data['name']} 的音频")

            wavs, sr = tts.generate_voice_clone(
                text=text,
                language="Auto",
                ref_audio=audio_path,
                ref_text=ref_text_to_use,
                x_vector_only_mode=effective_x_vector_mode
            )
            wav = wavs[0] if isinstance(wavs, list) else wavs

        elif mode == "custom_voice":
            # 预设音色模式
            if not speaker:
                speaker = "vivian"
                system_logger.warning(f"speaker 参数为空，使用默认音色: {speaker}")

            system_logger.info(f"custom_voice模式：使用预设音色 {speaker}, 指令: {instruct_text or '无'}")

            # 尝试使用 generate_custom_voice
            custom_voice_success = False
            try:
                if hasattr(tts, 'generate_custom_voice'):
                    wavs, sr = tts.generate_custom_voice(
                        text=text,
                        language="Chinese",
                        speaker=speaker,
                        instruct=instruct_text or "",
                        do_sample=True,
                        temperature=0.9,
                        top_k=50,
                        top_p=1.0
                    )
                    wav = wavs[0] if isinstance(wavs, list) else wavs
                    custom_voice_success = True
                    system_logger.info(f"使用 CustomVoice 模型生成成功 | 音色: {speaker}")
            except (ValueError, NotImplementedError) as e:
                if "does not support generate_custom_voice" in str(e) or "not implemented" in str(e).lower():
                    system_logger.warning(f"CustomVoice 模型不支持: {e}")
                else:
                    raise

            if not custom_voice_success:
                # 回退到 Base 模型的 voice_clone
                system_logger.warning(f"当前模型不支持 generate_custom_voice，回退到 Base 模型")
                tts_base = get_qwen3tts_model(model_size, "Base")
                default_ref_audio = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone_1.wav"
                default_ref_text = "甚至出现交易几乎停滞的情况。"
                wavs, sr = tts_base.generate_voice_clone(
                    text=text,
                    language="Auto",
                    ref_audio=default_ref_audio,
                    ref_text=default_ref_text,
                    x_vector_only_mode=True
                )
                wav = wavs[0] if isinstance(wavs, list) else wavs

        elif mode == "voice_design":
            # 音色设计模式
            if not voice_design_prompt:
                raise HTTPException(status_code=400, detail="voice_design 模式需要提供 voice_design_prompt 参数")

            system_logger.info(f"voice_design模式：音色描述: {voice_design_prompt}")

            voice_design_success = False
            try:
                if hasattr(tts, 'generate_voice_design'):
                    wavs, sr = tts.generate_voice_design(
                        text=text,
                        language="Auto",
                        instruct=voice_design_prompt
                    )
                    wav = wavs[0] if isinstance(wavs, list) else wavs
                    voice_design_success = True
                    system_logger.info("使用 VoiceDesign 模型生成成功")
            except ValueError as e:
                if "does not support generate_voice_design" in str(e):
                    system_logger.warning(f"VoiceDesign 模型不支持: {e}")
                else:
                    raise

            if not voice_design_success:
                # 回退到 Base 模型
                system_logger.warning(f"当前模型不支持 generate_voice_design，回退到 Base 模型")
                tts_base = get_qwen3tts_model(model_size, "Base")
                default_ref_audio = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone_1.wav"
                default_ref_text = "甚至出现交易几乎停滞的情况。"
                wavs, sr = tts_base.generate_voice_clone(
                    text=text,
                    language="Auto",
                    ref_audio=default_ref_audio,
                    ref_text=default_ref_text,
                    x_vector_only_mode=True
                )
                wav = wavs[0] if isinstance(wavs, list) else wavs

        else:
            # base 模式 - 使用默认音色
            system_logger.info("base模式：使用默认音色")
            default_ref_audio = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone_1.wav"
            default_ref_text = "甚至出现交易几乎停滞的情况。"
            wavs, sr = tts.generate_voice_clone(
                text=text,
                language="Auto",
                ref_audio=default_ref_audio,
                ref_text=default_ref_text,
                x_vector_only_mode=True
            )
            wav = wavs[0] if isinstance(wavs, list) else wavs

        # 保存音频
        import numpy as np
        if isinstance(wav, list):
            audio_data = np.array(wav)
        else:
            audio_data = wav

        audio_path = save_temp_audio(audio_data, sr)

        # 清理显存 - 防止内存泄漏
        if torch.cuda.is_available():
            del audio_data, wav
            cleanup_memory()
            log_gpu_memory_usage("Qwen3-TTS")

        total_duration = time.time() - start_time
        OperationLogger.log_tts_request("Qwen3-TTS", text, {
            "model_size": model_size,
            "mode": mode
        }, total_duration, "成功")

        system_logger.info(f"【Qwen3-TTS】合成完成 | 耗时: {total_duration:.3f}s")

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
        OperationLogger.log_error("Qwen3-TTS合成错误", str(e))
        OperationLogger.log_tts_request("Qwen3-TTS", text, {}, total_duration, f"失败: {str(e)}")
        system_logger.error(f"【Qwen3-TTS】错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def qwen3tts_status():
    """获取Qwen3-TTS模型状态"""
    try:
        import transformers
        tv = transformers.__version__.split('.')
        major, minor = int(tv[0]), int(tv[1])
        meets_req = major > 4 or (major == 4 and minor >= 57)

        return Qwen3TTSModelStatus(
            available=meets_req,
            model_size="1.7B (仅支持)",
            model_type="Base/CustomVoice/VoiceDesign",
            transformers_version=transformers.__version__,
            meets_requirement=meets_req,
            message="模型可用" if meets_req else f"需要 transformers >= 4.57.0，当前为 {transformers.__version__}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取状态失败: {str(e)}")


@router.get("/speakers")
async def get_qwen3tts_speakers():
    """获取Qwen3-TTS可用的预设音色列表"""
    speakers = [
        {"id": "vivian", "name": "Vivian", "gender": "female", "description": "温柔女声"},
        {"id": "serena", "name": "Serena", "gender": "female", "description": "活泼女声"},
        {"id": "uncle_fu", "name": "Uncle Fu", "gender": "male", "description": "稳重男声"},
        {"id": "dylan", "name": "Dylan", "gender": "male", "description": "阳光男声"},
        {"id": "eric", "name": "Eric", "gender": "male", "description": "成熟男声"},
        {"id": "ryan", "name": "Ryan", "gender": "male", "description": "年轻男声"},
        {"id": "aiden", "name": "Aiden", "gender": "male", "description": "磁性男声"},
        {"id": "ono_anna", "name": "Ono Anna", "gender": "female", "description": "甜美女声"},
        {"id": "sohee", "name": "Sohee", "gender": "female", "description": "韩语女声"},
    ]
    return {
        "success": True,
        "speakers": speakers,
        "total": len(speakers)
    }
