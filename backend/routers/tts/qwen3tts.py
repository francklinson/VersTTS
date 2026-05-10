#!/usr/bin/env python3
"""
Qwen3-TTS 路由
"""

import os
import time
import torch
import transformers
import numpy as np
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
    
    # 详细日志：请求开始
    system_logger.info(f"【Qwen3-TTS】{'='*60}")
    system_logger.info(f"【Qwen3-TTS】请求开始 | 模式: {mode} | 客户端: {client_ip}")
    system_logger.info(f"【Qwen3-TTS】输入文本: {text[:100]}...")

    try:
        # 记录API请求参数
        request_params = {
            "text_preview": text[:50],
            "model_size": model_size,
            "mode": mode,
            "speaker": speaker,
            "clone_speaker_id": clone_speaker_id,
            "voice_design_prompt": voice_design_prompt[:50] if voice_design_prompt else None,
            "instruct_text": instruct_text[:50] if instruct_text else None,
            "x_vector_only_mode": x_vector_only_mode
        }
        OperationLogger.log_api_request("/tts/qwen3tts", "POST", request_params, client_ip)
        system_logger.info(f"【Qwen3-TTS】请求参数: {request_params}")

        # 检查 transformers 版本
        tv = transformers.__version__.split('.')
        major, minor = int(tv[0]), int(tv[1])
        system_logger.info(f"【Qwen3-TTS】Transformers版本: {transformers.__version__}")
        if major < 4 or (major == 4 and minor < 57):
            system_logger.error(f"【Qwen3-TTS】版本不兼容，需要 >= 4.57.0")
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
        system_logger.info(f"【Qwen3-TTS】模式映射: {mode} -> {model_type}")
        
        # 记录模型信息
        system_logger.info(f"【Qwen3-TTS】模型信息:")
        system_logger.info(f"【Qwen3-TTS】   - 模型名称: Qwen3-TTS")
        system_logger.info(f"【Qwen3-TTS】   - 模型大小: {model_size}")
        system_logger.info(f"【Qwen3-TTS】   - 模型类型: {model_type}")
        system_logger.info(f"【Qwen3-TTS】   - 模型来源: ModelScope")
        system_logger.info(f"【Qwen3-TTS】   - 模型路径: /home/zhouchenghao/PycharmProjects/VersTTS/models/Qwen/Qwen3-TTS-1.7B")
        system_logger.info(f"【Qwen3-TTS】   - Transformers版本: {transformers.__version__}")
        system_logger.info(f"【Qwen3-TTS】   - 采样率: 24000 Hz")
        system_logger.info(f"【Qwen3-TTS】   - 支持语言: 中文、英文、日语、韩语、粤语、法语、德语、意大利语、俄语、西班牙语")

        # 获取模型
        system_logger.info(f"【Qwen3-TTS】加载模型: {model_size} / {model_type}")
        model_load_start = time.time()
        tts = get_qwen3tts_model(model_size, model_type)
        model_load_duration = time.time() - model_load_start
        system_logger.info(f"【Qwen3-TTS】模型加载完成 | 耗时: {model_load_duration:.3f}s")

        # 验证模型类型
        actual_model_type = getattr(tts.model, 'tts_model_type', 'unknown')
        system_logger.info(f"【Qwen3-TTS】实际模型类型: {actual_model_type}")

        wav = None
        sr = 24000
        actual_params = {"mode": mode}

        # 根据模式调用不同的生成方法
        if mode == "voice_clone":
            # 声音克隆模式
            system_logger.info(f"【Qwen3-TTS】进入Voice Clone模式")
            if not clone_speaker_id:
                system_logger.error(f"【Qwen3-TTS】缺少clone_speaker_id参数")
                raise HTTPException(status_code=400, detail="voice_clone 模式需要选择说话人")

            system_logger.info(f"【Qwen3-TTS】查找说话人: {clone_speaker_id}")
            db = load_speakers_db()
            speaker_data = None
            for s in db.get("speakers", []):
                if s["id"] == clone_speaker_id:
                    speaker_data = s
                    break

            if not speaker_data:
                system_logger.error(f"【Qwen3-TTS】说话人不存在: {clone_speaker_id}")
                raise HTTPException(status_code=404, detail=f"说话人不存在: {clone_speaker_id}")

            audio_path = speaker_data.get("audio_path")
            system_logger.info(f"【Qwen3-TTS】找到说话人 | 名称: {speaker_data.get('name')} | 音频: {audio_path}")
            
            if not audio_path or not os.path.exists(audio_path):
                system_logger.error(f"【Qwen3-TTS】参考音频不存在: {audio_path}")
                raise HTTPException(status_code=404, detail=f"说话人音频文件不存在: {audio_path}")

            # 使用说话人保存的参考文本
            ref_text_to_use = speaker_data.get("reference_text", "")
            system_logger.info(f"【Qwen3-TTS】参考文本: {ref_text_to_use[:50] if ref_text_to_use else '无'}...")
            
            # 如果没有参考文本，强制使用 x_vector_only_mode
            effective_x_vector_mode = x_vector_only_mode
            if not ref_text_to_use and not x_vector_only_mode:
                system_logger.info(f"【Qwen3-TTS】说话人无参考文本，自动切换x_vector_only_mode=True")
                effective_x_vector_mode = True

            system_logger.info(f"【Qwen3-TTS】调用generate_voice_clone | ref_audio={audio_path} | x_vector_only={effective_x_vector_mode}")
            gen_start = time.time()
            wavs, sr = tts.generate_voice_clone(
                text=text,
                language="Auto",
                ref_audio=audio_path,
                ref_text=ref_text_to_use,
                x_vector_only_mode=effective_x_vector_mode
            )
            gen_duration = time.time() - gen_start
            wav = wavs[0] if isinstance(wavs, list) else wavs
            system_logger.info(f"【Qwen3-TTS】Voice Clone生成完成 | 耗时: {gen_duration:.3f}s")
            
            actual_params.update({
                "clone_speaker_id": clone_speaker_id,
                "speaker_name": speaker_data.get('name'),
                "ref_audio": audio_path,
                "ref_text": ref_text_to_use[:50] if ref_text_to_use else None,
                "x_vector_only_mode": effective_x_vector_mode,
                "generation_duration": round(gen_duration, 3)
            })

        elif mode == "custom_voice":
            # 预设音色模式
            system_logger.info(f"【Qwen3-TTS】进入Custom Voice模式")
            if not speaker:
                speaker = "vivian"
                system_logger.warning(f"【Qwen3-TTS】speaker为空，使用默认: {speaker}")

            system_logger.info(f"【Qwen3-TTS】预设音色: {speaker} | 指令: {instruct_text or '无'}")
            actual_params["speaker"] = speaker
            actual_params["instruct_text"] = instruct_text[:50] if instruct_text else None

            # 尝试使用 generate_custom_voice
            custom_voice_success = False
            try:
                if hasattr(tts, 'generate_custom_voice'):
                    system_logger.info(f"【Qwen3-TTS】调用generate_custom_voice")
                    gen_start = time.time()
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
                    gen_duration = time.time() - gen_start
                    wav = wavs[0] if isinstance(wavs, list) else wavs
                    custom_voice_success = True
                    actual_params["generation_duration"] = round(gen_duration, 3)
                    system_logger.info(f"【Qwen3-TTS】CustomVoice生成成功 | 耗时: {gen_duration:.3f}s")
            except (ValueError, NotImplementedError) as e:
                if "does not support generate_custom_voice" in str(e) or "not implemented" in str(e).lower():
                    system_logger.warning(f"【Qwen3-TTS】CustomVoice不支持: {e}")
                else:
                    raise

            if not custom_voice_success:
                # 回退到 Base 模型的 voice_clone
                system_logger.warning(f"【Qwen3-TTS】回退到Base模型voice_clone")
                tts_base = get_qwen3tts_model(model_size, "Base")
                default_ref_audio = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone_1.wav"
                default_ref_text = "甚至出现交易几乎停滞的情况。"
                gen_start = time.time()
                wavs, sr = tts_base.generate_voice_clone(
                    text=text,
                    language="Auto",
                    ref_audio=default_ref_audio,
                    ref_text=default_ref_text,
                    x_vector_only_mode=True
                )
                gen_duration = time.time() - gen_start
                wav = wavs[0] if isinstance(wavs, list) else wavs
                actual_params["fallback"] = "Base.voice_clone"
                actual_params["generation_duration"] = round(gen_duration, 3)
                system_logger.info(f"【Qwen3-TTS】Base VoiceClone生成成功 | 耗时: {gen_duration:.3f}s")

        elif mode == "voice_design":
            # 音色设计模式
            system_logger.info(f"【Qwen3-TTS】进入Voice Design模式")
            if not voice_design_prompt:
                system_logger.error(f"【Qwen3-TTS】缺少voice_design_prompt参数")
                raise HTTPException(status_code=400, detail="voice_design 模式需要提供 voice_design_prompt 参数")

            system_logger.info(f"【Qwen3-TTS】音色描述: {voice_design_prompt}")
            actual_params["voice_design_prompt"] = voice_design_prompt[:100]

            voice_design_success = False
            try:
                if hasattr(tts, 'generate_voice_design'):
                    system_logger.info(f"【Qwen3-TTS】调用generate_voice_design")
                    gen_start = time.time()
                    wavs, sr = tts.generate_voice_design(
                        text=text,
                        language="Auto",
                        instruct=voice_design_prompt
                    )
                    gen_duration = time.time() - gen_start
                    wav = wavs[0] if isinstance(wavs, list) else wavs
                    voice_design_success = True
                    actual_params["generation_duration"] = round(gen_duration, 3)
                    system_logger.info(f"【Qwen3-TTS】VoiceDesign生成成功 | 耗时: {gen_duration:.3f}s")
            except ValueError as e:
                if "does not support generate_voice_design" in str(e):
                    system_logger.warning(f"【Qwen3-TTS】VoiceDesign不支持: {e}")
                else:
                    raise

            if not voice_design_success:
                # 回退到 Base 模型
                system_logger.warning(f"【Qwen3-TTS】回退到Base模型voice_clone")
                tts_base = get_qwen3tts_model(model_size, "Base")
                default_ref_audio = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone_1.wav"
                default_ref_text = "甚至出现交易几乎停滞的情况。"
                gen_start = time.time()
                wavs, sr = tts_base.generate_voice_clone(
                    text=text,
                    language="Auto",
                    ref_audio=default_ref_audio,
                    ref_text=default_ref_text,
                    x_vector_only_mode=True
                )
                gen_duration = time.time() - gen_start
                wav = wavs[0] if isinstance(wavs, list) else wavs
                actual_params["fallback"] = "Base.voice_clone"
                actual_params["generation_duration"] = round(gen_duration, 3)
                system_logger.info(f"【Qwen3-TTS】Base VoiceClone生成成功 | 耗时: {gen_duration:.3f}s")

        else:
            # base 模式 - 使用默认音色
            system_logger.info(f"【Qwen3-TTS】进入Base模式（默认音色）")
            default_ref_audio = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone_1.wav"
            default_ref_text = "甚至出现交易几乎停滞的情况。"
            gen_start = time.time()
            wavs, sr = tts.generate_voice_clone(
                text=text,
                language="Auto",
                ref_audio=default_ref_audio,
                ref_text=default_ref_text,
                x_vector_only_mode=True
            )
            gen_duration = time.time() - gen_start
            wav = wavs[0] if isinstance(wavs, list) else wavs
            actual_params["generation_duration"] = round(gen_duration, 3)
            system_logger.info(f"【Qwen3-TTS】Base生成成功 | 耗时: {gen_duration:.3f}s")

        # 保存音频
        system_logger.info(f"【Qwen3-TTS】保存音频文件...")
        save_start = time.time()
        if isinstance(wav, list):
            audio_data = np.array(wav)
        else:
            audio_data = wav

        audio_path = save_temp_audio(audio_data, sr, prefix="qwen3tts")
        save_duration = time.time() - save_start
        system_logger.info(f"【Qwen3-TTS】音频保存完成: {audio_path} | 耗时: {save_duration:.3f}s")
        
        actual_params["output_path"] = audio_path
        actual_params["sample_rate"] = sr

        # 清理显存 - 防止内存泄漏
        if torch.cuda.is_available():
            system_logger.info(f"【Qwen3-TTS】清理GPU显存...")
            del audio_data, wav
            cleanup_memory()
            log_gpu_memory_usage("Qwen3-TTS")

        total_duration = time.time() - start_time
        actual_params["total_duration"] = round(total_duration, 3)
        
        OperationLogger.log_tts_request("Qwen3-TTS", text, actual_params, total_duration, "成功")
        system_logger.info(f"【Qwen3-TTS】请求完成 | 总耗时: {total_duration:.3f}s | 输出: {audio_path}")
        system_logger.info(f"【Qwen3-TTS】{'='*60}")

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
        OperationLogger.log_tts_request("Qwen3-TTS", text, {"mode": mode}, total_duration, f"失败: {str(e)}")
        system_logger.error(f"【Qwen3-TTS】错误: {e}")
        system_logger.error(f"【Qwen3-TTS】{'='*60}")
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
