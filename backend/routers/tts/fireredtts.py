#!/usr/bin/env python3
"""
FireRedTTS2 路由
"""

import time
import os
import torch
import torchaudio
from datetime import datetime
from fastapi import APIRouter, Form, Request, HTTPException, UploadFile, File
from typing import Optional

from backend.logger_config import OperationLogger, system_logger
from backend.models import TTSResponse
from backend.engines import get_fireredtts2_model
from backend.services import get_speaker_by_id
from backend.core import audio_to_base64, cleanup_memory, log_gpu_memory_usage

router = APIRouter()


@router.post("/")
async def tts_fireredtts(
        request: Request,
        text: str = Form(...),
        mode: str = Form("clone"),
        ref_text: Optional[str] = Form(None),
        clone_speaker_id: Optional[str] = Form(None),
        temperature: float = Form(0.9),
        topk: int = Form(30),
        output_format: str = Form("url"),
        ref_audio: Optional[UploadFile] = File(None)
):
    """FireRedTTS2语音合成 - 按照原始GitHub代码方式调用

    使用方法与官方一致:
    - generate_monologue: 独白生成（支持参考音频和随机音色）
    - 使用torchaudio.save()保存音频，采样率24000Hz
    """
    start_time = time.time()
    client_ip = request.client.host if request.client else "unknown"

    try:
        OperationLogger.log_api_request("/tts/fireredtts", "POST", {
            "text_preview": text[:50],
            "mode": mode,
            "clone_speaker_id": clone_speaker_id
        }, client_ip)

        system_logger.info(f"【FireRedTTS2】请求 | 模式: {mode} | 文本: {text[:50]}...")

        # 加载模型
        model = get_fireredtts2_model()

        # 保存参考音频（如果需要）
        ref_path = None
        final_ref_text = ref_text or ""

        if mode == "clone":
            # 方式1: 通过clone_speaker_id使用本地说话人音频
            if clone_speaker_id:
                speaker = get_speaker_by_id(clone_speaker_id)
                if not speaker:
                    raise HTTPException(status_code=404, detail=f"说话人不存在: {clone_speaker_id}")

                ref_path = speaker.get("audio_path")
                if not ref_path or not os.path.exists(ref_path):
                    raise HTTPException(status_code=404, detail=f"说话人音频文件不存在: {ref_path}")

                # 使用说话人管理中的参考文本
                final_ref_text = speaker.get("reference_text", "") or ref_text or ""
                if not final_ref_text:
                    raise HTTPException(status_code=400, detail=f"说话人 {speaker['name']} 没有参考文本，无法用于声音克隆。请在说话人管理中添加参考文本。")
                system_logger.info(f"【FireRedTTS2】clone模式: 使用说话人 {speaker['name']} 的音频: {ref_path}")

            # 方式2: 通过上传的音频文件
            elif ref_audio:
                ref_path = f"uploads/fireredtts_ref_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.wav"
                with open(ref_path, "wb") as f:
                    f.write(await ref_audio.read())
                system_logger.info(f"【FireRedTTS2】clone模式: 使用上传的参考音频: {ref_path}")

            else:
                raise HTTPException(status_code=400, detail="clone模式需要提供clone_speaker_id或上传参考音频")

            # 使用generate_monologue进行克隆 - 按照GitHub README示例
            audio = model.generate_monologue(
                text=text,
                prompt_wav=ref_path,
                prompt_text=final_ref_text,
                temperature=temperature,
                topk=topk
            )
        else:  # random模式
            # 使用generate_monologue生成随机音色（不传prompt_wav）- 按照GitHub README示例
            audio = model.generate_monologue(
                text=text,
                temperature=temperature,
                topk=topk
            )

        # 按照GitHub示例，采样率为24000Hz
        sr = 24000

        # 保存音频 - 使用有意义的文件名
        import re
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        text_summary = re.sub(r'[^\w一-鿿]', '', text[:8].strip()) or "audio"
        speaker_part = ""
        if speaker and speaker.get("name"):
            clean_name = re.sub(r'[^\w一-鿿]', '', speaker.get("name", ""))[:6]
            if clean_name:
                speaker_part = f"_{clean_name}"
        audio_path = f"outputs/fireredtts_{mode}{speaker_part}_{text_summary}_{timestamp}.wav"

        # 确保音频是torch tensor并移至CPU
        if hasattr(audio, 'cpu'):
            audio = audio.cpu()

        torchaudio.save(audio_path, audio, sr)
        system_logger.info(f"【FireRedTTS2】生成完成: {audio_path}")

        # 清理临时文件（仅清理上传的临时文件，不清理说话人管理中的文件）
        if ref_path and ref_path.startswith("uploads/") and os.path.exists(ref_path):
            os.remove(ref_path)

        # 清理显存 - 防止内存泄漏
        if torch.cuda.is_available():
            del audio
            cleanup_memory()
            log_gpu_memory_usage("FireRedTTS2")

        total_duration = time.time() - start_time
        OperationLogger.log_tts_request("FireRedTTS2", text, {
            "mode": mode,
            "temperature": temperature
        }, total_duration, "成功")

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
                audio_url=f"/audio/{os.path.basename(audio_path)}",
                sample_rate=sr
            )

    except HTTPException:
        raise
    except Exception as e:
        total_duration = time.time() - start_time
        OperationLogger.log_error("FireRedTTS2合成错误", str(e))
        OperationLogger.log_tts_request("FireRedTTS2", text, {}, total_duration, f"失败: {str(e)}")
        system_logger.error(f"【FireRedTTS2】错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))
