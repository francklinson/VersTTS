#!/usr/bin/env python3
"""
GPT-SoVITS 路由
"""

import os
import time
from datetime import datetime

import soundfile as sf
from fastapi import APIRouter, Request, HTTPException

import torch
from backend.logger_config import OperationLogger, system_logger
from backend.config import models
from backend.core import normalize_audio_volume, save_temp_audio, audio_to_base64
from backend.engines import get_gpt_sovits_model, init_gpt_sovits_pipeline
from backend.services import get_speaker_by_id
from backend.models import TTSResponse

router = APIRouter()


@router.post("/")
async def tts_gptsovits(request: Request):
    """GPT-SoVITS语音合成"""
    start_time = time.time()
    ref_path = None
    
    try:
        # 解析表单数据
        form = await request.form()
        
        text = form.get("text")
        if not text:
            raise HTTPException(status_code=400, detail="请提供要合成的文本")
        
        text_lang = form.get("text_lang", "zh")
        prompt_lang = form.get("prompt_lang", "zh")
        clone_speaker_id = form.get("clone_speaker_id")
        top_k = int(form.get("top_k", 15))
        top_p = float(form.get("top_p", 1.0))
        temperature = float(form.get("temperature", 1.0))
        text_split_method = form.get("text_split_method", "cut5")
        batch_size = int(form.get("batch_size", 1))
        speed_factor = float(form.get("speed_factor", 1.0))
        version = form.get("version", "v2")
        output_format = form.get("output_format", "url")
        prompt_text = form.get("prompt_text")
        
        # 获取文件上传
        prompt_wav = form.get("prompt_wav")
        
        # 记录API请求
        client_ip = request.client.host if request.client else "unknown"
        OperationLogger.log_api_request("/tts/gptsovits", "POST", {
            "text_preview": text[:50],
            "text_lang": text_lang,
            "prompt_lang": prompt_lang,
            "version": version,
            "clone_speaker_id": clone_speaker_id
        }, client_ip)
        
        system_logger.info(f"【GPT-SoVITS】开始合成 | 文本: {text[:50]}... | 版本: {version}")
        
        # 处理参考音频
        if clone_speaker_id:
            speaker = get_speaker_by_id(clone_speaker_id)
            if not speaker:
                raise HTTPException(status_code=404, detail=f"说话人不存在: {clone_speaker_id}")
            
            ref_path = speaker.get("audio_path")
            if not ref_path or not os.path.exists(ref_path):
                raise HTTPException(status_code=404, detail="说话人音频文件不存在")
            
            if not prompt_text and speaker.get("reference_text"):
                prompt_text = speaker.get("reference_text")
                
        elif prompt_wav:
            ref_path = f"uploads/gptsovits_ref_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.wav"
            with open(ref_path, "wb") as f:
                f.write(await prompt_wav.read())
        else:
            raise HTTPException(status_code=400, detail="请提供参考音频或选择说话人")
        
        if not prompt_text:
            raise HTTPException(status_code=400, detail="请提供参考音频文本")
        
        # 获取模型
        model_info = get_gpt_sovits_model(version)
        
        # 初始化管道
        infer_start = time.time()
        pipeline = init_gpt_sovits_pipeline(model_info, ref_path)
        
        # 构建请求参数
        req = {
            "text": text,
            "text_lang": text_lang.lower(),
            "ref_audio_path": ref_path,
            "prompt_text": prompt_text,
            "prompt_lang": prompt_lang.lower(),
            "top_k": top_k,
            "top_p": top_p,
            "temperature": temperature,
            "text_split_method": text_split_method,
            "batch_size": batch_size,
            "speed_factor": speed_factor,
            "media_type": "wav",
            "streaming_mode": False,
            "parallel_infer": True,
        }
        
        # 执行推理
        tts_generator = pipeline.run(req)
        sr, audio_data = next(tts_generator)
        infer_duration = time.time() - infer_start
        
        # 音量归一化
        audio_data = normalize_audio_volume(audio_data)
        
        # 保存音频
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        audio_path = f"outputs/gptsovits_{timestamp}.wav"
        sf.write(audio_path, audio_data, sr)
        
        # 清理临时参考音频
        if not clone_speaker_id and ref_path and os.path.exists(ref_path) and ref_path.startswith("uploads/"):
            os.remove(ref_path)
        
        audio_size = os.path.getsize(audio_path)
        OperationLogger.log_file_operation("保存音频", audio_path, audio_size, "成功")
        
        total_duration = time.time() - start_time
        OperationLogger.log_tts_request("GPT-SoVITS", text, {
            "version": version,
            "text_lang": text_lang,
            "prompt_lang": prompt_lang
        }, total_duration, "成功")
        
        system_logger.info(f"【GPT-SoVITS】合成完成 | 耗时: {total_duration:.3f}s")
        
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
        OperationLogger.log_error("GPT-SoVITS合成错误", str(e))
        # 清理临时文件
        if 'ref_path' in locals() and ref_path and os.path.exists(ref_path) and ref_path.startswith("uploads/"):
            os.remove(ref_path)
        raise HTTPException(status_code=500, detail=str(e))
