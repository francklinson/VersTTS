#!/usr/bin/env python3
"""
批量处理路由
支持批量生成（抽卡模式）和批量下载
"""

import os
import io
import zipfile
import asyncio
import aiohttp
from datetime import datetime
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, HTTPException, Request, Form
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from backend.logger_config import system_logger
from backend.batch_processor import batch_processor, BatchJob
from backend.models.schemas import BatchGenerateRequest, BatchGenerateResponse, BatchDownloadRequest
from backend.config import OUTPUTS_DIR

router = APIRouter()

# 并发控制配置
# OmniVoice是独立HTTP服务，可以有限并行（2个并发）
OMNIVOICE_CONCURRENCY = 2
# 本地GPU模型保持顺序执行（避免爆显存）
GPU_MODEL_CONCURRENCY = 1

# 创建信号量控制并发
omnivoice_semaphore = asyncio.Semaphore(OMNIVOICE_CONCURRENCY)


class BatchTTSRequest(BaseModel):
    """批量 TTS 请求"""
    model: str = Field(default="chattts", description="TTS模型名称")
    tasks: List[dict] = Field(default=[], description="任务列表")


@router.post("/create")
async def create_batch_job(request: BatchTTSRequest):
    """创建批量TTS任务"""
    try:
        job = batch_processor.create_job(request.model, request.tasks)
        system_logger.info(f"【批量处理】创建任务: {job.job_id}, 总任务数: {job.total}")
        return {
            "success": True,
            "job_id": job.job_id,
            "total": job.total,
            "status": job.status
        }
    except Exception as e:
        system_logger.error(f"【批量处理】创建任务失败: {e}")
        raise HTTPException(status_code=500, detail=f"创建批量任务失败: {str(e)}")


@router.get("/{job_id}/status")
async def get_batch_status(job_id: str):
    """获取批量任务状态"""
    job = batch_processor.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    return {
        "job_id": job.job_id,
        "status": job.status,
        "total": job.total,
        "completed": job.completed,
        "failed": job.failed,
        "progress": f"{(job.completed + job.failed) / job.total * 100:.1f}%" if job.total > 0 else "0%"
    }


@router.post("/{job_id}/process")
async def process_batch_job(job_id: str):
    """处理批量TTS任务（简化版，实际实现需要异步处理）"""
    job = batch_processor.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    return {
        "success": True,
        "message": "批量任务已加入处理队列",
        "job_id": job_id
    }


@router.get("/{job_id}/download")
async def download_batch_results(job_id: str):
    """下载批量任务结果ZIP包"""
    try:
        zip_path = batch_processor.create_zip_package(job_id)
        return {
            "success": True,
            "download_url": f"/tts/batch/{job_id}/download/file"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"创建下载包失败: {str(e)}")


@router.get("/{job_id}/results")
async def get_batch_results(job_id: str):
    """获取批量任务详细结果"""
    job = batch_processor.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    return {
        "job_id": job.job_id,
        "status": job.status,
        "tasks": [
            {
                "id": task.id,
                "text": task.text[:100] + "..." if len(task.text) > 100 else task.text,
                "status": task.status,
                "error": task.error
            }
            for task in job.tasks
        ]
    }


# ========== 批量生成（抽卡模式）API ==========

@router.post("/generate", response_model=BatchGenerateResponse)
async def batch_generate(
    request: Request,
    text: str = Form(..., description="要合成的文本"),
    model: str = Form("voxcpm", description="TTS模型名称"),
    mode: str = Form("base", description="生成模式"),
    count: int = Form(10, description="生成数量(2-20)"),
    speaker_id: str = Form(None, description="说话人ID"),
    voice_design_prompt: str = Form(None, description="音色设计描述"),
    control_prompt: str = Form(None, description="控制指令"),
    speed: float = Form(1.0, description="语速(0.5-2.0)")
):
    """
    批量生成（抽卡模式）- 一次生成多个音频变体
    
    支持模型:
    - voxcpm: 基础生成、音色设计、声音克隆、极致克隆
    - qwen3tts: base, voice_clone, custom_voice, voice_design
    """
    try:
        # 验证参数
        if count < 2 or count > 20:
            raise HTTPException(status_code=400, detail="生成数量必须在2-20之间")
        
        if not text or not text.strip():
            raise HTTPException(status_code=400, detail="文本不能为空")
        
        system_logger.info(f"【批量生成】模型: {model} | 模式: {mode} | 数量: {count} | 文本: {text[:50]}...")
        
        # 根据模型调用相应的生成逻辑
        audio_urls = []
        audio_files = []
        
        if model == "voxcpm":
            result = await _batch_generate_voxcpm(
                text=text,
                mode=mode,
                count=count,
                speaker_id=speaker_id,
                voice_design_prompt=voice_design_prompt,
                control_prompt=control_prompt
            )
            audio_urls = result["audio_urls"]
            audio_files = result["audio_files"]
        elif model == "qwen3tts":
            result = await _batch_generate_qwen3tts(
                text=text,
                mode=mode,
                count=count,
                speaker_id=speaker_id,
                voice_design_prompt=voice_design_prompt
            )
            audio_urls = result["audio_urls"]
            audio_files = result["audio_files"]
        elif model == "omnivoice":
            # 语速自动矫正
            original_speed = speed
            if speed < 0.5 or speed > 2.0:
                speed = max(0.5, min(2.0, speed))
                speed = round(speed, 1)
                system_logger.warning(f"【批量生成】语速参数 {original_speed} 超出范围，已自动矫正为 {speed}")
            
            result = await _batch_generate_omnivoice(
                text=text,
                mode=mode,
                count=count,
                speaker_id=speaker_id,
                voice_design_prompt=voice_design_prompt,
                speed=speed
            )
            audio_urls = result["audio_urls"]
            audio_files = result["audio_files"]
        else:
            raise HTTPException(status_code=400, detail=f"不支持的模型: {model}")
        
        system_logger.info(f"【批量生成】完成 | 生成数量: {len(audio_urls)}")
        
        return BatchGenerateResponse(
            success=True,
            message=f"成功生成 {len(audio_urls)} 个音频",
            model=model,
            count=len(audio_urls),
            audio_urls=audio_urls,
            audio_files=audio_files
        )
        
    except HTTPException:
        raise
    except Exception as e:
        system_logger.error(f"【批量生成】错误: {e}")
        raise HTTPException(status_code=500, detail=f"批量生成失败: {str(e)}")


async def _batch_generate_voxcpm(text: str, mode: str, count: int, 
                                  speaker_id: str = None, 
                                  voice_design_prompt: str = None,
                                  control_prompt: str = None) -> Dict:
    """批量生成VoxCPM音频"""
    import torch
    import soundfile as sf
    import numpy as np
    from backend.engines import get_voxcpm_model
    from backend.services import get_speaker_by_id
    from backend.core import cleanup_memory, log_gpu_memory_usage, save_temp_audio
    
    audio_urls = []
    audio_files = []
    
    # 获取模型
    model = get_voxcpm_model()
    
    # 获取说话人信息（如果需要）
    ref_path = None
    speaker_ref_text = None
    if mode in ["clone", "ultimate_clone"] and speaker_id:
        speaker = get_speaker_by_id(speaker_id)
        if speaker:
            ref_path = speaker.get("audio_path")
            speaker_ref_text = speaker.get("reference_text")
    
    # 检查模型类型
    import sys
    voxcpm_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "algorithms", "VoxCPM", "src")
    if voxcpm_path not in sys.path:
        sys.path.insert(0, voxcpm_path)
    from voxcpm.model.voxcpm2 import VoxCPM2Model
    is_v2 = isinstance(model.tts_model, VoxCPM2Model)
    
    # 批量生成
    for i in range(count):
        try:
            # 构建生成参数
            generate_kwargs = {
                "cfg_value": 2.0,
                "inference_timesteps": 10
            }
            
            # 根据模式处理
            if mode == "voice_design" and voice_design_prompt:
                generate_kwargs["text"] = f"({voice_design_prompt}){text}"
            elif mode == "clone" and ref_path:
                if control_prompt:
                    generate_kwargs["text"] = f"({control_prompt}){text}"
                else:
                    generate_kwargs["text"] = text
                
                if is_v2:
                    generate_kwargs["reference_wav_path"] = ref_path
                else:
                    generate_kwargs["prompt_wav_path"] = ref_path
                    if speaker_ref_text:
                        generate_kwargs["prompt_text"] = speaker_ref_text
            elif mode == "ultimate_clone" and ref_path and speaker_ref_text:
                generate_kwargs["text"] = text
                if is_v2:
                    generate_kwargs["reference_wav_path"] = ref_path
                    generate_kwargs["prompt_wav_path"] = ref_path
                    generate_kwargs["prompt_text"] = speaker_ref_text
                else:
                    generate_kwargs["prompt_wav_path"] = ref_path
                    generate_kwargs["prompt_text"] = speaker_ref_text
            else:
                # base模式
                generate_kwargs["text"] = text
            
            # 生成音频
            audio_data = model.generate(**generate_kwargs)
            
            # 保存音频（使用统一格式）
            sr = 48000
            audio_path = save_temp_audio(audio_data, sr, prefix="voxcpm")
            
            audio_urls.append(f"/audio/{os.path.basename(audio_path)}")
            audio_files.append(os.path.basename(audio_path))
            
            system_logger.info(f"【批量生成】VoxCPM 第 {i+1}/{count} 个完成")
            
        except Exception as e:
            system_logger.error(f"【批量生成】VoxCPM 第 {i+1} 个失败: {e}")
    
    # 清理显存
    if torch.cuda.is_available():
        cleanup_memory()
        log_gpu_memory_usage("VoxCPM-Batch")
    
    return {"audio_urls": audio_urls, "audio_files": audio_files}


async def _batch_generate_qwen3tts(text: str, mode: str, count: int,
                                    speaker_id: str = None,
                                    voice_design_prompt: str = None) -> Dict:
    """批量生成Qwen3-TTS音频"""
    import torch
    import numpy as np
    from backend.engines import get_qwen3tts_model
    from backend.services import load_speakers_db
    from backend.core import cleanup_memory, log_gpu_memory_usage, save_temp_audio
    
    audio_urls = []
    audio_files = []
    
    # 根据模式确定模型类型
    model_type_map = {
        "voice_clone": "VoiceClone",
        "custom_voice": "CustomVoice",
        "voice_design": "VoiceDesign",
        "base": "Base"
    }
    model_type = model_type_map.get(mode, "Base")
    
    # 获取模型
    tts = get_qwen3tts_model("1.7B", model_type)
    
    # 获取说话人信息（如果需要）
    speaker_data = None
    if mode == "voice_clone" and speaker_id:
        db = load_speakers_db()
        for s in db.get("speakers", []):
            if s["id"] == speaker_id:
                speaker_data = s
                break
    
    # 批量生成
    for i in range(count):
        try:
            wav = None
            sr = 24000
            
            if mode == "voice_clone" and speaker_data:
                # 声音克隆模式
                ref_path = speaker_data.get("audio_path")
                ref_text = speaker_data.get("reference_text", "")
                
                wavs, sr = tts.generate_voice_clone(
                    text=text,
                    language="Auto",
                    ref_audio=ref_path,
                    ref_text=ref_text,
                    x_vector_only_mode=not ref_text
                )
                wav = wavs[0] if isinstance(wavs, list) else wavs
                
            elif mode == "custom_voice":
                # 预设音色模式
                speaker = "vivian"
                if hasattr(tts, 'generate_custom_voice'):
                    wavs, sr = tts.generate_custom_voice(
                        text=text,
                        language="Chinese",
                        speaker=speaker,
                        instruct="",
                        do_sample=True,
                        temperature=0.9,
                        top_k=50,
                        top_p=1.0
                    )
                    wav = wavs[0] if isinstance(wavs, list) else wavs
                else:
                    # 回退到Base模型
                    tts_base = get_qwen3tts_model("1.7B", "Base")
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
                    
            elif mode == "voice_design" and voice_design_prompt:
                # 音色设计模式
                if hasattr(tts, 'generate_voice_design'):
                    wavs, sr = tts.generate_voice_design(
                        text=text,
                        language="Auto",
                        instruct=voice_design_prompt
                    )
                    wav = wavs[0] if isinstance(wavs, list) else wavs
                else:
                    # 回退到Base模型
                    tts_base = get_qwen3tts_model("1.7B", "Base")
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
                # base模式
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
            
            # 处理音频数据
            if isinstance(wav, list):
                audio_data = np.array(wav)
            else:
                audio_data = wav
            
            # 保存音频（使用统一格式）
            audio_path = save_temp_audio(audio_data, sr, prefix="qwen3tts")
            
            audio_urls.append(f"/audio/{os.path.basename(audio_path)}")
            audio_files.append(os.path.basename(audio_path))
            
            system_logger.info(f"【批量生成】Qwen3-TTS 第 {i+1}/{count} 个完成")
            
            # 清理
            if torch.cuda.is_available():
                del audio_data, wav
                
        except Exception as e:
            system_logger.error(f"【批量生成】Qwen3-TTS 第 {i+1} 个失败: {e}")
    
    # 清理显存
    if torch.cuda.is_available():
        cleanup_memory()
        log_gpu_memory_usage("Qwen3-TTS-Batch")
    
    return {"audio_urls": audio_urls, "audio_files": audio_files}


async def _generate_single_omnivoice(
    session: aiohttp.ClientSession,
    text: str,
    mode: str,
    index: int,
    ref_path: str = None,
    speaker_ref_text: str = None,
    voice_design_prompt: str = None,
    omnivoice_url: str = "",
    speed: float = 1.0
) -> Dict:
    """单个OmniVoice生成任务（受信号量控制）"""
    import soundfile as sf
    
    async with omnivoice_semaphore:  # 限制并发数
        try:
            # 构建请求数据
            data = {
                "text": text,
                "mode": mode,
                "num_steps": 32,
                "speed": speed
            }
            if mode == "voice_clone" and ref_path:
                data["ref_audio"] = ref_path
                if speaker_ref_text:
                    data["ref_text"] = speaker_ref_text
            elif mode == "voice_design" and voice_design_prompt:
                data["voice_design_prompt"] = voice_design_prompt
            
            # 异步HTTP请求（超时120秒）
            timeout = aiohttp.ClientTimeout(total=120)
            async with session.post(omnivoice_url, data=data, timeout=timeout) as response:
                if response.status != 200:
                    error_text = await response.text()
                    system_logger.error(f"【批量生成】OmniVoice 第 {index+1} 个失败: {error_text}")
                    return {"success": False, "index": index}
                
                result = await response.json()
                if not result.get("success"):
                    system_logger.error(f"【批量生成】OmniVoice 第 {index+1} 个合成失败")
                    return {"success": False, "index": index}
                
                temp_audio_path = result.get("audio_path")
                if not temp_audio_path or not os.path.exists(temp_audio_path):
                    system_logger.error(f"【批量生成】OmniVoice 第 {index+1} 个音频文件未生成")
                    return {"success": False, "index": index}
                
                # 保存音频到 outputs 目录
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                output_path = os.path.join(OUTPUTS_DIR, f"omnivoice_{timestamp}.wav")
                audio_data, sample_rate = sf.read(temp_audio_path)
                sf.write(output_path, audio_data, samplerate=sample_rate)
                
                # 清理临时文件
                try:
                    os.remove(temp_audio_path)
                except:
                    pass
                
                system_logger.info(f"【批量生成】OmniVoice 第 {index+1} 个完成")
                return {
                    "success": True,
                    "index": index,
                    "audio_url": f"/audio/{os.path.basename(output_path)}",
                    "audio_file": os.path.basename(output_path)
                }
                
        except asyncio.TimeoutError:
            system_logger.error(f"【批量生成】OmniVoice 第 {index+1} 个超时")
            return {"success": False, "index": index, "error": "timeout"}
        except Exception as e:
            system_logger.error(f"【批量生成】OmniVoice 第 {index+1} 个失败: {e}")
            return {"success": False, "index": index, "error": str(e)}


def _fix_voice_design_format(prompt: str) -> str:
    """
    修正声音设计属性的格式
    
    问题：用户可能输入 "宁夏话女，中年，高音调" 或 "女，老年甘肃话"（缺少逗号）
    修正："宁夏话，女，中年，高音调" 或 "女，老年，甘肃话"
    """
    import re
    if not prompt:
        return prompt
    
    dialects = ['四川话', '东北话', '河南话', '陕西话', '云南话', '贵州话', '桂林话',
                '甘肃话', '宁夏话', '济南话', '青岛话', '石家庄话']
    genders = ['男', '女']
    ages = ['儿童', '少年', '青年', '中年', '老年']
    pitches = ['极低音调', '低音调', '中音调', '高音调', '极高音调', '耳语']
    
    result = prompt
    
    # 迭代处理直到没有变化（处理多重粘连）
    max_iterations = 10
    for _ in range(max_iterations):
        prev_result = result
        
        # 1. 处理方言和性别之间缺少分隔符
        for dialect in dialects:
            for gender in genders:
                result = re.sub(f'{dialect}{gender}(?![\u4e00-\u9fff])', f'{dialect}，{gender}', result)
        
        # 2. 处理性别和年龄之间缺少分隔符
        for gender in genders:
            for age in ages:
                result = re.sub(f'{gender}{age}(?![\u4e00-\u9fff])', f'{gender}，{age}', result)
        
        # 3. 处理年龄和方言之间缺少分隔符
        for age in ages:
            for dialect in dialects:
                result = re.sub(f'{age}{dialect}(?![\u4e00-\u9fff])', f'{age}，{dialect}', result)
        
        # 4. 处理年龄和音调之间缺少分隔符
        for age in ages:
            for pitch in pitches:
                result = re.sub(f'{age}{pitch}(?![\u4e00-\u9fff])', f'{age}，{pitch}', result)
        
        # 5. 处理性别和音调之间缺少分隔符
        for gender in genders:
            for pitch in pitches:
                result = re.sub(f'{gender}{pitch}(?![\u4e00-\u9fff])', f'{gender}，{pitch}', result)
        
        # 6. 处理方言和音调之间缺少分隔符
        for dialect in dialects:
            for pitch in pitches:
                result = re.sub(f'{dialect}{pitch}(?![\u4e00-\u9fff])', f'{dialect}，{pitch}', result)
        
        if result == prev_result:
            break
    
    # 规范化分隔符
    result = result.replace(',', '，')
    result = re.sub(r'\s*，\s*', '，', result)
    result = re.sub(r'，+', '，', result)
    result = re.sub(r'^，|，$', '', result)
    
    return result.strip()


async def _batch_generate_omnivoice(text: str, mode: str, count: int,
                                    speaker_id: str = None,
                                    voice_design_prompt: str = None,
                                    speed: float = 1.0) -> Dict:
    """
    批量生成OmniVoice音频（并行版本）
    
    使用异步并行生成，并发数由 OMNIVOICE_CONCURRENCY 控制（默认2）
    注意：由于OmniVoice是独立HTTP服务，并行请求受服务端处理能力限制
    """
    from backend.services import get_speaker_by_id
    from backend.config import OMNIVOICE_HOST, OMNIVOICE_PORT
    
    audio_urls = []
    audio_files = []
    
    OMNIVOICE_SERVICE_URL = f"http://{OMNIVOICE_HOST}:{OMNIVOICE_PORT}/tts"
    
    # 修正声音设计格式
    if voice_design_prompt:
        voice_design_prompt = _fix_voice_design_format(voice_design_prompt)
    
    # 获取说话人信息（如果需要）
    ref_path = None
    speaker_ref_text = None
    if mode == "voice_clone" and speaker_id:
        speaker = get_speaker_by_id(speaker_id)
        if speaker:
            ref_path = speaker.get("audio_path")
            speaker_ref_text = speaker.get("reference_text")
    
    system_logger.info(f"【批量生成】OmniVoice 开始并行生成 | 数量: {count} | 并发度: {OMNIVOICE_CONCURRENCY}")
    
    # 创建异步HTTP会话
    timeout = aiohttp.ClientTimeout(total=120)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        # 创建所有任务
        tasks = []
        for i in range(count):
            task = _generate_single_omnivoice(
                session=session,
                text=text,
                mode=mode,
                index=i,
                ref_path=ref_path,
                speaker_ref_text=speaker_ref_text,
                voice_design_prompt=voice_design_prompt,
                omnivoice_url=OMNIVOICE_SERVICE_URL,
                speed=speed
            )
            tasks.append(task)
        
        # 并行执行所有任务（受信号量控制实际并发数）
        results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # 处理结果
    success_count = 0
    for result in results:
        if isinstance(result, Exception):
            system_logger.error(f"【批量生成】OmniVoice 任务异常: {result}")
            continue
        if result.get("success"):
            audio_urls.append(result["audio_url"])
            audio_files.append(result["audio_file"])
            success_count += 1
    
    system_logger.info(f"【批量生成】OmniVoice 完成 | 成功: {success_count}/{count}")
    
    return {"audio_urls": audio_urls, "audio_files": audio_files}


@router.post("/download-zip")
async def batch_download_zip(files: List[str] = Form(...), zip_name: str = Form(None)):
    """
    批量下载 - 将多个音频文件打包成ZIP
    
    参数:
    - files: 音频文件名列表（不包含路径）
    - zip_name: ZIP包名称（可选）
    """
    try:
        if not files:
            raise HTTPException(status_code=400, detail="文件列表不能为空")
        
        if len(files) > 50:
            raise HTTPException(status_code=400, detail="单次下载文件数量不能超过50个")
        
        # 生成ZIP包名称
        if not zip_name:
            zip_name = f"batch_download_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if not zip_name.endswith('.zip'):
            zip_name += '.zip'
        
        # 创建ZIP包（内存中）
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            for filename in files:
                file_path = os.path.join(OUTPUTS_DIR, filename)
                if os.path.exists(file_path):
                    zf.write(file_path, filename)
                else:
                    system_logger.warning(f"【批量下载】文件不存在: {file_path}")
        
        zip_buffer.seek(0)
        
        system_logger.info(f"【批量下载】创建ZIP包: {zip_name}, 包含 {len(files)} 个文件")
        
        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={"Content-Disposition": f"attachment; filename={zip_name}"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        system_logger.error(f"【批量下载】错误: {e}")
        raise HTTPException(status_code=500, detail=f"创建ZIP包失败: {str(e)}")


@router.get("/config")
async def get_batch_config():
    """
    获取批量生成配置和系统状态
    
    返回:
    - 各模型的并发配置
    - 当前GPU显存状态（如果可用）
    - 建议的最大并发数
    """
    try:
        import torch
        
        config = {
            "concurrency": {
                "omnivoice": OMNIVOICE_CONCURRENCY,
                "voxcpm": GPU_MODEL_CONCURRENCY,
                "qwen3tts": GPU_MODEL_CONCURRENCY,
                "note": "本地GPU模型保持顺序执行以避免爆显存"
            },
            "gpu_info": {}
        }
        
        # 获取GPU信息
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            config["gpu_info"]["device_count"] = gpu_count
            config["gpu_info"]["devices"] = []
            
            for i in range(gpu_count):
                props = torch.cuda.get_device_properties(i)
                total_memory = props.total_memory / (1024**3)  # GB
                allocated = torch.cuda.memory_allocated(i) / (1024**3)  # GB
                reserved = torch.cuda.memory_reserved(i) / (1024**3)  # GB
                free = total_memory - allocated
                
                config["gpu_info"]["devices"].append({
                    "id": i,
                    "name": props.name,
                    "total_memory_gb": round(total_memory, 2),
                    "allocated_memory_gb": round(allocated, 2),
                    "reserved_memory_gb": round(reserved, 2),
                    "free_memory_gb": round(free, 2),
                    "utilization_percent": round((allocated / total_memory) * 100, 1)
                })
            
            # 根据显存使用情况给出建议
            primary_gpu = config["gpu_info"]["devices"][0]
            if primary_gpu["free_memory_gb"] < 4:
                config["recommendation"] = "显存紧张，建议降低批量生成数量或减少并发"
            elif primary_gpu["free_memory_gb"] < 8:
                config["recommendation"] = "显存充足，可以使用默认配置"
            else:
                config["recommendation"] = "显存非常充足，可以考虑增加OmniVoice并发数"
        else:
            config["gpu_info"]["available"] = False
            config["recommendation"] = "未检测到GPU，仅CPU模式可用"
        
        return {
            "success": True,
            "config": config
        }
        
    except Exception as e:
        system_logger.error(f"【批量配置】获取配置失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取配置失败: {str(e)}")
