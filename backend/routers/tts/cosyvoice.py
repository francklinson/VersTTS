#!/usr/bin/env python3
"""
CosyVoice 路由
通过独立服务调用 CosyVoice（需要 transformers 4.51.3）
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
from backend.services import load_speakers_db
from backend.core import save_temp_audio, audio_to_base64, cleanup_memory, log_gpu_memory_usage
from backend.config import COSYVOICE_HOST, COSYVOICE_PORT

router = APIRouter()

COSYVOICE_SERVICE_URL = f"http://{COSYVOICE_HOST}:{COSYVOICE_PORT}/tts"
COSYVOICE_HEALTH_URL = f"http://{COSYVOICE_HOST}:{COSYVOICE_PORT}/health"


def _check_cosyvoice_service():
    """检查 CosyVoice 独立服务是否运行"""
    try:
        response = requests.get(COSYVOICE_HEALTH_URL, timeout=3)
        return response.status_code == 200
    except:
        return False


@router.post("/")
async def tts_cosyvoice(
        request: Request,
        text: str = Form(...),
        mode: str = Form("sft"),
        speaker_id: str = Form("中文女"),
        prompt_text: Optional[str] = Form(None),
        instruct_text: Optional[str] = Form(None),
        prompt_wav: Optional[UploadFile] = File(None),
        clone_speaker_id: Optional[str] = Form(None),
        output_format: str = Form("url")
):
    """CosyVoice语音合成 - 通过独立服务"""
    ref_path = None
    start_time = time.time()
    client_ip = request.client.host if request.client else "unknown"
    
    # 详细日志：请求开始
    system_logger.info(f"【CosyVoice】{'='*60}")
    system_logger.info(f"【CosyVoice】请求开始 | 模式: {mode} | 客户端: {client_ip}")
    system_logger.info(f"【CosyVoice】输入文本: {text[:100]}...")

    try:
        # 记录API请求参数
        request_params = {
            "text_preview": text[:50],
            "mode": mode,
            "speaker_id": speaker_id,
            "clone_speaker_id": clone_speaker_id,
            "instruct_text": instruct_text[:50] if instruct_text else None,
            "has_upload": prompt_wav is not None
        }
        OperationLogger.log_api_request("/tts/cosyvoice", "POST", request_params, client_ip)
        system_logger.info(f"【CosyVoice】请求参数: {request_params}")

        # 检查独立服务状态
        system_logger.info(f"【CosyVoice】检查独立服务状态...")
        if not _check_cosyvoice_service():
            system_logger.error(f"【CosyVoice】独立服务未运行")
            raise HTTPException(
                status_code=503,
                detail="CosyVoice 独立服务未运行。请执行: nohup python cosyvoice_service.py > logs/cosyvoice_service.log 2>&1 &"
            )
        system_logger.info(f"【CosyVoice】独立服务运行正常")
        
        # 记录模型信息
        system_logger.info(f"【CosyVoice】模型信息:")
        system_logger.info(f"【CosyVoice】   - 模型名称: Fun-CosyVoice3-0.5B-2512")
        system_logger.info(f"【CosyVoice】   - 模型来源: ModelScope / HuggingFace")
        system_logger.info(f"【CosyVoice】   - 模型路径: /home/zhouchenghao/PycharmProjects/VersTTS/models/FunAudioLLM/Fun-CosyVoice3-0.5B-2512")
        system_logger.info(f"【CosyVoice】   - 支持模式: zero_shot, instruct")
        system_logger.info(f"【CosyVoice】   - 采样率: 22050 Hz")
        system_logger.info(f"【CosyVoice】   - 支持方言: 18+种中文方言")

        if mode == "sft":
            system_logger.warning(f"【CosyVoice】拒绝SFT模式请求")
            raise HTTPException(status_code=400, detail="CosyVoice 3.0 不支持SFT预训练音色模式，请使用Zero-shot克隆模式")

        elif mode == "zero_shot":
            system_logger.info(f"【CosyVoice】进入Zero-shot模式处理...")
            # 获取参考音频路径
            if clone_speaker_id:
                system_logger.info(f"【CosyVoice】查找说话人: {clone_speaker_id}")
                db = load_speakers_db()
                speaker = None
                for s in db.get("speakers", []):
                    if s["id"] == clone_speaker_id:
                        speaker = s
                        break

                if not speaker:
                    system_logger.error(f"【CosyVoice】说话人不存在: {clone_speaker_id}")
                    raise HTTPException(status_code=404, detail=f"说话人不存在: {clone_speaker_id}")

                audio_path = speaker.get("audio_path")
                system_logger.info(f"【CosyVoice】找到说话人 | 名称: {speaker.get('name')} | 音频: {audio_path}")
                
                if not audio_path or not os.path.exists(audio_path):
                    system_logger.error(f"【CosyVoice】参考音频文件不存在: {audio_path}")
                    raise HTTPException(status_code=404, detail=f"说话人音频文件不存在")

                prompt_text = speaker.get("reference_text", "")
                ref_path = audio_path
                system_logger.info(f"【CosyVoice】参考文本: {prompt_text[:50] if prompt_text else '无'}...")

            elif prompt_wav:
                # 保存上传的音频文件
                system_logger.info(f"【CosyVoice】处理上传的参考音频...")
                file_content = await prompt_wav.read()
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                    tmp_file.write(file_content)
                    ref_path = tmp_file.name
                system_logger.info(f"【CosyVoice】上传音频已保存: {ref_path}")
            else:
                system_logger.error(f"【CosyVoice】缺少参考音频")
                raise HTTPException(status_code=400, detail="zero_shot模式需要提供clone_speaker_id或上传参考音频")

            # 调用独立服务
            data = {
                "text": text,
                "mode": "zero_shot",
                "prompt_text": prompt_text or "",
                "prompt_wav_path": ref_path,
                "output_format": output_format
            }
            system_logger.info(f"【CosyVoice】Zero-shot请求数据: text={text[:50]}..., prompt_wav_path={ref_path}")

        elif mode == "instruct":
            system_logger.info(f"【CosyVoice】进入Instruct模式处理...")
            if not instruct_text:
                system_logger.error(f"【CosyVoice】缺少指令文本")
                raise HTTPException(status_code=400, detail="instruct模式需要提供instruct_text指令文本")
            
            system_logger.info(f"【CosyVoice】指令文本: {instruct_text}")

            # 获取参考音频路径
            if clone_speaker_id:
                system_logger.info(f"【CosyVoice】查找说话人: {clone_speaker_id}")
                db = load_speakers_db()
                speaker = None
                for s in db.get("speakers", []):
                    if s["id"] == clone_speaker_id:
                        speaker = s
                        break

                if not speaker:
                    system_logger.error(f"【CosyVoice】说话人不存在: {clone_speaker_id}")
                    raise HTTPException(status_code=404, detail=f"说话人不存在: {clone_speaker_id}")

                audio_path = speaker.get("audio_path")
                system_logger.info(f"【CosyVoice】找到说话人 | 名称: {speaker.get('name')} | 音频: {audio_path}")
                
                if not audio_path or not os.path.exists(audio_path):
                    system_logger.error(f"【CosyVoice】参考音频文件不存在: {audio_path}")
                    raise HTTPException(status_code=404, detail=f"说话人音频文件不存在")
                ref_path = audio_path

            elif prompt_wav:
                # 保存上传的音频文件
                system_logger.info(f"【CosyVoice】处理上传的参考音频...")
                file_content = await prompt_wav.read()
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                    tmp_file.write(file_content)
                    ref_path = tmp_file.name
                system_logger.info(f"【CosyVoice】上传音频已保存: {ref_path}")
            else:
                system_logger.error(f"【CosyVoice】缺少参考音频")
                raise HTTPException(status_code=400, detail="instruct模式需要提供clone_speaker_id或上传参考音频")

            # 调用独立服务
            data = {
                "text": text,
                "mode": "instruct",
                "instruct_text": instruct_text,
                "prompt_wav_path": ref_path,
                "output_format": output_format
            }
            system_logger.info(f"【CosyVoice】Instruct请求数据: text={text[:50]}..., instruct={instruct_text}, prompt_wav_path={ref_path}")

        else:
            system_logger.error(f"【CosyVoice】不支持的模式: {mode}")
            raise HTTPException(status_code=400, detail=f"不支持的模式: {mode}")

        # 调用独立服务
        system_logger.info(f"【CosyVoice】调用独立服务: {COSYVOICE_SERVICE_URL}")
        service_start = time.time()
        response = requests.post(COSYVOICE_SERVICE_URL, data=data, timeout=120)
        service_duration = time.time() - service_start
        system_logger.info(f"【CosyVoice】独立服务响应 | 状态码: {response.status_code} | 耗时: {service_duration:.3f}s")

        if response.status_code != 200:
            error_detail = response.json().get("detail", "未知错误")
            system_logger.error(f"【CosyVoice】独立服务错误: {error_detail}")
            raise HTTPException(status_code=response.status_code, detail=error_detail)

        result = response.json()
        if not result.get("success"):
            system_logger.error(f"【CosyVoice】合成失败: {result}")
            raise HTTPException(status_code=500, detail="CosyVoice 合成失败")

        audio_path = result.get("audio_path")
        system_logger.info(f"【CosyVoice】独立服务返回音频: {audio_path}")
        
        if not audio_path or not os.path.exists(audio_path):
            system_logger.error(f"【CosyVoice】音频文件未生成: {audio_path}")
            raise HTTPException(status_code=500, detail="CosyVoice 音频文件未生成")

        # 保存音频到 outputs 目录
        sr = result.get("sample_rate", 22050)
        system_logger.info(f"【CosyVoice】读取音频文件: {audio_path} | 采样率: {sr}")
        audio_data, sample_rate = sf.read(audio_path)
        output_path = save_temp_audio(audio_data, sr, prefix="cosyvoice")
        system_logger.info(f"【CosyVoice】音频保存完成: {output_path}")

        # 清理临时文件
        if prompt_wav and ref_path and "/tmp" in ref_path:
            try:
                os.remove(ref_path)
                system_logger.info(f"【CosyVoice】清理临时上传文件: {ref_path}")
            except Exception as e:
                system_logger.warning(f"【CosyVoice】清理临时文件失败: {e}")
        try:
            os.remove(audio_path)
            system_logger.info(f"【CosyVoice】清理服务临时文件: {audio_path}")
        except Exception as e:
            system_logger.warning(f"【CosyVoice】清理服务临时文件失败: {e}")

        total_duration = time.time() - start_time
        
        # 记录实际使用的参数
        actual_params = {
            "mode": mode,
            "clone_speaker_id": clone_speaker_id,
            "speaker_name": speaker.get('name') if 'speaker' in locals() and speaker else None,
            "instruct_text": instruct_text[:50] if instruct_text else None,
            "ref_path": ref_path,
            "output_path": output_path,
            "sample_rate": sr,
            "service_duration": round(service_duration, 3),
            "total_duration": round(total_duration, 3)
        }
        OperationLogger.log_tts_request("CosyVoice", text, actual_params, total_duration, "成功")
        system_logger.info(f"【CosyVoice】请求完成 | 总耗时: {total_duration:.3f}s | 输出: {output_path}")
        system_logger.info(f"【CosyVoice】{'='*60}")

        if output_format == "base64":
            audio_b64 = audio_to_base64(output_path)
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
                audio_url=f"/audio/{output_path.split('/')[-1]}",
                sample_rate=sr
            )

    except HTTPException:
        raise
    except Exception as e:
        total_duration = time.time() - start_time
        OperationLogger.log_error("CosyVoice合成错误", str(e))
        OperationLogger.log_tts_request("CosyVoice", text, {"mode": mode, "clone_speaker_id": clone_speaker_id}, total_duration, f"失败: {str(e)}")
        system_logger.error(f"【CosyVoice】合成错误: {e}")
        system_logger.error(f"【CosyVoice】{'='*60}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/speakers")
async def get_cosyvoice_speakers():
    """获取CosyVoice可用的说话人列表"""
    return {
        "success": True,
        "message": "CosyVoice 3.0 不支持预设音色，请使用Zero-shot克隆模式上传参考音频",
        "speakers": []
    }
