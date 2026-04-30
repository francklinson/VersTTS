#!/usr/bin/env python3
"""
说话人管理路由
"""

import os
import subprocess
import time
from typing import Optional

from fastapi import APIRouter, Form, UploadFile, File, HTTPException
from fastapi.responses import FileResponse

from backend.logger_config import system_logger, OperationLogger
from backend.config import SPEAKERS_DIR
from backend.services.speaker_service import (
    load_speakers_db,
    get_speaker_by_id,
    check_speaker_name_exists,
    add_speaker,
    delete_speaker,
)

router = APIRouter()


@router.get("/")
async def get_speakers():
    """获取所有已保存的说话人列表"""
    try:
        db = load_speakers_db()
        # 返回时不包含完整的 embedding 字符串（太长），只返回基本信息
        speakers_list = []
        for speaker in db["speakers"]:
            speakers_list.append({
                "id": speaker["id"],
                "name": speaker["name"],
                "created_at": speaker["created_at"],
                "model_type": speaker.get("model_type", "chattts"),
                "audio_path": speaker.get("audio_path"),
                "has_embedding": bool(speaker.get("embedding")),
                "has_reference_text": bool(speaker.get("reference_text")),
                "reference_text": speaker.get("reference_text", "")[:100] if speaker.get("reference_text") else ""
            })
        return {
            "success": True,
            "speakers": speakers_list,
            "total": len(speakers_list)
        }
    except Exception as e:
        system_logger.error(f"【说话人管理】获取列表失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取说话人列表失败: {str(e)}")


@router.get("/{speaker_id}")
async def get_speaker_detail(speaker_id: str):
    """获取指定说话人的详细信息（包含embedding）"""
    try:
        db = load_speakers_db()
        for speaker in db["speakers"]:
            if speaker["id"] == speaker_id:
                return {
                    "success": True,
                    "speaker": speaker
                }
        raise HTTPException(status_code=404, detail="说话人不存在")
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"获取说话人详情失败: {str(e)}")


@router.get("/{speaker_id}/audio")
async def get_speaker_audio(speaker_id: str):
    """获取指定说话人的参考音频文件"""
    try:
        db = load_speakers_db()
        for speaker in db["speakers"]:
            if speaker["id"] == speaker_id:
                audio_path = speaker.get("audio_path")
                if not audio_path or not os.path.exists(audio_path):
                    raise HTTPException(status_code=404, detail="音频文件不存在")

                # 根据文件扩展名确定媒体类型
                ext = os.path.splitext(audio_path)[1].lower()
                media_type = {
                    '.wav': 'audio/wav',
                    '.mp3': 'audio/mpeg',
                    '.ogg': 'audio/ogg',
                    '.webm': 'audio/webm',
                    '.m4a': 'audio/mp4'
                }.get(ext, 'audio/wav')

                filename = os.path.basename(audio_path)
                return FileResponse(audio_path, media_type=media_type, filename=filename)

        raise HTTPException(status_code=404, detail="说话人不存在")
    except HTTPException:
        raise
    except Exception as e:
        system_logger.error(f"获取说话人音频失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取说话人音频失败: {str(e)}")


@router.post("/check-name")
async def check_name(name: str = Form(...)):
    """检查说话人名称是否可用"""
    exists = check_speaker_name_exists(name)
    return {
        "success": True,
        "available": not exists,
        "exists": exists,
        "message": "名称已被使用" if exists else "名称可用"
    }


@router.post("/upload")
async def upload_speaker_audio(
        audio: UploadFile = File(...),
        speaker_name: str = Form(...),
        reference_text: Optional[str] = Form(None)
):
    """
    上传说话人音频文件（与模型解耦，只保存音频和文本）
    """
    start_time = time.time()

    # 验证名称
    if not speaker_name or len(speaker_name.strip()) == 0:
        raise HTTPException(status_code=400, detail="说话人名称不能为空")

    if check_speaker_name_exists(speaker_name):
        raise HTTPException(status_code=400, detail=f"说话人名称 '{speaker_name}' 已存在")

    # 验证文件格式
    allowed_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.webm'}
    file_ext = os.path.splitext(audio.filename)[1].lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的音频格式: {file_ext}。支持的格式: {', '.join(allowed_extensions)}"
        )

    try:
        system_logger.info(f"【说话人管理】开始上传: {speaker_name}, 文件: {audio.filename}")

        # 读取音频文件
        audio_bytes = await audio.read()

        # 保存上传的音频文件
        timestamp = int(time.time())
        audio_filename = f"speaker_{timestamp}_{speaker_name}.wav"
        audio_path = os.path.join(SPEAKERS_DIR, audio_filename)

        # 对于 webm/ogg 格式，使用 ffmpeg 转换为 wav
        if file_ext in ['.webm', '.ogg']:
            temp_path = os.path.join(SPEAKERS_DIR, f"temp_{timestamp}{file_ext}")
            try:
                # 先保存原始文件
                with open(temp_path, 'wb') as f:
                    f.write(audio_bytes)

                # 使用 ffmpeg 转换
                cmd = [
                    'ffmpeg', '-y', '-i', temp_path,
                    '-ar', '24000', '-ac', '1', '-acodec', 'pcm_s16le',
                    audio_path
                ]

                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

                if result.returncode != 0:
                    raise Exception(f"ffmpeg 转换失败: {result.stderr}")

                # 删除临时文件
                os.remove(temp_path)

            except Exception as e:
                system_logger.error(f"【说话人管理】转换失败: {e}")
                raise HTTPException(status_code=400, detail=f"音频转换失败: {e}")
        else:
            # 直接保存文件
            with open(audio_path, 'wb') as f:
                f.write(audio_bytes)

        duration = time.time() - start_time
        system_logger.info(f"【说话人管理】上传成功: {speaker_name}, 路径: {audio_path}, 耗时: {duration:.2f}s")

        return {
            "success": True,
            "message": "音频上传成功",
            "speaker_name": speaker_name,
            "audio_path": audio_path,
            "reference_text": reference_text,
            "duration": duration
        }

    except Exception as e:
        # 清理已保存的音频文件
        if 'audio_path' in locals() and os.path.exists(audio_path):
            os.remove(audio_path)

        system_logger.error(f"【说话人管理】上传失败: {e}")
        raise HTTPException(status_code=500, detail=f"上传失败: {str(e)}")


@router.post("/save")
async def save_speaker(
        name: str = Form(...),
        audio_path: str = Form(...),
        reference_text: Optional[str] = Form(None)
):
    """保存说话人信息（与模型解耦，只保存音频和文本）"""
    try:
        system_logger.info(f"【说话人管理】开始保存: {name}, 音频: {audio_path}")

        # 验证名称
        if not name or len(name.strip()) == 0:
            raise HTTPException(status_code=400, detail="说话人名称不能为空")

        if check_speaker_name_exists(name):
            raise HTTPException(status_code=400, detail=f"说话人名称 '{name}' 已存在")

        # 验证音频文件是否存在
        if not audio_path or not os.path.exists(audio_path):
            raise HTTPException(status_code=400, detail="音频文件不存在")

        # 保存说话人（embedding 设为 None，与模型解耦）
        speaker = add_speaker(name, None, audio_path, reference_text)

        # 记录审计日志
        OperationLogger.log_speaker_operation("创建", speaker["name"], speaker["id"])

        system_logger.info(f"【说话人管理】保存成功: {name}, ID: {speaker['id']}")

        return {
            "success": True,
            "message": "说话人保存成功",
            "speaker": {
                "id": speaker["id"],
                "name": speaker["name"],
                "audio_path": speaker["audio_path"],
                "reference_text": speaker.get("reference_text"),
                "created_at": speaker["created_at"]
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        system_logger.error(f"【说话人管理】保存失败: {e}")
        raise HTTPException(status_code=500, detail=f"保存说话人失败: {str(e)}")


@router.delete("/{speaker_id}")
async def delete_speaker_api(speaker_id: str):
    """删除指定的说话人"""
    try:
        # 获取说话人信息以删除关联的音频文件
        db = load_speakers_db()
        speaker = None
        for s in db["speakers"]:
            if s["id"] == speaker_id:
                speaker = s
                break

        if not speaker:
            raise HTTPException(status_code=404, detail="说话人不存在")

        # 删除关联的音频文件
        audio_path = speaker.get("audio_path")
        if audio_path and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
                system_logger.info(f"【说话人管理】删除音频文件: {audio_path}")
            except Exception as e:
                system_logger.warning(f"【说话人管理】删除音频文件失败: {e}")

        # 删除说话人记录
        if delete_speaker(speaker_id):
            OperationLogger.log_speaker_operation("删除", speaker.get("name", "unknown"), speaker_id)
            return {
                "success": True,
                "message": "说话人已删除"
            }
        else:
            raise HTTPException(status_code=500, detail="删除说话人失败")

    except HTTPException:
        raise
    except Exception as e:
        system_logger.error(f"【说话人管理】删除失败: {e}")
        raise HTTPException(status_code=500, detail=f"删除说话人失败: {str(e)}")
