#!/usr/bin/env python3
"""
参考人声路由
"""

import os
import json
from typing import List, Dict, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from backend.logger_config import system_logger
from backend.config import PROJECT_ROOT, ALGORITHMS_DIR

router = APIRouter()


# 参考人声存储路径
REFERENCE_VOICES_DIR = os.path.join(PROJECT_ROOT, "reference_voices")


def load_reference_voices() -> List[Dict]:
    """加载参考人声列表"""
    voices = []
    
    # 从参考人声目录加载
    if os.path.exists(REFERENCE_VOICES_DIR):
        for category in os.listdir(REFERENCE_VOICES_DIR):
            category_path = os.path.join(REFERENCE_VOICES_DIR, category)
            if os.path.isdir(category_path):
                for filename in os.listdir(category_path):
                    if filename.endswith(('.wav', '.mp3', '.ogg')):
                        voice_id = f"{category}_{filename.replace('.', '_')}"
                        voices.append({
                            "id": voice_id,
                            "category": category,
                            "filename": filename,
                            "path": os.path.join(category_path, filename),
                            "name": filename.rsplit('.', 1)[0]
                        })
    
    return voices


def get_reference_voice_by_id(voice_id: str) -> Optional[Dict]:
    """根据ID获取参考人声"""
    voices = load_reference_voices()
    for voice in voices:
        if voice["id"] == voice_id:
            return voice
    return None


@router.get("/")
async def get_reference_voices():
    """获取所有参考人声列表"""
    try:
        voices = load_reference_voices()
        
        # 按分类组织
        categories = {}
        for voice in voices:
            category = voice["category"]
            if category not in categories:
                categories[category] = []
            categories[category].append({
                "id": voice["id"],
                "name": voice["name"],
                "filename": voice["filename"]
            })
        
        return {
            "success": True,
            "categories": categories,
            "total": len(voices)
        }
    except Exception as e:
        system_logger.error(f"【参考人声】获取列表失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取参考人声列表失败: {str(e)}")


@router.get("/categories")
async def get_reference_voice_categories():
    """获取参考人声分类列表"""
    try:
        voices = load_reference_voices()
        categories = list(set(v["category"] for v in voices))
        
        return {
            "success": True,
            "categories": categories
        }
    except Exception as e:
        system_logger.error(f"【参考人声】获取分类失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取分类列表失败: {str(e)}")


@router.get("/{category}/{filename}")
async def get_reference_audio(category: str, filename: str):
    """获取参考音频文件"""
    try:
        file_path = os.path.join(REFERENCE_VOICES_DIR, category, filename)
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="音频文件不存在")
        
        # 根据文件扩展名确定媒体类型
        ext = os.path.splitext(filename)[1].lower()
        media_type = {
            '.wav': 'audio/wav',
            '.mp3': 'audio/mpeg',
            '.ogg': 'audio/ogg',
        }.get(ext, 'audio/wav')
        
        return FileResponse(file_path, media_type=media_type, filename=filename)
    except HTTPException:
        raise
    except Exception as e:
        system_logger.error(f"【参考人声】获取音频失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取音频失败: {str(e)}")
