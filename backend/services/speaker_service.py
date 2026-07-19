#!/usr/bin/env python3
"""
说话人管理服务
与模型解耦的说话人管理模块
"""

import os
import time
import json
from typing import Optional, Dict
from datetime import datetime

from fastapi import HTTPException

from backend.logger_config import system_logger, OperationLogger
from backend.config import SPEAKERS_DIR, SPEAKERS_DB_FILE


def load_speakers_db() -> Dict:
    """加载说话人数据库"""
    if os.path.exists(SPEAKERS_DB_FILE):
        try:
            with open(SPEAKERS_DB_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            system_logger.error(f"【说话人管理】加载数据库失败: {e}")
    return {"speakers": [], "version": "1.0"}


def save_speakers_db(db: Dict) -> bool:
    """保存说话人数据库"""
    try:
        with open(SPEAKERS_DB_FILE, 'w', encoding='utf-8') as f:
            json.dump(db, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        system_logger.error(f"【说话人管理】保存数据库失败: {e}")
        return False


def get_speaker_by_name(name: str) -> Optional[Dict]:
    """根据名称获取说话人"""
    db = load_speakers_db()
    for speaker in db["speakers"]:
        if speaker["name"] == name:
            return speaker
    return None


def get_speaker_by_id(speaker_id: str) -> Optional[Dict]:
    """根据ID获取说话人"""
    db = load_speakers_db()
    for speaker in db["speakers"]:
        if speaker["id"] == speaker_id:
            return speaker
    return None


def check_speaker_name_exists(name: str) -> bool:
    """检查说话人名称是否已存在"""
    return get_speaker_by_name(name) is not None


def add_speaker(name: str, embedding: Optional[str] = None, 
                audio_path: Optional[str] = None,
                reference_text: Optional[str] = None) -> Dict:
    """添加新说话人（与模型解耦）
    
    Args:
        name: 说话人名称
        embedding: 说话人embedding (base64编码，可选，与模型解耦后可为None)
        audio_path: 参考音频路径
        reference_text: 参考音频对应的文本（可选）
    """
    db = load_speakers_db()

    speaker = {
        "id": f"spk_{int(time.time() * 1000)}",
        "name": name,
        "embedding": embedding,  # 可为None，与模型解耦
        "audio_path": audio_path,
        "reference_text": reference_text,
        "created_at": datetime.now().isoformat(),
        "model_type": "universal"  # 改为通用类型，不再绑定特定模型
    }

    db["speakers"].append(speaker)

    if save_speakers_db(db):
        system_logger.info(
            f"【说话人管理】添加说话人成功: {name}, 文本: {reference_text[:30] if reference_text else '无'}")
        return speaker
    else:
        raise HTTPException(status_code=500, detail="保存说话人失败")


def delete_speaker(speaker_id: str) -> bool:
    """删除说话人"""
    db = load_speakers_db()
    original_count = len(db["speakers"])
    db["speakers"] = [s for s in db["speakers"] if s["id"] != speaker_id]

    if len(db["speakers"]) < original_count:
        save_speakers_db(db)
        system_logger.info(f"【说话人管理】删除说话人: {speaker_id}")
        return True
    return False


def update_speaker(speaker_id: str, **updates) -> Optional[Dict]:
    """更新说话人信息（如参考文本等）"""
    db = load_speakers_db()
    for speaker in db["speakers"]:
        if speaker["id"] == speaker_id:
            for key, value in updates.items():
                if key in ("reference_text", "name", "audio_path"):
                    speaker[key] = value
            save_speakers_db(db)
            system_logger.info(f"【说话人管理】更新说话人: {speaker_id}, 字段: {list(updates.keys())}")
            return speaker
    return None
