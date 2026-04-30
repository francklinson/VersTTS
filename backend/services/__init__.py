#!/usr/bin/env python3
"""
业务逻辑服务模块
"""

from .speaker_service import (
    load_speakers_db,
    save_speakers_db,
    get_speaker_by_name,
    get_speaker_by_id,
    check_speaker_name_exists,
    add_speaker,
    delete_speaker,
)

__all__ = [
    'load_speakers_db',
    'save_speakers_db',
    'get_speaker_by_name',
    'get_speaker_by_id',
    'check_speaker_name_exists',
    'add_speaker',
    'delete_speaker',
]
