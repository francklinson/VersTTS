#!/usr/bin/env python3
"""
API 路由模块
"""

from fastapi import APIRouter

from .health import router as health_router
from .speakers import router as speakers_router
from .tts import router as tts_router
from .batch import router as batch_router
from .reference_voices import router as reference_voices_router
from .recording_scripts import router as recording_scripts_router

# 主路由
router = APIRouter()

# 注册子路由
router.include_router(health_router, tags=["Health"])
router.include_router(speakers_router, prefix="/speakers", tags=["Speakers"])
router.include_router(tts_router, prefix="/tts", tags=["TTS"])
router.include_router(batch_router, prefix="/tts/batch", tags=["Batch"])
router.include_router(reference_voices_router, prefix="/reference_voices", tags=["Reference Voices"])
router.include_router(recording_scripts_router, prefix="/recording_scripts", tags=["Recording Scripts"])

__all__ = ['router']
