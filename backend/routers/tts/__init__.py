#!/usr/bin/env python3
"""
TTS 路由模块
"""

from fastapi import APIRouter

from .chattts import router as chattts_router
from .cosyvoice import router as cosyvoice_router
from .f5tts import router as f5tts_router
from .qwen3tts import router as qwen3tts_router
from .openvoice import router as openvoice_router
from .gptsovits import router as gptsovits_router
from .voxcpm import router as voxcpm_router
from .indextts import router as indextts_router
from .fireredtts import router as fireredtts_router
from .omnivoice import router as omnivoice_router
from .pilottts import router as pilottts_router
from .dotstts import router as dotstts_router
from .fishspeech import router as fishspeech_router

# TTS 主路由
router = APIRouter()

# 注册各引擎路由
router.include_router(chattts_router, prefix="/chattts")
router.include_router(cosyvoice_router, prefix="/cosyvoice")
router.include_router(f5tts_router, prefix="/f5tts")
router.include_router(qwen3tts_router, prefix="/qwen3tts")
router.include_router(openvoice_router, prefix="/openvoice")
router.include_router(gptsovits_router, prefix="/gptsovits")
router.include_router(voxcpm_router, prefix="/voxcpm")
router.include_router(indextts_router, prefix="/indextts")
router.include_router(fireredtts_router, prefix="/fireredtts")
router.include_router(omnivoice_router, prefix="/omnivoice")
router.include_router(pilottts_router, prefix="/pilottts")
router.include_router(dotstts_router, prefix="/dotstts")
router.include_router(fishspeech_router, prefix="/fishspeech")

__all__ = ['router']
