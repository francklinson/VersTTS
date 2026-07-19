#!/usr/bin/env python3
"""
FastAPI 应用生命周期管理
"""

import os
import sys
import gc
import time
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI

from backend.logger_config import OperationLogger, system_logger
from backend.config import models, PROJECT_ROOT, ensure_directories
from backend.core.memory_utils import cleanup_memory, get_gpu_memory_info
from backend.core.concurrency import initialize_concurrency, shutdown_concurrency
from backend.core.model_manager import model_manager


def get_preload_models():
    """从环境变量获取预加载模型配置"""
    preload_config = os.environ.get('PRELOAD_MODELS', '')

    if not preload_config or preload_config.lower() == 'none':
        return []

    if preload_config.lower() == 'all':
        return ['qwen3tts_base', 'qwen3tts_custom', 'qwen3tts_design', 'voxcpm']

    # 解析逗号分隔的列表
    return [m.strip() for m in preload_config.split(',') if m.strip()]


def _register_all_models():
    """向 ModelManager 注册主进程内所有模型"""
    model_manager.register("chattts", "ChatTTS", 1500)
    model_manager.register("f5tts", "F5-TTS", 1500)
    model_manager.register("indextts", "IndexTTS2", 2000)
    model_manager.register("openvoice", "OpenVoice V1", 1500)
    model_manager.register("openvoice_v2", "OpenVoice V2", 1500)
    model_manager.register("fireredtts2", "FireRedTTS2", 2000)
    model_manager.register("qwen3tts_1.7B_Base", "Qwen3-TTS-1.7B-Base", 3000)
    model_manager.register("qwen3tts_1.7B_CustomVoice", "Qwen3-TTS-1.7B-CustomVoice", 3000)
    model_manager.register("qwen3tts_1.7B_VoiceDesign", "Qwen3-TTS-1.7B-VoiceDesign", 3500)
    model_manager.register("voxcpm", "VoxCPM", 4000)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    init_start_time = time.time()

    system_logger.info("=" * 80)
    system_logger.info("【服务启动】初始化应用生命周期")
    system_logger.info("=" * 80)

    # 记录系统环境信息
    system_logger.info(f"【环境信息】Python版本: {sys.version}")
    system_logger.info(f"【环境信息】项目路径: {PROJECT_ROOT}")

    # 检查CUDA
    if torch.cuda.is_available():
        cuda_info = f"{torch.cuda.get_device_name(0)} (CUDA {torch.version.cuda})"
        system_logger.info(f"【硬件信息】CUDA可用: {cuda_info}")
        system_logger.info(f"【硬件信息】GPU数量: {torch.cuda.device_count()}")
        mem_info = get_gpu_memory_info()
        system_logger.info(
            f"【硬件信息】当前GPU内存: {mem_info['total']:.2f} GB")
        system_logger.info(
            f"【硬件信息】当前显存使用: {mem_info['allocated']:.2f} GB")
    else:
        system_logger.warning("【硬件信息】CUDA不可用,将使用CPU模式")

    # 创建输出目录
    ensure_directories()
    system_logger.info(f"【目录初始化】完成")

    # 初始化并发控制系统
    await initialize_concurrency()

    # 记录配置信息
    OperationLogger.log_config_load("CORS配置", "成功", "允许所有来源")
    OperationLogger.log_config_load("FastAPI配置", "成功", f"版本: {app.version}")
    OperationLogger.log_config_load("并发控制", "成功", "GPU锁+限流器+任务队列")

    # ========== 模型管理初始化 ==========
    _register_all_models()
    model_manager.register_to_main_service()
    await model_manager.start_idle_check()
    await model_manager.start_heartbeat()
    system_logger.info("【模型管理】ModelManager 已启动（按需加载 + 空闲超时卸载）")

    # ========== 预加载模型（可选） ==========
    preload_models = get_preload_models()

    if preload_models:
        system_logger.info("=" * 80)
        system_logger.info(f"【模型预加载】配置: {', '.join(preload_models)}")
        system_logger.info("=" * 80)

        # 预加载 Qwen3-TTS Base 模型
        if 'qwen3tts_base' in preload_models:
            try:
                from backend.engines import get_qwen3tts_model
                system_logger.info("【模型预加载】正在加载 Qwen3-TTS Base 模型...")
                get_qwen3tts_model("1.7B", "Base")
                system_logger.info("【模型预加载】Qwen3-TTS Base 模型加载完成")
            except Exception as e:
                system_logger.warning(f"【模型预加载】Qwen3-TTS Base 加载失败: {e}")

        # 预加载 Qwen3-TTS CustomVoice 模型
        if 'qwen3tts_custom' in preload_models:
            try:
                from backend.engines import get_qwen3tts_model
                system_logger.info("【模型预加载】正在加载 Qwen3-TTS CustomVoice 模型...")
                get_qwen3tts_model("1.7B", "CustomVoice")
                system_logger.info("【模型预加载】Qwen3-TTS CustomVoice 模型加载完成")
            except Exception as e:
                system_logger.warning(f"【模型预加载】Qwen3-TTS CustomVoice 加载失败: {e}")

        # 预加载 Qwen3-TTS VoiceDesign 模型
        if 'qwen3tts_design' in preload_models:
            try:
                from backend.engines import get_qwen3tts_model
                system_logger.info("【模型预加载】正在加载 Qwen3-TTS VoiceDesign 模型...")
                get_qwen3tts_model("1.7B", "VoiceDesign")
                system_logger.info("【模型预加载】Qwen3-TTS VoiceDesign 模型加载完成")
            except Exception as e:
                system_logger.warning(f"【模型预加载】Qwen3-TTS VoiceDesign 加载失败: {e}")

        # 预加载 VoxCPM 模型
        if 'voxcpm' in preload_models:
            try:
                from backend.engines import get_voxcpm_model
                system_logger.info("【模型预加载】正在加载 VoxCPM 模型...")
                get_voxcpm_model()
                system_logger.info("【模型预加载】VoxCPM 模型加载完成")
            except Exception as e:
                system_logger.warning(f"【模型预加载】VoxCPM 加载失败: {e}")

        # 显示预加载后的显存状态
        if torch.cuda.is_available():
            mem_after = get_gpu_memory_info()
            system_logger.info(f"【模型预加载】预加载后显存使用: {mem_after['allocated']:.2f} GB")

        system_logger.info("=" * 80)
    else:
        system_logger.info("【模型预加载】已禁用（按需加载模式）")

    init_duration = time.time() - init_start_time
    OperationLogger.log_init_complete(init_duration, "成功")

    # 启动 outputs/ 目录定时清理任务（每小时执行一次）
    async def _outputs_cleanup_loop():
        from backend.core.audio_utils import cleanup_old_outputs
        while True:
            try:
                await asyncio.sleep(3600)  # 每小时
                cleanup_old_outputs(max_age_hours=24)
            except asyncio.CancelledError:
                break
            except Exception as e:
                system_logger.warning(f"【清理】定时清理任务出错: {e}")

    cleanup_task = asyncio.create_task(_outputs_cleanup_loop())
    system_logger.info("【清理】outputs/ 定时清理已启动（每1小时，清理24h前文件）")

    yield

    # 服务关闭
    system_logger.info("=" * 80)
    system_logger.info("【服务关闭】正在清理资源...")

    # 停止 outputs 定时清理任务
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass

    # 停止模型管理后台任务
    await model_manager.stop()

    # 从主服务注销
    model_manager.unregister_from_main_service()

    # 关闭并发控制系统
    await shutdown_concurrency()

    # 记录已加载的模型
    loaded_models = list(models.keys())
    if loaded_models:
        system_logger.info(f"【服务关闭】清理已加载模型: {', '.join(loaded_models)}")

    # 获取清理前的显存信息
    mem_before = get_gpu_memory_info()

    # 清理模型缓存
    for key, model in list(models.items()):
        try:
            del model
        except Exception as e:
            system_logger.warning(f"【服务关闭】清理模型 {key} 时出错: {e}")

    models.clear()

    # 强制垃圾回收和显存清理
    cleanup_memory(force_gc=True, empty_cache=True, synchronize=True)

    # 获取清理后的显存信息
    mem_after = get_gpu_memory_info()
    released = mem_before["allocated"] - mem_after["allocated"]

    system_logger.info(f"【服务关闭】显存释放: {released:.2f} GB")
    system_logger.info("【服务关闭】TTS服务已关闭")
    system_logger.info("=" * 80)
