#!/usr/bin/env python3
"""
FastAPI 应用生命周期管理
"""

import sys
import time
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI

from backend.logger_config import OperationLogger, system_logger
from backend.config import models, PROJECT_ROOT, ensure_directories


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
        system_logger.info(
            f"【硬件信息】当前GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")
    else:
        system_logger.warning("【硬件信息】CUDA不可用,将使用CPU模式")

    # 创建输出目录
    ensure_directories()
    system_logger.info(f"【目录初始化】完成")

    # 记录配置信息
    OperationLogger.log_config_load("CORS配置", "成功", "允许所有来源")
    OperationLogger.log_config_load("FastAPI配置", "成功", f"版本: {app.version}")

    init_duration = time.time() - init_start_time
    OperationLogger.log_init_complete(init_duration, "成功")

    yield

    # 服务关闭
    system_logger.info("=" * 80)
    system_logger.info("【服务关闭】正在清理资源...")

    # 记录已加载的模型
    loaded_models = list(models.keys())
    if loaded_models:
        system_logger.info(f"【服务关闭】清理已加载模型: {', '.join(loaded_models)}")

    models.clear()

    # 清理CUDA缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        system_logger.info("【服务关闭】CUDA缓存已清理")

    system_logger.info("【服务关闭】TTS服务已关闭")
    system_logger.info("=" * 80)
