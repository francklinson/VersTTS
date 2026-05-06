#!/usr/bin/env python3
"""
内存和显存管理工具模块
用于防止内存泄漏和显存资源未释放
"""

import gc
import time
import logging
from typing import Optional
from functools import wraps

import torch

from backend.logger_config import system_logger


def cleanup_memory(force_gc: bool = True, empty_cache: bool = True, synchronize: bool = False):
    """
    清理内存和显存资源

    Args:
        force_gc: 是否强制运行Python垃圾回收
        empty_cache: 是否清空PyTorch CUDA缓存
        synchronize: 是否同步CUDA操作（会等待所有CUDA操作完成）
    """
    try:
        # 强制垃圾回收
        if force_gc:
            gc.collect()

        # 清理CUDA缓存
        if empty_cache and torch.cuda.is_available():
            if synchronize:
                torch.cuda.synchronize()
            torch.cuda.empty_cache()

    except Exception as e:
        system_logger.warning(f"【内存清理】清理过程中出现警告: {e}")


def get_gpu_memory_info() -> dict:
    """
    获取GPU内存使用信息

    Returns:
        dict: 包含allocated、reserved、total的字典（单位：GB）
    """
    if not torch.cuda.is_available():
        return {"allocated": 0, "reserved": 0, "total": 0}

    try:
        allocated = torch.cuda.memory_allocated() / 1024 ** 3
        reserved = torch.cuda.memory_reserved() / 1024 ** 3
        total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3

        return {
            "allocated": allocated,
            "reserved": reserved,
            "total": total,
            "free": total - allocated
        }
    except Exception as e:
        system_logger.warning(f"【显存监控】获取显存信息失败: {e}")
        return {"allocated": 0, "reserved": 0, "total": 0, "free": 0}


def log_gpu_memory_usage(label: str = ""):
    """记录GPU内存使用情况"""
    if not torch.cuda.is_available():
        return

    mem_info = get_gpu_memory_info()
    prefix = f"【{label}】" if label else "【显存监控】"
    system_logger.info(
        f"{prefix} GPU显存使用: "
        f"已分配={mem_info['allocated']:.2f}GB, "
        f"预留={mem_info['reserved']:.2f}GB, "
        f"总计={mem_info['total']:.2f}GB, "
        f"可用={mem_info['free']:.2f}GB"
    )


def release_tensor(tensor, name: str = ""):
    """
    安全释放PyTorch张量

    Args:
        tensor: 要释放的张量
        name: 张量名称（用于日志）
    """
    if tensor is None:
        return

    try:
        if isinstance(tensor, torch.Tensor):
            # 移动到CPU以减少显存占用
            if tensor.is_cuda:
                tensor = tensor.cpu()
            # 删除张量
            del tensor
    except Exception as e:
        if name:
            system_logger.warning(f"【内存清理】释放张量 '{name}' 时出错: {e}")


class GPUMemoryTracker:
    """GPU内存使用追踪器"""

    def __init__(self, label: str = ""):
        self.label = label
        self.start_mem = 0
        self.peak_mem = 0

    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            self.start_mem = torch.cuda.memory_allocated() / 1024 ** 3
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if torch.cuda.is_available():
            self.peak_mem = torch.cuda.max_memory_allocated() / 1024 ** 3
            current_mem = torch.cuda.memory_allocated() / 1024 ** 3
            used_mem = current_mem - self.start_mem

            prefix = f"【{self.label}】" if self.label else "【显存追踪】"
            system_logger.info(
                f"{prefix} 显存变化: "
                f"起始={self.start_mem:.2f}GB, "
                f"当前={current_mem:.2f}GB, "
                f"使用={used_mem:.2f}GB, "
                f"峰值={self.peak_mem:.2f}GB"
            )


def with_memory_cleanup(func):
    """装饰器：在函数执行后自动清理内存"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        finally:
            cleanup_memory()
    return wrapper


def clear_model_cache(model_key: Optional[str] = None):
    """
    清理模型缓存

    Args:
        model_key: 指定要清理的模型key，为None时清理所有
    """
    from backend.config import models

    try:
        if model_key and model_key in models:
            system_logger.info(f"【模型缓存】清理模型: {model_key}")
            model = models.pop(model_key)
            del model
        elif model_key is None:
            system_logger.info(f"【模型缓存】清理所有模型，共 {len(models)} 个")
            for key, model in list(models.items()):
                del model
            models.clear()

        cleanup_memory()

    except Exception as e:
        system_logger.error(f"【模型缓存】清理失败: {e}")
