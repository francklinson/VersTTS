#!/usr/bin/env python3
"""
ChatTTS 模型加载器
"""

import os
import time

import torch
from fastapi import HTTPException

from backend.logger_config import OperationLogger, system_logger
from backend.config import models, ALGORITHM_PATHS, PROJECT_ROOT


def get_chattts_model():
    """获取或加载ChatTTS模型"""
    if "chattts" not in models:
        start_time = time.time()
        OperationLogger.log_model_load("ChatTTS", "开始加载")

        # 清理CUDA缓存和状态，避免与之前加载的模型（如CosyVoice）产生冲突
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                system_logger.info("【模型加载】ChatTTS CUDA缓存已清理")
            except Exception as e:
                system_logger.warning(f"【模型加载】ChatTTS CUDA缓存清理警告: {e}")

        import ChatTTS
        chat = ChatTTS.Chat()
        model_path = os.path.join(ALGORITHM_PATHS['chattts'], "models")
        system_logger.info(f"【模型加载】ChatTTS 从路径: {model_path}")

        # 显式指定设备，避免ChatTTS自动检测时出现问题
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        system_logger.info(f"【模型加载】ChatTTS 使用设备: {device}")

        try:
            if not chat.load(source="custom", custom_path=model_path, device=device):
                OperationLogger.log_model_load("ChatTTS", "失败", 0, "模型加载错误")
                raise HTTPException(status_code=500, detail="ChatTTS模型加载失败")
        except RuntimeError as e:
            if "CUDA" in str(e) or "cuda" in str(e).lower():
                system_logger.error(f"【模型加载】ChatTTS CUDA错误: {e}")
                # 尝试强制重置CUDA状态
                if torch.cuda.is_available():
                    try:
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        # 等待一点时间让CUDA恢复
                        time.sleep(1)
                        system_logger.info("【模型加载】ChatTTS 尝试重新加载...")
                        if not chat.load(source="custom", custom_path=model_path, device=device):
                            raise HTTPException(status_code=500, detail="ChatTTS模型加载失败（CUDA恢复后重试）")
                    except Exception as retry_e:
                        system_logger.error(f"【模型加载】ChatTTS CUDA恢复失败: {retry_e}")
                        raise HTTPException(status_code=500, detail=f"ChatTTS模型加载失败: {str(e)}")
                else:
                    raise HTTPException(status_code=500, detail=f"ChatTTS模型加载失败: {str(e)}")
            else:
                raise

        models["chattts"] = chat
        duration = time.time() - start_time

        # 记录GPU内存使用
        gpu_mem = torch.cuda.memory_allocated() / 1024 ** 3 if torch.cuda.is_available() else 0
        OperationLogger.log_model_load("ChatTTS", "成功", duration, f"GPU内存: {gpu_mem:.2f}GB")
        OperationLogger.log_performance("ChatTTS加载", duration, 0, gpu_mem)

    return models["chattts"]
