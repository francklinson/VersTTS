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
        model_name = "ChatTTS"
        
        # 记录加载开始
        OperationLogger.log_model_load(model_name, "开始加载")
        
        # 确定模型路径和设备
        model_path = os.path.join(ALGORITHM_PATHS['chattts'], "models")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        device_str = str(device)
        
        # 记录路径和设备信息
        OperationLogger.log_model_load_detail(
            model_name, 
            "路径确认",
            model_path=model_path,
            device=device_str
        )
        
        # 检查模型路径并计算大小
        if os.path.exists(model_path):
            total_size = 0
            file_count = 0
            for dirpath, dirnames, filenames in os.walk(model_path):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    total_size += os.path.getsize(fp)
                    file_count += 1
            size_gb = total_size / (1024**3)
            OperationLogger.log_model_load_detail(
                model_name,
                "文件检查",
                model_path=model_path,
                model_size=f"{size_gb:.2f}GB",
                extra_info={"文件数": file_count}
            )
        else:
            OperationLogger.log_model_load_detail(
                model_name,
                "文件检查",
                model_path=model_path,
                extra_info={"状态": "路径不存在"}
            )

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
        
        # 记录开始加载模型
        OperationLogger.log_model_load_detail(
            model_name,
            "加载中",
            device=device_str
        )

        try:
            if not chat.load(source="custom", custom_path=model_path, device=device):
                OperationLogger.log_model_load(model_name, "失败", 0, "模型加载错误")
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
        
        # 记录加载完成详细信息
        OperationLogger.log_model_load_detail(
            model_name,
            "完成",
            device=device_str,
            memory_usage=gpu_mem,
            extra_info={"耗时": f"{duration:.3f}s"}
        )
        
        OperationLogger.log_model_load(model_name, "成功", duration, f"GPU内存: {gpu_mem:.2f}GB")
        OperationLogger.log_performance("ChatTTS加载", duration, 0, gpu_mem)

    return models["chattts"]
