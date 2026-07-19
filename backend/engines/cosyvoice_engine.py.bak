#!/usr/bin/env python3
"""
CosyVoice 模型加载器
"""

import os
import sys
import time

import torch

from backend.logger_config import OperationLogger, system_logger
from backend.config import models, ALGORITHM_PATHS, PROJECT_ROOT


def get_cosyvoice_model(model_dir: str = "Fun-CosyVoice3-0.5B"):
    """获取或加载CosyVoice模型，使用独立的 transformers 4.51.3"""
    key = f"cosyvoice_{model_dir}"
    if key not in models:
        start_time = time.time()
        model_name = f"CosyVoice-{model_dir}"
        
        # 记录加载开始
        OperationLogger.log_model_load(model_name, "开始加载")
        
        # 添加 CosyVoice 路径
        cosyvoice_path = ALGORITHM_PATHS['cosyvoice']
        if cosyvoice_path not in sys.path:
            sys.path.insert(0, cosyvoice_path)
        matchatts_path = ALGORITHM_PATHS['matchatts']
        if matchatts_path not in sys.path:
            sys.path.insert(0, matchatts_path)
        
        # 确定模型路径
        model_path = os.path.join(ALGORITHM_PATHS['cosyvoice'], "models", "iic", model_dir)
        if not os.path.exists(model_path):
            model_path = model_path.replace("-", "___")
        
        # 确定设备
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        # 记录路径和设备
        OperationLogger.log_model_load_detail(
            model_name,
            "路径确认",
            model_path=model_path,
            device=device
        )
        
        # 检查模型文件大小
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
                extra_info={"状态": "路径不存在，将尝试自动下载"}
            )
        
        # CosyVoice 源码已修改，直接使用本地 transformers 4.51.3
        OperationLogger.log_model_load_detail(
            model_name,
            "加载中",
            device=device
        )
        
        from cosyvoice.cli.cosyvoice import AutoModel
        models[key] = AutoModel(model_dir=model_path)

        duration = time.time() - start_time
        gpu_mem = torch.cuda.memory_allocated() / 1024 ** 3 if torch.cuda.is_available() else 0
        
        # 记录加载完成详细信息
        OperationLogger.log_model_load_detail(
            model_name,
            "完成",
            device=device,
            memory_usage=gpu_mem,
            extra_info={"耗时": f"{duration:.3f}s"}
        )
        
        OperationLogger.log_model_load(model_name, "成功", duration, f"GPU内存: {gpu_mem:.2f}GB")
        OperationLogger.log_performance("CosyVoice加载", duration, 0, gpu_mem)

    return models[key]
