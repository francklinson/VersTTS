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
        OperationLogger.log_model_load(f"CosyVoice-{model_dir}", "开始加载")

        # 添加 CosyVoice 路径
        cosyvoice_path = ALGORITHM_PATHS['cosyvoice']
        if cosyvoice_path not in sys.path:
            sys.path.insert(0, cosyvoice_path)
        matchatts_path = ALGORITHM_PATHS['matchatts']
        if matchatts_path not in sys.path:
            sys.path.insert(0, matchatts_path)

        # CosyVoice 源码已修改，直接使用本地 transformers 4.51.3
        from cosyvoice.cli.cosyvoice import AutoModel
        model_path = os.path.join(ALGORITHM_PATHS['cosyvoice'], "models", "iic", model_dir)
        if not os.path.exists(model_path):
            model_path = model_path.replace("-", "___")

        system_logger.info(f"【模型加载】CosyVoice 从路径: {model_path}")
        models[key] = AutoModel(model_dir=model_path)

        duration = time.time() - start_time
        gpu_mem = torch.cuda.memory_allocated() / 1024 ** 3 if torch.cuda.is_available() else 0
        OperationLogger.log_model_load(f"CosyVoice-{model_dir}", "成功", duration, f"GPU内存: {gpu_mem:.2f}GB")
        OperationLogger.log_performance("CosyVoice加载", duration, 0, gpu_mem)

    return models[key]
