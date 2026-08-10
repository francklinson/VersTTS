#!/usr/bin/env python3
"""
IndexTTS 模型加载器
"""

import os
import sys
import time

import torch
from fastapi import HTTPException

from backend.logger_config import OperationLogger, system_logger
from backend.config import models, ALGORITHM_PATHS, MODEL_PATHS
from backend.core.model_manager import model_manager


def get_indextts_model():
    """获取或加载IndexTTS2模型 - 使用原始GitHub代码方式"""
    if "indextts" not in models:
        start_time = time.time()
        OperationLogger.log_model_load("IndexTTS2", "开始加载")

        try:
            # 添加IndexTTS路径
            indextts_path = ALGORITHM_PATHS['indextts']
            if indextts_path not in sys.path:
                sys.path.insert(0, indextts_path)

            # 使用IndexTTS2 (infer_v2) - 按照GitHub README示例
            from indextts.infer_v2 import IndexTTS2

            model_dir = MODEL_PATHS['indextts']
            cfg_path = os.path.join(model_dir, "config.yaml")

            system_logger.info(f"【模型加载】IndexTTS2 从路径: {model_dir}")

            device = "cuda" if torch.cuda.is_available() else "cpu"

            max_retries = 2
            for attempt in range(max_retries):
                try:
                    # 按照GitHub示例初始化: use_fp16=False, use_cuda_kernel=False, use_deepspeed=False
                    models["indextts"] = IndexTTS2(
                        cfg_path=cfg_path,
                        model_dir=model_dir,
                        use_fp16=False,
                        device=device,
                        use_cuda_kernel=False,
                        use_deepspeed=False
                    )
                    break
                except RuntimeError as e:
                    if "CUDA" in str(e) or "out of memory" in str(e).lower():
                        system_logger.warning("【模型加载】IndexTTS2 OOM，尝试驱逐其他模型...")
                        model_manager.request_eviction(needed_mb=2000, exclude_key="indextts")
                        time.sleep(3)
                        if attempt == max_retries - 1:
                            raise
                    else:
                        raise

            model_manager.touch("indextts")
            duration = time.time() - start_time
            gpu_mem = torch.cuda.memory_allocated() / 1024 ** 3 if torch.cuda.is_available() else 0
            OperationLogger.log_model_load("IndexTTS2", "成功", duration, f"GPU内存: {gpu_mem:.2f}GB")
            OperationLogger.log_performance("IndexTTS2加载", duration, 0, gpu_mem)
        except Exception as e:
            OperationLogger.log_model_load("IndexTTS2", "失败", 0, str(e))
            system_logger.error(f"【模型加载】IndexTTS2 失败: {e}")
            raise HTTPException(status_code=500, detail=f"IndexTTS2模型加载失败: {str(e)}")

    model_manager.touch("indextts")
    return models["indextts"]
