#!/usr/bin/env python3
"""
PilotTTS 模型加载器
支持基础模型（零样本声音克隆）和指令模型（情感/副语言/方言）
"""

import os
import sys
import time
import torch
from fastapi import HTTPException

from backend.logger_config import OperationLogger, system_logger
from backend.config import models, ALGORITHM_PATHS, MODEL_PATHS

# PilotTTS 引擎缓存
_pilottts_engines = {}


def get_pilottts_engine(model_type: str = "base"):
    """
    获取或加载PilotTTS推理引擎

    Args:
        model_type: "base" (零样本克隆) 或 "instruct" (情感/方言/副语言)

    Returns:
        (InferenceEngine, config_dict)
    """
    if model_type in _pilottts_engines:
        return _pilottts_engines[model_type]

    # 显存有限，切换模型类型时自动释放另一个模型 (base ↔ instruct 互斥加载)
    other_type = "instruct" if model_type == "base" else "base"
    if other_type in _pilottts_engines:
        system_logger.info(f"【PilotTTS】显存限制，自动释放 {other_type} 模型以加载 {model_type}")
        cleanup_pilottts_engine(other_type)

    start_time = time.time()
    OperationLogger.log_model_load(f"PilotTTS-{model_type}", "开始加载")
    pilottts_dir = ALGORITHM_PATHS.get('pilottts', '')

    try:
        # 添加路径: PilotTTS根目录 (import pilot_voice) + third_party (cosyvoice/matcha)
        for p in [
            pilottts_dir,
            os.path.join(pilottts_dir, "third_party"),
            os.path.join(pilottts_dir, "third_party", "Matcha-TTS"),
        ]:
            if p not in sys.path:
                sys.path.insert(0, p)

        # 修复 CUDA_VISIBLE_DEVICES 为空字符串导致 CUDA 初始化失败的问题
        if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip() == "":
            if "CUDA_VISIBLE_DEVICES" in os.environ:
                del os.environ["CUDA_VISIBLE_DEVICES"]

        import yaml
        from pilot_voice.engine import InferenceEngine

        pretrained_dir = MODEL_PATHS.get('pilottts', os.path.join(pilottts_dir, 'pretrained_models'))

        if model_type == "instruct":
            config_path = os.path.join(pilottts_dir, "configs", "infer_pilot_tts_instruct.yaml")
            checkpoint = os.path.join(pretrained_dir, "pilot_tts_instruct.pt")
        else:
            config_path = os.path.join(pilottts_dir, "configs", "infer_pilot_tts.yaml")
            checkpoint = os.path.join(pretrained_dir, "pilot_tts.pt")

        if not os.path.exists(checkpoint):
            raise FileNotFoundError(f"PilotTTS模型文件不存在: {checkpoint}")

        with open(config_path) as f:
            config = yaml.safe_load(f)
        config["checkpoint_path"] = checkpoint

        # 修正模型路径为绝对路径
        for key in ["model", "vocoder"]:
            if key in config:
                if isinstance(config[key], dict):
                    for subkey in config[key]:
                        val = config[key][subkey]
                        if isinstance(val, str) and val.startswith("pretrained_models/"):
                            abs_path = os.path.join(pretrained_dir, val.replace("pretrained_models/", ""))
                            if os.path.exists(abs_path):
                                config[key][subkey] = abs_path

        # 修正 tokenizer 路径为绝对路径
        tokenizer_path = config.get("tokenizer", {}).get("path", "")
        if tokenizer_path and not os.path.isabs(tokenizer_path):
            abs_tokenizer = os.path.join(pilottts_dir, tokenizer_path)
            if os.path.exists(abs_tokenizer):
                config["tokenizer"]["path"] = abs_tokenizer

        # 修正 campplus 路径为绝对路径
        campplus_path = config.get("spk_embedding", {}).get("campplus_path", "")
        if campplus_path and not os.path.isabs(campplus_path):
            abs_campplus = os.path.join(pilottts_dir, campplus_path)
            if os.path.exists(abs_campplus):
                config["spk_embedding"]["campplus_path"] = abs_campplus

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        engine = InferenceEngine(config, device)

        _pilottts_engines[model_type] = (engine, config, device)

        duration = time.time() - start_time
        gpu_mem = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
        OperationLogger.log_model_load(f"PilotTTS-{model_type}", "成功", duration, f"GPU内存: {gpu_mem:.2f}GB")
        system_logger.info(f"【PilotTTS】模型加载成功: {model_type} | 耗时: {duration:.2f}s | GPU: {gpu_mem:.2f}GB")

        return _pilottts_engines[model_type]

    except Exception as e:
        OperationLogger.log_model_load(f"PilotTTS-{model_type}", "失败", 0, str(e))
        system_logger.error(f"【PilotTTS】模型加载失败: {e}")
        raise HTTPException(status_code=500, detail=f"PilotTTS模型加载失败: {str(e)}")


def cleanup_pilottts_engine(model_type: str = None):
    """清理PilotTTS引擎显存"""
    if model_type:
        if model_type in _pilottts_engines:
            del _pilottts_engines[model_type]
    else:
        _pilottts_engines.clear()

    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
