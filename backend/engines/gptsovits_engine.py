#!/usr/bin/env python3
"""
GPT-SoVITS 模型加载器
"""

import os
import sys

import torch

from backend.logger_config import system_logger
from backend.config import models, ALGORITHM_PATHS, PROJECT_ROOT


def _setup_gpt_sovits_path():
    """设置GPT-SoVITS所需的系统路径"""
    gpt_sovits_root = ALGORITHM_PATHS['gptsovits']
    gpt_sovits_module = ALGORITHM_PATHS['gptsovits_module']

    if gpt_sovits_root not in sys.path:
        sys.path.append(gpt_sovits_root)
    if gpt_sovits_module not in sys.path:
        sys.path.append(gpt_sovits_module)

    # 设置BERT模型路径环境变量（使用绝对路径）
    bert_path = os.path.join(gpt_sovits_module, "pretrained_models", "chinese-roberta-wwm-ext-large")
    os.environ["bert_path"] = bert_path

    # 设置G2PW模型路径环境变量（使用绝对路径，避免相对路径问题）
    g2pw_model_path = os.path.join(gpt_sovits_module, "text", "G2PWModel")
    os.environ["g2pw_model"] = g2pw_model_path

    # 确保G2PW模型目录存在（避免自动下载逻辑触发）
    os.makedirs(g2pw_model_path, exist_ok=True)

    # 保存当前工作目录并切换到GPT-SoVITS目录
    original_cwd = os.getcwd()
    if os.getcwd() != gpt_sovits_root:
        os.chdir(gpt_sovits_root)

    return original_cwd


def get_gpt_sovits_model(version: str = "v2"):
    """获取或加载GPT-SoVITS模型"""
    key = f"gpt_sovits_{version}"
    if key not in models:
        import time
        from backend.logger_config import OperationLogger
        start_time = time.time()
        OperationLogger.log_model_load(f"GPT-SoVITS-{version}", "开始加载")

        # 设置路径并保存原工作目录
        original_cwd = _setup_gpt_sovits_path()

        try:
            from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config

            # 加载配置文件
            config_path = os.path.join(ALGORITHM_PATHS['gptsovits_module'], "configs", "tts_infer.yaml")
            tts_config = TTS_Config(config_path)

            # 根据版本选择配置
            if version in tts_config.default_configs:
                # 使用指定版本的配置
                tts_config.configs = tts_config.default_configs[version].copy()
                tts_config.version = version

            # 使用CUDA
            if torch.cuda.is_available():
                tts_config.configs["device"] = "cuda"
                tts_config.configs["is_half"] = True
                tts_config.device = "cuda"
                tts_config.is_half = True
            else:
                tts_config.configs["device"] = "cpu"
                tts_config.configs["is_half"] = False
                tts_config.device = "cpu"
                tts_config.is_half = False

            system_logger.info(f"【模型加载】GPT-SoVITS 版本: {tts_config.version}, 设备: {tts_config.device}")

            models[key] = {
                "config": tts_config,
                "pipeline": None,  # 延迟初始化
                "version": version
            }

            duration = time.time() - start_time
            gpu_mem = torch.cuda.memory_allocated() / 1024 ** 3 if torch.cuda.is_available() else 0
            OperationLogger.log_model_load(f"GPT-SoVITS-{version}", "成功", duration, f"GPU内存: {gpu_mem:.2f}GB")
            OperationLogger.log_performance("GPT-SoVITS加载", duration, 0, gpu_mem)
        finally:
            # 恢复工作目录
            os.chdir(original_cwd)

    return models[key]


def init_gpt_sovits_pipeline(model_info, ref_audio_path: str = None):
    """初始化GPT-SoVITS推理管道"""
    # 设置路径并保存原工作目录
    original_cwd = _setup_gpt_sovits_path()

    try:
        from GPT_SoVITS.TTS_infer_pack.TTS import TTS

        # 检查是否需要重新初始化管道（版本变化或未初始化）
        pipeline = model_info.get("pipeline")
        cached_version = model_info.get("pipeline_version")
        current_version = model_info.get("version")

        if pipeline is None or cached_version != current_version:
            # 清理旧的管道
            if pipeline is not None:
                del pipeline
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                system_logger.info(f"【GPT-SoVITS】版本切换: {cached_version} -> {current_version}")

            # 创建新管道
            pipeline = TTS(model_info["config"])
            model_info["pipeline"] = pipeline
            model_info["pipeline_version"] = current_version
            system_logger.info(f"【GPT-SoVITS】管道初始化完成 | 版本: {current_version}")

        if ref_audio_path and os.path.exists(ref_audio_path):
            model_info["pipeline"].set_ref_audio(ref_audio_path)

        return model_info["pipeline"]
    finally:
        # 恢复工作目录
        os.chdir(original_cwd)
