#!/usr/bin/env python3
"""
GPT-SoVITS 模型加载器
"""

import os
import sys
import time

import torch

from backend.logger_config import OperationLogger, system_logger
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
        start_time = time.time()
        model_name = f"GPT-SoVITS-{version}"
        
        OperationLogger.log_model_load(model_name, "开始加载")

        # 设置路径并保存原工作目录
        original_cwd = _setup_gpt_sovits_path()
        
        # 确定设备
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 记录配置信息
        gpt_sovits_module = ALGORITHM_PATHS['gptsovits_module']
        pretrained_path = os.path.join(gpt_sovits_module, "pretrained_models")
        
        OperationLogger.log_model_load_detail(
            model_name,
            "路径确认",
            model_path=pretrained_path,
            device=device,
            extra_info={"版本": version, "G2PW路径": os.environ.get("g2pw_model", "")}
        )
        
        # 计算预训练模型大小
        if os.path.exists(pretrained_path):
            total_size = 0
            file_count = 0
            for dirpath, dirnames, filenames in os.walk(pretrained_path):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    total_size += os.path.getsize(fp)
                    file_count += 1
            size_gb = total_size / (1024**3)
            OperationLogger.log_model_load_detail(
                model_name,
                "文件检查",
                model_path=pretrained_path,
                model_size=f"{size_gb:.2f}GB",
                extra_info={"文件数": file_count}
            )

        try:
            from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config

            # 加载配置文件
            config_path = os.path.join(gpt_sovits_module, "configs", "tts_infer.yaml")
            OperationLogger.log_model_load_detail(
                model_name,
                "配置加载",
                model_path=config_path
            )
            
            tts_config = TTS_Config(config_path)

            # 根据版本选择配置
            if version in tts_config.default_configs:
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

            OperationLogger.log_model_load_detail(
                model_name,
                "加载中",
                device=device,
                extra_info={
                    "版本": tts_config.version,
                    "半精度": tts_config.is_half,
                    "T2S权重": tts_config.configs.get("t2s_weights_path", "默认"),
                    "VITS权重": tts_config.configs.get("vits_weights_path", "默认")
                }
            )

            models[key] = {
                "config": tts_config,
                "pipeline": None,  # 延迟初始化
                "version": version
            }

            duration = time.time() - start_time
            gpu_mem = torch.cuda.memory_allocated() / 1024 ** 3 if torch.cuda.is_available() else 0
            
            OperationLogger.log_model_load_detail(
                model_name,
                "完成",
                device=device,
                memory_usage=gpu_mem,
                extra_info={"耗时": f"{duration:.3f}s"}
            )
            
            OperationLogger.log_model_load(model_name, "成功", duration, f"GPU内存: {gpu_mem:.2f}GB")
            OperationLogger.log_performance("GPT-SoVITS加载", duration, 0, gpu_mem)
        finally:
            # 恢复工作目录
            os.chdir(original_cwd)

    return models[key]


def init_gpt_sovits_pipeline(model_info, ref_audio_path: str = None):
    """初始化GPT-SoVITS推理管道"""
    # 设置路径并保存原工作目录
    original_cwd = _setup_gpt_sovits_path()
    
    pipeline_start = time.time()
    version = model_info.get("version", "unknown")

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
            OperationLogger.log_model_load_detail(
                f"GPT-SoVITS-{version}",
                "管道初始化",
                extra_info={"版本": current_version}
            )
            
            pipeline = TTS(model_info["config"])
            model_info["pipeline"] = pipeline
            model_info["pipeline_version"] = current_version
            
            pipeline_duration = time.time() - pipeline_start
            OperationLogger.log_model_load_detail(
                f"GPT-SoVITS-{version}",
                "管道完成",
                extra_info={"耗时": f"{pipeline_duration:.3f}s"}
            )
            
            system_logger.info(f"【GPT-SoVITS】管道初始化完成 | 版本: {current_version}")

        if ref_audio_path and os.path.exists(ref_audio_path):
            model_info["pipeline"].set_ref_audio(ref_audio_path)

        return model_info["pipeline"]
    finally:
        # 恢复工作目录
        os.chdir(original_cwd)
