#!/usr/bin/env python3
"""
统一TTS后端API服务
支持: ChatTTS, CosyVoice, F5-TTS, Qwen3-TTS, OpenVoice, GPT-SoVITS
"""

import os
import sys
import io
import json
import base64
import time
import tempfile
from typing import Optional, List, Dict
from contextlib import asynccontextmanager
from datetime import datetime

import torch
import torchaudio
import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Request, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn

# 导入自定义日志配置
from backend.logger_config import (
    OperationLogger, log_operation, log_api_call,
    system_logger, operation_logger, audit_logger, 
    performance_logger, logger
)

# 导入批量处理模块
from backend.batch_processor import batch_processor, BatchJob

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 记录项目启动
OperationLogger.log_init_start()

# 全局模型缓存
models = {}

# 添加项目路径
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'ChatTTS'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'CosyVoice'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'CosyVoice', 'third_party', 'Matcha-TTS'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'F5-TTS'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'OpenVoice'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'Qwen3-TTS'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'GPT-SoVITS'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'GPT-SoVITS', 'GPT_SoVITS'))

# ==================== 生命周期管理 ====================

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
        system_logger.info(f"【硬件信息】当前GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        system_logger.warning("【硬件信息】CUDA不可用,将使用CPU模式")

    # 创建输出目录
    directories = ["output", "uploads", "logs"]
    for dir_name in directories:
        dir_path = os.path.join(PROJECT_ROOT, dir_name)
        os.makedirs(dir_path, exist_ok=True)
        system_logger.info(f"【目录初始化】{dir_name}: {dir_path}")

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

# ==================== FastAPI应用 ====================

app = FastAPI(
    title="VersTTS API",
    description="统一文本转语音API服务",
    version="1.0.0",
    lifespan=lifespan
)

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== 数据模型 ====================

class BaseTTSRequest(BaseModel):
    text: str = Field(..., description="要合成的文本")
    language: str = Field(default="zh", description="语言: zh, en, auto")

class ChatTTSRequest(BaseTTSRequest):
    speaker_emb: Optional[str] = Field(default=None, description="说话人embedding (base64)")
    temperature: float = Field(default=0.3, description="采样温度")
    top_P: float = Field(default=0.7, description="Top P采样")
    top_K: float = Field(default=20, description="Top K采样")

class CosyVoiceRequest(BaseTTSRequest):
    mode: str = Field(default="sft", description="模式: sft, zero_shot, cross_lingual, instruct")
    speaker_id: str = Field(default="中文女", description="说话人ID")
    prompt_text: Optional[str] = Field(default=None, description="参考文本")
    instruct_text: Optional[str] = Field(default=None, description="指令文本")

class F5TTSRequest(BaseTTSRequest):
    ref_text: str = Field(..., description="参考文本")
    nfe_step: int = Field(default=32, description="NFE步数")
    cfg_strength: float = Field(default=2.0, description="CFG强度")
    speed: float = Field(default=1.0, description="语速")

class Qwen3TTSRequest(BaseTTSRequest):
    model_size: str = Field(default="1.7B", description="模型大小: 0.6B, 1.7B")
    mode: str = Field(default="base", description="模式: base, voice_clone")
    ref_audio: Optional[str] = Field(default=None, description="参考音频URL/base64")
    ref_text: Optional[str] = Field(default=None, description="参考文本")

class OpenVoiceRequest(BaseTTSRequest):
    style: str = Field(default="default", description="风格: default, whispering")
    speed: float = Field(default=1.0, description="语速")
    speaker: str = Field(default="default", description="说话人")

class GPTSoVITSRequest(BaseTTSRequest):
    text_lang: str = Field(default="zh", description="文本语言: zh, en, ja, ko, yue")
    ref_audio_path: Optional[str] = Field(default=None, description="参考音频路径")
    prompt_text: Optional[str] = Field(default=None, description="参考音频文本")
    prompt_lang: str = Field(default="zh", description="参考音频语言")
    top_k: int = Field(default=15, description="Top K采样")
    top_p: float = Field(default=1.0, description="Top P采样")
    temperature: float = Field(default=1.0, description="温度")
    text_split_method: str = Field(default="cut5", description="文本分割方法")
    batch_size: int = Field(default=1, description="批处理大小")
    speed_factor: float = Field(default=1.0, description="语速因子")
    version: str = Field(default="v2", description="版本: v1, v2, v3, v4, v2Pro, v2ProPlus")

class TTSResponse(BaseModel):
    success: bool
    message: str
    audio_url: Optional[str] = None
    audio_base64: Optional[str] = None
    sample_rate: Optional[int] = None

# ==================== 模型加载函数 ====================

def get_chattts_model():
    """获取或加载ChatTTS模型"""
    if "chattts" not in models:
        start_time = time.time()
        OperationLogger.log_model_load("ChatTTS", "开始加载")
        
        import ChatTTS
        chat = ChatTTS.Chat()
        model_path = os.path.join(PROJECT_ROOT, "ChatTTS", "models")
        system_logger.info(f"【模型加载】ChatTTS 从路径: {model_path}")
        
        if not chat.load(source="custom", custom_path=model_path):
            OperationLogger.log_model_load("ChatTTS", "失败", 0, "模型加载错误")
            raise HTTPException(status_code=500, detail="ChatTTS模型加载失败")
        
        models["chattts"] = chat
        duration = time.time() - start_time
        
        # 记录GPU内存使用
        gpu_mem = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
        OperationLogger.log_model_load("ChatTTS", "成功", duration, f"GPU内存: {gpu_mem:.2f}GB")
        OperationLogger.log_performance("ChatTTS加载", duration, 0, gpu_mem)
    
    return models["chattts"]

def get_cosyvoice_model(model_dir: str = "CosyVoice-300M-SFT"):
    """获取或加载CosyVoice模型"""
    key = f"cosyvoice_{model_dir}"
    if key not in models:
        start_time = time.time()
        OperationLogger.log_model_load(f"CosyVoice-{model_dir}", "开始加载")
        
        from cosyvoice.cli.cosyvoice import AutoModel
        model_path = os.path.join(PROJECT_ROOT, "CosyVoice", "models", "iic", model_dir)
        if not os.path.exists(model_path):
            model_path = model_path.replace("-", "___")
        
        system_logger.info(f"【模型加载】CosyVoice 从路径: {model_path}")
        models[key] = AutoModel(model_dir=model_path)
        
        duration = time.time() - start_time
        gpu_mem = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
        OperationLogger.log_model_load(f"CosyVoice-{model_dir}", "成功", duration, f"GPU内存: {gpu_mem:.2f}GB")
        OperationLogger.log_performance("CosyVoice加载", duration, 0, gpu_mem)
    
    return models[key]

def get_f5tts_model():
    """获取或加载F5-TTS模型"""
    if "f5tts" not in models:
        start_time = time.time()
        OperationLogger.log_model_load("F5-TTS", "开始加载")
        
        from f5_tts.api import F5TTS
        ckpt_path = os.path.join(PROJECT_ROOT, "F5-TTS", "models", "model_1200000.pt")
        system_logger.info(f"【模型加载】F5-TTS 检查点: {ckpt_path}")
        
        models["f5tts"] = F5TTS(model="F5TTS_Base", ckpt_file=ckpt_path)
        
        duration = time.time() - start_time
        gpu_mem = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
        OperationLogger.log_model_load("F5-TTS", "成功", duration, f"GPU内存: {gpu_mem:.2f}GB")
        OperationLogger.log_performance("F5-TTS加载", duration, 0, gpu_mem)
    
    return models["f5tts"]

def get_qwen3tts_model(model_size: str = "1.7B"):
    """获取或加载Qwen3-TTS模型"""
    key = f"qwen3tts_{model_size}"
    if key not in models:
        start_time = time.time()
        OperationLogger.log_model_load(f"Qwen3-TTS-{model_size}", "开始加载")
        
        from qwen_tts import Qwen3TTSModel
        size_map = {
            "0.6B": "0___6B",
            "1.7B": "1___7B"
        }
        size_str = size_map.get(model_size, model_size.replace('.', '___'))
        model_path = os.path.join(PROJECT_ROOT, "Qwen3-TTS", "models", "Qwen", f"Qwen3-TTS-12Hz-{size_str}-Base")
        
        system_logger.info(f"【模型加载】Qwen3-TTS 路径: {model_path}")
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        attn_impl = "flash_attention_2" if torch.cuda.is_available() else "eager"
        
        models[key] = Qwen3TTSModel.from_pretrained(
            model_path,
            device_map=device,
            dtype=dtype,
            attn_implementation=attn_impl
        )
        
        duration = time.time() - start_time
        gpu_mem = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
        OperationLogger.log_model_load(f"Qwen3-TTS-{model_size}", "成功", duration, f"GPU内存: {gpu_mem:.2f}GB")
        OperationLogger.log_performance("Qwen3-TTS加载", duration, 0, gpu_mem)
    
    return models[key]

def get_openvoice_models():
    """获取或加载OpenVoice模型"""
    if "openvoice" not in models:
        start_time = time.time()
        OperationLogger.log_model_load("OpenVoice", "开始加载")
        
        from openvoice.api import BaseSpeakerTTS, ToneColorConverter
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        # V1版本路径
        ckpt_base_en = os.path.join(PROJECT_ROOT, "OpenVoice", "checkpoints_v1", "checkpoints", "base_speakers", "EN")
        ckpt_base_zh = os.path.join(PROJECT_ROOT, "OpenVoice", "checkpoints_v1", "checkpoints", "base_speakers", "ZH")
        ckpt_converter = os.path.join(PROJECT_ROOT, "OpenVoice", "checkpoints_v1", "checkpoints", "converter")
        
        # 如果V1不存在,尝试V2
        if not os.path.exists(ckpt_base_en):
            ckpt_base_en = os.path.join(PROJECT_ROOT, "OpenVoice", "checkpoints_v2", "checkpoints_v2", "base_speakers")
            ckpt_base_zh = os.path.join(PROJECT_ROOT, "OpenVoice", "checkpoints_v2", "checkpoints_v2", "base_speakers")
            ckpt_converter = os.path.join(PROJECT_ROOT, "OpenVoice", "checkpoints_v2", "checkpoints_v2", "converter")
            system_logger.info(f"【模型加载】OpenVoice 使用V2版本")
        else:
            system_logger.info(f"【模型加载】OpenVoice 使用V1版本")
        
        base_speaker_tts = BaseSpeakerTTS(f'{ckpt_base_en}/config.json', device=device)
        base_speaker_tts.load_ckpt(f'{ckpt_base_en}/checkpoint.pth')
        
        tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
        tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')
        
        # 加载音色嵌入
        source_se = {}
        if os.path.exists(f'{ckpt_base_en}/en_default_se.pth'):
            source_se['en'] = torch.load(f'{ckpt_base_en}/en_default_se.pth').to(device)
            source_se['zh'] = torch.load(f'{ckpt_base_zh}/zh_default_se.pth').to(device)
        elif os.path.exists(f'{ckpt_base_en}/ses/en-default.pth'):
            source_se['en'] = torch.load(f'{ckpt_base_en}/ses/en-default.pth').to(device)
            source_se['zh'] = torch.load(f'{ckpt_base_zh}/ses/zh.pth').to(device)
        
        models["openvoice"] = {
            "tts": base_speaker_tts,
            "converter": tone_color_converter,
            "source_se": source_se,
            "device": device,
            "ckpt_base_en": ckpt_base_en,
            "ckpt_base_zh": ckpt_base_zh,
        }
        
        duration = time.time() - start_time
        gpu_mem = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
        OperationLogger.log_model_load("OpenVoice", "成功", duration, f"GPU内存: {gpu_mem:.2f}GB")
        OperationLogger.log_performance("OpenVoice加载", duration, 0, gpu_mem)
    
    return models["openvoice"]

def _setup_gpt_sovits_path():
    """设置GPT-SoVITS所需的系统路径"""
    gpt_sovits_root = os.path.join(PROJECT_ROOT, "GPT-SoVITS")
    gpt_sovits_module = os.path.join(gpt_sovits_root, "GPT_SoVITS")
    
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
        OperationLogger.log_model_load(f"GPT-SoVITS-{version}", "开始加载")
        
        # 设置路径并保存原工作目录
        original_cwd = _setup_gpt_sovits_path()
        
        try:
            from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
            
            # 加载配置文件
            config_path = os.path.join(PROJECT_ROOT, "GPT-SoVITS", "GPT_SoVITS", "configs", "tts_infer.yaml")
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
            gpu_mem = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
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
        
        if model_info["pipeline"] is None:
            pipeline = TTS(model_info["config"])
            model_info["pipeline"] = pipeline
            
        if ref_audio_path and os.path.exists(ref_audio_path):
            model_info["pipeline"].set_ref_audio(ref_audio_path)
            
        return model_info["pipeline"]
    finally:
        # 恢复工作目录
        os.chdir(original_cwd)

# ==================== 工具函数 ====================

def save_temp_audio(audio_data: np.ndarray, sample_rate: int, suffix: str = ".wav") -> str:
    """保存临时音频文件"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    temp_path = f"output/tts_{timestamp}{suffix}"
    sf.write(temp_path, audio_data, sample_rate)
    return temp_path

def audio_to_base64(audio_path: str) -> str:
    """将音频文件转为base64"""
    with open(audio_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# ==================== API路由 ====================

@app.get("/")
async def root(request: Request):
    """根路径"""
    client_ip = request.client.host if request.client else "unknown"
    OperationLogger.log_api_request("/", "GET", {}, client_ip, 0)
    
    return {
        "name": "VersTTS API",
        "version": "1.2.0",
        "endpoints": [
            "/tts/chattts",
            "/tts/cosyvoice",
            "/tts/f5tts",
            "/tts/qwen3tts",
            "/tts/openvoice",
            "/tts/gptsovits",
            "/tts/batch/create",
            "/tts/batch/{job_id}/status",
            "/tts/batch/{job_id}/process",
            "/tts/batch/{job_id}/download",
        ]
    }

@app.get("/health")
async def health(request: Request):
    """健康检查"""
    client_ip = request.client.host if request.client else "unknown"
    
    # 获取系统状态
    import psutil
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()
    
    gpu_info = ""
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.memory_allocated() / 1024**3
        gpu_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        gpu_info = f"{gpu_mem:.2f}GB / {gpu_total:.2f}GB"
    
    OperationLogger.log_api_request("/health", "GET", {}, client_ip, 0)
    OperationLogger.log_system_status(cpu_percent, memory.percent, gpu_info)
    
    return {
        "status": "healthy",
        "cuda_available": torch.cuda.is_available(),
        "models_loaded": list(models.keys()),
        "system": {
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "gpu_memory": gpu_info
        }
    }

# ==================== ChatTTS API ====================

@app.post("/tts/chattts")
async def tts_chattts(
    request: Request,
    text: str = Form(...),
    speaker_emb: Optional[str] = Form(None),
    temperature: float = Form(0.3),
    top_P: float = Form(0.7),
    top_K: float = Form(20),
    output_format: str = Form("url")
):
    """ChatTTS语音合成"""
    start_time = time.time()
    client_ip = request.client.host if request.client else "unknown"
    
    try:
        # 记录API请求
        OperationLogger.log_api_request("/tts/chattts", "POST", {
            "text_preview": text[:50],
            "temperature": temperature,
            "top_P": top_P,
            "top_K": top_K
        }, client_ip)
        
        system_logger.info(f"【ChatTTS】开始合成 | 文本: {text[:50]}... | 客户端: {client_ip}")
        chat = get_chattts_model()

        # 准备说话人embedding
        if speaker_emb:
            spk = speaker_emb
            system_logger.info(f"【ChatTTS】使用自定义说话人embedding")
        else:
            spk = chat.sample_random_speaker()
            system_logger.info(f"【ChatTTS】使用随机说话人")

        # 推理
        params = chat.InferCodeParams(
            spk_emb=spk,
            temperature=float(temperature),
            top_P=float(top_P),
            top_K=int(top_K),
        )
        
        infer_start = time.time()
        wavs = chat.infer(
            [text],
            stream=False,
            params_infer_code=params,
        )
        infer_duration = time.time() - infer_start

        # 保存音频
        audio_path = save_temp_audio(wavs[0], 24000)
        audio_size = os.path.getsize(audio_path)
        
        # 记录文件操作
        OperationLogger.log_file_operation("保存音频", audio_path, audio_size, "成功")

        total_duration = time.time() - start_time
        
        # 记录TTS请求完成
        OperationLogger.log_tts_request("ChatTTS", text, {
            "temperature": temperature,
            "top_P": top_P,
            "top_K": top_K,
            "output_format": output_format
        }, total_duration, "成功")
        
        system_logger.info(f"【ChatTTS】合成完成 | 耗时: {total_duration:.3f}s | 音频: {audio_path}")

        if output_format == "base64":
            audio_b64 = audio_to_base64(audio_path)
            return TTSResponse(
                success=True,
                message="合成成功",
                audio_base64=audio_b64,
                sample_rate=24000
            )
        else:
            return TTSResponse(
                success=True,
                message="合成成功",
                audio_url=f"/audio/{os.path.basename(audio_path)}",
                sample_rate=24000
            )
    except Exception as e:
        total_duration = time.time() - start_time
        OperationLogger.log_error("ChatTTS合成错误", str(e))
        OperationLogger.log_tts_request("ChatTTS", text, {}, total_duration, f"失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tts/chattts/speakers")
async def chattts_speakers():
    """获取随机说话人"""
    try:
        chat = get_chattts_model()
        speaker = chat.sample_random_speaker()
        return {"speaker_emb": speaker}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================== CosyVoice API ====================

@app.post("/tts/cosyvoice")
async def tts_cosyvoice(
    text: str = Form(...),
    mode: str = Form("sft"),
    speaker_id: str = Form("中文女"),
    prompt_text: Optional[str] = Form(None),
    instruct_text: Optional[str] = Form(None),
    prompt_wav: Optional[UploadFile] = File(None),
    output_format: str = Form("url")
):
    """CosyVoice语音合成"""
    try:
        logger.info(f"CosyVoice请求: {text[:50]}... 模式: {mode}")

        # 根据模式选择模型
        model_map = {
            "sft": "CosyVoice-300M-SFT",
            "zero_shot": "CosyVoice-300M",
            "cross_lingual": "CosyVoice-300M",
            "instruct": "CosyVoice-300M-Instruct",
        }
        model_dir = model_map.get(mode, "CosyVoice-300M-SFT")
        cosyvoice = get_cosyvoice_model(model_dir)

        # 根据模式推理
        if mode == "sft":
            model_output = cosyvoice.inference_sft(text, speaker_id)
        elif mode == "zero_shot":
            if not prompt_wav:
                raise HTTPException(status_code=400, detail="zero_shot模式需要提供参考音频")
            from cosyvoice.utils.file_utils import load_wav
            prompt_speech = load_wav(prompt_wav.file, 16000)
            model_output = cosyvoice.inference_zero_shot(text, prompt_text or "", prompt_speech)
        elif mode == "cross_lingual":
            if not prompt_wav:
                raise HTTPException(status_code=400, detail="cross_lingual模式需要提供参考音频")
            from cosyvoice.utils.file_utils import load_wav
            prompt_speech = load_wav(prompt_wav.file, 16000)
            model_output = cosyvoice.inference_cross_lingual(text, prompt_speech)
        elif mode == "instruct":
            if not instruct_text:
                raise HTTPException(status_code=400, detail="instruct模式需要提供指令文本")
            model_output = cosyvoice.inference_instruct(text, speaker_id, instruct_text)
        else:
            raise HTTPException(status_code=400, detail=f"不支持的模式: {mode}")

        # 获取音频 - CosyVoice返回的是tensor，需要正确处理
        for result in model_output:
            audio_data = result['tts_speech']
            # 如果是tensor，转换为numpy并确保形状正确
            if hasattr(audio_data, 'numpy'):
                audio_np = audio_data.numpy()
            else:
                audio_np = audio_data

            # 确保是1D或2D数组
            if audio_np.ndim > 2:
                audio_np = audio_np.squeeze()
            if audio_np.ndim == 1:
                audio_np = audio_np.reshape(1, -1)

            # 使用torchaudio保存
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            audio_path = f"output/tts_{timestamp}.wav"
            torchaudio.save(audio_path, torch.from_numpy(audio_np), cosyvoice.sample_rate)
            break

        if output_format == "base64":
            audio_b64 = audio_to_base64(audio_path)
            return TTSResponse(
                success=True,
                message="合成成功",
                audio_base64=audio_b64,
                sample_rate=cosyvoice.sample_rate
            )
        else:
            return TTSResponse(
                success=True,
                message="合成成功",
                audio_url=f"/audio/{os.path.basename(audio_path)}",
                sample_rate=cosyvoice.sample_rate
            )
    except Exception as e:
        logger.error(f"CosyVoice错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tts/cosyvoice/speakers")
async def cosyvoice_speakers():
    """获取可用的说话人列表"""
    try:
        cosyvoice = get_cosyvoice_model("CosyVoice-300M-SFT")
        speakers = cosyvoice.list_available_spks()
        return {"speakers": speakers}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================== F5-TTS API ====================

# 默认参考音频路径
DEFAULT_F5TTS_REF_ZH = os.path.join(PROJECT_ROOT, "F5-TTS", "src", "f5_tts", "infer", "examples", "basic", "basic_ref_zh.wav")
DEFAULT_F5TTS_REF_EN = os.path.join(PROJECT_ROOT, "F5-TTS", "src", "f5_tts", "infer", "examples", "basic", "basic_ref_en.wav")
DEFAULT_F5TTS_TEXT_ZH = "在一无所知中，梦里的一天结束了，一个新的轮回便会开始。"
DEFAULT_F5TTS_TEXT_EN = "Some call me nature, others call me mother nature."

@app.post("/tts/f5tts")
async def tts_f5tts(
    text: str = Form(...),
    ref_text: Optional[str] = Form(None),
    ref_wav: Optional[UploadFile] = File(None),
    nfe_step: int = Form(32),
    cfg_strength: float = Form(2.0),
    speed: float = Form(1.0),
    output_format: str = Form("url")
):
    """F5-TTS语音合成"""
    try:
        logger.info(f"F5-TTS请求: {text[:50]}...")

        # 判断是否使用默认参考音频
        if ref_wav:
            # 使用上传的参考音频
            ref_path = f"uploads/ref_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.wav"
            with open(ref_path, "wb") as f:
                f.write(await ref_wav.read())
            use_ref_text = ref_text or "参考音频文本"
            is_temp = True
        else:
            # 使用默认参考音频（根据语言选择）
            has_chinese = any('\u4e00' <= char <= '\u9fff' for char in text)
            if has_chinese and os.path.exists(DEFAULT_F5TTS_REF_ZH):
                ref_path = DEFAULT_F5TTS_REF_ZH
                use_ref_text = ref_text or DEFAULT_F5TTS_TEXT_ZH
                logger.info(f"使用默认中文参考音频: {ref_path}")
            else:
                ref_path = DEFAULT_F5TTS_REF_EN
                use_ref_text = ref_text or DEFAULT_F5TTS_TEXT_EN
                logger.info(f"使用默认英文参考音频: {ref_path}")
            is_temp = False

        # 加载模型并推理
        f5tts = get_f5tts_model()
        wav, sr, _ = f5tts.infer(
            ref_file=ref_path,
            ref_text=use_ref_text,
            gen_text=text,
            nfe_step=nfe_step,
            cfg_strength=cfg_strength,
            speed=speed,
        )

        # 保存音频
        audio_path = save_temp_audio(wav, sr)

        # 清理临时参考音频
        if is_temp and os.path.exists(ref_path):
            os.remove(ref_path)

        if output_format == "base64":
            audio_b64 = audio_to_base64(audio_path)
            return TTSResponse(
                success=True,
                message="合成成功",
                audio_base64=audio_b64,
                sample_rate=sr
            )
        else:
            return TTSResponse(
                success=True,
                message="合成成功",
                audio_url=f"/audio/{os.path.basename(audio_path)}",
                sample_rate=sr
            )
    except Exception as e:
        logger.error(f"F5-TTS错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== Qwen3-TTS API ====================

@app.post("/tts/qwen3tts")
async def tts_qwen3tts(
    text: str = Form(...),
    model_size: str = Form("1.7B"),
    mode: str = Form("base"),
    ref_wav: Optional[UploadFile] = File(None),
    ref_text: Optional[str] = Form(None),
    output_format: str = Form("url")
):
    """Qwen3-TTS语音合成"""
    try:
        logger.info(f"Qwen3-TTS请求: {text[:50]}... 模型: {model_size}")

        tts = get_qwen3tts_model(model_size)

        if mode == "voice_clone" and ref_wav:
            # 保存参考音频
            ref_path = f"uploads/ref_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.wav"
            with open(ref_path, "wb") as f:
                f.write(await ref_wav.read())

            wavs, sr = tts.generate_voice_clone(
                text=text,
                language="Auto",
                ref_audio=ref_path,
                ref_text=ref_text or "",
                x_vector_only_mode=False
            )
            os.remove(ref_path)
            wav = wavs[0] if isinstance(wavs, list) else wavs
        else:
            # 基础模式：使用默认参考音频进行voice_clone
            # 使用官方示例中的默认参考音频URL
            default_ref_audio = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone_1.wav"
            default_ref_text = "甚至出现交易几乎停滞的情况。"
            
            logger.info(f"基础模式：使用默认参考音频")
            wavs, sr = tts.generate_voice_clone(
                text=text,
                language="Auto",
                ref_audio=default_ref_audio,
                ref_text=default_ref_text,
                x_vector_only_mode=True  # 只使用音色特征
            )
            wav = wavs[0] if isinstance(wavs, list) else wavs

        # 保存音频
        audio_path = save_temp_audio(wav, sr)

        if output_format == "base64":
            audio_b64 = audio_to_base64(audio_path)
            return TTSResponse(
                success=True,
                message="合成成功",
                audio_base64=audio_b64,
                sample_rate=sr
            )
        else:
            return TTSResponse(
                success=True,
                message="合成成功",
                audio_url=f"/audio/{os.path.basename(audio_path)}",
                sample_rate=sr
            )
    except Exception as e:
        logger.error(f"Qwen3-TTS错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== OpenVoice API ====================

@app.post("/tts/openvoice")
async def tts_openvoice(
    text: str = Form(...),
    language: str = Form("zh"),
    style: str = Form("default"),
    speed: float = Form(1.0),
    ref_wav: Optional[UploadFile] = File(None),
    output_format: str = Form("url")
):
    """OpenVoice语音合成"""
    try:
        logger.info(f"OpenVoice请求: {text[:50]}...")

        ov = get_openvoice_models()
        tts = ov["tts"]
        converter = ov["converter"]
        source_se = ov["source_se"]
        device = ov["device"]

        lang_map = {"zh": "Chinese", "en": "English"}
        language_full = lang_map.get(language, "Chinese")

        # 提取目标音色
        target_se = source_se.get("zh" if language == "zh" else "en")
        if ref_wav:
            from openvoice import se_extractor
            ref_path = f"uploads/ref_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.wav"
            with open(ref_path, "wb") as f:
                f.write(await ref_wav.read())
            target_se, _ = se_extractor.get_se(ref_path, converter, target_dir='processed', vad=True)
            os.remove(ref_path)

        # 生成基础音频
        temp_path = f"output/temp_openvoice_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.wav"
        tts.tts(text, temp_path, speaker=style, language=language_full, speed=speed)

        # 转换音色
        audio_path = f"output/openvoice_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.wav"
        src_se = source_se.get("zh" if language == "zh" else "en")
        converter.convert(
            audio_src_path=temp_path,
            src_se=src_se,
            tgt_se=target_se,
            output_path=audio_path,
            message="@VersTTS"
        )

        # 清理临时文件
        os.remove(temp_path)

        # 获取采样率
        from openvoice import utils
        hps = utils.get_hparams_from_file(f'{ov["ckpt_base_en"]}/config.json')
        sample_rate = hps.data.sampling_rate

        if output_format == "base64":
            audio_b64 = audio_to_base64(audio_path)
            return TTSResponse(
                success=True,
                message="合成成功",
                audio_base64=audio_b64,
                sample_rate=sample_rate
            )
        else:
            return TTSResponse(
                success=True,
                message="合成成功",
                audio_url=f"/audio/{os.path.basename(audio_path)}",
                sample_rate=sample_rate
            )
    except Exception as e:
        logger.error(f"OpenVoice错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== GPT-SoVITS API ====================

@app.post("/tts/gptsovits")
async def tts_gptsovits(
    request: Request,
    text: str = Form(...),
    text_lang: str = Form("zh"),
    prompt_wav: UploadFile = File(...),
    prompt_text: str = Form(...),
    prompt_lang: str = Form("zh"),
    top_k: int = Form(15),
    top_p: float = Form(1.0),
    temperature: float = Form(1.0),
    text_split_method: str = Form("cut5"),
    batch_size: int = Form(1),
    speed_factor: float = Form(1.0),
    version: str = Form("v2"),
    output_format: str = Form("url")
):
    """GPT-SoVITS语音合成 - 必须提供参考音频"""
    start_time = time.time()
    client_ip = request.client.host if request.client else "unknown"
    
    try:
        # 记录API请求
        OperationLogger.log_api_request("/tts/gptsovits", "POST", {
            "text_preview": text[:50],
            "text_lang": text_lang,
            "prompt_lang": prompt_lang,
            "version": version
        }, client_ip)
        
        system_logger.info(f"【GPT-SoVITS】开始合成 | 文本: {text[:50]}... | 版本: {version} | 客户端: {client_ip}")
        
        # 保存参考音频
        ref_path = f"uploads/gptsovits_ref_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.wav"
        with open(ref_path, "wb") as f:
            f.write(await prompt_wav.read())
        system_logger.info(f"【GPT-SoVITS】参考音频已保存: {ref_path}")
        
        # 获取模型
        model_info = get_gpt_sovits_model(version)
        
        # 初始化管道
        infer_start = time.time()
        pipeline = init_gpt_sovits_pipeline(model_info, ref_path)
        
        # 构建请求参数
        req = {
            "text": text,
            "text_lang": text_lang.lower(),
            "ref_audio_path": ref_path,
            "prompt_text": prompt_text,
            "prompt_lang": prompt_lang.lower(),
            "top_k": top_k,
            "top_p": top_p,
            "temperature": temperature,
            "text_split_method": text_split_method,
            "batch_size": batch_size,
            "speed_factor": speed_factor,
            "media_type": "wav",
            "streaming_mode": False,
            "parallel_infer": True,
        }
        
        # 执行推理
        tts_generator = pipeline.run(req)
        sr, audio_data = next(tts_generator)
        infer_duration = time.time() - infer_start
        
        # 保存音频
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        audio_path = f"output/gptsovits_{timestamp}.wav"
        sf.write(audio_path, audio_data, sr)
        
        # 清理临时参考音频
        if os.path.exists(ref_path):
            os.remove(ref_path)
            
        audio_size = os.path.getsize(audio_path)
        OperationLogger.log_file_operation("保存音频", audio_path, audio_size, "成功")
        
        total_duration = time.time() - start_time
        
        # 记录TTS请求完成
        OperationLogger.log_tts_request("GPT-SoVITS", text, {
            "version": version,
            "text_lang": text_lang,
            "prompt_lang": prompt_lang,
            "output_format": output_format
        }, total_duration, "成功")
        
        system_logger.info(f"【GPT-SoVITS】合成完成 | 耗时: {total_duration:.3f}s | 音频: {audio_path}")
        
        if output_format == "base64":
            audio_b64 = audio_to_base64(audio_path)
            return TTSResponse(
                success=True,
                message="合成成功",
                audio_base64=audio_b64,
                sample_rate=sr
            )
        else:
            return TTSResponse(
                success=True,
                message="合成成功",
                audio_url=f"/audio/{os.path.basename(audio_path)}",
                sample_rate=sr
            )
    except Exception as e:
        total_duration = time.time() - start_time
        OperationLogger.log_error("GPT-SoVITS合成错误", str(e))
        OperationLogger.log_tts_request("GPT-SoVITS", text, {}, total_duration, f"失败: {str(e)}")
        # 清理临时文件
        if 'ref_path' in locals() and os.path.exists(ref_path):
            os.remove(ref_path)
        raise HTTPException(status_code=500, detail=str(e))

# ==================== 批量处理 API ====================

class BatchTTSRequest(BaseModel):
    model: str = Field(..., description="TTS模型: chattts, cosyvoice, f5tts, qwen3tts, openvoice, gptsovits")
    options: Optional[Dict] = Field(default={}, description="模型特定选项")

@app.post("/tts/batch/create")
async def batch_create(
    model: str = Form(...),
    text_file: UploadFile = File(...),
    ref_wav: Optional[UploadFile] = File(None)
):
    """创建批量TTS任务
    
    支持上传文件格式:
    - .txt: 每行一个文本
    - .csv: text,speaker_id(可选) 列
    - .json: [{"text": "...", "speaker_id": "..."}, ...]
    """
    try:
        # 读取并解析文件
        content = await text_file.read()
        tasks_data = batch_processor.parse_text_file(content, text_file.filename)
        
        if not tasks_data:
            raise HTTPException(status_code=400, detail="文件内容为空")
        
        # 保存参考音频(如果需要)
        ref_path = None
        if ref_wav:
            ref_path = f"uploads/batch_ref_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.wav"
            with open(ref_path, "wb") as f:
                f.write(await ref_wav.read())
        
        # 创建批量任务
        job = batch_processor.create_job(model, tasks_data)
        
        system_logger.info(f"【批量任务】创建成功 | ID: {job.job_id} | 模型: {model} | 任务数: {len(tasks_data)}")
        
        return {
            "success": True,
            "job_id": job.job_id,
            "model": model,
            "total": len(tasks_data),
            "message": "批量任务创建成功"
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        system_logger.error(f"【批量任务】创建失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tts/batch/{job_id}/status")
async def batch_status(job_id: str):
    """查询批量任务状态"""
    job = batch_processor.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    return {
        "job_id": job.job_id,
        "model": job.model,
        "status": job.status,
        "total": job.total,
        "completed": job.completed,
        "failed": job.failed,
        "progress": f"{((job.completed + job.failed) / job.total * 100):.1f}%" if job.total > 0 else "0%"
    }

@app.post("/tts/batch/{job_id}/process")
async def batch_process(
    job_id: str,
    background_tasks: BackgroundTasks,
    options: Optional[str] = Form("{}")
):
    """开始处理批量任务"""
    import json
    from fastapi import BackgroundTasks
    
    job = batch_processor.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    if job.status == "processing":
        raise HTTPException(status_code=400, detail="任务正在处理中")
    
    if job.status == "completed":
        raise HTTPException(status_code=400, detail="任务已完成")
    
    try:
        opts = json.loads(options)
    except:
        opts = {}
    
    # 启动后台处理
    background_tasks.add_task(process_batch_job, job_id, opts)
    
    job.status = "processing"
    
    return {
        "success": True,
        "message": "批量任务开始处理",
        "job_id": job_id
    }

async def process_batch_job(job_id: str, options: dict):
    """后台处理批量任务(简化版框架)"""
    job = batch_processor.get_job(job_id)
    if not job:
        return
    
    model_name = job.model
    system_logger.info(f"【批量任务】开始处理 | ID: {job_id} | 模型: {model_name}")
    
    # TODO: 实现完整的批量处理逻辑
    # 这里需要根据不同的模型调用相应的合成函数
    # 为简化代码，当前版本仅提供API框架
    
    job.status = "completed"
    system_logger.info(f"【批量任务】框架已创建 | ID: {job_id}")

@app.get("/tts/batch/{job_id}/download")
async def batch_download(job_id: str):
    """下载批量任务结果ZIP包"""
    job = batch_processor.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    if job.status != "completed":
        raise HTTPException(status_code=400, detail="任务尚未完成")
    
    try:
        zip_path = batch_processor.create_zip_package(job_id)
        
        if not os.path.exists(zip_path):
            raise HTTPException(status_code=500, detail="ZIP包生成失败")
        
        return FileResponse(
            zip_path,
            media_type="application/zip",
            filename=f"batch_tts_{job_id}.zip"
        )
    except Exception as e:
        system_logger.error(f"【批量任务】下载失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tts/batch/{job_id}/results")
async def batch_results(job_id: str):
    """获取批量任务详细结果"""
    job = batch_processor.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    return {
        "job_id": job.job_id,
        "model": job.model,
        "status": job.status,
        "total": job.total,
        "completed": job.completed,
        "failed": job.failed,
        "tasks": [
            {
                "id": task.id,
                "text": task.text[:100] + "..." if len(task.text) > 100 else task.text,
                "status": task.status,
                "audio_url": f"/audio/{os.path.basename(task.audio_path)}" if task.audio_path else None,
                "error": task.error
            }
            for task in job.tasks
        ]
    }

# ==================== 参考人声 API ====================

# 加载参考人声元数据
REFERENCE_AUDIO_DIR = os.path.join(PROJECT_ROOT, "reference_audio")
REFERENCE_METADATA_PATH = os.path.join(REFERENCE_AUDIO_DIR, "metadata.json")

def load_reference_voices() -> List[Dict]:
    """加载参考人声列表"""
    try:
        if os.path.exists(REFERENCE_METADATA_PATH):
            with open(REFERENCE_METADATA_PATH, "r", encoding="utf-8") as f:
                metadata = json.load(f)
                return metadata.get("samples", [])
    except Exception as e:
        system_logger.error(f"加载参考人声元数据失败: {e}")
    return []

def get_reference_voice_by_id(voice_id: str) -> Optional[Dict]:
    """根据ID获取参考人声"""
    voices = load_reference_voices()
    for voice in voices:
        if voice.get("id") == voice_id:
            return voice
    return None

@app.get("/reference_voices")
async def list_reference_voices(
    category: Optional[str] = None,
    gender: Optional[str] = None,
    model: Optional[str] = None
):
    """
    获取参考人声列表
    
    参数:
    - category: 分类过滤 (children/teenagers/adults)
    - gender: 性别过滤 (male/female)
    - model: 模型兼容性过滤 (chattts/cosyvoice/f5tts/qwen3tts/openvoice/gptsovits)
    """
    try:
        voices = load_reference_voices()
        
        # 应用过滤
        filtered_voices = voices
        if category:
            filtered_voices = [v for v in filtered_voices if v.get("category") == category]
        if gender:
            filtered_voices = [v for v in filtered_voices if v.get("gender") == gender]
        if model:
            filtered_voices = [
                v for v in filtered_voices 
                if v.get("compatible_models", {}).get(model, False)
            ]
        
        # 添加音频URL
        for voice in filtered_voices:
            voice["audio_url"] = f"/reference_audio/{voice['filename']}"
        
        return {
            "success": True,
            "count": len(filtered_voices),
            "voices": filtered_voices
        }
    except Exception as e:
        system_logger.error(f"获取参考人声列表失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/reference_voices/categories")
async def list_reference_categories():
    """获取参考人声分类列表"""
    try:
        if os.path.exists(REFERENCE_METADATA_PATH):
            with open(REFERENCE_METADATA_PATH, "r", encoding="utf-8") as f:
                metadata = json.load(f)
                categories = metadata.get("categories", {})
                return {
                    "success": True,
                    "categories": [
                        {
                            "id": cat_id,
                            "name": cat_data.get("description", cat_id),
                            "count": cat_data.get("count", 0)
                        }
                        for cat_id, cat_data in categories.items()
                    ]
                }
    except Exception as e:
        system_logger.error(f"获取参考人声分类失败: {e}")
    
    return {
        "success": True,
        "categories": [
            {"id": "children", "name": "儿童声音(3-12岁)", "count": 0},
            {"id": "teenagers", "name": "中学生声音(13-18岁)", "count": 0},
            {"id": "adults", "name": "成人声音(18岁以上)", "count": 0}
        ]
    }

@app.get("/reference_audio/{category}/{filename}")
async def get_reference_audio(category: str, filename: str):
    """获取参考人声音频文件"""
    file_path = os.path.join(REFERENCE_AUDIO_DIR, category, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="音频文件不存在")
    return FileResponse(file_path, media_type="audio/wav")

# ==================== 静态文件服务 ====================

@app.get("/audio/{filename}")
async def get_audio(filename: str):
    """获取音频文件"""
    file_path = f"output/{filename}"
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="文件不存在")
    return FileResponse(file_path, media_type="audio/wav")

# 挂载前端静态文件 - 使用frontend目录作为静态文件根目录
# 并添加一个专门的前端路由
@app.get("/app", response_class=FileResponse)
async def serve_frontend():
    """提供前端界面"""
    frontend_path = os.path.join(PROJECT_ROOT, "frontend", "index.html")
    if os.path.exists(frontend_path):
        return FileResponse(frontend_path)
    raise HTTPException(status_code=404, detail="前端文件不存在")

# 如果frontend/dist存在则挂载
frontend_dist = os.path.join(PROJECT_ROOT, "frontend", "dist")
if os.path.exists(frontend_dist):
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")

# ==================== 主函数 ====================

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
