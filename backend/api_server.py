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
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Request, BackgroundTasks, Query
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn

# 文本预处理函数 - 用于 ChatTTS
def preprocess_text_for_chattts(text: str) -> str:
    """
    预处理文本以适应 ChatTTS 模型
    - 转换全角字符为半角
    - 移除用户手动添加的控制标签（避免与系统自动添加的冲突）
    - 移除或替换不支持的字符
    """
    import re
    
    # 第一步：移除用户手动添加的 ChatTTS 控制标签
    # 这些标签会通过 API 参数自动添加，手动添加会导致冲突和 CUDA 错误
    control_tags = [
        r'\[oral_\d+\]',      # [oral_0] 到 [oral_9]
        r'\[laugh_\d+\]',     # [laugh_0] 到 [laugh_2]
        r'\[break_\d+\]',     # [break_0] 到 [break_7]
        r'\[speed_\d+\]',     # [speed_0] 到 [speed_9]
        r'\[laugh\]',         # [laugh]
        r'\[uv_break\]',      # [uv_break]
        r'\[v_break\]',       # [v_break]
        r'\[break\]',         # [break]
    ]
    
    for tag_pattern in control_tags:
        text = re.sub(tag_pattern, '', text)
    
    # 全角到半角映射
    fullwidth_to_halfwidth = {
        '（': '(', '）': ')', '【': '[', '】': ']',
        '｛': '{', '｝': '}', '「': '"', '」': '"',
        '『': '"', '』': '"', '［': '[', '］': ']',
        '。': '.', '，': ',', '！': '!', '？': '?',
        '；': ';', '：': ':', '"': '"', '"': '"',
        ''': "'", ''': "'", '、': ',', '·': '.',
        '～': '~', '＠': '@', '＃': '#', '＄': '$',
        '％': '%', '＆': '&', '＊': '*', '＋': '+',
        '－': '-', '／': '/', '＜': '<', '＝': '=',
        '＞': '>', '？': '?', '＾': '^', '＿': '_',
        '｀': '`', '｜': '|', '～': '~',
        '０': '0', '１': '1', '２': '2', '３': '3', '４': '4',
        '５': '5', '６': '6', '７': '7', '８': '8', '９': '9',
        'Ａ': 'A', 'Ｂ': 'B', 'Ｃ': 'C', 'Ｄ': 'D', 'Ｅ': 'E',
        'Ｆ': 'F', 'Ｇ': 'G', 'Ｈ': 'H', 'Ｉ': 'I', 'Ｊ': 'J',
        'Ｋ': 'K', 'Ｌ': 'L', 'Ｍ': 'M', 'Ｎ': 'N', 'Ｏ': 'O',
        'Ｐ': 'P', 'Ｑ': 'Q', 'Ｒ': 'R', 'Ｓ': 'S', 'Ｔ': 'T',
        'Ｕ': 'U', 'Ｖ': 'V', 'Ｗ': 'W', 'Ｘ': 'X', 'Ｙ': 'Y',
        'Ｚ': 'Z',
        'ａ': 'a', 'ｂ': 'b', 'ｃ': 'c', 'ｄ': 'd', 'ｅ': 'e',
        'ｆ': 'f', 'ｇ': 'g', 'ｈ': 'h', 'ｉ': 'i', 'ｊ': 'j',
        'ｋ': 'k', 'ｌ': 'l', 'ｍ': 'm', 'ｎ': 'n', 'ｏ': 'o',
        'ｐ': 'p', 'ｑ': 'q', 'ｒ': 'r', 'ｓ': 's', 'ｔ': 't',
        'ｕ': 'u', 'ｖ': 'v', 'ｗ': 'w', 'ｘ': 'x', 'ｙ': 'y',
        'ｚ': 'z',
    }
    
    # 转换全角到半角
    for fw, hw in fullwidth_to_halfwidth.items():
        text = text.replace(fw, hw)
    
    # 移除其他可能不支持的字符（保留中英文、数字、常用标点）
    # 注意：不再保留方括号 []，因为控制标签已经被移除
    allowed_pattern = r'[^\u4e00-\u9fa5a-zA-Z0-9\s\(\)\{\}\<\>\,\.\!\?\;\:\'\"\-\_\~\@\#\$\%\&\*\+\=\|\/\\\n]'
    text = re.sub(allowed_pattern, '', text)
    
    # 清理多余空格
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


# 导入自定义日志配置
from backend.logger_config import (
    OperationLogger, log_operation, log_api_call,
    system_logger, operation_logger, audit_logger, logger
)

# 导入批量处理模块
from backend.batch_processor import batch_processor, BatchJob

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 记录项目启动
OperationLogger.log_init_start()

# 全局模型缓存
models = {}

# ==================== 说话人管理 ====================

# 说话人数据存储路径
SPEAKERS_DIR = os.path.join(PROJECT_ROOT, "speakers")
SPEAKERS_DB_FILE = os.path.join(SPEAKERS_DIR, "speakers_db.json")

# 确保说话人目录存在
os.makedirs(SPEAKERS_DIR, exist_ok=True)

def load_speakers_db() -> Dict:
    """加载说话人数据库"""
    if os.path.exists(SPEAKERS_DB_FILE):
        try:
            with open(SPEAKERS_DB_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            system_logger.error(f"【说话人管理】加载数据库失败: {e}")
    return {"speakers": [], "version": "1.0"}

def save_speakers_db(db: Dict):
    """保存说话人数据库"""
    try:
        with open(SPEAKERS_DB_FILE, 'w', encoding='utf-8') as f:
            json.dump(db, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        system_logger.error(f"【说话人管理】保存数据库失败: {e}")
        return False

def get_speaker_by_name(name: str) -> Optional[Dict]:
    """根据名称获取说话人"""
    db = load_speakers_db()
    for speaker in db["speakers"]:
        if speaker["name"] == name:
            return speaker
    return None

def check_speaker_name_exists(name: str) -> bool:
    """检查说话人名称是否已存在"""
    return get_speaker_by_name(name) is not None

def add_speaker(name: str, embedding: Optional[str] = None, audio_path: Optional[str] = None, reference_text: Optional[str] = None) -> Dict:
    """添加新说话人（与模型解耦）
    
    Args:
        name: 说话人名称
        embedding: 说话人embedding (base64编码，可选，与模型解耦后可为None)
        audio_path: 参考音频路径
        reference_text: 参考音频对应的文本（可选）
    """
    db = load_speakers_db()
    
    speaker = {
        "id": f"spk_{int(time.time() * 1000)}",
        "name": name,
        "embedding": embedding,  # 可为None，与模型解耦
        "audio_path": audio_path,
        "reference_text": reference_text,
        "created_at": datetime.now().isoformat(),
        "model_type": "universal"  # 改为通用类型，不再绑定特定模型
    }
    
    db["speakers"].append(speaker)
    
    if save_speakers_db(db):
        system_logger.info(f"【说话人管理】添加说话人成功: {name}, 文本: {reference_text[:30] if reference_text else '无'}")
        return speaker
    else:
        raise HTTPException(status_code=500, detail="保存说话人失败")

def delete_speaker(speaker_id: str) -> bool:
    """删除说话人"""
    db = load_speakers_db()
    original_count = len(db["speakers"])
    db["speakers"] = [s for s in db["speakers"] if s["id"] != speaker_id]
    
    if len(db["speakers"]) < original_count:
        save_speakers_db(db)
        system_logger.info(f"【说话人管理】删除说话人: {speaker_id}")
        return True
    return False

# 添加项目路径
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'algorithms', 'ChatTTS'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'algorithms', 'CosyVoice'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'algorithms', 'CosyVoice', 'third_party', 'Matcha-TTS'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'algorithms', 'F5-TTS'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'algorithms', 'OpenVoice'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'algorithms', 'Qwen3-TTS'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'algorithms', 'GPT-SoVITS'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'algorithms', 'GPT-SoVITS', 'GPT_SoVITS'))

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
    mode: str = Field(default="base", description="模式: base, voice_clone, custom_voice, voice_design")
    speaker: Optional[str] = Field(default=None, description="预设音色: Vivian, Serena, Uncle_Fu, Dylan, Eric, Ryan, Aiden, Ono_Anna, Sohee")
    ref_audio: Optional[str] = Field(default=None, description="参考音频URL/base64")
    ref_text: Optional[str] = Field(default=None, description="参考文本")
    voice_design_prompt: Optional[str] = Field(default=None, description="音色设计描述（voice_design模式使用）")
    instruct_text: Optional[str] = Field(default=None, description="指令控制文本，用于控制语音风格")
    streaming: bool = Field(default=False, description="是否使用流式生成")
    x_vector_only_mode: bool = Field(default=False, description="是否仅使用说话人嵌入模式（voice_clone）")

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
        model_path = os.path.join(PROJECT_ROOT, "algorithms", "ChatTTS", "models")
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
        model_path = os.path.join(PROJECT_ROOT, "algorithms", "CosyVoice", "models", "iic", model_dir)
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
        ckpt_path = os.path.join(PROJECT_ROOT, "algorithms", "F5-TTS", "models", "model_1200000.pt")
        system_logger.info(f"【模型加载】F5-TTS 检查点: {ckpt_path}")
        
        models["f5tts"] = F5TTS(model="F5TTS_Base", ckpt_file=ckpt_path)
        
        duration = time.time() - start_time
        gpu_mem = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
        OperationLogger.log_model_load("F5-TTS", "成功", duration, f"GPU内存: {gpu_mem:.2f}GB")
        OperationLogger.log_performance("F5-TTS加载", duration, 0, gpu_mem)
    
    return models["f5tts"]

def get_qwen3tts_model(model_size: str = "1.7B", model_type: str = "Base"):
    """获取或加载Qwen3-TTS模型
    
    Args:
        model_size: 模型大小 "0.6B" 或 "1.7B"
        model_type: 模型类型 "Base", "CustomVoice", "VoiceDesign"
    """
    key = f"qwen3tts_{model_size}_{model_type}"
    if key not in models:
        start_time = time.time()
        OperationLogger.log_model_load(f"Qwen3-TTS-{model_size}-{model_type}", "开始加载")
        
        from qwen_tts import Qwen3TTSModel
        size_map = {
            "0.6B": "0___6B",
            "1.7B": "1___7B"
        }
        size_str = size_map.get(model_size, model_size.replace('.', '___'))
        model_path = os.path.join(PROJECT_ROOT, "algorithms", "Qwen3-TTS", "models", "Qwen", f"Qwen3-TTS-12Hz-{size_str}-{model_type}")
        
        # 如果指定类型模型不存在，尝试加载 Base 模型
        if not os.path.exists(model_path):
            if model_type != "Base":
                system_logger.warning(f"【模型加载】{model_type} 模型不存在，尝试加载 Base 模型")
                model_path = os.path.join(PROJECT_ROOT, "algorithms", "Qwen3-TTS", "models", "Qwen", f"Qwen3-TTS-12Hz-{size_str}-Base")
            if not os.path.exists(model_path):
                raise HTTPException(status_code=500, detail=f"Qwen3-TTS 模型不存在: {model_path}")
        
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
        OperationLogger.log_model_load(f"Qwen3-TTS-{model_size}-{model_type}", "成功", duration, f"GPU内存: {gpu_mem:.2f}GB")
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
        ckpt_base_en = os.path.join(PROJECT_ROOT, "algorithms", "OpenVoice", "checkpoints_v1", "checkpoints", "base_speakers", "EN")
        ckpt_base_zh = os.path.join(PROJECT_ROOT, "algorithms", "OpenVoice", "checkpoints_v1", "checkpoints", "base_speakers", "ZH")
        ckpt_converter = os.path.join(PROJECT_ROOT, "algorithms", "OpenVoice", "checkpoints_v1", "checkpoints", "converter")
        
        # 如果V1不存在,尝试V2
        if not os.path.exists(ckpt_base_en):
            ckpt_base_en = os.path.join(PROJECT_ROOT, "algorithms", "OpenVoice", "checkpoints_v2", "checkpoints_v2", "base_speakers")
            ckpt_base_zh = os.path.join(PROJECT_ROOT, "algorithms", "OpenVoice", "checkpoints_v2", "checkpoints_v2", "base_speakers")
            ckpt_converter = os.path.join(PROJECT_ROOT, "algorithms", "OpenVoice", "checkpoints_v2", "checkpoints_v2", "converter")
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
    gpt_sovits_root = os.path.join(PROJECT_ROOT, "algorithms", "GPT-SoVITS")
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
            config_path = os.path.join(PROJECT_ROOT, "algorithms", "GPT-SoVITS", "GPT_SoVITS", "configs", "tts_infer.yaml")
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
    """根路径 - 返回前端页面"""
    client_ip = request.client.host if request.client else "unknown"
    OperationLogger.log_api_request("/", "GET", {}, client_ip, 0)
    
    # 返回前端页面
    frontend_path = os.path.join(PROJECT_ROOT, "frontend", "index.html")
    if os.path.exists(frontend_path):
        return FileResponse(frontend_path)
    
    # 如果前端文件不存在，返回API信息
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

# ==================== 说话人管理 API ====================

@app.get("/speakers")
async def get_speakers():
    """获取所有已保存的说话人列表"""
    try:
        db = load_speakers_db()
        # 返回时不包含完整的 embedding 字符串（太长），只返回基本信息
        speakers_list = []
        for speaker in db["speakers"]:
            speakers_list.append({
                "id": speaker["id"],
                "name": speaker["name"],
                "created_at": speaker["created_at"],
                "model_type": speaker.get("model_type", "chattts"),
                "audio_path": speaker.get("audio_path"),  # 返回音频路径
                "has_embedding": bool(speaker.get("embedding")),
                "has_reference_text": bool(speaker.get("reference_text")),
                "reference_text": speaker.get("reference_text", "")[:100] if speaker.get("reference_text") else ""  # 只返回前100字
            })
        return {
            "success": True,
            "speakers": speakers_list,
            "total": len(speakers_list)
        }
    except Exception as e:
        system_logger.error(f"【说话人管理】获取列表失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取说话人列表失败: {str(e)}")

@app.get("/speakers/{speaker_id}")
async def get_speaker_detail(speaker_id: str):
    """获取指定说话人的详细信息（包含embedding）"""
    try:
        db = load_speakers_db()
        for speaker in db["speakers"]:
            if speaker["id"] == speaker_id:
                return {
                    "success": True,
                    "speaker": speaker
                }
        raise HTTPException(status_code=404, detail="说话人不存在")
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"获取说话人详情失败: {str(e)}")


@app.get("/speakers/{speaker_id}/audio")
async def get_speaker_audio(speaker_id: str):
    """获取指定说话人的参考音频文件"""
    try:
        db = load_speakers_db()
        for speaker in db["speakers"]:
            if speaker["id"] == speaker_id:
                audio_path = speaker.get("audio_path")
                if not audio_path or not os.path.exists(audio_path):
                    raise HTTPException(status_code=404, detail="音频文件不存在")
                
                # 根据文件扩展名确定媒体类型
                ext = os.path.splitext(audio_path)[1].lower()
                media_type = {
                    '.wav': 'audio/wav',
                    '.mp3': 'audio/mpeg',
                    '.ogg': 'audio/ogg',
                    '.webm': 'audio/webm',
                    '.m4a': 'audio/mp4'
                }.get(ext, 'audio/wav')
                
                return FileResponse(audio_path, media_type=media_type)
        
        raise HTTPException(status_code=404, detail="说话人不存在")
    except HTTPException:
        raise
    except Exception as e:
        system_logger.error(f"获取说话人音频失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取说话人音频失败: {str(e)}")


@app.post("/speakers/check-name")
async def check_name(name: str = Form(...)):
    """检查说话人名称是否可用"""
    exists = check_speaker_name_exists(name)
    return {
        "success": True,
        "available": not exists,
        "exists": exists,
        "message": "名称已被使用" if exists else "名称可用"
    }

@app.post("/speakers/upload")
async def upload_speaker_audio(
    audio: UploadFile = File(...),
    speaker_name: str = Form(...),
    reference_text: Optional[str] = Form(None)
):
    """
    上传说话人音频文件（与模型解耦，只保存音频和文本）
    
    - audio: 音频文件 (MP3, WAV, WEBM等)
    - speaker_name: 说话人名称
    - reference_text: 参考音频对应的文本
    """
    start_time = time.time()
    
    # 验证名称
    if not speaker_name or len(speaker_name.strip()) == 0:
        raise HTTPException(status_code=400, detail="说话人名称不能为空")
    
    if check_speaker_name_exists(speaker_name):
        raise HTTPException(status_code=400, detail=f"说话人名称 '{speaker_name}' 已存在")
    
    # 验证文件格式
    allowed_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.webm'}
    file_ext = os.path.splitext(audio.filename)[1].lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"不支持的音频格式: {file_ext}。支持的格式: {', '.join(allowed_extensions)}"
        )
    
    try:
        system_logger.info(f"【说话人管理】开始上传: {speaker_name}, 文件: {audio.filename}")
        
        # 读取音频文件
        audio_bytes = await audio.read()
        
        # 保存上传的音频文件
        timestamp = int(time.time())
        # 统一转换为 wav 格式以便兼容各种TTS模型
        audio_filename = f"speaker_{timestamp}_{speaker_name}.wav"
        audio_path = os.path.join(SPEAKERS_DIR, audio_filename)
        
        # 对于 webm/ogg 格式，使用 ffmpeg 转换为 wav
        if file_ext in ['.webm', '.ogg']:
            import subprocess
            temp_path = os.path.join(SPEAKERS_DIR, f"temp_{timestamp}{file_ext}")
            try:
                # 先保存原始文件
                with open(temp_path, 'wb') as f:
                    f.write(audio_bytes)
                
                # 使用 ffmpeg 转换
                cmd = [
                    'ffmpeg', '-y', '-i', temp_path,
                    '-ar', '24000', '-ac', '1', '-acodec', 'pcm_s16le',
                    audio_path
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                
                if result.returncode != 0:
                    raise Exception(f"ffmpeg 转换失败: {result.stderr}")
                
                # 删除临时文件
                os.remove(temp_path)
                
            except Exception as e:
                system_logger.error(f"【说话人管理】转换失败: {e}")
                raise HTTPException(status_code=400, detail=f"音频转换失败: {e}")
        else:
            # 直接保存文件
            with open(audio_path, 'wb') as f:
                f.write(audio_bytes)
        
        duration = time.time() - start_time
        system_logger.info(f"【说话人管理】上传成功: {speaker_name}, 路径: {audio_path}, 耗时: {duration:.2f}s")
        
        return {
            "success": True,
            "message": "音频上传成功",
            "speaker_name": speaker_name,
            "audio_path": audio_path,
            "reference_text": reference_text,
            "duration": duration
        }
        
    except Exception as e:
        # 清理已保存的音频文件
        if 'audio_path' in locals() and os.path.exists(audio_path):
            os.remove(audio_path)
        
        system_logger.error(f"【说话人管理】上传失败: {e}")
        raise HTTPException(status_code=500, detail=f"上传失败: {str(e)}")

@app.post("/speakers/save")
async def save_speaker(
    name: str = Form(...),
    audio_path: str = Form(...),
    reference_text: Optional[str] = Form(None)
):
    """
    保存说话人信息（与模型解耦，只保存音频和文本）
    
    - name: 说话人名称
    - audio_path: 参考音频路径
    - reference_text: 参考音频对应的文本（可选）
    """
    try:
        system_logger.info(f"【说话人管理】开始保存: {name}, 音频: {audio_path}")
        
        # 验证名称
        if not name or len(name.strip()) == 0:
            raise HTTPException(status_code=400, detail="说话人名称不能为空")
        
        if check_speaker_name_exists(name):
            raise HTTPException(status_code=400, detail=f"说话人名称 '{name}' 已存在")
        
        # 验证音频文件是否存在
        if not audio_path or not os.path.exists(audio_path):
            raise HTTPException(status_code=400, detail="音频文件不存在")
        
        # 保存说话人（embedding 设为 None，与模型解耦）
        speaker = add_speaker(name, None, audio_path, reference_text)
        
        # 记录审计日志
        OperationLogger.log_speaker_operation("创建", speaker["name"], speaker["id"])
        
        system_logger.info(f"【说话人管理】保存成功: {name}, ID: {speaker['id']}")
        
        return {
            "success": True,
            "message": "说话人保存成功",
            "speaker": {
                "id": speaker["id"],
                "name": speaker["name"],
                "audio_path": speaker["audio_path"],
                "reference_text": speaker.get("reference_text"),
                "created_at": speaker["created_at"]
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        system_logger.error(f"【说话人管理】保存失败: {e}")
        raise HTTPException(status_code=500, detail=f"保存说话人失败: {str(e)}")

@app.delete("/speakers/{speaker_id}")
async def delete_speaker_api(speaker_id: str):
    """删除指定的说话人"""
    try:
        # 获取说话人信息以删除关联的音频文件
        db = load_speakers_db()
        speaker = None
        for s in db["speakers"]:
            if s["id"] == speaker_id:
                speaker = s
                break
        
        if not speaker:
            raise HTTPException(status_code=404, detail="说话人不存在")
        
        # 删除关联的音频文件
        if speaker.get("audio_path") and os.path.exists(speaker["audio_path"]):
            os.remove(speaker["audio_path"])
        
        # 删除数据库记录
        if delete_speaker(speaker_id):
            # 记录审计日志
            OperationLogger.log_speaker_operation("删除", speaker.get("name", "unknown"), speaker_id)
            
            return {
                "success": True,
                "message": "说话人删除成功"
            }
        else:
            raise HTTPException(status_code=500, detail="删除说话人失败")
            
    except HTTPException:
        raise
    except Exception as e:
        system_logger.error(f"【说话人管理】删除失败: {e}")
        raise HTTPException(status_code=500, detail=f"删除说话人失败: {str(e)}")

# ==================== ChatTTS API ====================

@app.post("/tts/chattts")
async def tts_chattts(
    request: Request,
    text: str = Form(...),
    temperature: float = Form(0.3),
    top_P: float = Form(0.7),
    top_K: float = Form(20),
    output_format: str = Form("url")
):
    """ChatTTS语音合成 - 使用随机说话人"""
    start_time = time.time()
    client_ip = request.client.host if request.client else "unknown"
    
    try:
        # 预处理文本 - 移除控制标签和转换全角字符
        original_text = text
        text = preprocess_text_for_chattts(text)
        if text != original_text:
            system_logger.info(f"【ChatTTS】文本预处理 | 原始: {original_text[:50]}... | 处理后: {text[:50]}...")
        
        # 记录API请求
        OperationLogger.log_api_request("/tts/chattts", "POST", {
            "text_preview": text[:50],
            "temperature": temperature,
            "top_P": top_P,
            "top_K": top_K,
        }, client_ip)

        system_logger.info(f"【ChatTTS】开始合成 | 文本: {text[:50]}... | 客户端: {client_ip}")
        chat = get_chattts_model()

        # 使用随机说话人
        spk_emb = chat.sample_random_speaker()
        system_logger.info(f"【ChatTTS】使用随机说话人")

        # 记录推理参数
        system_logger.info(f"【ChatTTS】推理参数 - temperature={temperature}, top_P={top_P}, top_K={top_K}")

        # 清理 GPU 缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gpu_mem_before = torch.cuda.memory_allocated() / 1024**3
            system_logger.info(f"【ChatTTS】GPU内存清理完成，当前使用: {gpu_mem_before:.2f}GB")

        # 推理
        # 限制 temperature 最小值为 0.1（ChatTTS 不支持 temperature=0）
        safe_temperature = max(float(temperature), 0.1)
        if safe_temperature != float(temperature):
            system_logger.info(f"【ChatTTS】temperature 从 {temperature} 调整为 {safe_temperature}（最小值限制）")

        params = chat.InferCodeParams(
            spk_emb=spk_emb,
            temperature=safe_temperature,
            top_P=float(top_P),
            top_K=int(top_K),
        )

        system_logger.info(f"【ChatTTS】开始推理...")
        infer_start = time.time()
        try:
            wavs = chat.infer(
                [text],
                stream=False,
                params_infer_code=params,
            )
            system_logger.info(f"【ChatTTS】infer 返回成功")
        except RecursionError as e:
            system_logger.error(f"【ChatTTS】递归错误: {e}")
            import traceback
            system_logger.error(f"【ChatTTS】堆栈跟踪:\n{traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"语音合成失败(递归错误): {str(e)[:100]}")
        except ValueError as e:
            if "need at least one array to concatenate" in str(e):
                system_logger.error(f"【ChatTTS】生成结果为空: {e}")
                raise HTTPException(status_code=500, detail="语音合成失败: 模型生成结果为空。请尝试：1. 提高temperature参数 2. 检查参考音频质量 3. 缩短文本长度")
            raise HTTPException(status_code=500, detail=f"语音合成失败: {str(e)[:100]}")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                system_logger.error(f"【ChatTTS】GPU内存不足: {e}")
                # 尝试清理GPU内存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    system_logger.info(f"【ChatTTS】GPU内存已清理，请重试")
                raise HTTPException(status_code=503, detail="GPU内存不足，请稍后重试")
            system_logger.error(f"【ChatTTS】RuntimeError: {e}")
            import traceback
            system_logger.error(f"【ChatTTS】堆栈跟踪:\n{traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"语音合成失败: {str(e)[:100]}")
        except Exception as e:
            system_logger.error(f"【ChatTTS】未预期的错误: {type(e).__name__}: {e}")
            import traceback
            system_logger.error(f"【ChatTTS】堆栈跟踪:\n{traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"语音合成失败: {str(e)[:100]}")
        infer_duration = time.time() - infer_start
        system_logger.info(f"【ChatTTS】推理完成，耗时: {infer_duration:.3f}s")
        
        # 检查合成结果
        if not wavs or len(wavs) == 0:
            system_logger.error(f"【ChatTTS】合成结果为空列表")
            raise HTTPException(status_code=500, detail="语音合成失败: 返回结果为空")
        
        if wavs[0] is None:
            system_logger.error(f"【ChatTTS】合成结果第一个元素为None")
            raise HTTPException(status_code=500, detail="语音合成失败: 音频数据为None")
        
        wav_length = len(wavs[0])
        system_logger.info(f"【ChatTTS】合成音频长度: {wav_length} 样本")
        
        if wav_length == 0:
            system_logger.error(f"【ChatTTS】合成音频长度为0")
            raise HTTPException(status_code=500, detail="语音合成失败: 音频长度为0。建议：1. 提高temperature至0.5-0.7 2. 检查参考音频是否包含有效语音 3. 使用更短的文本测试")
        
        # 检查音频是否全为静音
        wav_max = np.abs(wavs[0]).max()
        system_logger.info(f"【ChatTTS】合成音频最大振幅: {wav_max:.6f}")
        
        if wav_max < 1e-5:
            system_logger.error(f"【ChatTTS】合成音频几乎全为静音")
            raise HTTPException(status_code=500, detail="语音合成失败: 音频几乎全为静音")

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
DEFAULT_F5TTS_REF_ZH = os.path.join(PROJECT_ROOT, "algorithms", "F5-TTS", "src", "f5_tts", "infer", "examples", "basic", "basic_ref_zh.wav")
DEFAULT_F5TTS_REF_EN = os.path.join(PROJECT_ROOT, "algorithms", "F5-TTS", "src", "f5_tts", "infer", "examples", "basic", "basic_ref_en.wav")
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
    mode: str = Form("voice_clone"),
    speaker: Optional[str] = Form(None),
    ref_wav: Optional[UploadFile] = File(None),
    ref_text: Optional[str] = Form(None),
    voice_design_prompt: Optional[str] = Form(None),
    instruct_text: Optional[str] = Form(None),
    streaming: bool = Form(False),
    x_vector_only_mode: bool = Form(False),
    output_format: str = Form("url")
):
    """Qwen3-TTS语音合成
    
    支持模式：
    - voice_clone: 音色克隆（需要参考音频）
    - custom_voice: 预设音色（需要选择 speaker）
    - voice_design: 音色设计（需要 voice_design_prompt）
    """
    try:
        logger.info(f"Qwen3-TTS请求: {text[:50]}... 模型: {model_size}, 模式: {mode}")

        # 根据模式选择模型类型
        model_type_map = {
            "voice_clone": "Base",
            "custom_voice": "CustomVoice",
            "voice_design": "VoiceDesign"
        }
        model_type = model_type_map.get(mode, "Base")
        
        tts = get_qwen3tts_model(model_size, model_type)

        if mode == "voice_clone":
            if ref_wav:
                # 保存参考音频
                ref_path = f"uploads/ref_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.wav"
                with open(ref_path, "wb") as f:
                    f.write(await ref_wav.read())

                wavs, sr = tts.generate_voice_clone(
                    text=text,
                    language="Auto",
                    ref_audio=ref_path,
                    ref_text=ref_text or "",
                    x_vector_only_mode=x_vector_only_mode
                )
                os.remove(ref_path)
                wav = wavs[0] if isinstance(wavs, list) else wavs
            else:
                # 使用默认参考音频
                default_ref_audio = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone_1.wav"
                default_ref_text = "甚至出现交易几乎停滞的情况。"
                
                logger.info(f"voice_clone模式：使用默认参考音频")
                wavs, sr = tts.generate_voice_clone(
                    text=text,
                    language="Auto",
                    ref_audio=default_ref_audio,
                    ref_text=default_ref_text,
                    x_vector_only_mode=True
                )
                wav = wavs[0] if isinstance(wavs, list) else wavs
                
        elif mode == "custom_voice":
            # 预设音色模式
            if not speaker:
                # 默认使用 Vivian
                speaker = "Vivian"
            
            logger.info(f"custom_voice模式：使用预设音色 {speaker}, 指令: {instruct_text or '无'}")
            
            # 尝试使用 generate_custom_voice，如果不支持则回退
            custom_voice_success = False
            try:
                if hasattr(tts, 'generate_custom_voice'):
                    wavs, sr = tts.generate_custom_voice(
                        text=text,
                        language="Auto",
                        speaker=speaker,
                        instruct=instruct_text or ""
                    )
                    wav = wavs[0] if isinstance(wavs, list) else wavs
                    custom_voice_success = True
                    logger.info("使用 CustomVoice 模型生成成功")
            except ValueError as e:
                if "does not support generate_custom_voice" in str(e):
                    logger.warning(f"CustomVoice 模型不支持该方法: {e}")
                else:
                    raise
            
            if not custom_voice_success:
                # 如果模型不支持，回退到 Base 模型的 voice_clone
                logger.warning(f"当前模型不支持 generate_custom_voice，回退到 Base 模型使用默认音色")
                tts_base = get_qwen3tts_model(model_size, "Base")
                default_ref_audio = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone_1.wav"
                default_ref_text = "甚至出现交易几乎停滞的情况。"
                wavs, sr = tts_base.generate_voice_clone(
                    text=text,
                    language="Auto",
                    ref_audio=default_ref_audio,
                    ref_text=default_ref_text,
                    x_vector_only_mode=True
                )
                wav = wavs[0] if isinstance(wavs, list) else wavs
                
        elif mode == "voice_design":
            # 音色设计模式
            if not voice_design_prompt:
                raise HTTPException(status_code=400, detail="voice_design 模式需要提供 voice_design_prompt 参数")
            
            logger.info(f"voice_design模式：音色描述: {voice_design_prompt}")
            
            # 尝试使用 generate_voice_design，如果不支持则回退
            voice_design_success = False
            try:
                if hasattr(tts, 'generate_voice_design'):
                    wavs, sr = tts.generate_voice_design(
                        text=text,
                        language="Auto",
                        instruct=voice_design_prompt
                    )
                    wav = wavs[0] if isinstance(wavs, list) else wavs
                    voice_design_success = True
                    logger.info("使用 VoiceDesign 模型生成成功")
            except ValueError as e:
                if "does not support generate_voice_design" in str(e):
                    logger.warning(f"VoiceDesign 模型不支持该方法: {e}")
                else:
                    raise
            
            if not voice_design_success:
                # 如果模型不支持，回退到 Base 模型的 voice_clone
                logger.warning(f"当前模型不支持 generate_voice_design，回退到 Base 模型使用默认音色")
                tts_base = get_qwen3tts_model(model_size, "Base")
                default_ref_audio = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone_1.wav"
                default_ref_text = "甚至出现交易几乎停滞的情况。"
                wavs, sr = tts_base.generate_voice_clone(
                    text=text,
                    language="Auto",
                    ref_audio=default_ref_audio,
                    ref_text=default_ref_text,
                    x_vector_only_mode=True
                )
                wav = wavs[0] if isinstance(wavs, list) else wavs
        else:
            raise HTTPException(status_code=400, detail=f"不支持的模式: {mode}")

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
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

class Qwen3TTSModelStatus(BaseModel):
    base_available: bool
    custom_voice_available: bool
    voice_design_available: bool
    model_sizes: List[str]
    message: str

@app.get("/tts/qwen3tts/status")
async def get_qwen3tts_status():
    """获取 Qwen3-TTS 模型状态"""
    try:
        status = {
            "base_available": False,
            "custom_voice_available": False,
            "voice_design_available": False,
            "model_sizes": [],
            "message": ""
        }
        
        size_configs = ["0.6B", "1.7B"]
        available_sizes = []
        
        for size in size_configs:
            size_map = {"0.6B": "0___6B", "1.7B": "1___7B"}
            size_str = size_map.get(size, size.replace('.', '___'))
            
            # 检查 Base 模型
            base_path = os.path.join(PROJECT_ROOT, "algorithms", "Qwen3-TTS", "models", "Qwen", f"Qwen3-TTS-12Hz-{size_str}-Base")
            if os.path.exists(base_path):
                status["base_available"] = True
                if size not in available_sizes:
                    available_sizes.append(size)
            
            # 检查 CustomVoice 模型
            custom_path = os.path.join(PROJECT_ROOT, "algorithms", "Qwen3-TTS", "models", "Qwen", f"Qwen3-TTS-12Hz-{size_str}-CustomVoice")
            if os.path.exists(custom_path):
                status["custom_voice_available"] = True
                if size not in available_sizes:
                    available_sizes.append(size)
            
            # 检查 VoiceDesign 模型
            design_path = os.path.join(PROJECT_ROOT, "algorithms", "Qwen3-TTS", "models", "Qwen", f"Qwen3-TTS-12Hz-{size_str}-VoiceDesign")
            if os.path.exists(design_path):
                status["voice_design_available"] = True
                if size not in available_sizes:
                    available_sizes.append(size)
        
        status["model_sizes"] = available_sizes
        
        # 生成状态消息
        if status["base_available"]:
            if status["custom_voice_available"] and status["voice_design_available"]:
                status["message"] = "所有模型已就绪"
            elif status["custom_voice_available"]:
                status["message"] = "Base 和 CustomVoice 模型可用，VoiceDesign 模型缺失"
            elif status["voice_design_available"]:
                status["message"] = "Base 和 VoiceDesign 模型可用，CustomVoice 模型缺失"
            else:
                status["message"] = "仅 Base 模型可用，CustomVoice 和 VoiceDesign 功能将使用默认音色"
        else:
            status["message"] = "Qwen3-TTS 模型未找到"
        
        return status
        
    except Exception as e:
        logger.error(f"获取 Qwen3-TTS 状态失败: {e}")
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


# ==================== 录音参考文本 API ====================

# 预定义的朗读文本片段
RECORDING_SCRIPTS = {
    "short": [
        # 经典诗词 - 唐诗
        {
            "id": "poem_001",
            "text": "床前明月光，疑是地上霜。举头望明月，低头思故乡。",
            "type": "唐诗",
            "source": "李白《静夜思》",
            "duration": "6-9秒"
        },
        {
            "id": "poem_002",
            "text": "春眠不觉晓，处处闻啼鸟。夜来风雨声，花落知多少。",
            "type": "唐诗",
            "source": "孟浩然《春晓》",
            "duration": "6-9秒"
        },
        {
            "id": "poem_003",
            "text": "白日依山尽，黄河入海流。欲穷千里目，更上一层楼。",
            "type": "唐诗",
            "source": "王之涣《登鹳雀楼》",
            "duration": "6-9秒"
        },
        {
            "id": "poem_004",
            "text": "千山鸟飞绝，万径人踪灭。孤舟蓑笠翁，独钓寒江雪。",
            "type": "唐诗",
            "source": "柳宗元《江雪》",
            "duration": "6-9秒"
        },
        {
            "id": "poem_005",
            "text": "空山不见人，但闻人语响。返景入深林，复照青苔上。",
            "type": "唐诗",
            "source": "王维《鹿柴》",
            "duration": "6-9秒"
        },
        # 经典诗词 - 宋词
        {
            "id": "poem_006",
            "text": "明月几时有？把酒问青天。不知天上宫阙，今夕是何年。",
            "type": "宋词",
            "source": "苏轼《水调歌头》",
            "duration": "7-10秒"
        },
        {
            "id": "poem_007",
            "text": "大江东去，浪淘尽，千古风流人物。故垒西边，人道是，三国周郎赤壁。",
            "type": "宋词",
            "source": "苏轼《念奴娇·赤壁怀古》",
            "duration": "8-11秒"
        },
        {
            "id": "poem_008",
            "text": "寻寻觅觅，冷冷清清，凄凄惨惨戚戚。乍暖还寒时候，最难将息。",
            "type": "宋词",
            "source": "李清照《声声慢》",
            "duration": "7-10秒"
        },
        # 经典散文
        {
            "id": "prose_001",
            "text": "燕子去了，有再来的时候；杨柳枯了，有再青的时候；桃花谢了，有再开的时候。",
            "type": "散文",
            "source": "朱自清《匆匆》",
            "duration": "8-11秒"
        },
        {
            "id": "prose_002",
            "text": "盼望着，盼望着，东风来了，春天的脚步近了。一切都像刚睡醒的样子，欣欣然张开了眼。",
            "type": "散文",
            "source": "朱自清《春》",
            "duration": "8-11秒"
        },
        {
            "id": "prose_003",
            "text": "不必说碧绿的菜畦，光滑的石井栏，高大的皂荚树，紫红的桑椹；也不必说鸣蝉在树叶里长吟。",
            "type": "散文",
            "source": "鲁迅《从百草园到三味书屋》",
            "duration": "9-12秒"
        },
        # 经典名言
        {
            "id": "quote_001",
            "text": "学而时习之，不亦说乎？有朋自远方来，不亦乐乎？",
            "type": "名言",
            "source": "《论语》",
            "duration": "6-9秒"
        },
        {
            "id": "quote_002",
            "text": "天行健，君子以自强不息；地势坤，君子以厚德载物。",
            "type": "名言",
            "source": "《周易》",
            "duration": "7-10秒"
        },
        {
            "id": "quote_003",
            "text": "路漫漫其修远兮，吾将上下而求索。",
            "type": "名言",
            "source": "屈原《离骚》",
            "duration": "5-8秒"
        },
        {
            "id": "quote_004",
            "text": "先天下之忧而忧，后天下之乐而乐。",
            "type": "名言",
            "source": "范仲淹《岳阳楼记》",
            "duration": "5-8秒"
        },
        # 实用场景
        {
            "id": "scene_001",
            "text": "你好，很高兴为您服务。我是您的专属语音助手，请问今天有什么可以帮助您的吗？",
            "type": "对话",
            "source": "客服场景",
            "duration": "7-10秒"
        },
        {
            "id": "scene_002",
            "text": "欢迎收听今天的晚间新闻。首先来看主要内容：国内经济形势稳中向好，科技创新取得新突破。",
            "type": "新闻",
            "source": "新闻播报",
            "duration": "8-11秒"
        }
    ],
    "medium": [
        # 经典散文片段
        {
            "id": "prose_101",
            "text": "这几天心里颇不宁静。今晚在院子里坐着乘凉，忽然想起日日走过的荷塘，在这满月的光里，总该另有一番样子吧。月亮渐渐地升高了，墙外马路上孩子们的欢笑，已经听不见了；妻在屋里拍着闰儿，迷迷糊糊地哼着眠歌。我悄悄地披了大衫，带上门出去。",
            "type": "散文",
            "source": "朱自清《荷塘月色》",
            "duration": "15-20秒"
        },
        {
            "id": "prose_102",
            "text": "曲曲折折的荷塘上面，弥望的是田田的叶子。叶子出水很高，像亭亭的舞女的裙。层层的叶子中间，零星地点缀着些白花，有袅娜地开着的，有羞涩地打着朵儿的；正如一粒粒的明珠，又如碧天里的星星，又如刚出浴的美人。",
            "type": "散文",
            "source": "朱自清《荷塘月色》",
            "duration": "13-18秒"
        },
        {
            "id": "prose_103",
            "text": "从百草园到三味书屋，只有一堵墙的距离，却隔开了两个世界。一个是充满野趣和幻想的童年乐园，一个是严肃而规矩的读书之地。我坐在三味书屋的课桌前，耳边似乎还能听到百草园里蝉鸣的声音。",
            "type": "散文",
            "source": "鲁迅《从百草园到三味书屋》",
            "duration": "12-17秒"
        },
        {
            "id": "prose_104",
            "text": "北京的冬季，地上还有积雪，灰黑色的秃树枝丫叉于晴朗的天空中，而远处有一二风筝浮动，在我是一种惊异和悲哀。故乡的风筝时节，是春二月，倘听到沙沙的风轮声，仰头便能看见一个淡墨色的蟹风筝或嫩蓝色的蜈蚣风筝。",
            "type": "散文",
            "source": "鲁迅《风筝》",
            "duration": "14-19秒"
        },
        # 经典小说片段
        {
            "id": "novel_101",
            "text": "话说天下大势，分久必合，合久必分。周末七国分争，并入于秦。及秦灭之后，楚、汉分争，又并入于汉。汉朝自高祖斩白蛇而起义，一统天下，后来光武中兴，传至献帝，遂分为三国。",
            "type": "小说",
            "source": "罗贯中《三国演义》",
            "duration": "16-21秒"
        },
        {
            "id": "novel_102",
            "text": "满纸荒唐言，一把辛酸泪。都云作者痴，谁解其中味？开辟鸿蒙，谁为情种？都只为风月情浓。趁着这奈何天，伤怀日，寂寥时，试遣愚衷。因此上，演出这怀金悼玉的《红楼梦》。",
            "type": "小说",
            "source": "曹雪芹《红楼梦》",
            "duration": "14-19秒"
        },
        {
            "id": "novel_103",
            "text": "混沌未分天地乱，茫茫渺渺无人见。自从盘古破鸿蒙，开辟从兹清浊辨。覆载群生仰至仁，发明万物皆成善。欲知造化会元功，须看《西游释厄传》。",
            "type": "小说",
            "source": "吴承恩《西游记》",
            "duration": "13-18秒"
        },
        {
            "id": "novel_104",
            "text": "银烛秋光冷画屏，轻罗小扇扑流萤。天阶夜色凉如水，坐看牵牛织女星。这是唐代诗人杜牧的《秋夕》。在这首诗中，诗人描绘了一个宫女在秋夜里扑打萤火虫、仰望星空的孤寂场景。",
            "type": "诗词赏析",
            "source": "唐诗赏析",
            "duration": "12-17秒"
        },
        # 现代散文
        {
            "id": "prose_105",
            "text": "我爱月夜，但我也爱星天。从前在家乡，七、八月的夜晚，在庭院里纳凉的时候，我最爱看天上密密麻麻的繁星。望着星天，我就会忘记一切，仿佛回到了母亲的怀里似的。",
            "type": "散文",
            "source": "巴金《繁星》",
            "duration": "11-16秒"
        },
        {
            "id": "prose_106",
            "text": "秋天，无论在什么地方的秋天，总是好的；可是啊，北国的秋，却特别地来得清，来得静，来得悲凉。我的不远千里，要从杭州赶上青岛，更要从青岛赶上北平来的理由，也不过想饱尝一尝这秋，这故都的秋味。",
            "type": "散文",
            "source": "郁达夫《故都的秋》",
            "duration": "14-19秒"
        },
        # 演讲稿
        {
            "id": "speech_101",
            "text": "今天，我站在这里，想和大家分享一个关于梦想的故事。每个人都有梦想，有人想当科学家，有人想当艺术家，有人想当医生。而我，想成为一名声音的魔术师，用我的声音传递温暖，传递力量，传递那些文字无法表达的情感。",
            "type": "演讲",
            "source": "励志演讲",
            "duration": "13-18秒"
        }
    ],
    "long": [
        # 经典长文
        {
            "id": "classic_201",
            "text": "庆历四年春，滕子京谪守巴陵郡。越明年，政通人和，百废具兴，乃重修岳阳楼，增其旧制，刻唐贤今人诗赋于其上，属予作文以记之。予观夫巴陵胜状，在洞庭一湖。衔远山，吞长江，浩浩汤汤，横无际涯，朝晖夕阴，气象万千，此则岳阳楼之大观也，前人之述备矣。",
            "type": "古文",
            "source": "范仲淹《岳阳楼记》",
            "duration": "20-25秒"
        },
        {
            "id": "classic_202",
            "text": "若夫霪雨霏霏，连月不开，阴风怒号，浊浪排空，日星隐曜，山岳潜形，商旅不行，樯倾楫摧，薄暮冥冥，虎啸猿啼。登斯楼也，则有去国怀乡，忧谗畏讥，满目萧然，感极而悲者矣。至若春和景明，波澜不惊，上下天光，一碧万顷，沙鸥翔集，锦鳞游泳，岸芷汀兰，郁郁青青。",
            "type": "古文",
            "source": "范仲淹《岳阳楼记》",
            "duration": "22-28秒"
        },
        {
            "id": "classic_203",
            "text": "晋太元中，武陵人捕鱼为业。缘溪行，忘路之远近。忽逢桃花林，夹岸数百步，中无杂树，芳草鲜美，落英缤纷。渔人甚异之，复前行，欲穷其林。林尽水源，便得一山，山有小口，仿佛若有光。便舍船，从口入。初极狭，才通人。复行数十步，豁然开朗。",
            "type": "古文",
            "source": "陶渊明《桃花源记》",
            "duration": "21-27秒"
        },
        {
            "id": "classic_204",
            "text": "山不在高，有仙则名。水不在深，有龙则灵。斯是陋室，惟吾德馨。苔痕上阶绿，草色入帘青。谈笑有鸿儒，往来无白丁。可以调素琴，阅金经。无丝竹之乱耳，无案牍之劳形。南阳诸葛庐，西蜀子云亭。孔子云：何陋之有？",
            "type": "古文",
            "source": "刘禹锡《陋室铭》",
            "duration": "18-23秒"
        },
        # 现代长文
        {
            "id": "prose_201",
            "text": "我常坐在这棵老槐树下，看夕阳慢慢西沉。金色的余晖洒在青石板路上，照亮了路边那几株野菊花。远处传来牧童的笛声，悠扬而婉转，让人想起童年那些无忧无虑的日子。那时候，天是那么蓝，云是那么白，我们的笑声是那样清脆。如今，岁月已经在我们脸上刻下了痕迹，但心中那份对美好生活的向往，却从未改变。",
            "type": "散文",
            "source": "原创散文",
            "duration": "20-26秒"
        },
        {
            "id": "speech_201",
            "text": "各位老师、同学们，大家好！今天我演讲的题目是《青春与梦想》。青春是人生最美好的时光，它如同初升的朝阳，充满了活力和希望。在这段宝贵的岁月里，我们应该怀揣梦想，勇敢前行。梦想是什么？梦想是灯塔，指引我们前进的方向；梦想是翅膀，带我们飞向远方。无论前方的路有多么坎坷，只要我们心中有梦，脚下就有力量。让我们以梦为马，不负韶华，用青春的汗水浇灌梦想的花朵！",
            "type": "演讲",
            "source": "校园演讲",
            "duration": "22-28秒"
        },
        # 诗歌长诗
        {
            "id": "poem_201",
            "text": "轻轻的我走了，正如我轻轻的来；我轻轻的招手，作别西天的云彩。那河畔的金柳，是夕阳中的新娘；波光里的艳影，在我的心头荡漾。软泥上的青荇，油油的在水底招摇；在康河的柔波里，我甘心做一条水草！那榆荫下的一潭，不是清泉，是天上虹；揉碎在浮藻间，沉淀着彩虹似的梦。",
            "type": "现代诗",
            "source": "徐志摩《再别康桥》",
            "duration": "23-29秒"
        }
    ]
}

@app.get("/recording_scripts")
async def get_recording_scripts(
    length: str = Query("short", description="文本长度: short/medium/long"),
    type_filter: Optional[str] = Query(None, description="文本类型过滤")
):
    """
    获取供用户朗读录音的参考文本片段
    
    参数:
    - length: 文本长度 (short-短文本5-8秒, medium-中等10-15秒, long-长文本15-20秒)
    - type_filter: 文本类型过滤 (通用/描述/新闻/对话/叙述/科普/演讲/故事)
    """
    try:
        scripts = RECORDING_SCRIPTS.get(length, RECORDING_SCRIPTS["short"])
        
        # 应用类型过滤
        if type_filter:
            scripts = [s for s in scripts if s.get("type") == type_filter]
        
        return {
            "success": True,
            "length": length,
            "count": len(scripts),
            "scripts": scripts
        }
    except Exception as e:
        system_logger.error(f"获取录音文本失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取录音文本失败: {str(e)}")


@app.get("/recording_scripts/types")
async def get_recording_script_types():
    """获取所有可用的录音文本类型"""
    types = set()
    for scripts in RECORDING_SCRIPTS.values():
        for script in scripts:
            types.add(script.get("type", "通用"))
    
    return {
        "success": True,
        "types": sorted(list(types))
    }


@app.get("/recording_scripts/random")
async def get_random_recording_script(
    length: str = Query("short", description="文本长度: short/medium/long")
):
    """随机获取一条录音文本"""
    import random
    
    scripts = RECORDING_SCRIPTS.get(length, RECORDING_SCRIPTS["short"])
    if scripts:
        script = random.choice(scripts)
        return {
            "success": True,
            "script": script
        }
    else:
        raise HTTPException(status_code=404, detail="没有找到合适的录音文本")


# ==================== 静态文件服务 ====================

@app.get("/audio/{filename}")
async def get_audio(filename: str):
    """获取音频文件"""
    file_path = f"output/{filename}"
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="文件不存在")
    return FileResponse(file_path, media_type="audio/wav")

# 挂载前端静态文件 - 使用frontend目录作为静态文件根目录
frontend_dir = os.path.join(PROJECT_ROOT, "frontend")
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")

# ==================== 主函数 ====================

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
