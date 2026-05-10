#!/usr/bin/env python3
"""
VersTTS API 数据模型 (Pydantic)
"""

from typing import Optional, List
from pydantic import BaseModel, Field


class BaseTTSRequest(BaseModel):
    """基础 TTS 请求"""
    text: str = Field(..., description="要合成的文本")
    language: str = Field(default="zh", description="语言: zh, en, auto")


class ChatTTSRequest(BaseTTSRequest):
    """ChatTTS 请求"""
    speaker_emb: Optional[str] = Field(default=None, description="说话人embedding (base64)")
    temperature: float = Field(default=0.3, description="采样温度")
    top_P: float = Field(default=0.7, description="Top P采样")
    top_K: float = Field(default=20, description="Top K采样")


class CosyVoiceRequest(BaseTTSRequest):
    """CosyVoice 请求"""
    mode: str = Field(default="sft", description="模式: sft, zero_shot, cross_lingual, instruct")
    speaker_id: str = Field(default="中文女", description="说话人ID")
    prompt_text: Optional[str] = Field(default=None, description="参考文本")
    instruct_text: Optional[str] = Field(default=None, description="指令文本")


class F5TTSRequest(BaseTTSRequest):
    """F5-TTS 请求"""
    ref_text: str = Field(..., description="参考文本")
    nfe_step: int = Field(default=32, description="NFE步数")
    cfg_strength: float = Field(default=2.0, description="CFG强度")
    speed: float = Field(default=1.0, description="语速")


class Qwen3TTSRequest(BaseTTSRequest):
    """Qwen3-TTS 请求"""
    model_size: str = Field(default="1.7B", description="模型大小: 0.6B, 1.7B")
    mode: str = Field(default="base", description="模式: base, voice_clone, custom_voice, voice_design")
    speaker: Optional[str] = Field(
        default=None,
        description="预设音色: vivian, serena, uncle_fu, dylan, eric, ryan, aiden, ono_anna, sohee (小写)"
    )
    ref_audio: Optional[str] = Field(default=None, description="参考音频URL/base64")
    ref_text: Optional[str] = Field(default=None, description="参考文本")
    voice_design_prompt: Optional[str] = Field(default=None, description="音色设计描述（voice_design模式使用）")
    instruct_text: Optional[str] = Field(default=None, description="指令控制文本，用于控制语音风格")
    streaming: bool = Field(default=False, description="是否使用流式生成")
    x_vector_only_mode: bool = Field(default=False, description="是否仅使用说话人嵌入模式（voice_clone）")


class OpenVoiceRequest(BaseTTSRequest):
    """OpenVoice 请求"""
    style: str = Field(default="default", description="风格: default, whispering")
    speed: float = Field(default=1.0, description="语速")
    speaker: str = Field(default="default", description="说话人")


class GPTSoVITSRequest(BaseTTSRequest):
    """GPT-SoVITS 请求"""
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


class VoxCPMRequest(BaseTTSRequest):
    """VoxCPM 请求"""
    mode: str = Field(default="base", description="模式: base, voice_design, clone, ultimate_clone")
    voice_design_prompt: Optional[str] = Field(default=None, description="音色设计描述（voice_design模式使用）")
    ref_audio_path: Optional[str] = Field(default=None, description="参考音频路径（clone模式使用）")
    ref_text: Optional[str] = Field(default=None, description="参考文本（ultimate_clone模式使用）")
    cfg_value: float = Field(default=2.0, description="CFG值")
    inference_timesteps: int = Field(default=10, description="推理步数")


class IndexTTSRequest(BaseTTSRequest):
    """IndexTTS 请求"""
    prompt_wav: Optional[str] = Field(default=None, description="参考音频路径")
    emotion_text: Optional[str] = Field(default=None, description="情感描述文本")
    duration_tokens: Optional[int] = Field(default=None, description="时长控制token数")
    mode: str = Field(default="free", description="模式: free(自由), controlled(可控)")
    clone_speaker_id: Optional[str] = Field(default=None, description="说话人ID，用于从说话人管理模块获取参考音频")


class FireRedTTS2Request(BaseTTSRequest):
    """FireRedTTS2 请求"""
    mode: str = Field(default="clone", description="模式: clone(克隆), random(随机音色)")
    text_list: Optional[str] = Field(default=None, description="对话文本列表，JSON格式")
    prompt_wav_list: Optional[str] = Field(default=None, description="参考音频路径列表，JSON格式")
    prompt_text_list: Optional[str] = Field(default=None, description="参考文本列表，JSON格式")
    temperature: float = Field(default=0.9, description="温度")
    topk: int = Field(default=30, description="Top K采样")


class TTSResponse(BaseModel):
    """TTS 响应"""
    success: bool
    message: str
    audio_url: Optional[str] = None
    audio_base64: Optional[str] = None
    sample_rate: Optional[int] = None


class Qwen3TTSModelStatus(BaseModel):
    """Qwen3-TTS 模型状态"""
    available: bool
    model_size: str
    model_type: str
    transformers_version: str
    meets_requirement: bool
    message: str


class BatchTTSRequest(BaseModel):
    """批量 TTS 请求"""
    model: str = Field(default="chattts", description="TTS模型名称")
    tasks: List[dict] = Field(default=[], description="任务列表")


class BatchGenerateRequest(BaseModel):
    """批量生成（抽卡）请求"""
    text: str = Field(..., description="要合成的文本")
    model: str = Field(default="voxcpm", description="TTS模型名称: voxcpm, qwen3tts")
    mode: str = Field(default="base", description="生成模式")
    count: int = Field(default=10, ge=2, le=20, description="生成数量(2-20)")
    speaker_id: Optional[str] = Field(default=None, description="说话人ID(克隆模式使用)")
    voice_design_prompt: Optional[str] = Field(default=None, description="音色设计描述")
    control_prompt: Optional[str] = Field(default=None, description="控制指令")


class BatchGenerateResponse(BaseModel):
    """批量生成（抽卡）响应"""
    success: bool
    message: str
    model: str
    count: int
    audio_urls: List[str] = Field(default=[], description="生成的音频URL列表")
    audio_files: List[str] = Field(default=[], description="音频文件名列表")


class BatchDownloadRequest(BaseModel):
    """批量下载请求"""
    files: List[str] = Field(..., description="要下载的文件名列表")
    zip_name: Optional[str] = Field(default=None, description="ZIP包名称")
