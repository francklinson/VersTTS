#!/usr/bin/env python3
"""
音频处理工具函数
"""

import os
import base64
from datetime import datetime
from typing import Optional

import numpy as np
import soundfile as sf

from backend.logger_config import OperationLogger


def normalize_audio_volume(audio_data: np.ndarray, target_db: float = -0.5) -> np.ndarray:
    """
    归一化音频音量到目标dB级别
    
    Args:
        audio_data: 输入音频数组
        target_db: 目标dB级别，默认-0.5 dB（接近最大音量）
    
    Returns:
        归一化后的音频数组
    """
    # 确保音频是float32类型
    if audio_data.dtype != np.float32:
        audio_data = audio_data.astype(np.float32)

    # 计算当前峰值
    current_peak = np.max(np.abs(audio_data))

    if current_peak == 0:
        return audio_data  # 避免除零

    # 计算目标峰值（从dB转换为线性比例）
    target_peak = 10 ** (target_db / 20.0)

    # 计算增益因子
    gain = target_peak / current_peak

    # 应用增益
    normalized_audio = audio_data * gain

    # 确保不会溢出（硬限幅）
    normalized_audio = np.clip(normalized_audio, -1.0, 1.0)

    return normalized_audio


def save_temp_audio(audio_data: np.ndarray, sample_rate: int,
                    suffix: str = ".wav", normalize: bool = True,
                    prefix: str = "tts", mode: str = None,
                    text: str = None, speaker_name: str = None) -> str:
    """
    保存临时音频文件

    Args:
        audio_data: 音频数据数组
        sample_rate: 采样率
        suffix: 文件后缀
        normalize: 是否进行音量归一化，默认True
        prefix: 文件名前缀，默认"tts"
        mode: 生成模式（可选，用于更有意义的文件名）
        text: 合成文本（可选，用于提取文本摘要到文件名）
        speaker_name: 说话人名称（可选）
    """
    import re
    from backend.config import OUTPUTS_DIR
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 生成有意义的文件名
    if mode or text or speaker_name:
        parts = [prefix]
        if mode:
            parts.append(mode)
        if speaker_name:
            clean_name = re.sub(r'[^\w一-鿿]', '', speaker_name)[:6]
            if clean_name:
                parts.append(clean_name)
        if text:
            text_summary = text[:8].strip()
            text_summary = re.sub(r'[^\w一-鿿]', '', text_summary)
            if text_summary:
                parts.append(text_summary)
        parts.append(timestamp)
        filename = "_".join(parts) + suffix
    else:
        filename = f"{prefix}_{timestamp}{suffix}"

    temp_path = os.path.join(OUTPUTS_DIR, filename)

    # 音量归一化处理
    if normalize:
        audio_data = normalize_audio_volume(audio_data)

    sf.write(temp_path, audio_data, sample_rate)

    # 记录文件操作
    audio_size = os.path.getsize(temp_path)
    OperationLogger.log_file_operation("保存音频", temp_path, audio_size, "成功")

    return temp_path


def audio_to_base64(audio_path: str) -> str:
    """将音频文件转为base64"""
    with open(audio_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")
