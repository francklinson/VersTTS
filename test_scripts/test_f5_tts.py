#!/usr/bin/env python3
# F5-TTS 试用脚本

import os
import sys
import numpy as np
import soundfile as sf
from pathlib import Path

# 添加项目路径
PROJECT_ROOT = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(PROJECT_ROOT, '..', 'algorithms', 'F5-TTS'))

from f5_tts.infer.utils_infer import (
    infer_process,
    load_model,
    load_vocoder,
    preprocess_ref_audio_text,
    device as default_device,
    mel_spec_type as default_mel_spec_type,
    nfe_step as default_nfe_step,
    cfg_strength as default_cfg_strength,
    sway_sampling_coef as default_sway_sampling_coef,
    speed as default_speed,
    fix_duration as default_fix_duration,
    cross_fade_duration as default_cross_fade_duration,
    target_rms as default_target_rms,
)
from omegaconf import OmegaConf
from hydra.utils import get_class

def main():
    print("=== F5-TTS 试用脚本 ===")
    
    # 配置
    output_dir = "output_f5_tts"  # 输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 模型配置
    model_name = "F5TTS_Base"
    vocoder_name = "vocos"  # 可选: "vocos" 或 "bigvgan"
    device = default_device  # 默认使用CUDA
    
    # 模型路径
    model_dir = "../algorithms/F5-TTS/models"
    ckpt_file = os.path.join(model_dir, "model_1200000.pt")
    
    # 检查模型文件
    if not os.path.exists(ckpt_file):
        print(f"模型文件不存在: {ckpt_file}")
        print("请确保已下载F5-TTS模型文件")
        return
    
    # 参考音频（可以替换为本地音频文件）
    # 注意：需要提供一个实际的参考音频文件
    ref_audio = "https://example.com/reference_audio.wav"  # 替换为实际的参考音频
    ref_text = "Some call me nature, others call me mother nature."
    
    # 测试文本
    test_texts = [
        "你好，这是F5-TTS的测试示例。",
        "Hello, this is a test example for F5-TTS.",
        "F5-TTS 可以克隆参考音频的音色。"
    ]
    
    # 加载vocoder
    print(f"加载Vocoder: {vocoder_name}")
    if vocoder_name == "vocos":
        vocoder_local_path = "../checkpoints/vocos-mel-24khz"
    elif vocoder_name == "bigvgan":
        vocoder_local_path = "../checkpoints/bigvgan_v2_24khz_100band_256x"
    else:
        vocoder_local_path = ""
    
    vocoder = load_vocoder(
        vocoder_name=vocoder_name, 
        is_local=False, 
        local_path=vocoder_local_path, 
        device=device
    )
    print("Vocoder加载完成！")
    
    # 加载模型配置
    model_cfg = OmegaConf.load(
        os.path.join("..", "algorithms", "F5-TTS", "src", "f5_tts", "configs", f"{model_name}.yaml")
    )
    model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
    model_arc = model_cfg.model.arch
    
    # 加载模型
    print(f"加载模型: {model_name}")
    ema_model = load_model(
        model_cls, 
        model_arc, 
        ckpt_file, 
        mel_spec_type=vocoder_name, 
        vocab_file="", 
        device=device
    )
    print("模型加载完成！")
    
    # 预处理参考音频
    print("预处理参考音频...")
    ref_audio, ref_text = preprocess_ref_audio_text(ref_audio, ref_text)
    print(f"参考音频: {ref_audio}")
    print(f"参考文本: {ref_text}")
    
    # 语音合成
    print("\n=== 语音合成 ===")
    for i, text in enumerate(test_texts):
        print(f"合成文本: {text}")
        
        try:
            # 合成音频
            audio_segment, final_sample_rate, _ = infer_process(
                ref_audio,
                ref_text,
                text,
                ema_model,
                vocoder,
                mel_spec_type=vocoder_name,
                target_rms=default_target_rms,
                cross_fade_duration=default_cross_fade_duration,
                nfe_step=default_nfe_step,
                cfg_strength=default_cfg_strength,
                sway_sampling_coef=default_sway_sampling_coef,
                speed=default_speed,
                fix_duration=default_fix_duration,
                device=device,
            )
            
            # 保存音频
            output_file = os.path.join(output_dir, f"f5_tts_{i}.wav")
            sf.write(output_file, audio_segment, final_sample_rate)
            print(f"保存到: {output_file}")
        except Exception as e:
            print(f"合成失败: {e}")
            print("提示：请确保参考音频路径正确，并且网络连接正常")
    
    print("\n=== 测试完成 ===")
    print(f"所有输出文件保存在: {output_dir}")

if __name__ == "__main__":
    main()