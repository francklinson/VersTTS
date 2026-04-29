#!/usr/bin/env python3
# Qwen3-TTS 试用脚本

import os
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

def main():
    print("=== Qwen3-TTS 试用脚本 ===")

    # 检查CUDA可用性
    cuda_available = torch.cuda.is_available()
    print(f"CUDA可用: {cuda_available}")
    if cuda_available:
        print(f"CUDA设备: {torch.cuda.get_device_name(0)}")

    # 配置
    if cuda_available:
        device = "cuda:0"  # 使用GPU
        print("使用GPU模式（flash attention）")
    else:
        device = "cpu"  # 回退到CPU
        print("警告：CUDA不可用，使用CPU模式")

    model_path = "../algorithms/Qwen3-TTS/models/Qwen/Qwen3-TTS-12Hz-1.7B-Base"  # 模型路径
    output_dir = "output_qwen3_tts"  # 输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 初始化模型
    print(f"加载模型: {model_path}")

    try:
        if cuda_available:
            # 使用flash attention加速
            tts = Qwen3TTSModel.from_pretrained(
                model_path,
                device_map=device,
                dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
            )
        else:
            # CPU模式不使用flash attention
            tts = Qwen3TTSModel.from_pretrained(
                model_path,
                device_map=device,
                dtype=torch.bfloat16,
                attn_implementation="eager",
            )
        print("模型加载完成！")
    except Exception as e:
        print(f"模型加载失败: {e}")
        if cuda_available:
            print("尝试回退到CPU模式...")
            device = "cpu"
            tts = Qwen3TTSModel.from_pretrained(
                model_path,
                device_map=device,
                dtype=torch.float32,
                attn_implementation="eager",
            )
            print("模型加载完成（CPU模式）")

    # 测试文本
    test_texts = [
        "你好，这是Qwen3-TTS的测试示例。",
        "Hello, this is a test example for Qwen3-TTS.",
        "Qwen3-TTS 支持多种语言的语音合成。"
    ]

    # 基础语音合成
    print("\n=== 基础语音合成 ===")
    for i, text in enumerate(test_texts):
        print(f"合成文本: {text}")
        wav, sr = tts.generate(text=text, language="Auto")
        output_file = os.path.join(output_dir, f"basic_synthesis_{i}.wav")
        sf.write(output_file, wav, sr)
        print(f"保存到: {output_file}")

    # 语音克隆
    print("\n=== 语音克隆 ===")
    # 参考音频（可以替换为本地音频文件）
    ref_audio = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone_1.wav"
    ref_text = "甚至出现交易几乎停滞的情况。"

    # 克隆语音合成
    clone_texts = [
        "这是使用参考音频克隆的语音。",
        "This is a voice cloned from the reference audio."
    ]

    for i, text in enumerate(clone_texts):
        print(f"克隆合成文本: {text}")
        wav, sr = tts.generate_voice_clone(
            text=text,
            language="Auto",
            ref_audio=ref_audio,
            ref_text=ref_text,
            x_vector_only_mode=False
        )
        output_file = os.path.join(output_dir, f"voice_clone_{i}.wav")
        sf.write(output_file, wav, sr)
        print(f"保存到: {output_file}")

    print("\n=== 测试完成 ===")
    print(f"所有输出文件保存在: {output_dir}")

if __name__ == "__main__":
    main()