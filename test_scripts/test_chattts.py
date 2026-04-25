#!/usr/bin/env python3
# ChatTTS 试用脚本

import os
import sys
import numpy as np
import soundfile as sf

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'algorithms', 'ChatTTS'))

import ChatTTS

def save_wav_file(wav, filename, sample_rate=24000):
    """保存WAV文件"""
    sf.write(filename, wav, sample_rate)
    print(f"保存到: {filename}")

def main():
    print("=== ChatTTS 试用脚本 ===")
    
    # 配置
    output_dir = "output_chattts"  # 输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化ChatTTS
    print("初始化ChatTTS...")
    chat = ChatTTS.Chat()
    
    # 加载模型
    print("加载模型...")
    is_loaded = chat.load(source="local")
    if not is_loaded:
        print("模型加载失败！")
        return
    print("模型加载成功！")
    
    # 采样随机说话人
    print("采样随机说话人...")
    spk = chat.sample_random_speaker()
    print(f"使用说话人: {spk}")
    
    # 测试文本
    test_texts = [
        "你好，这是ChatTTS的测试示例。",
        "Hello, this is a test example for ChatTTS.",
        "ChatTTS 支持情感表达，例如：你好啊！[laughter] 哈哈哈。",
        "今天天气真好，适合出去散步。",
        "The quick brown fox jumps over the lazy dog."
    ]
    
    # 语音合成
    print("\n=== 语音合成 ===")
    for i, text in enumerate(test_texts):
        print(f"合成文本: {text}")
        
        try:
            # 合成音频
            wav = chat.infer(
                [text],
                stream=False,
                params_infer_code=ChatTTS.Chat.InferCodeParams(
                    spk_emb=spk,
                ),
            )[0]
            
            # 保存音频
            output_file = os.path.join(output_dir, f"chattts_{i}.wav")
            save_wav_file(wav, output_file)
        except Exception as e:
            print(f"合成失败: {e}")
    
    # 测试不同说话人
    print("\n=== 测试不同说话人 ===")
    for i in range(3):
        spk = chat.sample_random_speaker()
        print(f"说话人 {i+1}: {spk}")
        
        try:
            wav = chat.infer(
                ["你好，我是不同的说话人。"],
                stream=False,
                params_infer_code=ChatTTS.Chat.InferCodeParams(
                    spk_emb=spk,
                ),
            )[0]
            
            output_file = os.path.join(output_dir, f"chattts_speaker_{i}.wav")
            save_wav_file(wav, output_file)
        except Exception as e:
            print(f"合成失败: {e}")
    
    print("\n=== 测试完成 ===")
    print(f"所有输出文件保存在: {output_dir}")

if __name__ == "__main__":
    main()