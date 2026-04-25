#!/usr/bin/env python3
# CosyVoice 试用脚本

import os
import sys
import torchaudio

# 添加项目路径
PROJECT_ROOT = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(PROJECT_ROOT, '..', 'algorithms', 'CosyVoice'))

from cosyvoice.cli.cosyvoice import AutoModel

def main():
    print("=== CosyVoice 试用脚本 ===")
    
    # 配置
    output_dir = "output_cosyvoice"  # 输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 模型路径
    model_paths = {
        "sft": "../algorithms/CosyVoice/models/iic/CosyVoice-300M-SFT",
        "zero_shot": "../algorithms/CosyVoice/models/iic/CosyVoice-300M",
        "instruct": "../algorithms/CosyVoice/models/iic/CosyVoice-300M-Instruct",
        "cosyvoice2": "../algorithms/CosyVoice/models/iic/CosyVoice2-0.5B"
    }
    
    # 1. SFT 模型测试
    print("\n=== 1. SFT 模型测试 ===")
    try:
        cosyvoice = AutoModel(model_dir=model_paths["sft"])
        print("可用的说话人:")
        print(cosyvoice.list_available_spks())
        
        # 测试文本
        test_texts = [
            "你好，这是CosyVoice SFT模型的测试示例。",
            "Hello, this is a test example for CosyVoice SFT model."
        ]
        
        for i, text in enumerate(test_texts):
            print(f"合成文本: {text}")
            # 选择说话人
            if any(char >= '\u4e00' and char <= '\u9fff' for char in text):
                speaker = "中文女"
            else:
                speaker = "English"
            
            for j, result in enumerate(cosyvoice.inference_sft(text, speaker, stream=False)):
                output_file = os.path.join(output_dir, f"sft_{i}_{j}.wav")
                torchaudio.save(output_file, result['tts_speech'], cosyvoice.sample_rate)
                print(f"保存到: {output_file}")
    except Exception as e:
        print(f"SFT模型测试失败: {e}")
    
    # 2. Zero-shot 模型测试
    print("\n=== 2. Zero-shot 模型测试 ===")
    try:
        cosyvoice = AutoModel(model_dir=model_paths["zero_shot"])
        
        # 测试文本
        text = "收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。"
        prompt_text = "希望你以后能够做的比我还好呦。"
        # 注意：需要提供一个参考音频文件
        prompt_audio = "https://example.com/reference_audio.wav"  # 替换为实际的参考音频
        
        print(f"合成文本: {text}")
        for i, result in enumerate(cosyvoice.inference_zero_shot(text, prompt_text, prompt_audio)):
            output_file = os.path.join(output_dir, f"zero_shot_{i}.wav")
            torchaudio.save(output_file, result['tts_speech'], cosyvoice.sample_rate)
            print(f"保存到: {output_file}")
    except Exception as e:
        print(f"Zero-shot模型测试失败: {e}")
        print("提示：请确保参考音频路径正确")
    
    # 3. Instruct 模型测试
    print("\n=== 3. Instruct 模型测试 ===")
    try:
        cosyvoice = AutoModel(model_dir=model_paths["instruct"])
        
        # 测试文本
        text = "在面对挑战时，他展现了非凡的<strong>勇气</strong>与<strong>智慧</strong>。"
        speaker = "中文男"
        prompt = "Theo 'Crimson', is a fiery, passionate rebel leader. Fights with fervor for justice, but struggles with impulsiveness.<|endofprompt|>"
        
        print(f"合成文本: {text}")
        for i, result in enumerate(cosyvoice.inference_instruct(text, speaker, prompt)):
            output_file = os.path.join(output_dir, f"instruct_{i}.wav")
            torchaudio.save(output_file, result['tts_speech'], cosyvoice.sample_rate)
            print(f"保存到: {output_file}")
    except Exception as e:
        print(f"Instruct模型测试失败: {e}")
    
    # 4. CosyVoice2 模型测试
    print("\n=== 4. CosyVoice2 模型测试 ===")
    try:
        cosyvoice = AutoModel(model_dir=model_paths["cosyvoice2"])
        
        # 测试文本
        text = "收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。"
        prompt_text = "希望你以后能够做的比我还好呦。"
        prompt_audio = "https://example.com/reference_audio.wav"  # 替换为实际的参考音频
        
        print(f"合成文本: {text}")
        for i, result in enumerate(cosyvoice.inference_zero_shot(text, prompt_text, prompt_audio)):
            output_file = os.path.join(output_dir, f"cosyvoice2_{i}.wav")
            torchaudio.save(output_file, result['tts_speech'], cosyvoice.sample_rate)
            print(f"保存到: {output_file}")
    except Exception as e:
        print(f"CosyVoice2模型测试失败: {e}")
    
    print("\n=== 测试完成 ===")
    print(f"所有输出文件保存在: {output_dir}")
    print("注意：如果某些测试失败，请检查模型路径和参考音频是否正确")

if __name__ == "__main__":
    main()