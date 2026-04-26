#!/usr/bin/env python3
# CosyVoice 试用脚本

import os
import sys
import torchaudio

# 添加项目路径
PROJECT_ROOT = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(PROJECT_ROOT, '..', 'algorithms', 'CosyVoice'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, '..', 'algorithms', 'CosyVoice', 'third_party', 'Matcha-TTS'))

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
        "cosyvoice2": "../algorithms/CosyVoice/models/iic/CosyVoice2-0.5B",
        "cosyvoice3": "../algorithms/CosyVoice/models/iic/Fun-CosyVoice3-0.5B"
    }
    
    # 参考音频路径（使用项目内置资源）
    prompt_audio_path = "../algorithms/CosyVoice/asset/zero_shot_prompt.wav"
    
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
        import traceback
        traceback.print_exc()
    
    # 2. Zero-shot 模型测试 (CosyVoice-300M)
    print("\n=== 2. Zero-shot 模型测试 (CosyVoice-300M) ===")
    try:
        cosyvoice = AutoModel(model_dir=model_paths["zero_shot"])
        
        text = "收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。"
        prompt_text = "希望你以后能够做的比我还好呦。"
        
        if os.path.exists(prompt_audio_path):
            print(f"合成文本: {text}")
            for i, result in enumerate(cosyvoice.inference_zero_shot(text, prompt_text, prompt_audio_path, stream=False)):
                output_file = os.path.join(output_dir, f"zero_shot_{i}.wav")
                torchaudio.save(output_file, result['tts_speech'], cosyvoice.sample_rate)
                print(f"保存到: {output_file}")
        else:
            print(f"参考音频不存在: {prompt_audio_path}，跳过测试")
    except Exception as e:
        print(f"Zero-shot模型测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 3. Instruct 模型测试
    print("\n=== 3. Instruct 模型测试 ===")
    try:
        cosyvoice = AutoModel(model_dir=model_paths["instruct"])
        
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
        import traceback
        traceback.print_exc()
    
    # 4. CosyVoice2 模型测试
    print("\n=== 4. CosyVoice2 模型测试 ===")
    try:
        cosyvoice = AutoModel(model_dir=model_paths["cosyvoice2"])
        
        text = "收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。"
        prompt_text = "希望你以后能够做的比我还好呦。"
        
        if os.path.exists(prompt_audio_path):
            print(f"合成文本: {text}")
            for i, result in enumerate(cosyvoice.inference_zero_shot(text, prompt_text, prompt_audio_path, stream=False)):
                output_file = os.path.join(output_dir, f"cosyvoice2_{i}.wav")
                torchaudio.save(output_file, result['tts_speech'], cosyvoice.sample_rate)
                print(f"保存到: {output_file}")
        else:
            print(f"参考音频不存在: {prompt_audio_path}，跳过测试")
    except Exception as e:
        print(f"CosyVoice2模型测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 5. CosyVoice3 (Fun-CosyVoice3-0.5B) 模型测试 —— 重点测试
    print("\n=== 5. CosyVoice3 (Fun-CosyVoice3-0.5B) 模型测试 ===")
    try:
        cosyvoice = AutoModel(model_dir=model_paths["cosyvoice3"])
        print(f"模型加载成功，采样率: {cosyvoice.sample_rate}")
        print(f"可用预训练音色: {cosyvoice.list_available_spks()}")
        
        # 5.1 zero_shot 测试（带参考文本）
        if os.path.exists(prompt_audio_path):
            text = "八百标兵奔北坡，北坡炮兵并排跑，炮兵怕把标兵碰，标兵怕碰炮兵炮。"
            prompt_text = "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。"
            print(f"\n5.1 zero_shot 测试 | 合成文本: {text}")
            for i, result in enumerate(cosyvoice.inference_zero_shot(text, prompt_text, prompt_audio_path, stream=False)):
                output_file = os.path.join(output_dir, f"cosyvoice3_zero_shot_{i}.wav")
                torchaudio.save(output_file, result['tts_speech'], cosyvoice.sample_rate)
                print(f"保存到: {output_file}")
        else:
            print(f"参考音频不存在: {prompt_audio_path}，跳过 zero_shot 测试")
        
        # 5.2 cross_lingual 测试
        if os.path.exists(prompt_audio_path):
            text = "You are a helpful assistant.<|endofprompt|>在他们讲述那个故事的过程中，他突然停下来，因为大家也被逗笑了。"
            print(f"\n5.2 cross_lingual 测试 | 合成文本: {text}")
            for i, result in enumerate(cosyvoice.inference_cross_lingual(text, prompt_audio_path, stream=False)):
                output_file = os.path.join(output_dir, f"cosyvoice3_cross_lingual_{i}.wav")
                torchaudio.save(output_file, result['tts_speech'], cosyvoice.sample_rate)
                print(f"保存到: {output_file}")
        else:
            print(f"参考音频不存在: {prompt_audio_path}，跳过 cross_lingual 测试")
        
        # 5.3 instruct2 测试
        if os.path.exists(prompt_audio_path):
            text = "收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。"
            instruct = "You are a helpful assistant. 请用四川话说这句话。<|endofprompt|>"
            print(f"\n5.3 instruct2 测试 | 合成文本: {text}")
            for i, result in enumerate(cosyvoice.inference_instruct2(text, instruct, prompt_audio_path, stream=False)):
                output_file = os.path.join(output_dir, f"cosyvoice3_instruct_{i}.wav")
                torchaudio.save(output_file, result['tts_speech'], cosyvoice.sample_rate)
                print(f"保存到: {output_file}")
        else:
            print(f"参考音频不存在: {prompt_audio_path}，跳过 instruct2 测试")
            
    except Exception as e:
        print(f"CosyVoice3模型测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n=== 测试完成 ===")
    print(f"所有输出文件保存在: {output_dir}")
    print("注意：如果某些测试失败，请检查模型路径和参考音频是否正确")

if __name__ == "__main__":
    main()