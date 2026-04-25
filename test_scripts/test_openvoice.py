#!/usr/bin/env python3
# OpenVoice 试用脚本

import os
import sys
import torch
import soundfile as sf

# 添加项目路径
PROJECT_ROOT = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(PROJECT_ROOT, '..', 'algorithms', 'OpenVoice'))

from openvoice import se_extractor
from openvoice.api import BaseSpeakerTTS, ToneColorConverter

def main():
    print("=== OpenVoice 试用脚本 ===")

    # 配置
    output_dir = "output_openvoice"  # 输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 设备配置
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    # 模型路径（V1版本）
    # V1: checkpoints_v1/checkpoints/base_speakers/EN
    # V2: checkpoints_v2/checkpoints_v2/base_speakers (结构不同)
    ckpt_base_en = "../algorithms/OpenVoice/checkpoints_v1/checkpoints/base_speakers/EN"
    ckpt_base_zh = "../algorithms/OpenVoice/checkpoints_v1/checkpoints/base_speakers/ZH"
    ckpt_converter = "../algorithms/OpenVoice/checkpoints_v1/checkpoints/converter"

    # 检查模型文件
    if not os.path.exists(ckpt_base_en):
        print(f"英文基础模型不存在: {ckpt_base_en}")
        print("请确保已下载OpenVoice V1模型")
        # 尝试使用V2版本
        print("尝试使用V2版本...")
        ckpt_base_en = "../algorithms/OpenVoice/checkpoints_v2/checkpoints_v2/base_speakers"
        ckpt_base_zh = "../algorithms/OpenVoice/checkpoints_v2/checkpoints_v2/base_speakers"
        ckpt_converter = "../algorithms/OpenVoice/checkpoints_v2/checkpoints_v2/converter"
        if not os.path.exists(ckpt_base_en):
            print("V2版本也不存在，请重新下载模型")
            return

    # 初始化模型
    print("初始化基础说话人TTS...")
    base_speaker_tts = BaseSpeakerTTS(f'{ckpt_base_en}/config.json', device=device)
    base_speaker_tts.load_ckpt(f'{ckpt_base_en}/checkpoint.pth')

    print("初始化音色转换器...")
    tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
    tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')
    print("模型加载完成！")

    # 获取源音色嵌入
    print("获取源音色嵌入...")
    # V1版本使用不同的音色文件命名
    if os.path.exists(f'{ckpt_base_en}/en_default_se.pth'):
        source_se_en = torch.load(f'{ckpt_base_en}/en_default_se.pth').to(device)
        source_se_zh = torch.load(f'{ckpt_base_zh}/zh_default_se.pth').to(device)
    elif os.path.exists(f'{ckpt_base_en}/ses/en-default.pth'):
        source_se_en = torch.load(f'{ckpt_base_en}/ses/en-default.pth').to(device)
        source_se_zh = torch.load(f'{ckpt_base_zh}/ses/zh.pth').to(device)
    else:
        print("无法找到音色嵌入文件")
        return

    # 参考音频（可以替换为本地音频文件）
    # 注意：需要提供一个实际的参考音频文件
    reference_speaker = "https://example.com/reference_audio.mp3"  # 替换为实际的参考音频

    # 提取目标音色嵌入
    print(f"提取参考音频音色: {reference_speaker}")
    try:
        target_se, audio_name = se_extractor.get_se(
            reference_speaker,
            tone_color_converter,
            target_dir='processed',
            vad=True
        )
        print(f"音色提取成功: {audio_name}")
    except Exception as e:
        print(f"音色提取失败: {e}")
        print("使用默认音色进行测试...")
        target_se = source_se_en

    # 测试文本
    test_texts = {
        "English": [
            "Hello, this is a test example for OpenVoice.",
            "OpenVoice can clone voices in multiple languages."
        ],
        "Chinese": [
            "你好，这是OpenVoice的测试示例。",
            "OpenVoice可以克隆多种语言的声音。"
        ]
    }

    # 测试不同风格
    styles = [
        ("default", "默认风格"),
        ("whispering", "低语风格"),
    ]

    # 语音合成
    print("\n=== 语音合成 ===")

    # 英文测试
    print("\n1. 英文测试:")
    for i, text in enumerate(test_texts["English"]):
        for style_name, style_desc in styles:  # 测试两种风格
            print(f"合成文本: {text}")
            print(f"风格: {style_desc}")

            try:
                # 生成基础语音
                src_path = f'{output_dir}/tmp.wav'
                base_speaker_tts.tts(
                    text,
                    src_path,
                    speaker=style_name,
                    language='English',
                    speed=0.9
                )
                current_source_se = source_se_en if style_name == "default" else source_se_en

                # 转换音色
                save_path = f'{output_dir}/openvoice_en_{style_name}_{i}.wav'
                encode_message = "@MyShell"
                tone_color_converter.convert(
                    audio_src_path=src_path,
                    src_se=current_source_se,
                    tgt_se=target_se,
                    output_path=save_path,
                    message=encode_message
                )
                print(f"保存到: {save_path}")
            except Exception as e:
                print(f"合成失败: {e}")

    # 中文测试
    print("\n2. 中文测试:")
    for i, text in enumerate(test_texts["Chinese"]):
        print(f"合成文本: {text}")

        try:
            # 生成基础语音
            src_path = f'{output_dir}/tmp.wav'
            base_speaker_tts.tts(
                text,
                src_path,
                speaker='default',
                language='Chinese',
                speed=1.0
            )

            # 转换音色
            save_path = f'{output_dir}/openvoice_zh_{i}.wav'
            encode_message = "@MyShell"
            tone_color_converter.convert(
                audio_src_path=src_path,
                src_se=source_se_zh,
                tgt_se=target_se,
                output_path=save_path,
                message=encode_message
            )
            print(f"保存到: {save_path}")
        except Exception as e:
            print(f"合成失败: {e}")

    print("\n=== 测试完成 ===")
    print(f"所有输出文件保存在: {output_dir}")
    print("注意：如果某些测试失败，请检查参考音频路径是否正确")

if __name__ == "__main__":
    main()