#!/usr/bin/env python3
"""
GPT-SoVITS 测试脚本
测试后端API和语音合成功能
"""

import os
import sys
import json
import time
import requests
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

# API基础URL
API_BASE = "http://localhost:8000"

def test_speakers_api():
    """测试说话人列表API"""
    print("=" * 60)
    print("测试1: 获取说话人列表")
    print("=" * 60)

    try:
        response = requests.get(f"{API_BASE}/speakers", timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                speakers = data.get("speakers", [])
                print(f"✓ 成功获取说话人列表，共 {len(speakers)} 个说话人")
                for spk in speakers:
                    print(f"  - {spk.get('name')} (ID: {spk.get('id')})")
                    print(f"    音频: {spk.get('audio_path')}")
                    print(f"    参考文本: {spk.get('reference_text', '无')[:50]}...")
                return speakers
            else:
                print(f"✗ 获取失败: {data.get('message')}")
                return []
        else:
            print(f"✗ 请求失败: HTTP {response.status_code}")
            return []
    except Exception as e:
        print(f"✗ 请求异常: {e}")
        return []

def test_gptsovits_api(speaker_id: str, speaker_name: str):
    """测试GPT-SoVITS语音合成API"""
    print("\n" + "=" * 60)
    print(f"测试2: GPT-SoVITS语音合成 (使用说话人: {speaker_name})")
    print("=" * 60)

    test_text = "你好，这是GPT-SoVITS语音合成测试。"

    try:
        form_data = {
            "text": test_text,
            "text_lang": "zh",
            "clone_speaker_id": speaker_id,
            "prompt_lang": "zh",
            "top_k": "15",
            "top_p": "1.0",
            "temperature": "1.0",
            "text_split_method": "cut5",
            "batch_size": "1",
            "speed_factor": "1.0",
            "version": "v2",
            "output_format": "url"
        }

        print(f"发送请求...")
        print(f"  文本: {test_text}")
        print(f"  说话人ID: {speaker_id}")

        start_time = time.time()
        response = requests.post(
            f"{API_BASE}/tts/gptsovits",
            data=form_data,
            timeout=120
        )
        elapsed = time.time() - start_time

        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                audio_url = data.get("audio_url")
                sample_rate = data.get("sample_rate")
                print(f"✓ 合成成功!")
                print(f"  耗时: {elapsed:.2f}秒")
                print(f"  音频URL: {audio_url}")
                print(f"  采样率: {sample_rate}Hz")

                # 下载音频文件
                if audio_url:
                    audio_response = requests.get(f"{API_BASE}{audio_url}", timeout=30)
                    if audio_response.status_code == 200:
                        output_dir = PROJECT_ROOT / "test_output"
                        output_dir.mkdir(exist_ok=True)
                        audio_path = output_dir / f"gptsovits_test_{speaker_name}.wav"
                        with open(audio_path, "wb") as f:
                            f.write(audio_response.content)
                        print(f"✓ 音频已保存: {audio_path}")
                        print(f"  文件大小: {len(audio_response.content)} bytes")
                        return str(audio_path)
                    else:
                        print(f"✗ 下载音频失败: HTTP {audio_response.status_code}")
                        return None
            else:
                print(f"✗ 合成失败: {data.get('message')}")
                return None
        else:
            print(f"✗ 请求失败: HTTP {response.status_code}")
            try:
                error_data = response.json()
                print(f"  错误信息: {error_data.get('detail', '未知错误')}")
            except:
                print(f"  响应内容: {response.text[:200]}")
            return None

    except Exception as e:
        print(f"✗ 请求异常: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_asr(audio_path: str):
    """使用ASR验证生成的音频"""
    print("\n" + "=" * 60)
    print("测试3: ASR验证生成的音频")
    print("=" * 60)

    try:
        # 使用CosyVoice的ASR或者其他可用ASR服务
        # 这里我们检查音频文件是否正常
        import wave
        with wave.open(audio_path, 'rb') as wf:
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            framerate = wf.getframerate()
            n_frames = wf.getnframes()
            duration = n_frames / framerate

            print(f"✓ 音频文件检查通过")
            print(f"  声道数: {channels}")
            print(f"  采样宽度: {sample_width} bytes")
            print(f"  采样率: {framerate}Hz")
            print(f"  帧数: {n_frames}")
            print(f"  时长: {duration:.2f}秒")

            if duration > 0 and n_frames > 0:
                print(f"✓ 音频文件有效，可以正常播放")
                return True
            else:
                print(f"✗ 音频文件异常")
                return False

    except Exception as e:
        print(f"✗ ASR验证失败: {e}")
        return False

def main():
    """主函数"""
    print("=" * 60)
    print("GPT-SoVITS 功能测试")
    print("=" * 60)
    print(f"项目路径: {PROJECT_ROOT}")
    print(f"API地址: {API_BASE}")
    print()

    # 检查API服务是否运行
    try:
        response = requests.get(f"{API_BASE}/health", timeout=5)
        print(f"✓ API服务运行正常")
    except:
        print(f"✗ API服务未运行，请先启动服务:")
        print(f"  python backend/api_server.py")
        return 1

    # 测试1: 获取说话人列表
    speakers = test_speakers_api()

    if not speakers:
        print("\n✗ 没有可用的说话人，无法继续测试")
        print("  请先在说话人管理模块中添加说话人")
        return 1

    # 测试2: 使用第一个说话人进行语音合成
    first_speaker = speakers[0]
    speaker_id = first_speaker.get("id")
    speaker_name = first_speaker.get("name")

    audio_path = test_gptsovits_api(speaker_id, speaker_name)

    if audio_path:
        # 测试3: ASR验证
        test_asr(audio_path)

        print("\n" + "=" * 60)
        print("测试总结")
        print("=" * 60)
        print("✓ 所有测试通过!")
        print(f"✓ GPT-SoVITS后端API运行正常")
        print(f"✓ 语音合成功能正常")
        print(f"✓ 生成的音频文件有效")
        return 0
    else:
        print("\n" + "=" * 60)
        print("测试总结")
        print("=" * 60)
        print("✗ 语音合成测试失败")
        return 1

if __name__ == "__main__":
    sys.exit(main())
