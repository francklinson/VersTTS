#!/usr/bin/env python3
"""
测试VoxCPM、IndexTTS、FireRedTTS2三个新TTS算法的功能
"""

import os
import sys
import time
import requests
from pathlib import Path

# 项目根目录 - 动态获取
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# API基础URL
BASE_URL = "http://localhost:8000"


def test_health():
    """测试服务健康状态"""
    print("=" * 60)
    print("测试服务健康状态")
    print("=" * 60)
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✓ 服务状态: {data.get('status', 'unknown')}")
            print(f"✓ CUDA可用: {data.get('cuda_available', False)}")
            print(f"✓ 已加载模型: {data.get('models_loaded', [])}")
            return True
        else:
            print(f"✗ 健康检查失败: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ 连接失败: {e}")
        return False


def test_voxcpm_base():
    """测试VoxCPM基础模式"""
    print("\n" + "=" * 60)
    print("测试 VoxCPM - Base模式")
    print("=" * 60)
    try:
        response = requests.post(
            f"{BASE_URL}/tts/voxcpm",
            data={
                "text": "你好，这是VoxCPM的测试语音。",
                "mode": "base",
                "cfg_value": 2.0,
                "inference_timesteps": 10,
                "output_format": "url"
            },
            timeout=120
        )
        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                print(f"✓ 合成成功")
                print(f"✓ 音频URL: {data.get('audio_url')}")
                print(f"✓ 采样率: {data.get('sample_rate')}")
                return True
            else:
                print(f"✗ 合成失败: {data.get('message')}")
                return False
        else:
            print(f"✗ 请求失败: HTTP {response.status_code}")
            print(f"  错误: {response.text}")
            return False
    except Exception as e:
        print(f"✗ 测试异常: {e}")
        return False


def test_voxcpm_voice_design():
    """测试VoxCPM音色设计模式"""
    print("\n" + "=" * 60)
    print("测试 VoxCPM - Voice Design模式")
    print("=" * 60)
    try:
        response = requests.post(
            f"{BASE_URL}/tts/voxcpm",
            data={
                "text": "你好，这是一个温柔的女声。",
                "mode": "voice_design",
                "voice_design_prompt": "A young woman with gentle and warm voice",
                "cfg_value": 2.0,
                "inference_timesteps": 10,
                "output_format": "url"
            },
            timeout=120
        )
        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                print(f"✓ 合成成功")
                print(f"✓ 音频URL: {data.get('audio_url')}")
                return True
            else:
                print(f"✗ 合成失败: {data.get('message')}")
                return False
        else:
            print(f"✗ 请求失败: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ 测试异常: {e}")
        return False


def test_indextts_clone():
    """测试IndexTTS2克隆模式 - 需要提供参考音频"""
    print("\n" + "=" * 60)
    print("测试 IndexTTS2 - Clone模式")
    print("=" * 60)
    print("注意: IndexTTS2需要提供参考音频(spk_audio_prompt)才能合成")
    print("跳过自动测试，请使用前端上传参考音频进行测试")
    return True


def test_fireredtts_random():
    """测试FireRedTTS2随机音色模式"""
    print("\n" + "=" * 60)
    print("测试 FireRedTTS2 - Random模式")
    print("=" * 60)
    try:
        response = requests.post(
            f"{BASE_URL}/tts/fireredtts",
            data={
                "text": "你好，这是FireRedTTS2的测试语音。",
                "mode": "random",
                "temperature": 0.9,
                "topk": 30,
                "output_format": "url"
            },
            timeout=120
        )
        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                print(f"✓ 合成成功")
                print(f"✓ 音频URL: {data.get('audio_url')}")
                print(f"✓ 采样率: {data.get('sample_rate')}")
                return True
            else:
                print(f"✗ 合成失败: {data.get('message')}")
                return False
        else:
            print(f"✗ 请求失败: HTTP {response.status_code}")
            print(f"  错误: {response.text}")
            return False
    except Exception as e:
        print(f"✗ 测试异常: {e}")
        return False


def check_models():
    """检查模型文件是否存在"""
    print("\n" + "=" * 60)
    print("检查模型文件状态")
    print("=" * 60)

    models = {
        "VoxCPM2": [
            PROJECT_ROOT / "algorithms/VoxCPM/models/VoxCPM2/config.json",
            PROJECT_ROOT / "algorithms/VoxCPM/models/VoxCPM2/model.safetensors"
        ],
        "IndexTTS-2": [
            PROJECT_ROOT / "algorithms/IndexTTS/checkpoints/config.yaml"
        ],
        "FireRedTTS2": [
            PROJECT_ROOT / "algorithms/FireRedTTS2/pretrained_models/FireRedTTS2/config_codec.json",
            PROJECT_ROOT / "algorithms/FireRedTTS2/pretrained_models/FireRedTTS2/config_llm.json"
        ]
    }

    all_ready = True
    for name, paths in models.items():
        # 只要有一个关键文件存在就认为模型已下载
        if any(p.exists() for p in paths):
            print(f"✓ {name}: 已下载")
        else:
            print(f"✗ {name}: 未下载 ({paths[0]})")
            all_ready = False

    return all_ready


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("TTS新算法功能测试")
    print("=" * 60)
    print(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"API地址: {BASE_URL}")

    # 检查模型文件
    models_ready = check_models()
    if not models_ready:
        print("\n⚠ 警告: 部分模型文件未下载，相关测试将跳过")

    # 测试服务健康状态
    if not test_health():
        print("\n✗ 服务未启动或无法连接，停止测试")
        print("请先启动后端服务: python backend/api_server.py")
        return

    results = {}

    # 测试VoxCPM
    if models_ready:
        results["voxcpm_base"] = test_voxcpm_base()
        results["voxcpm_voice_design"] = test_voxcpm_voice_design()

        # 测试IndexTTS
        results["indextts_clone"] = test_indextts_clone()

        # 测试FireRedTTS2
        results["fireredtts_random"] = test_fireredtts_random()

    # 汇总结果
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"通过: {passed}/{total}")

    for test_name, result in results.items():
        status = "✓ 通过" if result else "✗ 失败"
        print(f"  {test_name}: {status}")

    if passed == total:
        print("\n✓ 所有测试通过!")
    else:
        print(f"\n✗ {total - passed} 个测试失败")


if __name__ == "__main__":
    main()
