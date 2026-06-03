#!/usr/bin/env python3
"""
VersTTS API 测试脚本
用于测试各个TTS模型的API接口
"""

import requests
import sys
import os
import argparse

API_BASE = "http://localhost:8000"
NON_INTERACTIVE = False

def test_health():
    """测试健康检查接口"""
    print("\n" + "=" * 50)
    print("测试: 健康检查")
    print("=" * 50)
    try:
        response = requests.get(f"{API_BASE}/health", timeout=5)
        print(f"状态码: {response.status_code}")
        print(f"响应: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"错误: {e}")
        return False

def test_root():
    """测试根路径"""
    print("\n" + "=" * 50)
    print("测试: 根路径")
    print("=" * 50)
    try:
        response = requests.get(f"{API_BASE}/", timeout=5)
        print(f"状态码: {response.status_code}")
        content_type = response.headers.get('Content-Type', '')
        print(f"Content-Type: {content_type}")
        
        if 'application/json' in content_type:
            print(f"响应: {response.json()}")
        else:
            print(f"响应: HTML 页面 ({len(response.text)} 字符)")
        
        return response.status_code == 200
    except Exception as e:
        print(f"错误: {e}")
        return False

def test_chattts():
    """测试ChatTTS接口"""
    print("\n" + "=" * 50)
    print("测试: ChatTTS")
    print("=" * 50)
    try:
        data = {
            "text": "你好,这是ChatTTS的测试。",
            "temperature": 0.3,
            "output_format": "url"
        }
        response = requests.post(f"{API_BASE}/tts/chattts", data=data, timeout=120)
        print(f"状态码: {response.status_code}")
        result = response.json()
        print(f"响应: {result}")
        return result.get("success", False)
    except Exception as e:
        print(f"错误: {e}")
        return False

def test_cosyvoice():
    """测试CosyVoice接口"""
    print("\n" + "=" * 50)
    print("测试: CosyVoice")
    print("=" * 50)
    try:
        data = {
            "text": "你好,这是CosyVoice的测试。",
            "mode": "sft",
            "speaker_id": "中文女",
            "output_format": "url"
        }
        response = requests.post(f"{API_BASE}/tts/cosyvoice", data=data, timeout=120)
        print(f"状态码: {response.status_code}")
        result = response.json()
        print(f"响应: {result}")
        return result.get("success", False)
    except Exception as e:
        print(f"错误: {e}")
        return False

def test_qwen3tts():
    """测试Qwen3-TTS接口"""
    print("\n" + "=" * 50)
    print("测试: Qwen3-TTS")
    print("=" * 50)
    try:
        data = {
            "text": "你好,这是Qwen3-TTS的测试。",
            "model_size": "0.6B",  # 使用小模型加快测试
            "mode": "base",
            "output_format": "url"
        }
        response = requests.post(f"{API_BASE}/tts/qwen3tts", data=data, timeout=120)
        print(f"状态码: {response.status_code}")
        result = response.json()
        print(f"响应: {result}")
        return result.get("success", False)
    except Exception as e:
        print(f"错误: {e}")
        return False

def test_openvoice():
    """测试OpenVoice接口"""
    print("\n" + "=" * 50)
    print("测试: OpenVoice")
    print("=" * 50)
    try:
        data = {
            "text": "你好,这是OpenVoice的测试。",
            "language": "zh",
            "style": "default",
            "speed": 1.0,
            "output_format": "url"
        }
        response = requests.post(f"{API_BASE}/tts/openvoice", data=data, timeout=120)
        print(f"状态码: {response.status_code}")
        result = response.json()
        print(f"响应: {result}")
        return result.get("success", False)
    except Exception as e:
        print(f"错误: {e}")
        return False

def test_f5tts():
    """测试F5-TTS接口 (需要参考音频)"""
    print("\n" + "=" * 50)
    print("测试: F5-TTS (需要参考音频)")
    print("=" * 50)
    print("跳过: F5-TTS需要上传参考音频文件")
    return None

def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("VersTTS API 测试")
    print("=" * 60)

    results = {}

    # 基础测试
    results["health"] = test_health()
    results["root"] = test_root()

    # TTS模型测试 (需要模型已加载)
    print("\n" + "=" * 60)
    print("注意: 以下测试需要服务已启动并加载模型")
    print("=" * 60)

    # 提示用户确认（非交互模式跳过）
    if not NON_INTERACTIVE:
        response = input("\n是否继续测试TTS接口? (y/n): ")
        if response.lower() != 'y':
            print("跳过TTS测试")
            return results

    results["chattts"] = test_chattts()
    results["cosyvoice"] = test_cosyvoice()
    results["qwen3tts"] = test_qwen3tts()
    results["openvoice"] = test_openvoice()
    results["f5tts"] = test_f5tts()

    # 打印结果汇总
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    for name, result in results.items():
        status = "✅ 通过" if result else ("⏭️ 跳过" if result is None else "❌ 失败")
        print(f"{name:15} {status}")

    passed = sum(1 for r in results.values() if r)
    total = sum(1 for r in results.values() if r is not None)
    print(f"\n总计: {passed}/{total} 通过")

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VersTTS API 测试")
    parser.add_argument("--non-interactive", action="store_true", help="非交互模式（自动执行所有测试）")
    parser.add_argument("--base-url", type=str, default="http://localhost:8000", help="API基础URL")
    args, unknown = parser.parse_known_args()

    NON_INTERACTIVE = args.non_interactive
    API_BASE = args.base_url
    if unknown:
        API_BASE = unknown[0]

    print(f"API地址: {API_BASE}")
    run_all_tests()
