#!/usr/bin/env python3
"""
OmniVoice 基础功能测试脚本
测试模式: auto_voice, voice_clone, voice_design
"""

import os
import sys
import json
import time
import argparse
import requests
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
API_PORT = int(os.environ.get("VERS_TTS_PORT", "8000"))
BASE_URL = f"http://localhost:{API_PORT}"
TTS_TIMEOUT = 300


def get_speaker():
    """动态获取可用说话人"""
    try:
        resp = requests.get(f"{BASE_URL}/speakers/list", timeout=10)
        if resp.status_code == 200:
            speakers = resp.json()
            if speakers and len(speakers) > 0:
                return speakers[0].get("id", speakers[0].get("speaker_id"))
    except:
        pass
    return None


def check_omnivoice_service() -> bool:
    """检查OmniVoice独立服务"""
    try:
        resp = requests.get("http://127.0.0.1:8001/health", timeout=5)
        return resp.status_code == 200
    except:
        return False


def test_auto_voice() -> bool:
    """测试自动音色模式"""
    print("\n--- OmniVoice: auto_voice模式 ---")
    try:
        data = {
            "text": "你好，这是OmniVoice自动音色测试。",
            "mode": "auto_voice",
            "output_format": "url",
        }
        resp = requests.post(f"{BASE_URL}/tts/omnivoice", data=data, timeout=TTS_TIMEOUT)
        result = resp.json()
        ok = result.get("success")
        print(f"  {'✅' if ok else '❌'} auto_voice: {result.get('message', result.get('detail', ''))}")
        return ok
    except Exception as e:
        print(f"  ❌ auto_voice: {e}")
        return False


def test_voice_clone(speaker_id: str) -> bool:
    """测试声音克隆模式"""
    print("\n--- OmniVoice: voice_clone模式 ---")
    if not speaker_id:
        print("  ⏭️  跳过: 无可用说话人")
        return None
    try:
        data = {
            "text": "这是OmniVoice声音克隆测试。",
            "mode": "voice_clone",
            "clone_speaker_id": speaker_id,
            "output_format": "url",
        }
        resp = requests.post(f"{BASE_URL}/tts/omnivoice", data=data, timeout=TTS_TIMEOUT)
        result = resp.json()
        ok = result.get("success")
        print(f"  {'✅' if ok else '❌'} voice_clone: {result.get('message', result.get('detail', ''))}")
        return ok
    except Exception as e:
        print(f"  ❌ voice_clone: {e}")
        return False


def test_voice_design() -> bool:
    """测试声音设计模式"""
    print("\n--- OmniVoice: voice_design模式 ---")
    try:
        data = {
            "text": "你好，这是OmniVoice音色设计测试。",
            "mode": "voice_design",
            "voice_design_prompt": "年轻女性，温柔的声音，普通话",
            "output_format": "url",
        }
        resp = requests.post(f"{BASE_URL}/tts/omnivoice", data=data, timeout=TTS_TIMEOUT)
        result = resp.json()
        ok = result.get("success")
        print(f"  {'✅' if ok else '❌'} voice_design: {result.get('message', result.get('detail', ''))}")
        return ok
    except Exception as e:
        print(f"  ❌ voice_design: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="OmniVoice 功能测试")
    parser.add_argument("--quick", action="store_true", help="仅测试auto_voice模式")
    args = parser.parse_args()

    print("=" * 50)
    print("  OmniVoice 功能测试")
    print("=" * 50)
    print(f"  API: {BASE_URL}")

    svc_ok = check_omnivoice_service()
    print(f"  OmniVoice独立服务: {'✅ 运行中' if svc_ok else '⚠️  未运行'}")

    speaker_id = get_speaker()
    print(f"  说话人: {speaker_id or '未找到'}")

    if not svc_ok:
        print("\n  ⚠️  OmniVoice独立服务未运行，跳过测试")
        sys.exit(2)

    results = {"auto_voice": test_auto_voice()}

    if not args.quick:
        results["voice_clone"] = test_voice_clone(speaker_id)
        results["voice_design"] = test_voice_design()

    # 汇总
    print("\n" + "=" * 50)
    valid = {k: v for k, v in results.items() if v is not None}
    passed = sum(1 for v in valid.values() if v)
    for k, v in results.items():
        if v is None:
            print(f"  ⏭️  {k}")
        else:
            print(f"  {'✅' if v else '❌'} {k}")
    print(f"  通过: {passed}/{len(valid)}")
    sys.exit(0 if passed == len(valid) else 1)


if __name__ == "__main__":
    main()
