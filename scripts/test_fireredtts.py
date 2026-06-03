#!/usr/bin/env python3
"""
FireRedTTS2 基础功能测试脚本
测试模式: random (随机音色), clone (声音克隆)
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


def test_random() -> bool:
    """测试随机音色模式"""
    print("\n--- FireRedTTS2: random模式 ---")
    try:
        data = {
            "text": "你好，这是FireRedTTS2随机音色测试。",
            "mode": "random",
            "temperature": 0.9,
            "topk": 30,
            "output_format": "url",
        }
        resp = requests.post(f"{BASE_URL}/tts/fireredtts", data=data, timeout=TTS_TIMEOUT)
        result = resp.json()
        ok = result.get("success")
        print(f"  {'✅' if ok else '❌'} random: {result.get('message', result.get('detail', ''))}")
        return ok
    except Exception as e:
        print(f"  ❌ random: {e}")
        return False


def test_clone(speaker_id: str) -> bool:
    """测试声音克隆模式"""
    print("\n--- FireRedTTS2: clone模式 ---")
    if not speaker_id:
        print("  ⏭️  跳过: 无可用说话人")
        return None
    try:
        data = {
            "text": "这是FireRedTTS2声音克隆测试。",
            "mode": "clone",
            "clone_speaker_id": speaker_id,
            "output_format": "url",
        }
        resp = requests.post(f"{BASE_URL}/tts/fireredtts", data=data, timeout=TTS_TIMEOUT)
        result = resp.json()
        ok = result.get("success")
        print(f"  {'✅' if ok else '❌'} clone: {result.get('message', result.get('detail', ''))}")
        return ok
    except Exception as e:
        print(f"  ❌ clone: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="FireRedTTS2 功能测试")
    parser.add_argument("--quick", action="store_true", help="仅测试random模式")
    args = parser.parse_args()

    print("=" * 50)
    print("  FireRedTTS2 功能测试")
    print("=" * 50)
    print(f"  API: {BASE_URL}")

    speaker_id = get_speaker()
    print(f"  说话人: {speaker_id or '未找到'}")

    results = {"random": test_random()}

    if not args.quick:
        results["clone"] = test_clone(speaker_id)

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
