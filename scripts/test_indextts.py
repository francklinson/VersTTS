#!/usr/bin/env python3
"""
IndexTTS 基础功能测试脚本
测试模式: free (自由生成), controlled (可控生成)
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


def test_free(speaker_id: str) -> bool:
    """测试自由生成模式"""
    print("\n--- IndexTTS: free模式 ---")
    if not speaker_id:
        print("  ⏭️  跳过: 无可用说话人")
        return None
    try:
        data = {
            "text": "你好，这是IndexTTS自由生成模式测试。",
            "mode": "free",
            "clone_speaker_id": speaker_id,
            "output_format": "url",
        }
        resp = requests.post(f"{BASE_URL}/tts/indextts", data=data, timeout=TTS_TIMEOUT)
        result = resp.json()
        ok = result.get("success")
        print(f"  {'✅' if ok else '❌'} free: {result.get('message', result.get('detail', ''))}")
        return ok
    except Exception as e:
        print(f"  ❌ free: {e}")
        return False


def test_controlled(speaker_id: str) -> bool:
    """测试可控生成模式（情感控制）"""
    print("\n--- IndexTTS: controlled模式 ---")
    if not speaker_id:
        print("  ⏭️  跳过: 无可用说话人")
        return None
    try:
        data = {
            "text": "今天是个好日子，我感到非常开心和满足。",
            "mode": "controlled",
            "clone_speaker_id": speaker_id,
            "emo_text": "开心",
            "output_format": "url",
        }
        resp = requests.post(f"{BASE_URL}/tts/indextts", data=data, timeout=TTS_TIMEOUT)
        result = resp.json()
        ok = result.get("success")
        print(f"  {'✅' if ok else '❌'} controlled: {result.get('message', result.get('detail', ''))}")
        return ok
    except Exception as e:
        print(f"  ❌ controlled: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="IndexTTS 功能测试")
    parser.add_argument("--quick", action="store_true", help="仅测试free模式")
    args = parser.parse_args()

    print("=" * 50)
    print("  IndexTTS 功能测试")
    print("=" * 50)
    print(f"  API: {BASE_URL}")

    speaker_id = get_speaker()
    print(f"  说话人: {speaker_id or '未找到'}")

    results = {"free": test_free(speaker_id)}

    if not args.quick:
        results["controlled"] = test_controlled(speaker_id)

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
