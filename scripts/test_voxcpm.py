#!/usr/bin/env python3
"""
VoxCPM 基础功能测试脚本
测试模式: base, clone, voice_design, ultimate_clone
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


def call_voxcpm(mode: str, text: str, **kwargs) -> dict:
    """调用VoxCPM API"""
    data = {"text": text, "mode": mode, "output_format": "url"}
    data.update(kwargs)
    resp = requests.post(f"{BASE_URL}/tts/voxcpm", data=data, timeout=TTS_TIMEOUT)
    return resp.json()


def test_base() -> bool:
    """测试Base模式（基础生成）"""
    print("\n--- VoxCPM: base模式 ---")
    try:
        data = call_voxcpm("base", "你好，这是VoxCPM基础模式测试。",
                           cfg_value=2.0, inference_timesteps=10)
        ok = data.get("success")
        print(f"  {'✅' if ok else '❌'} base: {data.get('message', data.get('detail', ''))}")
        return ok
    except Exception as e:
        print(f"  ❌ base: {e}")
        return False


def test_voice_design() -> bool:
    """测试Voice Design模式（音色设计）"""
    print("\n--- VoxCPM: voice_design模式 ---")
    try:
        data = call_voxcpm("voice_design", "你好，这是音色设计测试。",
                           voice_design_prompt="成熟稳重的男声，播音腔",
                           cfg_value=2.0, inference_timesteps=10)
        ok = data.get("success")
        print(f"  {'✅' if ok else '❌'} voice_design: {data.get('message', data.get('detail', ''))}")
        return ok
    except Exception as e:
        print(f"  ❌ voice_design: {e}")
        return False


def test_clone(speaker_id: str) -> bool:
    """测试Clone模式（声音克隆）"""
    print("\n--- VoxCPM: clone模式 ---")
    if not speaker_id:
        print("  ⏭️  跳过: 无可用说话人")
        return None
    try:
        data = call_voxcpm("clone", "这是声音克隆测试。",
                           clone_speaker_id=speaker_id,
                           cfg_value=2.0, inference_timesteps=10)
        ok = data.get("success")
        print(f"  {'✅' if ok else '❌'} clone: {data.get('message', data.get('detail', ''))}")
        return ok
    except Exception as e:
        print(f"  ❌ clone: {e}")
        return False


def test_ultimate_clone(speaker_id: str) -> bool:
    """测试Ultimate Clone模式（极致克隆）"""
    print("\n--- VoxCPM: ultimate_clone模式 ---")
    if not speaker_id:
        print("  ⏭️  跳过: 无可用说话人")
        return None
    try:
        data = call_voxcpm("ultimate_clone", "这是极致克隆测试。",
                           clone_speaker_id=speaker_id,
                           cfg_value=2.0, inference_timesteps=10)
        ok = data.get("success")
        print(f"  {'✅' if ok else '❌'} ultimate_clone: {data.get('message', data.get('detail', ''))}")
        return ok
    except Exception as e:
        print(f"  ❌ ultimate_clone: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="VoxCPM 功能测试")
    parser.add_argument("--quick", action="store_true", help="仅测试base模式")
    args = parser.parse_args()

    print("=" * 50)
    print("  VoxCPM 功能测试")
    print("=" * 50)
    print(f"  API: {BASE_URL}")

    speaker_id = get_speaker()
    print(f"  说话人: {speaker_id or '未找到'}")

    results = {"base": test_base()}

    if not args.quick:
        results["voice_design"] = test_voice_design()
        results["clone"] = test_clone(speaker_id)
        results["ultimate_clone"] = test_ultimate_clone(speaker_id)

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
