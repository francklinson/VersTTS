#!/usr/bin/env python3
"""
F5-TTS API 测试脚本 - 使用说话人管理模块
测试 /tts/f5tts 端点，使用 clone_speaker_id 参数
"""

import requests
import json
import os
from pathlib import Path

API_BASE = "http://localhost:8000"

def get_speakers():
    """获取说话人列表"""
    try:
        response = requests.get(f"{API_BASE}/speakers")
        if response.ok:
            data = response.json()
            if data.get("success"):
                return data.get("speakers", [])
    except Exception as e:
        print(f"获取说话人列表失败: {e}")
    return []

def test_f5tts_with_speaker():
    """测试 F5-TTS 使用说话人管理模块"""
    print("=== F5-TTS API 测试 (使用说话人管理模块) ===\n")

    # 1. 获取说话人列表
    print("1. 获取说话人列表...")
    speakers = get_speakers()

    if not speakers:
        print("⚠️ 暂无可用说话人，请先添加说话人")
        print("   可通过前端说话人管理页面添加")
        return False

    print(f"✓ 找到 {len(speakers)} 个说话人:")
    for spk in speakers:
        ref_text = spk.get('reference_text', '')
        ref_text_short = ref_text[:30] + '...' if ref_text and len(ref_text) > 30 else ref_text
        print(f"   - {spk['name']} (ID: {spk['id']})")
        print(f"     参考文本: {ref_text_short or '无'}")

    # 2. 选择第一个有参考文本的说话人
    speaker = None
    for spk in speakers:
        if spk.get('reference_text'):
            speaker = spk
            break

    if not speaker:
        speaker = speakers[0]
        print(f"\n⚠️ 警告: 选择的说话人没有参考文本，可能导致合成失败")

    print(f"\n2. 选择说话人: {speaker['name']}")
    print(f"   ID: {speaker['id']}")
    print(f"   音频路径: {speaker.get('audio_path', 'N/A')}")
    print(f"   参考文本: {speaker.get('reference_text', 'N/A')[:50]}...")

    # 3. 测试语音合成
    print("\n3. 测试语音合成...")

    test_texts = [
        "你好，这是F5-TTS使用说话人管理模块的测试。",
        "F5-TTS现在支持从说话人管理模块中选择参考音频进行克隆。",
    ]

    output_dir = Path("output_f5tts_test")
    output_dir.mkdir(exist_ok=True)

    success_count = 0

    for i, text in enumerate(test_texts):
        print(f"\n   测试 {i+1}/{len(test_texts)}: {text[:30]}...")

        try:
            form_data = {
                'gen_text': text,
                'clone_speaker_id': speaker['id'],
                'nfe_step': '32',
                'cfg_strength': '2.0',
                'speed': '1.0',
                'cross_lingual': 'false',
            }

            response = requests.post(f"{API_BASE}/tts/f5tts", data=form_data)

            if response.ok:
                data = response.json()
                if data.get("success"):
                    print(f"   ✓ 合成成功")
                    print(f"     音频URL: {data.get('audio_url', 'N/A')}")
                    success_count += 1
                else:
                    print(f"   ✗ 合成失败: {data.get('message', '未知错误')}")
            else:
                print(f"   ✗ 请求失败: HTTP {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"     错误: {error_data.get('detail', '未知错误')}")
                except:
                    print(f"     响应: {response.text[:200]}")

        except Exception as e:
            print(f"   ✗ 异常: {e}")

    # 4. 测试结果汇总
    print(f"\n=== 测试结果 ===")
    print(f"总测试数: {len(test_texts)}")
    print(f"成功数: {success_count}")
    print(f"失败数: {len(test_texts) - success_count}")

    if success_count == len(test_texts):
        print("\n✓ 所有测试通过！")
        return True
    else:
        print(f"\n⚠️ 部分测试失败")
        return False

def test_f5tts_no_speaker():
    """测试 F5-TTS 不提供说话人ID时的默认行为"""
    print("\n=== 测试默认参考音频 ===\n")

    test_text = "这是一个使用默认参考音频的测试。"

    try:
        form_data = {
            'gen_text': test_text,
            'nfe_step': '32',
            'cfg_strength': '2.0',
        }

        print(f"测试文本: {test_text}")
        response = requests.post(f"{API_BASE}/tts/f5tts", data=form_data)

        if response.ok:
            data = response.json()
            if data.get("success"):
                print("✓ 使用默认参考音频合成成功")
                return True
            else:
                print(f"✗ 合成失败: {data.get('message', '未知错误')}")
        else:
            print(f"✗ 请求失败: HTTP {response.status_code}")

    except Exception as e:
        print(f"✗ 异常: {e}")

    return False

if __name__ == "__main__":
    print("="*60)
    print("F5-TTS API 测试脚本")
    print("测试内容: 使用说话人管理模块进行语音克隆")
    print("="*60)

    # 检查服务是否运行
    try:
        response = requests.get(f"{API_BASE}/health", timeout=5)
        if response.ok:
            print("\n✓ 后端服务运行正常")
        else:
            print("\n✗ 后端服务未正常运行")
            exit(1)
    except Exception as e:
        print(f"\n✗ 无法连接到后端服务: {e}")
        print("请确保后端服务已启动: python backend/api_server.py")
        exit(1)

    # 运行测试
    test_f5tts_with_speaker()
    test_f5tts_no_speaker()

    print("\n" + "="*60)
    print("测试完成")
    print("="*60)
