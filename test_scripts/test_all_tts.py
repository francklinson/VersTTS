#!/usr/bin/env python3
"""
VersTTS 统一TTS功能测试脚本
测试所有六个TTS项目的基础功能
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALGORITHMS_DIR = os.path.join(PROJECT_ROOT, "algorithms")


def run_test(script_name: str, description: str) -> bool:
    """运行单个测试脚本"""
    print("\n" + "=" * 60)
    print(f"测试项目: {description}")
    print("=" * 60)

    script_path = os.path.join(PROJECT_ROOT, "test_scripts", script_name)
    if not os.path.exists(script_path):
        print(f"❌ 测试脚本不存在: {script_path}")
        return False

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=PROJECT_ROOT,
            capture_output=False,
            timeout=300,
        )
        success = result.returncode == 0
        if success:
            print(f"✅ {description} 测试通过")
        else:
            print(f"❌ {description} 测试失败 (返回码: {result.returncode})")
        return success
    except subprocess.TimeoutExpired:
        print(f"⏱️ {description} 测试超时")
        return False
    except Exception as e:
        print(f"❌ {description} 测试异常: {e}")
        return False


def test_api() -> bool:
    """测试API服务"""
    print("\n" + "=" * 60)
    print("测试项目: VersTTS API 服务")
    print("=" * 60)

    script_path = os.path.join(PROJECT_ROOT, "test_scripts", "test_api.py")
    if not os.path.exists(script_path):
        print(f"❌ 测试脚本不存在: {script_path}")
        return False

    try:
        # API测试需要交互，使用子进程运行
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=PROJECT_ROOT,
            capture_output=False,
            timeout=60,
        )
        success = result.returncode == 0
        if success:
            print("✅ API 测试通过")
        else:
            print(f"❌ API 测试失败 (返回码: {result.returncode})")
        return success
    except subprocess.TimeoutExpired:
        print("⏱️ API 测试超时")
        return False
    except Exception as e:
        print(f"❌ API 测试异常: {e}")
        return False


def test_environment() -> bool:
    """测试基础环境"""
    print("\n" + "=" * 60)
    print("测试项目: 基础环境检查")
    print("=" * 60)

    checks = []

    # 检查Python版本
    py_version = sys.version_info
    checks.append(("Python版本", f"{py_version.major}.{py_version.minor}.{py_version.micro}", py_version >= (3, 10)))

    # 检查CUDA
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        cuda_info = torch.cuda.get_device_name(0) if cuda_available else "不可用"
        checks.append(("CUDA可用", cuda_info, cuda_available))
    except ImportError:
        checks.append(("PyTorch", "未安装", False))

    # 检查关键依赖
    packages = [
        ("numpy", "numpy"),
        ("soundfile", "soundfile"),
        ("fastapi", "fastapi"),
        ("uvicorn", "uvicorn"),
        ("transformers", "transformers"),
        ("librosa", "librosa"),
    ]

    for name, module in packages:
        try:
            __import__(module)
            checks.append((name, "已安装", True))
        except ImportError:
            checks.append((name, "未安装", False))

    # 检查项目目录结构
    dirs_to_check = [
        ("algorithms/ChatTTS", os.path.join(ALGORITHMS_DIR, "ChatTTS")),
        ("algorithms/CosyVoice", os.path.join(ALGORITHMS_DIR, "CosyVoice")),
        ("algorithms/F5-TTS", os.path.join(ALGORITHMS_DIR, "F5-TTS")),
        ("algorithms/GPT-SoVITS", os.path.join(ALGORITHMS_DIR, "GPT-SoVITS")),
        ("algorithms/OpenVoice", os.path.join(ALGORITHMS_DIR, "OpenVoice")),
        ("algorithms/Qwen3-TTS", os.path.join(ALGORITHMS_DIR, "Qwen3-TTS")),
    ]

    for name, path in dirs_to_check:
        exists = os.path.exists(path)
        checks.append((name, "存在" if exists else "不存在", exists))

    # 打印结果
    all_pass = True
    for name, value, passed in checks:
        status = "✅" if passed else "❌"
        print(f"{status} {name}: {value}")
        if not passed:
            all_pass = False

    return all_pass


def main():
    parser = argparse.ArgumentParser(description="VersTTS 统一TTS功能测试")
    parser.add_argument("--env-only", action="store_true", help="只测试环境")
    parser.add_argument("--api-only", action="store_true", help="只测试API")
    parser.add_argument("--skip-api", action="store_true", help="跳过API测试")
    args = parser.parse_args()

    print("=" * 60)
    print("VersTTS 统一TTS功能测试")
    print("=" * 60)
    print(f"项目根目录: {PROJECT_ROOT}")
    print(f"算法目录: {ALGORITHMS_DIR}")

    results = {}

    # 环境测试
    results["environment"] = test_environment()

    if args.env_only:
        print("\n" + "=" * 60)
        print("仅执行环境测试")
        print("=" * 60)
        sys.exit(0 if results["environment"] else 1)

    # 各个TTS项目测试
    tts_tests = [
        ("test_chattts.py", "ChatTTS"),
        ("test_cosyvoice.py", "CosyVoice"),
        ("test_f5_tts.py", "F5-TTS"),
        ("test_openvoice.py", "OpenVoice"),
        ("test_qwen3_tts.py", "Qwen3-TTS"),
    ]

    if not args.api_only:
        for script, name in tts_tests:
            results[name.lower().replace("-", "")] = run_test(script, name)

    # API测试
    if not args.skip_api:
        results["api"] = test_api()

    # 汇总
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)

    for name, passed in results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{name:20} {status}")

    total = len(results)
    passed = sum(1 for v in results.values() if v)
    print(f"\n总计: {passed}/{total} 通过")

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
