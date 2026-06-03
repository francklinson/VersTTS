#!/usr/bin/env python3
"""
VersTTS 全量TTS功能测试脚本
测试所有TTS项目的基础功能，包括 PilotTTS / VoxCPM / IndexTTS / FireRedTTS2 / OmniVoice
"""

import os
import sys
import json
import subprocess
import argparse
import time
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")
ALGORITHMS_DIR = os.path.join(PROJECT_ROOT, "algorithms")

# 运行测试超时时间（秒）
DEFAULT_TIMEOUT = 300
SHORT_TIMEOUT = 60


def run_script(script_name: str, description: str, timeout: int = DEFAULT_TIMEOUT, extra_args: list = None) -> bool:
    """运行单个测试脚本（仅限Python）"""
    print("\n" + "=" * 60)
    print(f"测试: {description}")
    print("=" * 60)

    script_path = os.path.join(SCRIPTS_DIR, script_name)
    if not os.path.exists(script_path):
        print(f"⚠️  测试脚本不存在: {script_path}")
        return False

    try:
        cmd = [sys.executable, script_path]
        if extra_args:
            cmd.extend(extra_args)

        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            capture_output=False,
            timeout=timeout,
        )
        success = result.returncode == 0
        status = "✅ 通过" if success else f"❌ 失败 (返回码: {result.returncode})"
        print(f"\n{status} | {description}")
        return success
    except subprocess.TimeoutExpired:
        print(f"\n⏱️  超时 | {description}")
        return False
    except Exception as e:
        print(f"\n❌ 异常 | {description}: {e}")
        return False


def test_environment() -> bool:
    """基础环境检查"""
    print("\n" + "=" * 60)
    print("测试: 基础环境检查")
    print("=" * 60)

    checks = []

    # Python版本
    py_version = sys.version_info
    checks.append(("Python版本", f"{py_version.major}.{py_version.minor}.{py_version.micro}", py_version >= (3, 10)))

    # CUDA
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        cuda_info = torch.cuda.get_device_name(0) if cuda_available else "不可用"
        checks.append(("CUDA", cuda_info, cuda_available))
    except ImportError:
        checks.append(("PyTorch", "未安装", False))

    # 关键依赖
    packages = [
        ("numpy", "numpy"), ("soundfile", "soundfile"), ("fastapi", "fastapi"),
        ("uvicorn", "uvicorn"), ("transformers", "transformers"), ("librosa", "librosa"),
        ("requests", "requests"),
    ]
    for name, module in packages:
        try:
            __import__(module)
            checks.append((name, "已安装", True))
        except ImportError:
            checks.append((name, "未安装", False))

    # 项目目录结构
    dirs_to_check = [
        ("algorithms/ChatTTS", os.path.join(ALGORITHMS_DIR, "ChatTTS")),
        ("algorithms/CosyVoice", os.path.join(ALGORITHMS_DIR, "CosyVoice")),
        ("algorithms/F5-TTS", os.path.join(ALGORITHMS_DIR, "F5-TTS")),
        ("algorithms/GPT-SoVITS", os.path.join(ALGORITHMS_DIR, "GPT-SoVITS")),
        ("algorithms/OpenVoice", os.path.join(ALGORITHMS_DIR, "OpenVoice")),
        ("algorithms/Qwen3-TTS", os.path.join(ALGORITHMS_DIR, "Qwen3-TTS")),
        ("algorithms/VoxCPM", os.path.join(ALGORITHMS_DIR, "VoxCPM")),
        ("algorithms/IndexTTS", os.path.join(ALGORITHMS_DIR, "IndexTTS")),
        ("algorithms/FireRedTTS2", os.path.join(ALGORITHMS_DIR, "FireRedTTS2")),
        ("algorithms/OmniVoice", os.path.join(ALGORITHMS_DIR, "OmniVoice")),
        ("algorithms/PilotTTS", os.path.join(ALGORITHMS_DIR, "PilotTTS")),
    ]

    for name, path in dirs_to_check:
        exists = os.path.exists(path)
        checks.append((name, "存在" if exists else "不存在", exists))

    # 模型文件检查
    model_dirs = {
        "models/Qwen3-TTS": os.path.join(PROJECT_ROOT, "models", "Qwen3-TTS"),
        "models/VoxCPM": os.path.join(PROJECT_ROOT, "models", "VoxCPM"),
        "models/CosyVoice": os.path.join(PROJECT_ROOT, "models", "CosyVoice"),
        "models/OmniVoice": os.path.join(PROJECT_ROOT, "models", "OmniVoice"),
        "models/PilotTTS": os.path.join(PROJECT_ROOT, "models", "PilotTTS"),
    }
    for name, path in model_dirs.items():
        if os.path.exists(path):
            size = sum(os.path.getsize(os.path.join(dirpath, filename))
                       for dirpath, dirnames, filenames in os.walk(path)
                       for filename in filenames) / (1024**3)
            checks.append((name, f"{size:.1f}GB", True))
        else:
            checks.append((name, "目录不存在", False))

    # 打印结果
    all_pass = True
    for name, value, passed in checks:
        status = "✅" if passed else "❌"
        print(f"  {status} {name}: {value}")
        if not passed:
            all_pass = False

    return all_pass


def test_standalone_services() -> dict:
    """检查独立服务状态"""
    import requests

    print("\n" + "=" * 60)
    print("测试: 独立服务状态检查")
    print("=" * 60)

    services = {
        "PilotTTS": ("127.0.0.1", 8003),
        "CosyVoice": ("127.0.0.1", 8002),
        "OmniVoice": ("127.0.0.1", 8001),
    }

    results = {}
    for name, (host, port) in services.items():
        try:
            response = requests.get(f"http://{host}:{port}/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                print(f"  ✅ {name} (端口 {port}): 运行中 - {json.dumps(data, ensure_ascii=False)}")
                results[name] = True
            else:
                print(f"  ⚠️  {name} (端口 {port}): HTTP {response.status_code}")
                results[name] = False
        except Exception as e:
            print(f"  ⚠️  {name} (端口 {port}): 未运行 ({e})")
            results[name] = False

    return results


def test_main_service() -> bool:
    """检查主API服务"""
    import requests

    print("\n" + "=" * 60)
    print("测试: 主API服务状态")
    print("=" * 60)

    # 尝试多个端口
    for port in [8000, 18800]:
        try:
            response = requests.get(f"http://localhost:{port}/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                print(f"  ✅ 主服务 (端口 {port}): 运行中")
                print(f"     模型: {data.get('models_loaded', [])}")
                print(f"     CUDA: {data.get('cuda_available', False)}")
                # 记录下来供后续测试使用
                os.environ["VERS_TTS_PORT"] = str(port)
                return True
        except:
            pass

    print(f"  ❌ 主服务未运行 (已尝试端口 8000, 18800)")
    return False


def get_api_port() -> int:
    """获取API端口"""
    return int(os.environ.get("VERS_TTS_PORT", "8000"))


def test_tts_endpoint(name: str, endpoint: str, data: dict, timeout: int = 120) -> bool:
    """通用TTS API端点测试"""
    import requests
    port = get_api_port()
    url = f"http://localhost:{port}{endpoint}"

    print(f"\n--- {name} ---")
    print(f"  URL: POST {url}")
    print(f"  参数: {json.dumps({k: str(v)[:50] for k, v in data.items()}, ensure_ascii=False)}")

    try:
        response = requests.post(url, data=data, timeout=timeout)
        result = response.json()
        if response.status_code == 200 and result.get("success"):
            audio_url = result.get("audio_url", "N/A")
            print(f"  ✅ 成功 | 音频: {audio_url}")
            return True
        else:
            error = result.get("detail", result.get("message", "未知错误"))
            print(f"  ❌ 失败 (HTTP {response.status_code}): {error}")
            return False
    except requests.exceptions.Timeout:
        print(f"  ⏱️  超时 ({timeout}s)")
        return False
    except requests.exceptions.ConnectionError:
        print(f"  ❌ 无法连接服务")
        return False
    except Exception as e:
        print(f"  ❌ 异常: {e}")
        return False


def test_pilottts_all_modes() -> dict:
    """测试PilotTTS全部4种模式"""
    import requests
    port = get_api_port()

    print("\n" + "=" * 60)
    print("测试: PilotTTS (全部模式)")
    print("=" * 60)

    # 动态获取说话人
    speaker_id = None
    try:
        response = requests.get(f"http://localhost:{port}/speakers/list", timeout=10)
        if response.status_code == 200:
            speakers = response.json()
            if speakers:
                speaker_id = speakers[0].get("id", speakers[0].get("speaker_id"))
                print(f"  自动选择说话人: {speaker_id}")
    except Exception as e:
        print(f"  ⚠️  获取说话人列表失败: {e}")

    if not speaker_id:
        # 尝试从文件读取
        db_file = os.path.join(PROJECT_ROOT, "speakers", "speakers_db.json")
        if os.path.exists(db_file):
            try:
                with open(db_file) as f:
                    db = json.load(f)
                    speakers = db.get("speakers", [])
                    if speakers:
                        speaker_id = speakers[0].get("id")
                        print(f"  从数据库选择说话人: {speaker_id}")
            except:
                pass

    if not speaker_id:
        print("  ❌ 无可用说话人，跳过PilotTTS测试")
        return {"speaker_available": False}

    results = {}
    test_cases = [
        ("voice_clone", "今天天气真不错，适合出门散步。", {"mode": "voice_clone", "clone_speaker_id": speaker_id}),
        ("emotion_happy", "今天真是太开心了！阳光真好！", {"mode": "emotion", "emotion": "happy", "clone_speaker_id": speaker_id}),
        ("emotion_sad", "今天心情不太好，下雨了。", {"mode": "emotion", "emotion": "sad", "clone_speaker_id": speaker_id}),
        ("dialect", "中不中啊，咱俩一块儿去吃胡辣汤吧。", {"mode": "dialect", "language": "zh-henan", "clone_speaker_id": speaker_id}),
        ("paralanguage", "这个笑话太好笑了我真的忍不住。", {"mode": "paralanguage", "clone_speaker_id": speaker_id}),
    ]

    for mode_key, text, params in test_cases:
        params["text"] = text
        params["output_format"] = "url"
        results[mode_key] = test_tts_endpoint(
            f"PilotTTS-{mode_key}", "/tts/pilottts/", params, timeout=300
        )

    return results


def test_pilottts_task_queue(speaker_id: str = None) -> bool:
    """测试PilotTTS任务队列"""
    import requests
    port = get_api_port()

    print("\n" + "=" * 60)
    print("测试: PilotTTS - 任务队列")
    print("=" * 60)

    if not speaker_id:
        try:
            response = requests.get(f"http://localhost:{port}/speakers/list", timeout=10)
            if response.status_code == 200:
                speakers = response.json()
                if speakers:
                    speaker_id = speakers[0].get("id", speakers[0].get("speaker_id"))
        except:
            pass

    if not speaker_id:
        print("  ❌ 无可用说话人，跳过")
        return False

    try:
        response = requests.post(
            f"http://localhost:{port}/tasks/submit",
            data={
                "text": "任务队列测试：PilotTTS批量生成。",
                "model": "pilottts",
                "mode": "voice_clone",
                "speaker_id": speaker_id,
            },
            timeout=30,
        )
        data = response.json()
        if not data.get("success"):
            print(f"  ❌ 任务提交失败: {data.get('detail')}")
            return False

        task_id = data.get("task_id")
        print(f"  ✅ 任务已提交: {task_id}")

        # 轮询等待任务完成（最多3分钟）
        for i in range(60):
            time.sleep(3)
            status_resp = requests.get(
                f"http://localhost:{port}/tasks/{task_id}/status", timeout=10
            )
            status_data = status_resp.json()
            status = status_data.get("status", "unknown")
            print(f"    轮询 {i+1}/60: {status}")
            if status == "completed":
                print(f"  ✅ 任务队列完成: {status_data.get('audio_url')}")
                return True
            elif status == "failed":
                print(f"  ❌ 任务失败: {status_data.get('error_message')}")
                return False

        print("  ⏱️  任务超时")
        return False
    except Exception as e:
        print(f"  ❌ 异常: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="VersTTS 全量TTS功能测试")
    parser.add_argument("--env-only", action="store_true", help="仅测试环境")
    parser.add_argument("--pilottts-only", action="store_true", help="仅测试PilotTTS")
    parser.add_argument("--skip-heavy", action="store_true", help="跳过耗时测试（模型加载类）")
    parser.add_argument("--service-only", action="store_true", help="仅测试服务状态")
    args = parser.parse_args()

    print("=" * 70)
    print("  VersTTS 全量TTS功能测试")
    print("=" * 70)
    print(f"  时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  项目根目录: {PROJECT_ROOT}")

    results = {}

    # ==================== 环境测试 ====================
    results["environment"] = test_environment()

    if args.env_only:
        print("\n" + "=" * 60)
        print("仅执行环境测试，退出。")
        sys.exit(0 if results["environment"] else 1)

    if args.service_only:
        svc_results = test_standalone_services()
        main_ok = test_main_service()
        print("\n" + "=" * 60)
        print("服务状态汇总")
        print("=" * 60)
        for name, ok in svc_results.items():
            print(f"  {'✅' if ok else '❌'} {name}")
        print(f"  {'✅' if main_ok else '❌'} 主API服务")
        sys.exit(0 if main_ok else 1)

    # ==================== 服务状态 ====================
    results["services"] = test_standalone_services()
    main_ok = test_main_service()

    if not main_ok:
        print("\n" + "=" * 60)
        print("⚠️  主API服务未运行，跳过API测试")
        print("   请先启动服务: bash start_server.sh start")
        print("=" * 60)
        sys.exit(1)

    # ==================== PilotTTS 全模式测试 ====================
    print("\n" + "=" * 70)
    print("  PilotTTS 全模式功能测试")
    print("=" * 70)
    pilottts_results = test_pilottts_all_modes()
    speaker_avail = pilottts_results.pop("speaker_available", True)
    results["pilottts"] = pilottts_results

    if speaker_avail:
        results["pilottts_task_queue"] = {"task_queue": test_pilottts_task_queue()}

    if args.pilottts_only:
        print("\n" + "=" * 60)
        print("仅执行PilotTTS测试，退出。")
        all_pass = all(
            v for r in [results["pilottts"], results.get("pilottts_task_queue", {})]
            for v in (r.values() if isinstance(r, dict) else [r])
        )
        sys.exit(0 if all_pass else 1)

    # ==================== 其他TTS项目测试 ====================
    if not args.skip_heavy:
        # 统一TTS功能测试（ChatTTS/CosyVoice/F5-TTS/OpenVoice/Qwen3-TTS等）
        tts_tests = [
            ("test_chattts.py", "ChatTTS"),
            ("test_cosyvoice.py", "CosyVoice"),
            ("test_f5_tts.py", "F5-TTS"),
            ("test_openvoice.py", "OpenVoice"),
            ("test_qwen3_tts.py", "Qwen3-TTS"),
            ("test_gptsovits.py", "GPT-SoVITS"),
        ]
        for script, name in tts_tests:
            results[f"tts_{name.lower()}"] = {"api": run_script(script, name)}

        # VoxCPM / IndexTTS / FireRedTTS2 模式测试
        if os.path.exists(os.path.join(SCRIPTS_DIR, "test_all_modes.py")):
            results["tts_voxcpm_indextts_fireredtts"] = {
                "modes": run_script("test_all_modes.py", "VoxCPM/IndexTTS/FireRedTTS2 模式测试", timeout=180)
            }

    # ==================== API基础测试 ====================
    if os.path.exists(os.path.join(SCRIPTS_DIR, "test_api.py")):
        results["api"] = {"health": run_script("test_api.py", "API基础测试", timeout=SHORT_TIMEOUT, extra_args=["--non-interactive"])}

    # ==================== 结果汇总 ====================
    print("\n" + "=" * 70)
    print("  全量测试结果汇总")
    print("=" * 70)

    total_passed = 0
    total_tests = 0

    def count_results(r, prefix=""):
        nonlocal total_passed, total_tests
        if isinstance(r, dict):
            for k, v in r.items():
                count_results(v, f"{prefix}{k}.")
        elif isinstance(r, bool):
            total_tests += 1
            if r:
                total_passed += 1

    for category, items in results.items():
        print(f"\n  📂 {category}:")
        if isinstance(items, dict):
            for name, result in items.items():
                if isinstance(result, dict):
                    for sub_name, sub_result in result.items():
                        status = "✅" if sub_result else "❌"
                        print(f"     {status} {sub_name}")
                        total_tests += 1
                        if sub_result:
                            total_passed += 1
                else:
                    status = "✅" if result else "❌"
                    print(f"     {status} {name}")
                    total_tests += 1
                    if result:
                        total_passed += 1
        elif isinstance(items, bool):
            status = "✅" if items else "❌"
            print(f"     {status} 全部通过")

    print(f"\n  ────────────────────────")
    print(f"  总计: {total_passed}/{total_tests} 通过")

    sys.exit(0 if total_passed == total_tests else 1)


if __name__ == "__main__":
    main()
