#!/usr/bin/env python3
"""
PilotTTS 全功能测试脚本
测试全部4种模式: voice_clone, emotion, dialect, paralanguage
支持动态说话人发现，无需硬编码 speaker_id
"""

import os
import sys
import json
import time
import argparse
import requests
import urllib3
from pathlib import Path

# 禁用SSL警告（自签名证书）
urllib3.disable_warnings()

# 项目根目录 - 动态获取
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

# API基础URL — 自动检测HTTPS/HTTP
_API_PORT = int(os.environ.get("VERS_TTS_PORT", "8000"))
_VERIFY_SSL = False

# 统一会话（禁用SSL验证）— 必须在 _detect_base_url 之前定义
_SESSION = requests.Session()
_SESSION.verify = False


def _detect_base_url() -> str:
    """自动检测API的协议和端口"""
    candidates = [
        (f"https://localhost:{_API_PORT}", _API_PORT),
        (f"http://localhost:{_API_PORT}", _API_PORT),
    ]
    # 也尝试常见端口
    for port in [8008, 8000, 18800]:
        if port != _API_PORT:
            candidates.append((f"https://localhost:{port}", port))
            candidates.append((f"http://localhost:{port}", port))

    for url, port in candidates:
        try:
            resp = _SESSION.get(f"{url}/health", timeout=3)
            if resp.status_code == 200:
                os.environ["VERS_TTS_PORT"] = str(port)
                return url
        except:
            pass
    return f"http://localhost:{_API_PORT}"


BASE_URL = _detect_base_url()
API_PORT = int(os.environ.get("VERS_TTS_PORT", "8000"))
HEALTH_TIMEOUT = 10
TTS_TIMEOUT = 300

# 便捷别名
_GET = _SESSION.get
_POST = _SESSION.post


def print_header(title: str, width: int = 60):
    """打印分隔标题"""
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def get_available_speaker() -> dict:
    """
    动态获取可用说话人
    优先级: API > 本地数据库文件
    """
    # 方式1: 从API获取
    try:
        response = _GET(f"{BASE_URL}/speakers/list", timeout=10)
        if response.status_code == 200:
            speakers = response.json()
            if speakers and isinstance(speakers, list) and len(speakers) > 0:
                speaker = speakers[0]
                return {
                    "id": speaker.get("id", speaker.get("speaker_id")),
                    "name": speaker.get("name", "Unknown"),
                    "audio_path": speaker.get("audio_path"),
                }
    except Exception:
        pass

    # 方式2: 从本地数据库文件获取
    db_path = PROJECT_ROOT / "speakers" / "speakers_db.json"
    if db_path.exists():
        try:
            with open(db_path) as f:
                db = json.load(f)
                speakers = db.get("speakers", [])
                if speakers:
                    speaker = speakers[0]
                    return {
                        "id": speaker.get("id"),
                        "name": speaker.get("name", "Unknown"),
                        "audio_path": speaker.get("audio_path"),
                    }
        except Exception:
            pass

    return {}


def check_services() -> dict:
    """检查所有相关服务状态"""
    print_header("服务状态检查")
    services = {
        "main_api": (API_PORT, "/health"),
        "pilottts": (8003, "/health"),
        "cosyvoice": (8002, "/health"),
        "omnivoice": (8001, "/health"),
    }
    results = {}
    for name, (port, endpoint) in services.items():
        # 主服务使用检测到的协议，独立服务使用HTTP
        if name == "main_api":
            url = f"{BASE_URL}{endpoint}"
        else:
            url = f"http://127.0.0.1:{port}{endpoint}"
        try:
            resp = _SESSION.get(url, timeout=5)
            ok = resp.status_code == 200
            status = "✅" if ok else f"⚠️  HTTP {resp.status_code}"
            print(f"  {status} {name} (端口 {port})")
            results[name] = ok
        except Exception:
            print(f"  ⚠️  {name} (端口 {port}): 未运行")
            results[name] = False
    return results


def call_pilottts_api(mode: str, text: str, speaker_id: str = None,
                      emotion: str = None, language: str = "zh") -> dict:
    """调用PilotTTS主API"""
    data = {
        "text": text,
        "mode": mode,
        "output_format": "url",
    }
    if speaker_id:
        data["clone_speaker_id"] = speaker_id
    if emotion:
        data["emotion"] = emotion
    if language != "zh":
        data["language"] = language

    response = _POST(
        f"{BASE_URL}/tts/pilottts/",
        data=data,
        timeout=TTS_TIMEOUT,
    )
    return response.json()


def test_health():
    """测试主服务健康状态"""
    print_header("主服务健康检查")
    try:
        response = _GET(f"{BASE_URL}/health", timeout=HEALTH_TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            print(f"  ✅ 服务状态: {data.get('status', 'unknown')}")
            print(f"  ✅ CUDA可用: {data.get('cuda_available', False)}")
            print(f"  ✅ 已加载模型: {data.get('models_loaded', [])}")
            return True
        else:
            print(f"  ❌ 健康检查失败: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"  ❌ 连接失败: {e}")
        print(f"  提示: 请先启动主服务 bash start_server.sh start")
        return False


def test_pilottts_service_health():
    """测试PilotTTS独立服务健康状态"""
    print_header("PilotTTS独立服务健康检查")
    try:
        response = _GET("http://127.0.0.1:8003/health", timeout=HEALTH_TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            print(f"  ✅ 状态: {data.get('status', 'unknown')}")
            print(f"  ✅ Base模型已加载: {data.get('base_loaded', False)}")
            print(f"  ✅ Instruct模型已加载: {data.get('instruct_loaded', False)}")
            return True
        else:
            print(f"  ⚠️  HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"  ⚠️  未运行: {e}")
        print(f"  提示: nohup python pilottts_service.py > logs/pilottts_service.log 2>&1 &")
        return False


def test_voice_clone(speaker_id: str, speaker_name: str) -> bool:
    """测试零样本声音克隆"""
    print_header("PilotTTS - voice_clone (零样本声音克隆)", 50)

    text = "今天天气真不错，阳光明媚，微风拂面，非常适合出门散步。"
    print(f"  说话人: {speaker_name} ({speaker_id})")
    print(f"  文本: {text}")

    try:
        data = call_pilottts_api("voice_clone", text, speaker_id=speaker_id)
        if data.get("success"):
            print(f"  ✅ 合成成功")
            print(f"  音频URL: {data.get('audio_url')}")
            print(f"  采样率: {data.get('sample_rate')}Hz")
            return True
        else:
            print(f"  ❌ 合成失败: {data.get('detail', '未知错误')}")
            return False
    except Exception as e:
        print(f"  ❌ 请求失败: {e}")
        return False


def test_emotion(speaker_id: str, speaker_name: str) -> bool:
    """测试情感合成（多种情感）"""
    print_header("PilotTTS - emotion (情感合成)", 50)

    emotions = {
        "happy": "今天真是太开心了！阳光灿烂，心情愉快！",
        "sad": "下雨了，心情有些低落，不想出门。",
        "angry": "这太过分了！怎么能这样对待别人！",
        "surprise": "哇！这真是个惊喜！太不可思议了！",
        "serious": "我需要郑重地声明，这是一件非常重要的事情。",
    }

    all_ok = True
    for emotion, text in emotions.items():
        try:
            data = call_pilottts_api("emotion", text, speaker_id=speaker_id, emotion=emotion)
            if data.get("success"):
                print(f"  ✅ {emotion:12s}: 合成成功 → {data.get('audio_url')}")
            else:
                print(f"  ❌ {emotion:12s}: 失败 → {data.get('detail', '未知错误')}")
                all_ok = False
        except Exception as e:
            print(f"  ❌ {emotion:12s}: 异常 → {e}")
            all_ok = False

    return all_ok


def test_dialect(speaker_id: str, speaker_name: str) -> bool:
    """测试方言合成"""
    print_header("PilotTTS - dialect (方言合成)", 50)

    # 测试几种方言
    dialects = {
        "zh-henan": "中不中啊！咱俩一块儿去喝胡辣汤吧，可好喝了！",
        "zh-dongbei": "嘎哈呢？咱俩去整个铁锅炖大鹅，老好吃了！",
        "zh-sichuan": "走嘛走嘛，我们切吃火锅，巴适得板！",
    }

    all_ok = True
    for dialect_code, text in dialects.items():
        try:
            data = call_pilottts_api("dialect", text, speaker_id=speaker_id, language=dialect_code)
            if data.get("success"):
                print(f"  ✅ {dialect_code:15s}: 合成成功 → {data.get('audio_url')}")
            else:
                print(f"  ❌ {dialect_code:15s}: 失败 → {data.get('detail', '未知错误')}")
                all_ok = False
        except Exception as e:
            print(f"  ❌ {dialect_code:15s}: 异常 → {e}")
            all_ok = False

    return all_ok


def test_paralanguage(speaker_id: str, speaker_name: str) -> bool:
    """测试副语言合成"""
    print_header("PilotTTS - paralanguage (副语言合成)", 50)

    # PilotTTS支持特殊标记: [laughter], [crying], [breathing], [coughing]
    test_cases = [
        ("笑声", "哈哈哈！这个笑话太好笑了！真的忍不住！"),
        ("正常", "今天的会议讨论了很多重要的议题。"),
    ]

    all_ok = True
    for label, text in test_cases:
        try:
            data = call_pilottts_api("paralanguage", text, speaker_id=speaker_id)
            if data.get("success"):
                print(f"  ✅ {label}: 合成成功 → {data.get('audio_url')}")
            else:
                print(f"  ❌ {label}: 失败 → {data.get('detail', '未知错误')}")
                all_ok = False
        except Exception as e:
            print(f"  ❌ {label}: 异常 → {e}")
            all_ok = False

    return all_ok


def test_combined(speaker_id: str, speaker_name: str) -> bool:
    """测试情感+副语言合并模式"""
    print_header("PilotTTS - combined (情感+副语言合并)", 50)

    test_cases = [
        ("happy+LAUGH", "今天真是太开心了<|LAUGH|>哈哈哈忍不住笑出声来！", "happy", "emotion"),
        ("sad+CRY", "发生了一件非常令人难过的事<|CRY|>眼泪止不住地流下来。", "sad", "emotion"),
        ("serious+BREATH", "这是一个非常重要的声明<|BREATH|>请大家注意听。", "serious", "emotion"),
        ("纯副语言(无情感)", "大家好<|LAUGH|>这个笑话太好笑了<|LAUGH|>真的忍不住。", None, "paralanguage"),
        ("anger+COUGH", "简直太过分了<|COUGH|>我真的说不下去了<|COUGH|>。", "angry", "emotion"),
    ]

    all_ok = True
    for label, text, emotion, api_mode in test_cases:
        try:
            data = call_pilottts_api(api_mode, text, speaker_id=speaker_id, emotion=emotion)
            if data.get("success"):
                print(f"  ✅ {label:25s}: 合成成功 → {data.get('audio_url')}")
            else:
                print(f"  ❌ {label:25s}: 失败 → {data.get('detail', '未知错误')}")
                all_ok = False
        except Exception as e:
            print(f"  ❌ {label:25s}: 异常 → {e}")
            all_ok = False

    return all_ok


def test_task_queue(speaker_id: str) -> bool:
    """测试任务队列提交+轮询"""
    print_header("PilotTTS - 任务队列", 50)

    try:
        # 提交任务
        response = _POST(
            f"{BASE_URL}/tasks/submit",
            data={
                "text": "任务队列测试：验证PilotTTS批量生成功能。",
                "model": "pilottts",
                "mode": "voice_clone",
                "speaker_id": speaker_id,
            },
            timeout=30,
        )
        data = response.json()
        if not data.get("success"):
            print(f"  ❌ 任务提交失败: {data.get('detail', '未知错误')}")
            return False

        task_id = data.get("task_id")
        print(f"  ✅ 任务已提交: {task_id}")
        print(f"  等待任务执行...")

        # 轮询等待（最多等待5分钟）
        max_polls = 100
        for i in range(max_polls):
            time.sleep(3)
            try:
                status_resp = _GET(f"{BASE_URL}/tasks/{task_id}/status", timeout=10)
                status_data = status_resp.json()
                status = status_data.get("status", "unknown")

                # 显示进度
                progress = status_data.get("progress", 0)
                progress_bar = f"[{'#' * (progress // 5)}{' ' * (20 - progress // 5)}]"
                print(f"  [{i+1:3d}/{max_polls}] {status:12s} {progress_bar} {progress}%", end="\r")

                if status == "completed":
                    print()  # 换行
                    print(f"  ✅ 任务完成!")
                    print(f"  音频URL: {status_data.get('audio_url')}")
                    return True
                elif status == "failed":
                    print()  # 换行
                    print(f"  ❌ 任务失败: {status_data.get('error_message', '未知错误')}")
                    return False
            except Exception as e:
                print(f"\n  ⚠️  轮询异常: {e}")

        print(f"\n  ⏱️  任务超时（等待 {max_polls * 3}s）")
        return False

    except Exception as e:
        print(f"  ❌ 异常: {e}")
        return False


def test_batch_generate(speaker_id: str) -> bool:
    """测试批量生成"""
    print_header("PilotTTS - 批量生成", 50)

    try:
        response = _POST(
            f"{BASE_URL}/tasks/batch/submit",
            json={
                "model": "pilottts",
                "mode": "voice_clone",
                "speaker_id": speaker_id,
                "tasks": [
                    {"text": "第一句测试文本。", "mode": "voice_clone", "speaker_id": speaker_id},
                    {"text": "第二句测试文本。", "mode": "voice_clone", "speaker_id": speaker_id},
                    {"text": "第三句测试文本。", "mode": "voice_clone", "speaker_id": speaker_id},
                ],
            },
            timeout=30,
        )
        data = response.json()
        if data.get("success"):
            batch_id = data.get("batch_id", "")
            task_ids = data.get("task_ids", [])
            print(f"  ✅ 批量提交成功")
            print(f"  batch_id: {batch_id}")
            print(f"  任务数: {len(task_ids)}")
            return True
        else:
            print(f"  ❌ 批量提交失败: {data.get('detail', '未知错误')}")
            return False
    except Exception as e:
        print(f"  ❌ 异常: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="PilotTTS 全功能测试套件")
    parser.add_argument("--quick", action="store_true", help="快速测试（仅声音克隆模式）")
    parser.add_argument("--skip-task-queue", action="store_true", help="跳过任务队列测试")
    parser.add_argument("--speaker-id", type=str, help="指定说话人ID（默认自动发现）")
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════════════╗")
    print("║           PilotTTS 全功能测试套件 v2                      ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"  时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  主API: {BASE_URL}")
    print(f"  PilotTTS服务: http://127.0.0.1:8003")

    # 1. 服务状态检查
    results = {}
    svc = check_services()
    results["services"] = svc

    if not svc.get("main_api"):
        print("\n❌ 主API服务未运行，无法继续测试")
        print("  请先启动: bash start_server.sh start")
        sys.exit(1)

    if not test_health():
        sys.exit(1)

    # 2. PilotTTS独立服务检查
    pilottts_svc_ok = test_pilottts_service_health()

    # 3. 获取说话人
    speaker = None
    if args.speaker_id:
        speaker = {"id": args.speaker_id, "name": "手动指定"}
        print(f"\n  使用手动指定的说话人: {args.speaker_id}")
    else:
        speaker = get_available_speaker()

    if not speaker or not speaker.get("id"):
        print("\n❌ 无可用说话人！请先在说话人管理中添加说话人")
        sys.exit(1)

    speaker_id = speaker["id"]
    speaker_name = speaker.get("name", "Unknown")
    print(f"\n  使用说话人: {speaker_name} ({speaker_id})")

    # 4. 运行测试
    # 声音克隆模式（基础模型）
    results["voice_clone"] = test_voice_clone(speaker_id, speaker_name)

    if args.quick:
        # 快速模式
        print_header("快速测试结果", 50)
        print(f"  voice_clone: {'✅ 通过' if results['voice_clone'] else '❌ 失败'}")
        sys.exit(0 if results["voice_clone"] else 1)

    # 指令模型需要独立服务运行
    if pilottts_svc_ok:
        results["emotion"] = test_emotion(speaker_id, speaker_name)
        results["dialect"] = test_dialect(speaker_id, speaker_name)
        results["paralanguage"] = test_paralanguage(speaker_id, speaker_name)
        results["combined"] = test_combined(speaker_id, speaker_name)
    else:
        print("\n⚠️  PilotTTS独立服务未运行，跳过 emotion/dialect/paralanguage 测试")
        results["emotion"] = None
        results["dialect"] = None
        results["paralanguage"] = None

    # 任务队列测试
    if not args.skip_task_queue:
        results["task_queue"] = test_task_queue(speaker_id)
        results["batch_generate"] = test_batch_generate(speaker_id)

    # 5. 汇总结果
    print_header("测试结果汇总")
    for name, result in results.items():
        if result is None:
            print(f"  ⏭️  {name}: 跳过")
        elif isinstance(result, bool):
            status = "✅ 通过" if result else "❌ 失败"
            print(f"  {status} | {name}")
        elif isinstance(result, dict):
            print(f"  📊 {name}:")
            for k, v in result.items():
                if isinstance(v, bool):
                    print(f"     {'✅' if v else '❌'} {k}")

    # 统计
    test_results = []
    for v in results.values():
        if isinstance(v, bool):
            test_results.append(v)
        elif isinstance(v, dict):
            for sv in v.values():
                if isinstance(sv, bool):
                    test_results.append(sv)

    passed = sum(1 for r in test_results if r)
    total = len(test_results)
    print(f"\n  ────────────────────────")
    print(f"  通过率: {passed}/{total}")

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
