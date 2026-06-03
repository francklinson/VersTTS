#!/usr/bin/env python3
"""
VersTTS 测试工具模块
提供统一的 HTTP/HTTPS 请求和端口自动检测
"""

import os
import requests
import urllib3

# 禁用 SSL 警告（自签名证书）
urllib3.disable_warnings()

# 默认端口列表（按优先级）
_PORT_CANDIDATES = [8008, 8000, 18800]


def detect_api_port() -> int:
    """自动检测主API服务端口"""
    # 优先使用环境变量
    env_port = os.environ.get("VERS_TTS_PORT")
    if env_port:
        return int(env_port)

    # 尝试各端口
    for port in _PORT_CANDIDATES:
        for protocol in ["https", "http"]:
            try:
                url = f"{protocol}://localhost:{port}/health"
                resp = requests.get(url, verify=False, timeout=3)
                if resp.status_code == 200:
                    os.environ["VERS_TTS_PORT"] = str(port)
                    return port
            except:
                pass

    # 默认返回 8000
    return 8000


def get_base_url() -> str:
    """获取API基础URL"""
    port = detect_api_port()
    # 检测是否使用SSL
    for protocol in ["https", "http"]:
        try:
            url = f"{protocol}://localhost:{port}/health"
            resp = requests.get(url, verify=False, timeout=3)
            if resp.status_code == 200:
                return f"{protocol}://localhost:{port}"
        except:
            pass
    return f"http://localhost:{port}"


def api_get(endpoint: str, timeout: int = 10) -> dict:
    """发送GET请求到主API"""
    base = get_base_url()
    url = f"{base}{endpoint}"
    resp = requests.get(url, verify=False, timeout=timeout)
    return resp


def api_post(endpoint: str, data: dict = None, json_data: dict = None,
             timeout: int = 120) -> dict:
    """发送POST请求到主API"""
    base = get_base_url()
    url = f"{base}{endpoint}"
    if json_data:
        resp = requests.post(url, json=json_data, verify=False, timeout=timeout)
    else:
        resp = requests.post(url, data=data, verify=False, timeout=timeout)
    return resp


def check_service(host: str = "127.0.0.1", port: int = 8000,
                  endpoint: str = "/health", protocol: str = "http",
                  timeout: int = 5) -> bool:
    """检查服务是否运行"""
    try:
        url = f"{protocol}://{host}:{port}{endpoint}"
        resp = requests.get(url, verify=False, timeout=timeout)
        return resp.status_code == 200
    except:
        return False


def get_first_speaker() -> str:
    """获取第一个可用说话人ID"""
    try:
        resp = api_get("/speakers/", timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            speakers = data.get("speakers", [])
            if isinstance(speakers, list) and len(speakers) > 0:
                return speakers[0].get("id")
    except:
        pass
    return None


def print_header(title: str, width: int = 60):
    """打印测试标题"""
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def summarize(results: dict) -> tuple:
    """汇总测试结果，返回 (passed, total, exit_code)"""
    valid = {k: v for k, v in results.items() if v is not None}
    passed = sum(1 for v in valid.values() if v)
    total = len(valid)
    for k, v in results.items():
        if v is None:
            print(f"  ⏭️  {k}")
        else:
            print(f"  {'✅' if v else '❌'} {k}")
    print(f"\n  通过: {passed}/{total}")
    return passed, total, (0 if passed == total else 1)
