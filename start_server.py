#!/usr/bin/env python3
"""
VersTTS 服务启动脚本
用于启动统一TTS API服务
"""

import os
import sys
import argparse
import subprocess

def check_environment():
    """检查环境配置"""
    print("=" * 60)
    print("检查环境配置...")
    print("=" * 60)

    # 检查Python版本
    import platform
    py_version = platform.python_version()
    print(f"Python版本: {py_version}")

    # 检查CUDA
    try:
        import torch
        if torch.cuda.is_available():
            print(f"CUDA可用: {torch.cuda.get_device_name(0)}")
            print(f"CUDA版本: {torch.version.cuda}")
        else:
            print("警告: CUDA不可用,将使用CPU模式")
    except ImportError:
        print("错误: 未安装PyTorch")
        return False

    # 检查必要的库
    required_packages = [
        "fastapi", "uvicorn", "numpy", "soundfile",
        "pydantic", "torchaudio", "omegaconf", "hydra-core"
    ]

    missing = []
    for pkg in required_packages:
        try:
            __import__(pkg.replace("-", "_"))
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"警告: 缺少以下依赖包: {', '.join(missing)}")
        print("请运行: pip install " + " ".join(missing))
        return False

    print("环境检查通过!")
    return True

def create_directories():
    """创建必要的目录"""
    dirs = ["output", "uploads", "backend", "frontend", "records"]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print("目录结构检查完成")

def main():
    parser = argparse.ArgumentParser(description="VersTTS 服务启动脚本")
    parser.add_argument("--host", default="0.0.0.0", help="服务主机地址")
    parser.add_argument("--port", type=int, default=8000, help="服务端口")
    parser.add_argument("--skip-check", action="store_true", help="跳过环境检查")
    parser.add_argument("--reload", action="store_true", help="开发模式(自动重载)")

    args = parser.parse_args()

    # 检查环境
    if not args.skip_check:
        if not check_environment():
            print("环境检查失败,请安装依赖后重试")
            sys.exit(1)

    # 创建目录
    create_directories()

    print("\n" + "=" * 60)
    print("启动VersTTS服务...")
    print("=" * 60)
    print(f"服务地址: http://{args.host}:{args.port}")
    print(f"API文档: http://{args.host}:{args.port}/docs")
    print("=" * 60 + "\n")

    # 启动服务
    cmd = [
        sys.executable, "-m", "uvicorn",
        "backend.api_server:app",
        "--host", args.host,
        "--port", str(args.port),
    ]

    if args.reload:
        cmd.append("--reload")

    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n服务已停止")

if __name__ == "__main__":
    main()
