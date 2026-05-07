#!/usr/bin/env python3
"""
VersTTS 模型下载脚本
用于下载所有TTS算法所需的预训练模型

使用方法:
    python scripts/download_models.py --all          # 下载所有模型
    python scripts/download_models.py --chattts      # 仅下载ChatTTS
    python scripts/download_models.py --list         # 列出可下载的模型
    
注意:
    - 需要联网执行此脚本
    - 下载完成后可在离线环境使用
    - 总下载大小约 50-80GB，请确保磁盘空间充足
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Optional

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# 模型下载配置
MODEL_DOWNLOAD_CONFIG = {
    "ChatTTS": {
        "source": "huggingface",
        "repo_id": "2Noise/ChatTTS",
        "local_dir": "algorithms/ChatTTS/models",
        "description": "ChatTTS 语音合成模型 (~7GB)",
        "required": True
    },
    "CosyVoice": {
        "source": "modelscope",
        "model_id": "iic/CosyVoice-300M",
        "local_dir": "algorithms/CosyVoice/models/iic/CosyVoice-300M",
        "description": "CosyVoice 基础模型 (~2GB)",
        "required": True
    },
    "CosyVoice2": {
        "source": "modelscope",
        "model_id": "iic/CosyVoice2-0.5B",
        "local_dir": "algorithms/CosyVoice/models/iic/CosyVoice2-0.5B",
        "description": "CosyVoice2 流式模型 (~2GB)",
        "required": False
    },
    "F5-TTS": {
        "source": "huggingface",
        "repo_id": "SWivid/F5-TTS",
        "local_dir": "algorithms/F5-TTS/models",
        "description": "F5-TTS 流匹配模型 (~1.5GB)",
        "required": True
    },
    "F5-TTS Vocos": {
        "source": "huggingface",
        "repo_id": "charactr/vocos-mel-24khz",
        "local_dir": "algorithms/F5-TTS/checkpoints/vocos-mel-24khz",
        "description": "F5-TTS Vocos声码器 (~100MB)",
        "required": True
    },
    "GPT-SoVITS": {
        "source": "huggingface",
        "repo_id": "RVC-Boss/GPT-SoVITS",
        "local_dir": "algorithms/GPT-SoVITS/GPT_SoVITS/pretrained_models",
        "description": "GPT-SoVITS 预训练模型 (~4GB)",
        "required": True
    },
    "OpenVoice V1": {
        "source": "wget",
        "url": "https://myshell-public-repo-hosting.s3.amazonaws.com/openvoice/checkpoints_1226.zip",
        "local_dir": "algorithms/OpenVoice/checkpoints_v1",
        "description": "OpenVoice V1 模型 (~500MB)",
        "required": True
    },
    "OpenVoice V2": {
        "source": "wget",
        "url": "https://myshell-public-repo-hosting.s3.amazonaws.com/openvoice/checkpoints_v2_0417.zip",
        "local_dir": "algorithms/OpenVoice/checkpoints_v2",
        "description": "OpenVoice V2 模型 (~500MB)",
        "required": True
    },
    "OpenVoice Whisper": {
        "source": "huggingface",
        "repo_id": "Systran/faster-whisper-large-v3",
        "local_dir": "algorithms/OpenVoice/faster-whisper-large-v3",
        "description": "Faster-Whisper 语音识别模型 (~2.8GB)",
        "required": True
    },
    "Qwen3-TTS": {
        "source": "huggingface",
        "repo_id": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        "local_dir": "algorithms/Qwen3-TTS/models/Qwen3-TTS-12Hz-1.7B-Base",
        "description": "Qwen3-TTS 基础模型 (~3.5GB)",
        "required": True
    },
    "VoxCPM": {
        "source": "huggingface",
        "repo_id": "openbmb/VoxCPM2",
        "local_dir": "algorithms/VoxCPM/models/VoxCPM2",
        "description": "VoxCPM2 TTS模型 (~8GB)",
        "required": True
    },
    "IndexTTS": {
        "source": "modelscope",
        "model_id": "IndexTeam/IndexTTS-2",
        "local_dir": "algorithms/IndexTTS/checkpoints",
        "description": "IndexTTS 2.0 模型 (~6GB)",
        "required": True
    },
    "FireRedTTS2": {
        "source": "huggingface",
        "repo_id": "FireRedTeam/FireRedTTS2",
        "local_dir": "algorithms/FireRedTTS2/pretrained_models/FireRedTTS2",
        "description": "FireRedTTS2 对话TTS模型 (~20GB)",
        "required": True
    },
}


def print_header(title: str):
    """打印标题"""
    print("\n" + "="*80)
    print(f" {title:^76s} ")
    print("="*80)


def print_progress(message: str, done: bool = False):
    """打印进度信息"""
    icon = "✅" if done else "⏳"
    print(f"  {icon} {message}")


def download_huggingface(repo_id: str, local_dir: str, token: Optional[str] = None) -> bool:
    """从HuggingFace下载模型"""
    try:
        from huggingface_hub import snapshot_download
        
        full_local_dir = os.path.join(PROJECT_ROOT, local_dir)
        os.makedirs(full_local_dir, exist_ok=True)
        
        print_progress(f"开始下载 {repo_id} ...")
        
        snapshot_download(
            repo_id=repo_id,
            local_dir=full_local_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
            token=token
        )
        
        print_progress(f"下载完成: {local_dir}", done=True)
        return True
    except Exception as e:
        print_progress(f"下载失败: {e}", done=False)
        return False


def download_modelscope(model_id: str, local_dir: str) -> bool:
    """从ModelScope下载模型"""
    try:
        from modelscope import snapshot_download
        
        full_local_dir = os.path.join(PROJECT_ROOT, local_dir)
        os.makedirs(full_local_dir, exist_ok=True)
        
        print_progress(f"开始下载 {model_id} ...")
        
        snapshot_download(
            model_id=model_id,
            local_dir=full_local_dir
        )
        
        print_progress(f"下载完成: {local_dir}", done=True)
        return True
    except Exception as e:
        print_progress(f"下载失败: {e}", done=False)
        return False


def download_wget(url: str, local_dir: str) -> bool:
    """使用wget下载文件"""
    try:
        full_local_dir = os.path.join(PROJECT_ROOT, local_dir)
        os.makedirs(full_local_dir, exist_ok=True)
        
        print_progress(f"开始下载 {url} ...")
        
        # 下载文件
        filename = os.path.basename(url)
        filepath = os.path.join(full_local_dir, filename)
        
        result = subprocess.run(
            ["wget", "-c", "-O", filepath, url],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print_progress(f"下载失败: {result.stderr}", done=False)
            return False
        
        # 如果是zip文件，解压
        if filename.endswith('.zip'):
            print_progress(f"解压 {filename} ...")
            result = subprocess.run(
                ["unzip", "-o", filepath, "-d", full_local_dir],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                os.remove(filepath)  # 删除zip文件
        
        print_progress(f"下载完成: {local_dir}", done=True)
        return True
    except Exception as e:
        print_progress(f"下载失败: {e}", done=False)
        return False


def download_model(name: str, config: Dict, token: Optional[str] = None) -> bool:
    """下载单个模型"""
    print(f"\n📦 {name}: {config['description']}")
    print(f"   目标目录: {config['local_dir']}")
    
    source = config.get("source")
    
    if source == "huggingface":
        return download_huggingface(config["repo_id"], config["local_dir"], token)
    elif source == "modelscope":
        return download_modelscope(config["model_id"], config["local_dir"])
    elif source == "wget":
        return download_wget(config["url"], config["local_dir"])
    else:
        print_progress(f"未知的下载源: {source}", done=False)
        return False


def list_models():
    """列出所有可下载的模型"""
    print_header("可下载模型列表")
    
    total_size = 0
    for name, config in MODEL_DOWNLOAD_CONFIG.items():
        required = "必需" if config.get("required", False) else "可选"
        print(f"\n  • {name}")
        print(f"    描述: {config['description']}")
        print(f"    类型: {required}")
        print(f"    目录: {config['local_dir']}")
        print(f"    来源: {config['source']}")


def download_all_models(token: Optional[str] = None, skip_optional: bool = True):
    """下载所有模型"""
    print_header("批量下载模型")
    
    results = []
    for name, config in MODEL_DOWNLOAD_CONFIG.items():
        if skip_optional and not config.get("required", False):
            print(f"\n⏭️  跳过可选模型: {name}")
            continue
        
        success = download_model(name, config, token)
        results.append((name, success))
    
    # 打印摘要
    print_header("下载结果摘要")
    success_count = sum(1 for _, success in results if success)
    fail_count = len(results) - success_count
    
    for name, success in results:
        icon = "✅" if success else "❌"
        print(f"  {icon} {name}")
    
    print(f"\n统计: {success_count} 成功 / {fail_count} 失败")
    
    return fail_count == 0


def main():
    parser = argparse.ArgumentParser(
        description="VersTTS 模型下载工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s --all                    # 下载所有必需模型
  %(prog)s --all --include-optional # 下载所有模型(包括可选)
  %(prog)s --chattts                # 仅下载ChatTTS
  %(prog)s --list                   # 列出所有可下载模型
  %(prog)s --token YOUR_HF_TOKEN    # 使用HuggingFace Token
        """
    )
    
    parser.add_argument("--all", action="store_true", help="下载所有必需模型")
    parser.add_argument("--list", action="store_true", help="列出可下载的模型")
    parser.add_argument("--include-optional", action="store_true", help="同时下载可选模型")
    parser.add_argument("--token", type=str, help="HuggingFace访问Token")
    
    # 为每个模型添加单独的下载选项
    for name in MODEL_DOWNLOAD_CONFIG.keys():
        arg_name = "--" + name.lower().replace(" ", "-").replace(".", "")
        parser.add_argument(arg_name, action="store_true", help=f"下载 {name}")
    
    args = parser.parse_args()
    
    if args.list:
        list_models()
        return
    
    if args.all:
        success = download_all_models(args.token, skip_optional=not args.include_optional)
        sys.exit(0 if success else 1)
    
    # 检查是否有指定单个模型
    download_count = 0
    for name, config in MODEL_DOWNLOAD_CONFIG.items():
        arg_name = name.lower().replace(" ", "_").replace(".", "")
        if getattr(args, arg_name, False):
            download_model(name, config, args.token)
            download_count += 1
    
    if download_count == 0:
        parser.print_help()
        print("\n\n提示: 使用 --all 下载所有模型，或使用 --list 查看可用模型")


if __name__ == "__main__":
    main()
