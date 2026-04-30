#!/usr/bin/env python3
"""
VersTTS 模型文件检查脚本
用于验证所有TTS算法所需的模型文件是否已正确下载到指定目录

使用方法:
    python scripts/check_models.py
    python scripts/check_models.py --fix    # 尝试修复缺失的模型路径
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# 模型路径配置
MODEL_PATHS = {
    "ChatTTS": {
        "path": "algorithms/ChatTTS/models",
        "required_files": ["*.pt", "*.safetensors", "*.json", "*.yaml"],
        "min_size_gb": 7.0,
        "description": "ChatTTS 语音合成模型"
    },
    "CosyVoice": {
        "path": "algorithms/CosyVoice/models/iic",
        "required_files": ["*CosyVoice*", "*.pt", "*.safetensors", "*.bin"],
        "min_size_gb": 2.0,
        "description": "CosyVoice 多语言TTS模型"
    },
    "F5-TTS": {
        "path": "algorithms/F5-TTS/models",
        "required_files": ["*.pt", "model_*"],
        "min_size_gb": 1.5,
        "description": "F5-TTS 流匹配模型"
    },
    "F5-TTS Vocos": {
        "path": "algorithms/F5-TTS/checkpoints/vocos-mel-24khz",
        "required_files": ["*.pt", "*.json"],
        "min_size_gb": 0.1,
        "description": "F5-TTS Vocos声码器"
    },
    "GPT-SoVITS": {
        "path": "algorithms/GPT-SoVITS/GPT_SoVITS/pretrained_models",
        "required_files": ["*.pth", "*.ckpt", "*.bin", "*.pt"],
        "min_size_gb": 4.0,
        "description": "GPT-SoVITS 预训练模型"
    },
    "OpenVoice V1": {
        "path": "algorithms/OpenVoice/checkpoints_v1/checkpoints",
        "required_files": ["*.pth", "*.json"],
        "min_size_gb": 0.5,
        "description": "OpenVoice V1 模型"
    },
    "OpenVoice V2": {
        "path": "algorithms/OpenVoice/checkpoints_v2/checkpoints_v2",
        "required_files": ["*.pth", "*.json"],
        "min_size_gb": 0.5,
        "description": "OpenVoice V2 模型"
    },
    "OpenVoice Whisper": {
        "path": "algorithms/OpenVoice/faster-whisper-large-v3",
        "required_files": ["*.bin", "*.json", "*.txt"],
        "min_size_gb": 2.8,
        "description": "OpenVoice faster-whisper模型"
    },
    "Qwen3-TTS": {
        "path": "algorithms/Qwen3-TTS/models",
        "required_files": ["*.safetensors", "*.json", "*.bin"],
        "min_size_gb": 3.5,
        "description": "Qwen3-TTS 语言模型"
    },
    "VoxCPM": {
        "path": "algorithms/VoxCPM/models/VoxCPM2",
        "required_files": ["*.safetensors", "*.json", "*.bin"],
        "min_size_gb": 8.0,
        "description": "VoxCPM2 TTS模型"
    },
    "IndexTTS": {
        "path": "algorithms/IndexTTS/checkpoints",
        "required_files": ["*.pth", "*.pt", "*.yaml"],
        "min_size_gb": 4.0,
        "description": "IndexTTS GPT模型"
    },
    "FireRedTTS2": {
        "path": "algorithms/FireRedTTS2/pretrained_models/FireRedTTS2",
        "required_files": ["*.pt", "*.ckpt", "*.json"],
        "min_size_gb": 15.0,
        "description": "FireRedTTS2 对话TTS模型"
    },
}

# HuggingFace缓存检查
HF_CACHE_PATHS = {
    "HF Hub Cache": "models/hf_cache/hub",
    "Transformers Cache": "models/transformers_cache",
    "Torch Cache": "models/torch_cache",
    "Datasets Cache": "models/datasets_cache",
}


def get_dir_size(path: str) -> float:
    """计算目录大小（GB）"""
    total = 0
    try:
        for dirpath, dirnames, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if os.path.exists(fp):
                    total += os.path.getsize(fp)
    except Exception:
        pass
    return total / (1024**3)


def check_model_files(model_name: str, config: Dict) -> Tuple[bool, str, float]:
    """
    检查模型文件是否存在
    
    Returns:
        (是否存在, 状态信息, 实际大小GB)
    """
    path = os.path.join(PROJECT_ROOT, config["path"])
    min_size = config.get("min_size_gb", 0)
    
    if not os.path.exists(path):
        return False, f"❌ 路径不存在: {path}", 0.0
    
    # 检查是否有文件
    has_files = False
    for pattern in config["required_files"]:
        import glob
        files = glob.glob(os.path.join(path, "**", pattern), recursive=True)
        if files:
            has_files = True
            break
    
    if not has_files:
        return False, f"⚠️ 路径存在但无模型文件", 0.0
    
    # 计算实际大小
    size_gb = get_dir_size(path)
    
    if size_gb < min_size * 0.5:  # 小于预期大小的50%
        return False, f"⚠️ 模型文件不完整 ({size_gb:.2f}GB / 预期 {min_size}GB+)", size_gb
    
    return True, f"✅ 正常 ({size_gb:.2f}GB)", size_gb


def check_hf_cache() -> Dict[str, Tuple[bool, float]]:
    """检查HuggingFace缓存目录"""
    results = {}
    for name, rel_path in HF_CACHE_PATHS.items():
        path = os.path.join(PROJECT_ROOT, rel_path)
        if os.path.exists(path):
            size = get_dir_size(path)
            results[name] = (True, size)
        else:
            results[name] = (False, 0.0)
    return results


def print_summary(results: Dict[str, Tuple[bool, str, float]], hf_results: Dict[str, Tuple[bool, float]]):
    """打印检查结果摘要"""
    print("\n" + "="*80)
    print(" "*25 + "模型检查报告")
    print("="*80)
    
    print("\n📦 TTS 模型状态:")
    print("-"*80)
    
    ok_count = 0
    fail_count = 0
    total_size = 0.0
    
    for name, config in MODEL_PATHS.items():
        exists, status, size = results[name]
        total_size += size
        if exists:
            ok_count += 1
            status_icon = "✅"
        else:
            fail_count += 1
            status_icon = "❌"
        
        print(f"{status_icon} {name:20s} | {status}")
    
    print("-"*80)
    print(f"\n📊 统计: {ok_count} 个正常 / {fail_count} 个异常")
    print(f"💾 总模型大小: {total_size:.2f} GB")
    
    print("\n🗄️ HuggingFace 缓存状态:")
    print("-"*80)
    for name, (exists, size) in hf_results.items():
        status = f"✅ ({size:.2f}GB)" if exists else "❌ 不存在"
        print(f"  {'✅' if exists else '❌'} {name}: {status}")
    
    print("\n" + "="*80)
    
    if fail_count > 0:
        print("\n⚠️  发现以下问题:")
        print("-"*80)
        for name, config in MODEL_PATHS.items():
            exists, status, _ = results[name]
            if not exists:
                print(f"  • {name}: {config['description']}")
                print(f"    预期路径: {config['path']}")
        print("-"*80)
        print("\n💡 解决方案:")
        print("  1. 运行模型下载脚本: python scripts/download_models.py")
        print("  2. 或手动下载模型到对应目录")
        print("  3. 详细说明请参考: DEPLOYMENT_GUIDE.md")
        return False
    else:
        print("\n🎉 所有模型检查通过！可以进行离线部署。")
        return True


def main():
    parser = argparse.ArgumentParser(description="VersTTS 模型文件检查工具")
    parser.add_argument("--fix", action="store_true", help="尝试创建缺失的目录结构")
    args = parser.parse_args()
    
    print("="*80)
    print(" "*20 + "VersTTS 模型文件检查")
    print("="*80)
    print(f"\n📁 项目根目录: {PROJECT_ROOT}")
    print(f"🔍 开始检查 {len(MODEL_PATHS)} 个TTS模型...\n")
    
    # 检查各模型
    results = {}
    for name, config in MODEL_PATHS.items():
        exists, status, size = check_model_files(name, config)
        results[name] = (exists, status, size)
    
    # 检查HF缓存
    hf_results = check_hf_cache()
    
    # 打印摘要
    all_ok = print_summary(results, hf_results)
    
    # 如果需要修复，创建缺失的目录
    if args.fix:
        print("\n🔧 修复模式: 创建缺失的目录结构...")
        created = []
        for name, config in MODEL_PATHS.items():
            path = os.path.join(PROJECT_ROOT, config["path"])
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)
                created.append(config["path"])
        
        # 创建HF缓存目录
        for name, rel_path in HF_CACHE_PATHS.items():
            path = os.path.join(PROJECT_ROOT, rel_path)
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)
                created.append(rel_path)
        
        if created:
            print(f"  ✅ 已创建 {len(created)} 个目录:")
            for p in created:
                print(f"    - {p}")
        else:
            print("  ℹ️ 所有目录已存在")
    
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
