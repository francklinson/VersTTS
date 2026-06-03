#!/usr/bin/env python3
"""
PilotTTS 模型直接下载脚本（绕过 SSL 问题）
"""
import os
import sys
import requests
from tqdm import tqdm

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pretrained_models")
HF_MIRROR = "https://hf-mirror.com"

# 禁用 SSL 警告
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def download_file(url, dest_path, desc=""):
    """下载单个文件"""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if os.path.exists(dest_path):
        print(f"  ✓ 已存在: {dest_path}")
        return True

    print(f"  下载: {desc} -> {dest_path}")
    try:
        resp = requests.get(url, stream=True, verify=False, allow_redirects=True, timeout=3600)
        if resp.status_code != 200:
            print(f"  ✗ HTTP {resp.status_code}")
            return False

        total = int(resp.headers.get('content-length', 0))
        with open(dest_path, 'wb') as f:
            with tqdm(total=total, unit='B', unit_scale=True, desc=os.path.basename(dest_path)[:40]) as pbar:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
        print(f"  ✓ 下载完成: {dest_path}")
        return True
    except Exception as e:
        print(f"  ✗ 下载失败: {e}")
        return False


def download_qwen3_06b():
    """下载 Qwen3-0.6B 基座模型"""
    print("\n[1/4] 下载 Qwen3-0.6B (LLM backbone)...")
    dest = os.path.join(MODEL_DIR, "Qwen3-0.6B")
    base_url = f"{HF_MIRROR}/Qwen/Qwen3-0.6B/resolve/main"

    files = [
        "model.safetensors",
        "config.json",
        "generation_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "vocab.json",
        "merges.txt",
    ]

    success = True
    for f in files:
        url = f"{base_url}/{f}"
        path = os.path.join(dest, f)
        if not download_file(url, path, f"Qwen3-0.6B/{f}"):
            if f == "model.safetensors":
                success = False
    return success


def download_w2v_bert():
    """下载 w2v-bert-2.0 音频特征提取器"""
    print("\n[2/4] 下载 w2v-bert-2.0 (音频特征提取器)...")
    dest = os.path.join(MODEL_DIR, "w2v-bert-2.0")
    base_url = f"{HF_MIRROR}/facebook/w2v-bert-2.0/resolve/main"

    files = [
        "model.safetensors",
        "config.json",
        "preprocessor_config.json",
    ]

    success = True
    for f in files:
        url = f"{base_url}/{f}"
        path = os.path.join(dest, f)
        if not download_file(url, path, f"w2v-bert-2.0/{f}"):
            if f == "model.safetensors":
                success = False
    return success


def download_pilottts_weights():
    """下载 PilotTTS 模型权重"""
    print("\n[3/4] 下载 PilotTTS 模型权重...")
    dest = MODEL_DIR
    base_url = f"{HF_MIRROR}/AmapVoice/PilotTTS/resolve/main"

    files = [
        ("pilot_tts.pt", "基础模型 (零样本声音克隆)"),
        ("pilot_tts_instruct.pt", "指令模型 (情感/副语言/方言)"),
    ]

    success = True
    for fname, desc in files:
        url = f"{base_url}/{fname}"
        path = os.path.join(dest, fname)
        if not download_file(url, path, desc):
            success = False
    return success


def setup_cosyvoice_vocoder():
    """设置 CosyVoice3-0.5B 声码器（复用已有模型）"""
    print("\n[4/4] 设置 CosyVoice3-0.5B 声码器...")

    src = "/home/zhouchenghao/PycharmProjects/VersTTS/models/CosyVoice/Fun-CosyVoice3-0.5B"
    dest = os.path.join(MODEL_DIR, "Fun-CosyVoice3-0.5B")

    if not os.path.exists(src):
        print(f"  ✗ 源目录不存在: {src}")
        print("  需要从 ModelScope/HuggingFace 下载 Fun-CosyVoice3-0.5B")
        return False

    if os.path.exists(dest):
        print(f"  ✓ 已存在: {dest}")
        return True

    # 创建符号链接
    os.symlink(src, dest)
    print(f"  ✓ 符号链接创建: {dest} -> {src}")
    return True


def setup_wav2vec2bert_stats():
    """设置 wav2vec2bert_stats.pt（复用已有文件）"""
    src = "/home/zhouchenghao/PycharmProjects/VersTTS/algorithms/IndexTTS/indextts/utils/maskgct/models/tts/maskgct/ckpt/wav2vec2bert_stats.pt"
    dest = os.path.join(MODEL_DIR, "wav2vec2bert_stats.pt")

    if not os.path.exists(src):
        print(f"  ⚠ 源文件不存在: {src}")
        return False

    if os.path.exists(dest):
        print(f"  ✓ 已存在: {dest}")
        return True

    import shutil
    shutil.copy2(src, dest)
    print(f"  ✓ 复制: {dest}")
    return True


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    print("=" * 60)
    print("PilotTTS 模型下载")
    print(f"目标目录: {MODEL_DIR}")
    print("=" * 60)

    # 设置 wav2vec2bert_stats.pt
    setup_wav2vec2bert_stats()

    # 下载各个模型
    results = {}
    results['qwen3'] = download_qwen3_06b()
    results['w2v_bert'] = download_w2v_bert()
    results['pilottts'] = download_pilottts_weights()
    results['cosyvoice'] = setup_cosyvoice_vocoder()

    print("\n" + "=" * 60)
    print("下载结果:")
    for name, ok in results.items():
        status = "✓" if ok else "✗"
        print(f"  {status} {name}")
    print("=" * 60)

    # 验证文件
    required = [
        os.path.join(MODEL_DIR, "Qwen3-0.6B", "model.safetensors"),
        os.path.join(MODEL_DIR, "w2v-bert-2.0", "model.safetensors"),
        os.path.join(MODEL_DIR, "pilot_tts.pt"),
        os.path.join(MODEL_DIR, "pilot_tts_instruct.pt"),
        os.path.join(MODEL_DIR, "wav2vec2bert_stats.pt"),
    ]

    missing = [f for f in required if not os.path.exists(f)]
    if missing:
        print(f"\n⚠ 缺少以下文件 ({len(missing)}个):")
        for f in missing:
            print(f"  - {f}")
    else:
        print("\n✓ 所有模型文件就绪!")


if __name__ == "__main__":
    main()
