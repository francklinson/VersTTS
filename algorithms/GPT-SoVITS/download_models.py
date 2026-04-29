#!/usr/bin/env python3
"""
GPT-SoVITS 模型下载脚本
从 HuggingFace 下载预训练模型
"""

import os
import sys
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.absolute()
PRETRAINED_DIR = PROJECT_ROOT / "GPT_SoVITS" / "pretrained_models"

def ensure_dir(path):
    """确保目录存在"""
    os.makedirs(path, exist_ok=True)
    return path

def download_file(url, dest_path):
    """下载文件"""
    import urllib.request
    import ssl

    # 创建SSL上下文
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    print(f"下载: {url}")
    print(f"保存到: {dest_path}")

    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, context=ssl_context, timeout=300) as response:
            total_size = int(response.headers.get('Content-Length', 0))
            downloaded = 0
            chunk_size = 8192

            with open(dest_path, 'wb') as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"\r进度: {progress:.1f}%", end='', flush=True)

        print(f"\n✓ 下载完成: {dest_path}")
        return True
    except Exception as e:
        print(f"\n✗ 下载失败: {e}")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return False

def download_from_hf(repo_id, filename, dest_dir, local_filename=None):
    """从HuggingFace下载文件"""
    from huggingface_hub import hf_hub_download
    import shutil

    dest_dir = Path(dest_dir)
    ensure_dir(dest_dir)

    local_name = local_filename or filename
    dest_path = dest_dir / local_name

    if dest_path.exists():
        print(f"✓ 文件已存在: {dest_path}")
        return True

    print(f"从HuggingFace下载: {repo_id}/{filename}")
    try:
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=dest_dir,
            local_dir_use_symlinks=False
        )
        # 重命名为目标文件名（如果需要）
        if local_filename and Path(downloaded_path).name != local_filename:
            shutil.move(downloaded_path, dest_path)
        print(f"✓ 下载完成: {dest_path}")
        return True
    except Exception as e:
        print(f"✗ 下载失败: {e}")
        return False

def download_gpt_sovits_models():
    """下载GPT-SoVITS预训练模型"""
    print("=" * 60)
    print("GPT-SoVITS 模型下载工具")
    print("=" * 60)

    # 确保目录存在
    ensure_dir(PRETRAINED_DIR)

    # V2模型 (推荐)
    v2_dir = ensure_dir(PRETRAINED_DIR / "gsv-v2final-pretrained")

    models_to_download = [
        # BERT模型
        {
            "repo_id": "lj1995/GPT-SoVITS",
            "filename": "chinese-roberta-wwm-ext-large/pytorch_model.bin",
            "dest_dir": PRETRAINED_DIR / "chinese-roberta-wwm-ext-large",
            "required": True
        },
        {
            "repo_id": "lj1995/GPT-SoVITS",
            "filename": "chinese-roberta-wwm-ext-large/config.json",
            "dest_dir": PRETRAINED_DIR / "chinese-roberta-wwm-ext-large",
            "required": True
        },
        # HuBERT模型
        {
            "repo_id": "lj1995/GPT-SoVITS",
            "filename": "chinese-hubert-base/pytorch_model.bin",
            "dest_dir": PRETRAINED_DIR / "chinese-hubert-base",
            "required": True
        },
        {
            "repo_id": "lj1995/GPT-SoVITS",
            "filename": "chinese-hubert-base/config.json",
            "dest_dir": PRETRAINED_DIR / "chinese-hubert-base",
            "required": True
        },
        # V2 GPT模型
        {
            "repo_id": "lj1995/GPT-SoVITS",
            "filename": "gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt",
            "dest_dir": v2_dir,
            "required": True
        },
        # V2 SoVITS模型
        {
            "repo_id": "lj1995/GPT-SoVITS",
            "filename": "gsv-v2final-pretrained/s2G2333k.pth",
            "dest_dir": v2_dir,
            "required": True
        },
    ]

    success_count = 0
    failed_models = []

    for model in models_to_download:
        print(f"\n{'='*60}")
        if download_from_hf(
            model["repo_id"],
            model["filename"],
            model["dest_dir"]
        ):
            success_count += 1
        else:
            failed_models.append(model["filename"])
            if model.get("required"):
                print(f"警告: 必需模型下载失败: {model['filename']}")

    print(f"\n{'='*60}")
    print(f"下载完成: {success_count}/{len(models_to_download)} 个模型")

    if failed_models:
        print(f"\n失败的模型:")
        for m in failed_models:
            print(f"  - {m}")

    print("=" * 60)
    return len(failed_models) == 0

if __name__ == "__main__":
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("正在安装 huggingface_hub...")
        os.system(f"{sys.executable} -m pip install huggingface_hub -q")
        from huggingface_hub import hf_hub_download

    success = download_gpt_sovits_models()
    sys.exit(0 if success else 1)
