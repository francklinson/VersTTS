#!/usr/bin/env python3
"""
PilotTTS 模型下载脚本
"""

import os
import sys

def download_pilottts_models():
    """下载PilotTTS模型文件"""
    print("=" * 60)
    print("下载 PilotTTS 模型文件")
    print("=" * 60)

    # 创建模型目录
    model_dir = "pretrained_models"
    os.makedirs(model_dir, exist_ok=True)

    # 设置离线模式环境变量
    os.environ['HF_HUB_OFFLINE'] = '0'
    os.environ['HF_DATASETS_OFFLINE'] = '0'
    os.environ['TRANSFORMERS_OFFLINE'] = '0'

    try:
        # 1. 下载 PilotTTS 主模型 (从 ModelScope)
        print("\n[1/5] 下载 PilotTTS 主模型...")
        try:
            from modelscope import snapshot_download
            snapshot_download('AmapVoice/PilotTTS', local_dir=os.path.join(model_dir, 'PilotTTS'))
            print("✓ PilotTTS 模型下载完成")
        except Exception as e:
            print(f"✗ PilotTTS 模型下载失败: {e}")
            print("  请手动从 https://modelscope.cn/models/AmapVoice/PilotTTS 下载")

        # 2. 下载 Qwen3-0.6B (LLM backbone)
        print("\n[2/5] 下载 Qwen3-0.6B (LLM backbone)...")
        try:
            from modelscope import snapshot_download
            snapshot_download('Qwen/Qwen3-0.6B', local_dir=os.path.join(model_dir, 'Qwen3-0.6B'))
            print("✓ Qwen3-0.6B 下载完成")
        except Exception as e:
            print(f"✗ Qwen3-0.6B 下载失败: {e}")
            try:
                from huggingface_hub import snapshot_download
                snapshot_download('Qwen/Qwen3-0.6B', local_dir=os.path.join(model_dir, 'Qwen3-0.6B'))
                print("✓ Qwen3-0.6B 下载完成 (HuggingFace)")
            except Exception as e2:
                print(f"✗ Qwen3-0.6B 下载失败: {e2}")

        # 3. 下载 w2v-bert-2.0 (音频特征提取器)
        print("\n[3/5] 下载 w2v-bert-2.0 (音频特征提取器)...")
        try:
            from modelscope import snapshot_download
            snapshot_download('iic/multi-modal/w2v-bert-2.0', local_dir=os.path.join(model_dir, 'w2v-bert-2.0'))
            print("✓ w2v-bert-2.0 下载完成")
        except Exception as e:
            print(f"✗ w2v-bert-2.0 下载失败 (ModelScope): {e}")
            try:
                from huggingface_hub import snapshot_download
                snapshot_download('facebook/w2v-bert-2.0', local_dir=os.path.join(model_dir, 'w2v-bert-2.0'))
                print("✓ w2v-bert-2.0 下载完成 (HuggingFace)")
            except Exception as e2:
                print(f"✗ w2v-bert-2.0 下载失败: {e2}")

        # 4. 下载 Fun-CosyVoice3-0.5B (声码器)
        print("\n[4/5] 下载 Fun-CosyVoice3-0.5B (声码器)...")
        try:
            from modelscope import snapshot_download
            snapshot_download('iic/Fun-CosyVoice3-0.5B', local_dir=os.path.join(model_dir, 'Fun-CosyVoice3-0.5B'))
            print("✓ Fun-CosyVoice3-0.5B 下载完成")
        except Exception as e:
            print(f"✗ Fun-CosyVoice3-0.5B 下载失败 (ModelScope): {e}")
            try:
                from huggingface_hub import snapshot_download
                snapshot_download('funaudiollm/Fun-CosyVoice3-0.5B', local_dir=os.path.join(model_dir, 'Fun-CosyVoice3-0.5B'))
                print("✓ Fun-CosyVoice3-0.5B 下载完成 (HuggingFace)")
            except Exception as e2:
                print(f"✗ Fun-CosyVoice3-0.5B 下载失败: {e2}")

        # 5. 下载 wav2vec2bert_stats.pt (来自 MaskGCT)
        print("\n[5/5] 下载 wav2vec2bert_stats.pt...")
        try:
            import wget
            url = "https://huggingface.co/amphion/maskgct/resolve/main/wav2vec2bert_stats.pt"
            output_path = os.path.join(model_dir, 'wav2vec2bert_stats.pt')
            if not os.path.exists(output_path):
                wget.download(url, output_path)
                print(f"\n✓ wav2vec2bert_stats.pt 下载完成")
            else:
                print("✓ wav2vec2bert_stats.pt 已存在")
        except Exception as e:
            print(f"✗ wav2vec2bert_stats.pt 下载失败: {e}")
            print("  请手动下载并放置到 pretrained_models/ 目录")

        print("\n" + "=" * 60)
        print("模型下载完成!")
        print("=" * 60)

        # 检查文件结构
        print("\n模型目录结构:")
        for root, dirs, files in os.walk(model_dir):
            level = root.replace(model_dir, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f'{indent}{os.path.basename(root)}/')
            subindent = ' ' * 2 * (level + 1)
            for file in files[:5]:  # 只显示前5个文件
                print(f'{subindent}{file}')
            if len(files) > 5:
                print(f'{subindent}... 还有 {len(files) - 5} 个文件')

    except Exception as e:
        print(f"\n下载过程出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    download_pilottts_models()
