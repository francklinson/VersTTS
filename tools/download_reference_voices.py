#!/usr/bin/env python3
"""
从公开资源下载参考人声音频
支持: Common Voice, AIShell 等公开数据集
"""

import os
import sys
import json
import requests
import argparse
from pathlib import Path
from typing import List, Dict, Optional
import soundfile as sf
import librosa
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
REFERENCE_DIR = PROJECT_ROOT / "reference_audio"
METADATA_PATH = REFERENCE_DIR / "metadata.json"

class ReferenceVoiceDownloader:
    """参考人声下载器"""
    
    def __init__(self):
        self.reference_dir = REFERENCE_DIR
        self.metadata_path = METADATA_PATH
        self.ensure_directories()
        
    def ensure_directories(self):
        """确保目录存在"""
        for category in ["children", "teenagers", "adults"]:
            (self.reference_dir / category).mkdir(parents=True, exist_ok=True)
    
    def load_metadata(self) -> Dict:
        """加载元数据"""
        if self.metadata_path.exists():
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {
            "version": "1.0.0",
            "description": "VersTTS参考人声音频元数据",
            "created_at": "2026-04-25",
            "categories": {
                "children": {"description": "儿童声音(3-12岁)", "count": 0},
                "teenagers": {"description": "中学生声音(13-18岁)", "count": 0},
                "adults": {"description": "成人声音(18岁以上)", "count": 0}
            },
            "samples": [],
            "model_compatibility": {}
        }
    
    def save_metadata(self, metadata: Dict):
        """保存元数据"""
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    def validate_audio(self, audio_path: str) -> tuple:
        """
        验证音频质量
        返回: (是否有效, 信息字典)
        """
        try:
            info = sf.info(audio_path)
            duration = info.duration
            
            # 检查时长
            if duration < 3 or duration > 30:
                return False, {"error": f"时长不符合要求: {duration:.1f}s"}
            
            # 检查采样率
            if info.samplerate not in [16000, 22050, 44100, 48000]:
                return False, {"error": f"采样率不符合: {info.samplerate}Hz"}
            
            # 检查声道
            if info.channels != 1:
                return False, {"error": f"非单声道: {info.channels}声道"}
            
            # 读取音频进行质量检查
            audio, sr = sf.read(audio_path)
            
            # 检查音量
            rms = np.sqrt(np.mean(audio**2))
            if rms < 0.01:  # 太安静
                return False, {"error": "音量过低"}
            
            # 简单静音检测
            silent_ratio = np.sum(np.abs(audio) < 0.01) / len(audio)
            if silent_ratio > 0.5:
                return False, {"error": f"静音比例过高: {silent_ratio*100:.1f}%"}
            
            return True, {
                "duration": duration,
                "sample_rate": info.samplerate,
                "channels": info.channels,
                "rms": float(rms)
            }
            
        except Exception as e:
            return False, {"error": str(e)}
    
    def add_sample(self, metadata: Dict, sample_info: Dict) -> Dict:
        """添加样本到元数据"""
        # 检查是否已存在
        existing_ids = {s["id"] for s in metadata["samples"]}
        if sample_info["id"] in existing_ids:
            print(f"样本 {sample_info['id']} 已存在，跳过")
            return metadata
        
        metadata["samples"].append(sample_info)
        
        # 更新分类计数
        category = sample_info.get("category", "adults")
        if category in metadata["categories"]:
            metadata["categories"][category]["count"] = sum(
                1 for s in metadata["samples"] if s.get("category") == category
            )
        
        return metadata
    
    def download_common_voice_sample(self, output_path: str, age_group: str = "adults") -> bool:
        """
        从Common Voice下载示例音频
        注意: 这只是一个示例框架，实际需要访问Common Voice API或数据集
        """
        print(f"Common Voice下载功能需要访问其API或下载完整数据集")
        print(f"请访问 https://commonvoice.mozilla.org/ 下载数据集")
        return False
    
    def create_sample_from_existing(self, source_path: str, category: str, 
                                     name: str, gender: str, age: str, text: str) -> bool:
        """
        从现有音频创建参考人声样本
        用于演示和测试
        """
        if not os.path.exists(source_path):
            print(f"源文件不存在: {source_path}")
            return False
        
        # 验证音频
        is_valid, info = self.validate_audio(source_path)
        if not is_valid:
            print(f"音频验证失败: {info.get('error')}")
            return False
        
        # 生成ID
        existing_count = sum(1 for s in self.load_metadata()["samples"] 
                           if s.get("category") == category)
        voice_id = f"voice_{category}_{existing_count+1:03d}"
        
        # 目标路径
        target_filename = f"{voice_id}.wav"
        target_path = self.reference_dir / category / target_filename
        
        # 复制/转换音频
        try:
            audio, sr = sf.read(source_path)
            
            # 转换为单声道
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            
            # 重采样到22050Hz
            if sr != 22050:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=22050)
                sr = 22050
            
            # 保存
            sf.write(target_path, audio, sr)
            
            # 更新元数据
            metadata = self.load_metadata()
            sample_info = {
                "id": voice_id,
                "name": name,
                "filename": f"{category}/{target_filename}",
                "category": category,
                "gender": gender,
                "age_group": age,
                "language": "zh",
                "text": text,
                "duration": info["duration"],
                "sample_rate": sr,
                "tags": [category, gender, "clear"],
                "source": "processed",
                "license": "internal",
                "compatible_models": {
                    "chattts": False,
                    "cosyvoice": True,
                    "f5tts": True,
                    "qwen3tts": True,
                    "openvoice": True,
                    "gptsovits": True
                }
            }
            
            metadata = self.add_sample(metadata, sample_info)
            self.save_metadata(metadata)
            
            print(f"✅ 成功创建: {voice_id}")
            print(f"   路径: {target_path}")
            print(f"   时长: {info['duration']:.1f}s")
            return True
            
        except Exception as e:
            print(f"❌ 处理失败: {e}")
            return False
    
    def list_datasets(self):
        """列出可用的公开数据集信息"""
        datasets = [
            {
                "name": "Common Voice (Mozilla)",
                "url": "https://commonvoice.mozilla.org/",
                "license": "CC0 (Public Domain)",
                "description": "众包语音数据集，包含多种语言和年龄段",
                "how_to_download": "访问网站下载完整数据集或使用API",
                "notes": "需要筛选儿童和青少年音频片段"
            },
            {
                "name": "AIShell-1/2/3",
                "url": "http://www.aishelltech.com/",
                "license": "学术研究免费",
                "description": "中文语音数据集",
                "how_to_download": "网站注册申请下载",
                "notes": "AIShell-3包含多说话人数据"
            },
            {
                "name": "THCHS-30",
                "url": "http://www.openslr.org/18/",
                "license": "Apache-2.0",
                "description": "清华大学中文语音数据集",
                "how_to_download": "直接下载",
                "notes": "包含30小时中文语音"
            },
            {
                "name": "MagicData",
                "url": "http://www.openslr.org/68/",
                "license": "Apache-2.0",
                "description": "中文对话语音数据集",
                "how_to_download": "OpenSLR网站直接下载",
                "notes": "包含成人语音为主"
            }
        ]
        
        print("\n" + "="*60)
        print("可用的公开语音数据集")
        print("="*60)
        
        for i, ds in enumerate(datasets, 1):
            print(f"\n{i}. {ds['name']}")
            print(f"   网址: {ds['url']}")
            print(f"   许可: {ds['license']}")
            print(f"   说明: {ds['description']}")
            print(f"   下载: {ds['how_to_download']}")
            print(f"   备注: {ds['notes']}")
        
        print("\n" + "="*60)
        print("推荐步骤:")
        print("1. 访问 Common Voice 网站")
        print("2. 下载中文数据集")
        print("3. 筛选儿童和青少年音频")
        print("4. 使用本工具的 --process 参数处理")
        print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="参考人声下载工具")
    parser.add_argument("--list", action="store_true", help="列出可用数据集")
    parser.add_argument("--process", type=str, help="处理现有音频文件")
    parser.add_argument("--category", type=str, default="adults", 
                       choices=["children", "teenagers", "adults"],
                       help="音频分类")
    parser.add_argument("--name", type=str, help="人声名称")
    parser.add_argument("--gender", type=str, choices=["male", "female"], 
                       help="性别")
    parser.add_argument("--age", type=str, help="年龄段(如: 8-10)")
    parser.add_argument("--text", type=str, help="参考文本")
    
    args = parser.parse_args()
    
    downloader = ReferenceVoiceDownloader()
    
    if args.list:
        downloader.list_datasets()
    elif args.process:
        if not all([args.name, args.gender, args.age, args.text]):
            print("错误: --process 需要同时提供 --name, --gender, --age, --text")
            sys.exit(1)
        
        success = downloader.create_sample_from_existing(
            args.process,
            args.category,
            args.name,
            args.gender,
            args.age,
            args.text
        )
        sys.exit(0 if success else 1)
    else:
        parser.print_help()
        print("\n提示: 使用 --list 查看可用公开数据集")


if __name__ == "__main__":
    main()
