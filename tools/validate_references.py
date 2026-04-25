#!/usr/bin/env python3
"""
参考音频验证工具
用于验证参考音频是否符合质量要求
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf
import librosa


def check_audio_quality(audio_path: str) -> Tuple[bool, List[str], Dict]:
    """
    检查音频质量
    
    Returns:
        (is_valid, warnings, info)
    """
    warnings = []
    info = {}
    
    try:
        # 读取音频
        audio, sr = sf.read(audio_path)
        
        # 基本信息
        info['sample_rate'] = sr
        info['channels'] = len(audio.shape) if len(audio.shape) > 1 else 1
        info['duration'] = len(audio) / sr
        info['samples'] = len(audio)
        
        # 检查采样率
        if sr not in [16000, 22050, 44100, 48000]:
            warnings.append(f"建议采样率: 22050Hz 或 44100Hz, 当前: {sr}Hz")
        
        # 检查声道
        if info['channels'] != 1:
            warnings.append(f"建议单声道音频, 当前: {info['channels']}声道")
        
        # 检查时长
        if info['duration'] < 5:
            warnings.append(f"时长过短({info['duration']:.1f}s), 建议至少5秒")
        elif info['duration'] > 30:
            warnings.append(f"时长过长({info['duration']:.1f}s), 建议不超过30秒")
        
        # 检查音量
        rms = np.sqrt(np.mean(audio**2))
        db = 20 * np.log10(rms + 1e-10)
        info['rms_db'] = db
        
        if db < -40:
            warnings.append(f"音量过低({db:.1f}dB), 可能影响克隆效果")
        elif db > -10:
            warnings.append(f"音量过高({db:.1f}dB), 可能有削波失真")
        
        # 检查静音
        silent_frames = np.sum(np.abs(audio) < 0.01)
        silent_ratio = silent_frames / len(audio)
        if silent_ratio > 0.3:
            warnings.append(f"静音比例过高({silent_ratio*100:.1f}%), 请剪辑掉多余静音")
        
        # 简单信噪比估计
        # 假设信号中前100ms为静音段来估计噪声
        noise_samples = min(int(0.1 * sr), len(audio) // 10)
        if noise_samples > 100:
            noise_floor = np.std(audio[:noise_samples])
            signal_level = np.std(audio)
            snr = 20 * np.log10(signal_level / (noise_floor + 1e-10))
            info['estimated_snr'] = snr
            
            if snr < 10:
                warnings.append(f"信噪比较低({snr:.1f}dB), 背景噪音可能较明显")
        
        is_valid = len(warnings) == 0
        
    except Exception as e:
        return False, [f"读取音频失败: {e}"], {}
    
    return is_valid, warnings, info


def validate_reference_audio(audio_path: str, metadata: Dict = None) -> Dict:
    """验证单个参考音频"""
    result = {
        'path': audio_path,
        'filename': os.path.basename(audio_path),
        'valid': False,
        'warnings': [],
        'info': {}
    }
    
    # 检查文件存在
    if not os.path.exists(audio_path):
        result['warnings'].append("文件不存在")
        return result
    
    # 检查格式
    if not audio_path.endswith('.wav'):
        result['warnings'].append("仅支持WAV格式")
    
    # 检查音频质量
    is_valid, warnings, info = check_audio_quality(audio_path)
    result['valid'] = is_valid
    result['warnings'] = warnings
    result['info'] = info
    
    return result


def scan_directory(directory: str) -> List[Dict]:
    """扫描目录中的所有音频文件"""
    results = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.wav'):
                path = os.path.join(root, file)
                result = validate_reference_audio(path)
                results.append(result)
    
    return results


def print_report(results: List[Dict]):
    """打印验证报告"""
    print("=" * 80)
    print("参考音频验证报告")
    print("=" * 80)
    
    valid_count = sum(1 for r in results if r['valid'])
    invalid_count = len(results) - valid_count
    
    print(f"\n总计: {len(results)} 个文件")
    print(f"通过: {valid_count} 个")
    print(f"警告: {invalid_count} 个")
    
    print("\n" + "-" * 80)
    print("详细结果:")
    print("-" * 80)
    
    for result in results:
        status = "✅ 通过" if result['valid'] else "⚠️  警告"
        print(f"\n{status} {result['filename']}")
        
        if result['info']:
            info = result['info']
            print(f"   采样率: {info.get('sample_rate', 'N/A')}Hz")
            print(f"   声道: {info.get('channels', 'N/A')}")
            print(f"   时长: {info.get('duration', 0):.2f}s")
            print(f"   音量: {info.get('rms_db', 0):.1f}dB")
        
        if result['warnings']:
            for warning in result['warnings']:
                print(f"   ⚠️  {warning}")
    
    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description='参考音频验证工具')
    parser.add_argument('path', nargs='?', default='reference_audio',
                       help='要验证的音频文件或目录')
    parser.add_argument('--fix', action='store_true',
                       help='尝试自动修复问题')
    parser.add_argument('--json', action='store_true',
                       help='输出JSON格式结果')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.path):
        print(f"错误: 路径不存在 {args.path}")
        sys.exit(1)
    
    # 验证单个文件
    if os.path.isfile(args.path):
        result = validate_reference_audio(args.path)
        results = [result]
    else:
        # 验证整个目录
        results = scan_directory(args.path)
    
    # 输出结果
    if args.json:
        print(json.dumps(results, indent=2, ensure_ascii=False))
    else:
        print_report(results)
    
    # 返回码
    invalid_count = sum(1 for r in results if not r['valid'])
    sys.exit(0 if invalid_count == 0 else 1)


if __name__ == '__main__':
    main()
