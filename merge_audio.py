#!/usr/bin/env python3
"""
将一个文件夹中的大量 TTS 音频文件合并为单个大音频文件。

设计要点：
  - 输入：存放 wav/mp3/flac/ogg 等音频文件的文件夹路径。
  - 输出：合并后的单个音频文件（默认 wav，可指定 mp3 等）。
  - 不同 TTS 模型产出的采样率不一致（项目内已知 22050 / 24000 / 32000 Hz），
    直接拼接会变调/变速，因此脚本会统一重采样到目标采样率后再拼接。
  - 声道统一为单声道：多声道文件按下混（取均值）转为单声道，确保拼接维度一致。
  - 排序：优先按文件名中的 `_NNofMM` 批次序号排序；没有序号则按文件名字典序。
    理论上也可按文件 mtime 排序（保留为可选参数）。
  - 间隔静音：每段之间可插入若干毫秒静音，避免句首句尾粘连（默认 300ms）。
  - 音量归一化：复用项目 audio_utils.normalize_audio_volume，使各段响度一致。
  - 依赖：numpy / soundfile / soxr / scipy（项目 requirements 已含），无需 ffmpeg。

用法示例：
    python merge_audio.py "D:\\path\\to\\audio_folder"
    python merge_audio.py "D:\\path\\to\\audio_folder" -o merged.mp3 --sr 24000 --gap 400
    python merge_audio.py "D:\\path\\to\\audio_folder" --no-normalize --sort mtime --gap 0
"""

import argparse
import os
import re
import sys
from datetime import datetime

import numpy as np
import soundfile as sf

# 复用项目内的音量归一化与配置（若作为独立脚本脱离项目运行，下面有兜底实现）
try:
    from backend.core.audio_utils import normalize_audio_volume
except Exception:
    def normalize_audio_volume(audio_data: np.ndarray, target_db: float = -0.5) -> np.ndarray:
        """兜底实现：归一化音频音量到目标 dB。"""
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        current_peak = np.max(np.abs(audio_data)) if audio_data.size else 0.0
        if current_peak == 0:
            return audio_data
        gain = (10 ** (target_db / 20.0)) / current_peak
        return np.clip(audio_data * gain, -1.0, 1.0)


AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}

# 匹配文件名中的批次序号，如 "..._03of10.wav"  ->  3
_SEQ_RE = re.compile(r"_(\d+)of\d+", re.IGNORECASE)


def parse_args():
    p = argparse.ArgumentParser(
        description="合并一个文件夹中的 TTS 音频为单个音频文件。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("input_dir", help="存放音频文件的文件夹路径")
    p.add_argument("-o", "--output", default=None,
                   help="合并后输出文件路径（默认：输入目录旁的 merged_<时间>.wav）")
    p.add_argument("--sr", "--sample-rate", type=int, default=None, dest="target_sr",
                   help="目标采样率；默认取输入文件中的最大采样率，保证不降采样损失")
    p.add_argument("--gap", type=float, default=300.0,
                   help="每段之间插入的静音时长（毫秒），默认 300；设 0 表示不插入")
    p.add_argument("--sort", choices=["seq", "name", "mtime"], default="seq",
                   help="排序方式：seq=按文件名序号（默认）｜name=文件名字典序｜mtime=修改时间")
    p.add_argument("--no-normalize", action="store_true",
                   help="关闭音量归一化（默认开启）")
    p.add_argument("--target-db", type=float, default=-0.5,
                   help="归一化目标 dB，默认 -0.5")
    p.add_argument("--recursive", "-r", action="store_true",
                   help="递归收集子目录中的音频文件")
    return p.parse_args()


def collect_files(input_dir: str, recursive: bool):
    files = []
    if recursive:
        for root, _, names in os.walk(input_dir):
            for n in names:
                if os.path.splitext(n)[1].lower() in AUDIO_EXTS:
                    files.append(os.path.join(root, n))
    else:
        for n in os.listdir(input_dir):
            fp = os.path.join(input_dir, n)
            if os.path.isfile(fp) and os.path.splitext(n)[1].lower() in AUDIO_EXTS:
                files.append(fp)
    return files


def sort_key(path: str, mode: str):
    base = os.path.basename(path)
    if mode == "mtime":
        return (os.path.getmtime(path), base)
    if mode == "seq":
        m = _SEQ_RE.search(base)
        if m:
            # 有序号的优先按序号排，序号相同时兜底按文件名
            return (0, int(m.group(1)), base)
        # 没有序号的退化为字典序，但排在有序号文件之后（用 1 标记）
        return (1, 0, base)
    # name
    return (0, base)


def resample(audio: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """高质量重采样（输入为单声道 1D 数组）。优先 soxr，其次 scipy，最后 librosa。"""
    if sr_in == sr_out or audio.size == 0:
        return audio
    try:
        import soxr
        return soxr.resample(audio, sr_in, sr_out)
    except Exception:
        pass
    try:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(sr_in, sr_out)
        return resample_poly(audio, sr_out // g, sr_in // g).astype(np.float32)
    except Exception:
        pass
    try:
        import librosa
        return librosa.resample(audio, orig_sr=sr_in, target_sr=sr_out)
    except Exception as e:
        raise RuntimeError(f"重采样失败且无可用的重采样库：{e}")


def to_mono(audio: np.ndarray) -> np.ndarray:
    """多声道转单声道（取均值）；已是单声道则原样返回。"""
    if audio.ndim == 1:
        return audio
    return audio.mean(axis=1)


def main():
    args = parse_args()

    input_dir = args.input_dir
    if not os.path.isdir(input_dir):
        print(f"[错误] 输入路径不是文件夹：{input_dir}", file=sys.stderr)
        sys.exit(1)

    files = collect_files(input_dir, args.recursive)
    if not files:
        print(f"[错误] 文件夹中未找到音频文件：{input_dir}", file=sys.stderr)
        sys.exit(1)

    files.sort(key=lambda p: sort_key(p, args.sort))
    print(f"[信息] 共找到 {len(files)} 个音频文件，排序方式：{args.sort}")

    # 第一遍：读取元数据，确定目标采样率
    infos = []  # [(path, sr, channels)]
    for fp in files:
        try:
            info = sf.info(fp)
        except Exception as e:
            print(f"[警告] 跳过无法读取的文件 {fp}: {e}", file=sys.stderr)
            continue
        infos.append((fp, info.samplerate, info.channels))

    if not infos:
        print("[错误] 没有可合并的有效音频文件。", file=sys.stderr)
        sys.exit(1)

    sample_rates = {sr for _, sr, _ in infos}
    target_sr = args.target_sr or max(sample_rates)
    print(f"[信息] 输入采样率集合：{sorted(sample_rates)} -> 目标采样率：{target_sr}")

    # 第二遍：逐个读取、重采样、转单声道、归一化、拼接
    pieces = []
    total_sec = 0.0
    if args.gap and args.gap > 0:
        silence = np.zeros(int(target_sr * args.gap / 1000.0), dtype=np.float32)
    else:
        silence = None

    for idx, (fp, sr, _channels) in enumerate(infos, 1):
        audio, _ = sf.read(fp, dtype="float32", always_2d=False)
        audio = to_mono(audio).astype(np.float32, copy=False)
        if sr != target_sr:
            audio = resample(audio, sr, target_sr).astype(np.float32, copy=False)
        if not args.no_normalize:
            audio = normalize_audio_volume(audio, target_db=args.target_db)
        pieces.append(audio)
        total_sec += len(audio) / target_sr
        if silence is not None and idx < len(infos):
            pieces.append(silence)

    merged = np.concatenate(pieces, axis=0) if pieces else np.zeros(0, dtype=np.float32)
    print(f"[信息] 合并完成：总时长 {total_sec:.2f}s，总采样点 {len(merged)}，"
          f"采样率 {target_sr} Hz")

    # 输出路径
    if args.output:
        out_path = args.output
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(os.path.dirname(os.path.abspath(input_dir)),
                                f"merged_{ts}.wav")
    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    fmt_ext = os.path.splitext(out_path)[1].lower()
    if fmt_ext in (".mp3", ".ogg", ".m4a", ".aac"):
        # soundfile 不支持有损编码写出，转用 ffmpeg（若存在）做最后一步封装
        tmp_wav = out_path + ".tmp.wav"
        sf.write(tmp_wav, merged, target_sr, subtype="PCM_16")
        import shutil
        if shutil.which("ffmpeg"):
            import subprocess
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", tmp_wav, out_path],
                check=True,
            )
            os.remove(tmp_wav)
            print(f"[信息] 已通过 ffmpeg 转码为 {fmt_ext}")
        else:
            print(f"[警告] 未找到 ffmpeg，无法写出 {fmt_ext}，已保留临时 wav：{tmp_wav}",
                  file=sys.stderr)
            out_path = tmp_wav
    else:
        subtype = "PCM_16" if fmt_ext in (".wav", "") else None
        sf.write(out_path, merged, target_sr, subtype=subtype)

    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"[完成] 已输出：{out_path}（{size_mb:.2f} MB）")


if __name__ == "__main__":
    main()
