#!/usr/bin/env python3
"""
语音数据增强：在纯净人声上构造带噪 / 不同拾音距离 / 不同混响 的训练样本。

三类增强（按真实声学链路顺序执行）：
  1) 加混响      —— 合成房间脉冲响应(RIR)并 FFT 卷积，模拟不同 RT60 的房间。
  2) 拾音距离    —— 按距离衰减增益 (-20·log10(d/ref) dB)，可选空气吸收低通，
                    模仿话筒离声源远近。
  3) 加噪        —— 稳态噪声(white/pink/brown)或瞬态事件(click/clap/knock)，
                    按目标信噪比 SNR(dB) 缩放混入。

所有处理仅依赖 numpy + soundfile(+ 可选 scipy 加速)，无需额外安装。

参数规范（每个数值类参数都支持三种写法）：
    15          固定值
    5,10,15     从列表中随机采样一个
    5:20        在 [5, 20] 区间均匀随机采样（含端点）

用法示例：
    # 单文件，加粉噪 SNR 在 5~20 dB 随机，每个样本随机一组参数，生成 5 个副本
    python augment_audio.py clean.wav --noise-type pink --snr 5:20 --variants 5

    # 文件夹批量，叠混响 + 距离 + 瞬态敲击噪声，输出到 augmented/
    python augment_audio.py clean_dir/ -o augmented/ --rt60 0.2:0.8 \
        --distance 0.5:3.0 --transient knock --n-events 1:5 --snr 5:15

    # 只加距离衰减，不加噪不混响
    python augment_audio.py clean.wav --distance 1.0:4.0 --no-noise --no-reverb
"""

import argparse
import math
import os
import random
import sys

import numpy as np
import soundfile as sf


# 复用项目内峰值归一化；脱离项目运行时有兜底实现
try:
    from backend.core.audio_utils import normalize_audio_volume
except Exception:
    def normalize_audio_volume(audio_data: np.ndarray, target_db: float = -1.0) -> np.ndarray:
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        current_peak = np.max(np.abs(audio_data)) if audio_data.size else 0.0
        if current_peak == 0:
            return audio_data
        gain = (10 ** (target_db / 20.0)) / current_peak
        return np.clip(audio_data * gain, -1.0, 1.0)


# ----------------------------- 参数采样 ----------------------------- #

def parse_param(spec, rng: random.Random):
    """解析参数规范并采样：
        '15'      -> 15
        '5,10,15' -> 从列表随机
        '5:20'    -> [5,20] 均匀随机（浮点）
    支持字符串或已是数字的输入。
    """
    if spec is None:
        return None
    if isinstance(spec, (int, float)):
        return float(spec)
    s = str(spec).strip()
    if "," in s:
        vals = [float(x) for x in s.split(",")]
        return rng.choice(vals)
    if ":" in s:
        lo, hi = s.split(":")
        return rng.uniform(float(lo), float(hi))
    return float(s)


def sample_int(spec, rng: random.Random):
    """同 parse_param，但结果取整（用于事件个数等离散参数）。"""
    v = parse_param(spec, rng)
    return int(round(v)) if v is not None else None


# ----------------------------- 噪声生成 ----------------------------- #

def gen_steady_noise(n: int, sr: int, kind: str = "white",
                     rng: np.random.Generator = None) -> np.ndarray:
    """生成稳态噪声（单声道 float32）。
    white: 均匀白噪声；pink: 1/√f 幅度（粉噪近似）；brown: 1/f 幅度（褐噪近似）。
    用频域幅度整形实现，质量好于简单差分。"""
    if rng is None:
        rng = np.random.default_rng()
    if kind == "white":
        return rng.standard_normal(n).astype(np.float32) * 0.1

    # 频域幅度整形：白噪声 → FFT → 乘幅度谱 → IFFT
    white = rng.standard_normal(n).astype(np.float32)
    spec = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    # 避免 0 频除零
    freqs[0] = freqs[1] if n > 1 else 1.0
    if kind == "pink":
        amp = 1.0 / np.sqrt(np.abs(freqs))
    elif kind == "brown":
        amp = 1.0 / np.abs(freqs)
    else:
        amp = np.ones_like(freqs)
    spec = spec * amp
    noise = np.fft.irfft(spec, n=n).astype(np.float32)
    # 归一化到合理幅度
    peak = np.max(np.abs(noise)) or 1.0
    return (noise / peak * 0.3).astype(np.float32)


def gen_transient_event(n_event: int, sr: int, kind: str = "click",
                        rng: np.random.Generator = None) -> np.ndarray:
    """生成单个瞬态事件波形。
    click: 极短脉冲；clap: 宽带冲击 + 快速衰减；knock: 低频冲击 + 中速衰减（带轻微共振）。"""
    if rng is None:
        rng = np.random.default_rng()
    if kind == "click":
        length = max(1, int(sr * 0.003))  # 3ms
        env = np.ones(length, dtype=np.float32)
        sig = env * (rng.standard_normal(length) * 0.9)
        return sig.astype(np.float32)
    if kind == "clap":
        length = int(sr * 0.05)  # 50ms
        t = np.arange(length) / sr
        env = np.exp(-t * 60.0)
        sig = env * (rng.standard_normal(length) * 0.9)
        return sig.astype(np.float32)
    if kind == "knock":
        length = int(sr * 0.08)  # 80ms
        t = np.arange(length) / sr
        env = np.exp(-t * 30.0)
        # 150Hz 附近共振，模拟敲击木/门
        carrier = np.sin(2 * np.pi * 150 * t) + 0.4 * rng.standard_normal(length)
        sig = env * carrier * 0.8
        return sig.astype(np.float32)
    # 默认 click
    return gen_transient_event(n_event, sr, "click", rng)


def gen_transient_track(n: int, sr: int, n_events: int, kind: str = "clap",
                        rng: np.random.Generator = None) -> np.ndarray:
    """在长度 n 的轨道上随机放置 n_events 个瞬态事件，返回与信号等长的噪声轨。"""
    if rng is None:
        rng = np.random.default_rng()
    track = np.zeros(n, dtype=np.float32)
    if n_events <= 0:
        return track
    for _ in range(n_events):
        evt = gen_transient_event(n, sr, kind, rng)
        # 随机起始位置，留出事件长度
        start = rng.integers(0, max(1, n - len(evt)))
        track[start:start + len(evt)] += evt
    peak = np.max(np.abs(track)) or 1.0
    return (track / peak * 0.9).astype(np.float32)


# ----------------------------- SNR 混合 ----------------------------- #

def signal_power(x: np.ndarray) -> float:
    """信号功率（均方）。"""
    if x.size == 0:
        return 0.0
    return float(np.mean(x.astype(np.float64) ** 2))


def mix_at_snr(signal: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """按目标 SNR(dB) 将噪声缩放后混入信号。
    SNR = 10·log10(P_signal / P_noise)  =>  P_noise = P_signal · 10^(-SNR/10)
    noise 会被截断/补齐到与 signal 等长。"""
    n = len(signal)
    if len(noise) >= n:
        noise = noise[:n]
    else:
        noise = np.pad(noise, (0, n - len(noise)))
    ps = signal_power(signal)
    pn = signal_power(noise)
    if ps <= 0 or pn <= 0:
        return signal.astype(np.float32, copy=True)
    target_pn = ps * (10.0 ** (-snr_db / 10.0))
    scale = math.sqrt(target_pn / pn)
    return (signal + noise * scale).astype(np.float32)


# ----------------------------- 拾音距离 ----------------------------- #

def lowpass_biquad_simple(x: np.ndarray, sr: int, cutoff: float) -> np.ndarray:
    """简单一阶 RC 低通（空气吸收近似）。无需 scipy。"""
    if cutoff <= 0 or cutoff >= sr / 2:
        return x
    dt = 1.0 / sr
    rc = 1.0 / (2 * math.pi * cutoff)
    alpha = dt / (rc + dt)
    y = np.empty_like(x)
    prev = 0.0
    for i in range(len(x)):
        prev = prev + alpha * (x[i] - prev)
        y[i] = prev
    return y


def simulate_distance(audio: np.ndarray, sr: int, distance: float,
                      ref: float = 1.0, air_absorption: bool = True) -> np.ndarray:
    """模仿拾音距离：
      - 平方反比衰减近似为 -20·log10(d/ref) dB 增益；
      - 可选空气吸收：距离越远高频衰减越多，用一个距离相关的低通模拟。"""
    if distance <= 0:
        distance = 1e-3
    gain = (ref / distance) ** 1.0  # 等价于 -20log10(d/ref) 的线性增益
    out = audio * gain
    if air_absorption and distance > ref:
        # 距离越大，低通截止越低（粗略：每米距离 cutoff 下降）
        cutoff = max(2000.0, 12000.0 - 1500.0 * (distance - ref))
        out = lowpass_biquad_simple(out.astype(np.float32), sr, cutoff)
    return out.astype(np.float32)


# ----------------------------- 混响 ----------------------------- #

def synth_rir(sr: int, rt60: float, direct_gain: float = 1.0,
              early_reflections: bool = True,
              rng: np.random.Generator = None) -> np.ndarray:
    """合成房间脉冲响应(RIR)。
    用「直达冲激 + 衰减白噪声」构造指数衰减的晚期混响尾；
    RT60 决定衰减常数：alpha = 3 / (sr * rt60)（衰减到 -60dB）。
    可选叠加几个早期反射（延迟 + 衰减）。"""
    if rng is None:
        rng = np.random.default_rng()
    if rt60 <= 0:
        return np.array([direct_gain], dtype=np.float32)

    length = int(sr * rt60) + 1
    t = np.arange(length) / sr
    alpha = 3.0 / (sr * rt60)  # 使 exp(-alpha·length) ≈ 10^(-3) = -60dB
    env = np.exp(-alpha * np.arange(length))
    late = (rng.standard_normal(length) * env).astype(np.float32)

    rir = late.copy()
    # 直达声
    rir[0] += direct_gain
    # 早期反射：几个随机延迟的衰减冲激
    if early_reflections:
        for frac, g in [(0.012, 0.6), (0.022, 0.45), (0.035, 0.35), (0.05, 0.25)]:
            idx = int(frac * sr)
            if 0 < idx < length:
                rir[idx] += g * direct_gain
    # 归一化
    peak = np.max(np.abs(rir)) or 1.0
    return (rir / peak).astype(np.float32)


def fft_convolve(signal: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """FFT 卷积，返回全长输出（保留尾部混响）。优先 scipy，否则用 numpy。"""
    try:
        from scipy.signal import fftconvolve
        return fftconvolve(signal, kernel).astype(np.float32)
    except Exception:
        n = len(signal) + len(kernel) - 1
        # 零填充到 2 的幂加速
        nfft = 1 << (n - 1).bit_length()
        s = np.fft.rfft(signal, nfft)
        k = np.fft.rfft(kernel, nfft)
        out = np.fft.irfft(s * k, nfft)[:n]
        return out.astype(np.float32)


def add_reverb(audio: np.ndarray, sr: int, rt60: float,
               wet: float = 0.3, rng: np.random.Generator = None) -> np.ndarray:
    """给音频加混响：dry·(1-wet) + wet·conv(audio, rir)。"""
    if rt60 <= 0:
        return audio.astype(np.float32, copy=True)
    rir = synth_rir(sr, rt60, rng=rng)
    wet_sig = fft_convolve(audio.astype(np.float32), rir)
    # 对齐到原信号长度（裁掉尾部，保持时长不变）
    wet_sig = wet_sig[:len(audio)]
    dry = audio.astype(np.float32)
    return (dry * (1.0 - wet) + wet_sig * wet).astype(np.float32)


# ----------------------------- 主流程 ----------------------------- #

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}


def parse_args():
    p = argparse.ArgumentParser(
        description="语音数据增强：加噪 / 拾音距离 / 混响。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("input", help="输入文件或文件夹路径")
    p.add_argument("-o", "--output", default=None,
                   help="输出目录（默认：输入旁的 augmented/）")
    p.add_argument("--variants", "-n", type=int, default=1,
                   help="每个输入文件生成几个随机参数副本，默认 1")
    p.add_argument("--recursive", "-r", action="store_true", help="递归收集子目录音频")
    p.add_argument("--seed", type=int, default=None, help="随机种子，可复现")

    # 加噪
    p.add_argument("--noise-type", choices=["white", "pink", "brown", "click", "clap", "knock", "none"],
                   default="pink", help="稳态噪声类型或瞬态事件类型，默认 pink；none=不加噪")
    p.add_argument("--snr", default="10", help="信噪比 dB 规范，如 15 / 5,10,15 / 5:20，默认 10")
    p.add_argument("--transient", action="store_true",
                   help="启用瞬态噪声（--noise-type 解释为 click/clap/knock）")
    p.add_argument("--n-events", default="1:3",
                   help="瞬态事件个数规范（仅 --transient 时生效），默认 1:3")
    p.add_argument("--no-noise", action="store_true", help="跳过加噪")

    # 拾音距离
    p.add_argument("--distance", default=None,
                   help="拾音距离(米)规范，如 2.0 / 0.5,1.5,3.0 / 0.5:4.0；不填=不处理")
    p.add_argument("--ref-distance", type=float, default=1.0, help="参考距离(米)，默认 1.0")
    p.add_argument("--no-air-absorption", action="store_true", help="关闭空气吸收低通")
    p.add_argument("--no-distance", action="store_true", help="跳过距离模拟")

    # 混响
    p.add_argument("--rt60", default=None,
                   help="混响时间 RT60(秒)规范，如 0.4 / 0.2,0.5,0.8 / 0.1:0.9；不填=不加混响")
    p.add_argument("--wet", default="0.3", help="混响湿声比例规范，默认 0.3")
    p.add_argument("--no-reverb", action="store_true", help="跳过混响")

    # 输出
    p.add_argument("--target-db", type=float, default=-1.0, help="末端峰值归一化目标 dB，默认 -1.0")
    p.add_argument("--no-normalize", action="store_true", help="关闭末端归一化")
    return p.parse_args()


def collect_files(path: str, recursive: bool):
    if os.path.isfile(path):
        return [path]
    files = []
    if recursive:
        for root, _, names in os.walk(path):
            for n in names:
                if os.path.splitext(n)[1].lower() in AUDIO_EXTS:
                    files.append(os.path.join(root, n))
    else:
        for n in os.listdir(path):
            fp = os.path.join(path, n)
            if os.path.isfile(fp) and os.path.splitext(n)[1].lower() in AUDIO_EXTS:
                files.append(fp)
    return files


def make_tag(rng: random.Random, args, sr: int):
    """采样本次副本的全部参数，返回 (params_dict, tag_string)。"""
    params = {}

    # 加噪
    if not args.no_noise and args.noise_type != "none":
        params["noise_type"] = args.noise_type
        if args.transient:
            params["transient"] = True
            params["n_events"] = sample_int(args.n_events, rng)
        params["snr"] = round(parse_param(args.snr, rng), 1)

    # 距离
    if not args.no_distance and args.distance is not None:
        params["distance"] = round(parse_param(args.distance, rng), 2)

    # 混响
    if not args.no_reverb and args.rt60 is not None:
        params["rt60"] = round(parse_param(args.rt60, rng), 2)
        params["wet"] = round(parse_param(args.wet, rng), 2)

    # 生成标签
    parts = []
    if "noise_type" in params:
        if params.get("transient"):
            parts.append(f"{params['noise_type']}x{params['n_events']}")
        else:
            parts.append(params["noise_type"])
        parts.append(f"snr{params['snr']:.1f}".replace(".", "p"))
    if "distance" in params:
        parts.append(f"d{params['distance']:.1f}m".replace(".0", ""))
    if "rt60" in params:
        parts.append(f"rt{params['rt60']:.2f}".replace(".", ""))
    tag = "_".join(parts) if parts else "clean"
    return params, tag


def augment_one(audio: np.ndarray, sr: int, params: dict, rng: np.random.Generator) -> np.ndarray:
    """对单段音频按参数执行增强链路（混响→距离→加噪→归一化）。"""
    out = audio.astype(np.float32, copy=True)
    n = len(out)

    # 1) 混响
    if "rt60" in params:
        out = add_reverb(out, sr, params["rt60"], wet=params.get("wet", 0.3), rng=rng)

    # 2) 拾音距离
    if "distance" in params:
        out = simulate_distance(out, sr, params["distance"],
                                ref=1.0, air_absorption=not args_no_air)

    # 3) 加噪
    if "noise_type" in params:
        if params.get("transient"):
            noise = gen_transient_track(n, sr, params["n_events"],
                                        kind=params["noise_type"], rng=rng)
        else:
            noise = gen_steady_noise(n, sr, kind=params["noise_type"], rng=rng)
        out = mix_at_snr(out, noise, params["snr"])

    # 4) 末端归一化防削顶
    if not args_no_norm:
        out = normalize_audio_volume(out, target_db=args_target_db)
    else:
        out = np.clip(out, -1.0, 1.0)
    return out


# 模块级占位，避免在 augment_one 内重复传参；main 中赋值
args_no_air = False
args_no_norm = False
args_target_db = -1.0


def main():
    global args_no_air, args_no_norm, args_target_db
    args = parse_args()
    args_no_air = args.no_air_absorption
    args_no_norm = args.no_normalize
    args_target_db = args.target_db

    if not os.path.exists(args.input):
        print(f"[错误] 输入路径不存在：{args.input}", file=sys.stderr)
        sys.exit(1)

    files = collect_files(args.input, args.recursive)
    if not files:
        print(f"[错误] 未找到音频文件：{args.input}", file=sys.stderr)
        sys.exit(1)

    out_dir = args.output or os.path.join(
        os.path.dirname(os.path.abspath(args.input)) or ".",
        "augmented")
    os.makedirs(out_dir, exist_ok=True)

    seed_root = args.seed if args.seed is not None else random.randrange(2 ** 31)
    print(f"[信息] 输入 {len(files)} 个文件，每文件 {args.variants} 个副本，"
          f"输出目录 {out_dir}，种子根 {seed_root}")

    total = 0
    for fp in files:
        try:
            audio, sr = sf.read(fp, dtype="float32", always_2d=False)
        except Exception as e:
            print(f"[警告] 跳过无法读取的文件 {fp}: {e}", file=sys.stderr)
            continue
        if audio.ndim > 1:
            audio = audio.mean(axis=1)  # 增强统一在单声道进行
        base = os.path.splitext(os.path.basename(fp))[0]

        for v in range(args.variants):
            # 每个副本一个独立但可复现的种子流
            rng_py = random.Random(seed_root + total * 7919)
            rng_np = np.random.default_rng(seed_root + total * 7919)
            params, tag = make_tag(rng_py, args, sr)
            out = augment_one(audio, sr, params, rng_np)

            out_name = f"{base}__{tag}.wav"
            out_path = os.path.join(out_dir, out_name)
            sf.write(out_path, out, sr, subtype="PCM_16")
            total += 1
            print(f"  [{total}] {out_name}  (params={params})")

    print(f"[完成] 共生成 {total} 个增强样本 → {out_dir}")


if __name__ == "__main__":
    main()
