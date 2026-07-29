#!/usr/bin/env python3
"""
存量有效音频筛选脚本（需求1）。

扫描 outputs 输出目录下所有已生成的 .wav 音频，逐个调用 wenet ASR 识别文本，
将识别文本与预设的目标词清单逐一计算字符相似度，取最高分作为该音频的"内容契合度"：
    - 最高相似度 >= 阈值：内容包含目标词（即有效/想要的语音）→ 复制一份到 outputs/pass/
    - 最高相似度 <  阈值：内容跑偏（未说出目标词）           → 复制一份到 outputs/failed/

注意：仅"复制"，原 wav 文件保留在 outputs 根目录不动，便于复核与回溯。

用法:
    # 默认：阈值 0.5，命中的复制到 pass/，跑偏的复制到 failed/
    python scripts/filter_effective_audio.py

    # 仅预览（不复制，只生成清单），用于先核对
    python scripts/filter_effective_audio.py --dry-run

    # 指定输出目录、清单路径、阈值
    python scripts/filter_effective_audio.py --outputs-dir /path/to/outputs --report result.csv --threshold 0.6

    # 自定义目标词清单
    python scripts/filter_effective_audio.py --words 救命 打你 打人

注意:
    - 仅扫描 .wav 文件；outputs 中的 .zip 不会展开；pass/、failed/ 子目录不会被重复扫描。
    - ASR 模型首次加载较慢（wenetspeech），后续逐条识别为 CPU 推理。
    - 相似度基准为"目标词清单"：ASR 文本与清单中每个词算相似度取最高值。
"""

import argparse
import csv
import difflib
import os
import re
import shutil
import sys
from datetime import datetime

# 确保项目根目录在 sys.path 中，以便 `from backend...` 可导入
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# 默认目标词清单（需求1）：这些词正是期望在生成语音中出现的
DEFAULT_SENSITIVE_WORDS = [
    "救命",
    "打你",
    "找人弄你",
    "整死你",
    "打死你",
    "打人",
    "别打了",
    "放过我",
    "弄死你",
]

# 默认相似度阈值（需求1更宽松；需求2另用 0.6）
DEFAULT_THRESHOLD = 0.6


def _normalize_text(text: str) -> str:
    """文本归一化：去除标点、空白、常见全半角差异，统一用于相似度计算。

    ASR 结果可能带有逗号、句号、问号等标点，直接比对会拉低相似度；
    这里只保留中文与字母数字并转小写，使比对更稳健。
    """
    if not text:
        return ""
    # 仅保留中文、英文字母、数字，其余全部移除
    cleaned = re.sub(r"[^\w一-鿿]", "", text)
    return cleaned.lower()


def _substr_similarity(asr_text: str, word: str) -> float:
    """计算 ASR 文本对单个目标词的"包含相似度"。

    归一化后：
        - 若目标词作为子串精确出现在 ASR 文本中 → 1.0（确信说出该词）；
        - 否则在 ASR 文本上滑动一个与目标词等长的窗口，取与目标词最相似的一段
          的 SequenceMatcher 比值，用于容错 ASR 错字（如"救命"识别成"救民"）。
    返回 [0, 1]。相比整体 ratio()，不会被长 ASR 文本稀释短目标词的得分。
    """
    a, w = _normalize_text(asr_text), _normalize_text(word)
    if not a or not w:
        return 0.0
    if w in a:
        return 1.0
    L = len(w)
    if len(a) < L:
        # ASR 比目标词还短，直接整体比对
        return difflib.SequenceMatcher(None, w, a).ratio()
    best = 0.0
    for i in range(len(a) - L + 1):
        s = difflib.SequenceMatcher(None, w, a[i:i + L]).ratio()
        if s > best:
            best = s
            if best >= 1.0:
                break
    return best


def _best_match(asr_text: str, words: list) -> tuple:
    """返回 (最高相似度, 最匹配的目标词)。"""
    best_score = 0.0
    best_word = ""
    for w in words:
        score = _substr_similarity(asr_text, w)
        if score > best_score:
            best_score = score
            best_word = w
    return best_score, best_word


def _iter_wav_files(outputs_dir: str, skip_dirs=()):
    """遍历输出目录下的 .wav 文件（平铺，不递归），跳过指定子目录，按文件名排序便于复核。"""
    if not os.path.isdir(outputs_dir):
        return
    for name in sorted(os.listdir(outputs_dir)):
        if name in skip_dirs:
            continue
        if name.lower().endswith(".wav"):
            full = os.path.join(outputs_dir, name)
            if os.path.isfile(full):
                yield full


def _unique_dest(dest_dir: str, filename: str) -> str:
    """在目标目录下生成不冲突的文件名（同名则追加序号），避免复制时覆盖。"""
    dest = os.path.join(dest_dir, filename)
    if not os.path.exists(dest):
        return dest
    stem, ext = os.path.splitext(filename)
    i = 1
    while True:
        candidate = os.path.join(dest_dir, f"{stem}_{i}{ext}")
        if not os.path.exists(candidate):
            return candidate
        i += 1


def main():
    parser = argparse.ArgumentParser(
        description="扫描 outputs 目录，用 wenet ASR 识别音频，按相似度分流到 pass/failed（复制不移动）"
    )
    parser.add_argument(
        "--outputs-dir", default=None,
        help="音频输出目录，默认读取 backend.config.OUTPUTS_DIR"
    )
    parser.add_argument(
        "--words", nargs="*", default=None,
        help="目标词列表（空格分隔），作为相似度比对基准；不传则使用内置清单"
    )
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_THRESHOLD,
        help=f"相似度阈值，>= 该值判为 pass（含目标词），< 该值判为 failed（跑偏）；默认 {DEFAULT_THRESHOLD}"
    )
    parser.add_argument(
        "--report", default=None,
        help="判别结果 CSV 输出路径；默认 outputs 目录下 filter_report_<时间戳>.csv"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="仅预览：生成清单但不复制文件"
    )
    args = parser.parse_args()

    # 解析输出目录：优先命令行，其次配置
    if args.outputs_dir:
        outputs_dir = os.path.abspath(args.outputs_dir)
    else:
        from backend.config import OUTPUTS_DIR
        outputs_dir = OUTPUTS_DIR

    if not os.path.isdir(outputs_dir):
        print(f"[错误] 输出目录不存在: {outputs_dir}")
        sys.exit(1)

    words = args.words if args.words else DEFAULT_SENSITIVE_WORDS
    pass_dir = os.path.join(outputs_dir, "pass")
    failed_dir = os.path.join(outputs_dir, "failed")
    threshold = args.threshold
    print(f"[配置] 输出目录: {outputs_dir}")
    print(f"[配置] 目标词 ({len(words)}): {words}")
    print(f"[配置] 相似度阈值: {threshold}（>=pass，<failed）")
    print(f"[配置] pass 目录: {pass_dir}")
    print(f"[配置] failed 目录: {failed_dir}")
    print(f"[配置] 模式: {'预览(--dry-run，不复制)' if args.dry_run else '正式执行（复制一份到对应目录，原文件保留）'}")

    if not args.dry_run:
        os.makedirs(pass_dir, exist_ok=True)
        os.makedirs(failed_dir, exist_ok=True)

    # 懒加载 ASR（首次调用加载 wenetspeech，常驻）
    from backend.services.asr_service import transcribe
    print("[ASR] 首次调用将加载 wenetspeech 模型，请稍候...")

    # 跳过 pass/failed 子目录，避免重复扫描已分流的音频
    wav_files = list(_iter_wav_files(outputs_dir, skip_dirs=("pass", "failed")))
    total = len(wav_files)
    print(f"[扫描] 发现 .wav 文件: {total} 个（已排除 pass/、failed/ 子目录）")
    if total == 0:
        print("[完成] 无可扫描音频。")
        return

    # 清单路径
    if args.report:
        report_path = os.path.abspath(args.report)
    else:
        report_path = os.path.join(
            outputs_dir, f"filter_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )

    rows = []  # [(filename, verdict, best_score, best_word, asr_text)]
    scanned = 0
    pass_count = 0
    failed_count = 0

    for idx, wav_path in enumerate(wav_files, 1):
        filename = os.path.basename(wav_path)
        try:
            asr_text = transcribe(wav_path)
        except Exception as e:
            print(f"  [{idx}/{total}] {filename} | ASR失败: {e}")
            scanned += 1
            rows.append((filename, "asr_error", "", "", f"ASR失败: {e}"))
            continue

        scanned += 1
        best_score, best_word = _best_match(asr_text, words)
        is_pass = best_score >= threshold
        verdict = "pass" if is_pass else "failed"

        if is_pass:
            pass_count += 1
        else:
            failed_count += 1
        print(f"  [{idx}/{total}] {filename} | {verdict} | 相似度={best_score:.2f} "
              f"(目标词='{best_word}') | '{asr_text}'")

        rows.append((filename, verdict, f"{best_score:.3f}", best_word, asr_text))

        if args.dry_run:
            continue

        # 复制一份到对应目录，原文件保留
        target_dir = pass_dir if is_pass else failed_dir
        try:
            dest = _unique_dest(target_dir, filename)
            shutil.copy2(wav_path, dest)
        except OSError as e:
            print(f"           -> 复制失败: {filename} -> {target_dir} | {e}")

    # 写出 CSV 清单
    with open(report_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "verdict", "best_similarity", "best_match_word", "asr_text"])
        for row in rows:
            writer.writerow(row)
    print(f"\n[清单] 共 {len(rows)} 条，已写入: {report_path}")

    print(f"[统计] 扫描 {scanned}/{total}，pass={pass_count}，failed={failed_count}"
          f"{'' if args.dry_run else f'（已复制到 {pass_dir} / {failed_dir}，原文件保留）'}")
    print("[完成]")


if __name__ == "__main__":
    main()
