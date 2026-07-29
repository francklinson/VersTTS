#!/usr/bin/env python3
"""
音频处理工具函数
"""

import os
import time
import base64
from datetime import datetime
from typing import Optional

import numpy as np
import soundfile as sf

from backend.logger_config import OperationLogger, system_logger


def normalize_audio_volume(audio_data: np.ndarray, target_db: float = -0.5) -> np.ndarray:
    """
    归一化音频音量到目标dB级别
    
    Args:
        audio_data: 输入音频数组
        target_db: 目标dB级别，默认-0.5 dB（接近最大音量）
    
    Returns:
        归一化后的音频数组
    """
    # 确保音频是float32类型
    if audio_data.dtype != np.float32:
        audio_data = audio_data.astype(np.float32)

    # 计算当前峰值
    current_peak = np.max(np.abs(audio_data))

    if current_peak == 0:
        return audio_data  # 避免除零

    # 计算目标峰值（从dB转换为线性比例）
    target_peak = 10 ** (target_db / 20.0)

    # 计算增益因子
    gain = target_peak / current_peak

    # 应用增益
    normalized_audio = audio_data * gain

    # 确保不会溢出（硬限幅）
    normalized_audio = np.clip(normalized_audio, -1.0, 1.0)

    return normalized_audio


def build_meaningful_filename(model: str, mode: str, text: str, index: int = 0, batch_total: int = 1,
                               speaker_name: str = None, prefix: str = None,
                               instruct_prompt: str = None) -> str:
    """
    生成有意义的音频文件名 —— 全局命名规则单一来源。

    格式: {prefix}_{model}_{mode}_{指令摘要}_{speaker}_{text摘要}_{HHMMSS}{_NNofMM}.wav

    指令摘要紧跟 mode 之后（突出指令词），时间戳压缩为时分秒（HHMMSS）放到末尾兜底防重名。
    被 save_temp_audio（即时生成路径）与 task_handlers._generate_meaningful_filename
    （任务队列路径）共同复用，确保两条路径命名规则一致。

    Args:
        model: 模型名称（即时路径中由 prefix 充当）
        mode: 生成模式
        text: 合成文本
        index: 批量生成时的索引
        batch_total: 批量生成总数
        speaker_name: 说话人名称（可选）
        prefix: 前缀（可选，用于区分不同来源）
        instruct_prompt: 指令文本（可选，instruct_text/control_prompt/voice_design_prompt）
    """
    import re

    # 文件名清洗：保留中文、英文、数字、下划线，其余移除
    def _clean(s: str, limit: int) -> str:
        cleaned = re.sub(r'[^\w一-鿿]', '', s or '')
        return cleaned[:limit]

    # 文本摘要（前 8 字符）
    text_summary = _clean(text[:8].strip(), 8) if text else ""
    if not text_summary:
        text_summary = "audio"

    # 指令摘要（前 8 字符），无指令则跳过该段
    instruct_summary = _clean(instruct_prompt[:8].strip(), 8) if instruct_prompt else ""

    # 说话人名称（前 6 字符）
    speaker_part = ""
    if speaker_name:
        clean_name = _clean(speaker_name, 6)
        if clean_name:
            speaker_part = clean_name

    # 时间戳压缩为时分秒，放末尾兜底防重名
    timestamp = datetime.now().strftime("%H%M%S")

    # 构建文件名各部分（顺序：前缀 → 模型 → 模式 → 指令 → 说话人 → 文本 → 时间戳 → 批次）
    parts = []
    if prefix:
        parts.append(prefix)
    parts.append(model)
    if mode:
        parts.append(mode)
    if instruct_summary:
        parts.append(instruct_summary)
    if speaker_part:
        parts.append(speaker_part)
    parts.append(text_summary)
    parts.append(timestamp)

    # 批量生成时添加序号
    if batch_total > 1:
        parts.append(f"{index+1:02d}of{batch_total:02d}")

    return "_".join(parts) + ".wav"


def save_temp_audio(audio_data: np.ndarray, sample_rate: int,
                    suffix: str = ".wav", normalize: bool = True,
                    prefix: str = "tts", mode: str = None,
                    text: str = None, speaker_name: str = None,
                    instruct_prompt: str = None,
                    index: int = 0, batch_total: int = 1) -> str:
    """
    保存临时音频文件

    Args:
        audio_data: 音频数据数组
        sample_rate: 采样率
        suffix: 文件后缀
        normalize: 是否进行音量归一化，默认True
        prefix: 文件名前缀，默认"tts"（即时路径中充当 model 角色）
        mode: 生成模式（可选，用于更有意义的文件名）
        text: 合成文本（可选，用于提取文本摘要到文件名）
        speaker_name: 说话人名称（可选）
        instruct_prompt: 指令文本（可选，写入文件名指令段）
        index: 批量生成时的索引
        batch_total: 批量生成总数
    """
    from backend.config import OUTPUTS_DIR
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    # 统一调用共用命名函数（指令前置、时间戳压缩为 HHMMSS）
    filename = build_meaningful_filename(
        model=prefix, mode=mode, text=text, index=index, batch_total=batch_total,
        speaker_name=speaker_name, instruct_prompt=instruct_prompt
    )
    # 若调用方指定了非默认后缀（如 .flac），替换默认 .wav
    if suffix and suffix != ".wav":
        filename = filename[:-len(".wav")] + suffix if filename.endswith(".wav") else filename + suffix

    temp_path = os.path.join(OUTPUTS_DIR, filename)

    # 音量归一化处理
    if normalize:
        audio_data = normalize_audio_volume(audio_data)

    sf.write(temp_path, audio_data, sample_rate)

    # 记录文件操作
    audio_size = os.path.getsize(temp_path)
    OperationLogger.log_file_operation("保存音频", temp_path, audio_size, "成功")

    # 内容校验（需求2）：ASR 识别后与输入文本比对相似度，不达标则删除并抛 AudioVerifyError。
    # 用 text 参数作为期望文本；校验关闭/无 text/ASR 异常时放行不阻断。
    # 各即时路由的 except Exception 会将该异常透传为 HTTP 500。
    verify_and_cleanup(temp_path, text, model_tag=prefix or "tts")

    return temp_path


def audio_to_base64(audio_path: str) -> str:
    """将音频文件转为base64"""
    with open(audio_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def cleanup_old_outputs(max_age_hours: int = 24) -> int:
    """
    清理 outputs/ 目录中过期的音频文件。

    Args:
        max_age_hours: 最大保留时间（小时），超过此时间的文件将被删除

    Returns:
        清理的文件数量
    """
    from backend.config import OUTPUTS_DIR

    if not os.path.exists(OUTPUTS_DIR):
        return 0

    cutoff = time.time() - (max_age_hours * 3600)
    removed = 0
    total_size = 0

    try:
        for filename in os.listdir(OUTPUTS_DIR):
            filepath = os.path.join(OUTPUTS_DIR, filename)
            if not os.path.isfile(filepath):
                continue
            try:
                mtime = os.path.getmtime(filepath)
                if mtime < cutoff:
                    size = os.path.getsize(filepath)
                    os.remove(filepath)
                    removed += 1
                    total_size += size
            except OSError as e:
                system_logger.warning(f"【清理】删除文件失败 {filepath}: {e}")

        if removed > 0:
            mb = total_size / (1024 * 1024)
            system_logger.info(f"【清理】已清理 outputs/ 目录: {removed} 个文件, {mb:.1f} MB")
    except Exception as e:
        system_logger.error(f"【清理】扫描 outputs/ 目录出错: {e}")

    return removed


def get_outputs_disk_usage() -> dict:
    """获取 outputs/ 目录磁盘使用情况"""
    from backend.config import OUTPUTS_DIR

    result = {"path": OUTPUTS_DIR, "file_count": 0, "total_size_mb": 0, "oldest_file_hours": None}

    if not os.path.exists(OUTPUTS_DIR):
        return result

    now = time.time()
    total_size = 0
    oldest = None

    try:
        for filename in os.listdir(OUTPUTS_DIR):
            filepath = os.path.join(OUTPUTS_DIR, filename)
            if os.path.isfile(filepath):
                total_size += os.path.getsize(filepath)
                result["file_count"] += 1
                mtime = os.path.getmtime(filepath)
                if oldest is None or mtime < oldest:
                    oldest = mtime

        result["total_size_mb"] = round(total_size / (1024 * 1024), 2)
        if oldest:
            result["oldest_file_hours"] = round((now - oldest) / 3600, 1)
    except Exception as e:
        system_logger.error(f"【磁盘】统计 outputs/ 目录出错: {e}")

    return result


# ========== 音频内容校验（需求2）==========
# 新生成音频写入后，用 wenet ASR 识别文本，与原始输入文本计算字符相似度，
# 低于阈值则判定为生成不正确并删除文件。ASR 自身异常时放行不阻断，
# 避免 wenet 故障导致所有 TTS 不可用。

class AudioVerifyError(Exception):
    """音频内容校验失败（ASR 识别文本与期望文本相似度低于阈值）。

    带有 error_code 属性，供任务队列/前端识别为"可重试的内容校验失败"，
    区别于普通生成错误：前端据此显示特殊提示与重试按钮，重试后清除原任务。
    """

    # 错误码：内容校验失败（可重试）
    error_code = "AUDIO_VERIFY_FAILED"
    # 面向用户的特殊提示
    user_message = "音频内容校验未通过（生成的语音与输入文本不符），请重试"


def _normalize_text_for_verify(text: str) -> str:
    """文本归一化：去除标点、空白，保留中文与字母数字并转小写，统一用于相似度比对。"""
    import re
    if not text:
        return ""
    return re.sub(r"[^\w一-鿿]", "", text).lower()


def _substr_similarity_inner(short: str, long: str) -> float:
    """较短文本在较长文本中找最相似的等长子串，返回 [0,1]。

    短文本作为子串精确出现 → 1.0；否则滑动等长窗口取最相似段，
    用于容错 ASR 错字。避免短基准被长 ASR 文本稀释整体相似度。
    """
    s, l = _normalize_text_for_verify(short), _normalize_text_for_verify(long)
    if not s or not l:
        return 0.0
    if s in l:
        return 1.0
    if len(s) > len(l):
        import difflib
        return difflib.SequenceMatcher(None, s, l).ratio()
    import difflib
    best = 0.0
    for i in range(len(l) - len(s) + 1):
        v = difflib.SequenceMatcher(None, s, l[i:i + len(s)]).ratio()
        if v > best:
            best = v
            if best >= 1.0:
                break
    return best


def text_similarity(asr_text: str, expected_text: str) -> float:
    """计算 ASR 识别文本与期望文本的字符相似度（归一化后）。

    返回 [0, 1]，1 表示完全一致。取"整体 SequenceMatcher 比值"与
    "双向子串包含相似度"的较大值：
        - 整体比值对长文本容错漏字/多字/错字；
        - 子串相似度防止短输入文本被较长的 ASR 结果稀释（例如输入"救命"、
          ASR 识别成"救命啊救命"，整体比值仅 0.57 会误判，子串匹配为 1.0）。
    """
    a = _normalize_text_for_verify(asr_text)
    b = _normalize_text_for_verify(expected_text)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    import difflib
    overall = difflib.SequenceMatcher(None, a, b).ratio()
    if len(b) <= len(a):
        sub = _substr_similarity_inner(expected_text, asr_text)
    else:
        sub = _substr_similarity_inner(asr_text, expected_text)
    return max(overall, sub)


def verify_audio_content(audio_path: str, expected_text: str,
                         threshold: float = None, enabled: bool = None) -> tuple:
    """对已生成的音频做 ASR 校验，返回 (是否通过, 相似度, ASR文本, 原因)。

    Args:
        audio_path: 音频文件绝对路径
        expected_text: 原始输入文本（期望内容）
        threshold: 相似度阈值，低于此值判定失败；None 时读取 config 默认
        enabled: 是否启用校验；None 时读取 config 默认。关闭时直接返回通过

    Returns:
        (passed: bool, similarity: float, asr_text: str, reason: str)
        - 关闭校验：(True, -1.0, "", "disabled")
        - ASR 异常：(True, -1.0, "", "asr_error:<msg>")  ← 放行不阻断
        - 通过：(True, score, asr_text, "pass")
        - 失败：(False, score, asr_text, "below_threshold")
    """
    from backend.config import AUDIO_VERIFY_ENABLED, AUDIO_VERIFY_THRESHOLD
    if enabled is None:
        enabled = AUDIO_VERIFY_ENABLED
    if threshold is None:
        threshold = AUDIO_VERIFY_THRESHOLD

    if not enabled:
        return True, -1.0, "", "disabled"
    if not expected_text or not str(expected_text).strip():
        # 无期望文本无法校验，放行
        return True, -1.0, "", "no_expected_text"
    if not os.path.exists(audio_path):
        return False, -1.0, "", "audio_not_found"

    try:
        from backend.services.asr_service import transcribe
        asr_text = transcribe(audio_path)
    except Exception as e:
        system_logger.warning(f"【校验】ASR 异常，放行不阻断: {e}")
        return True, -1.0, "", f"asr_error:{type(e).__name__}"

    similarity = text_similarity(asr_text, expected_text)
    if similarity >= threshold:
        return True, round(similarity, 3), asr_text, "pass"
    return False, round(similarity, 3), asr_text, "below_threshold"


def verify_and_cleanup(audio_path: str, expected_text: str,
                       model_tag: str = "", threshold: float = None,
                       enabled: bool = None) -> str:
    """校验音频内容，失败则删除文件并抛出 AudioVerifyError；通过则原样返回路径。

    供即时生成路径（save_temp_audio）与批量生成路径共用：
        - 校验关闭 / 无期望文本 / ASR 异常 → 不删、不抛，原样返回路径
        - 相似度达标 → 不删，原样返回路径
        - 相似度不达标 → 删除文件，抛 AudioVerifyError（含 ASR 文本与相似度便于排查）
    """
    passed, similarity, asr_text, reason = verify_audio_content(
        audio_path, expected_text, threshold=threshold, enabled=enabled
    )
    if passed:
        if reason not in ("disabled", "no_expected_text", "asr_error") and similarity >= 0:
            system_logger.info(
                f"【校验】{model_tag} 通过 | 相似度={similarity:.2f} | "
                f"ASR='{asr_text}' | 期望='{expected_text[:30]}'"
            )
        return audio_path

    # 校验失败：删除文件
    try:
        if os.path.exists(audio_path):
            os.remove(audio_path)
            system_logger.info(f"【校验】{model_tag} 已删除不合格音频: {audio_path}")
    except OSError as e:
        system_logger.warning(f"【校验】删除失败 {audio_path}: {e}")

    raise AudioVerifyError(
        f"{model_tag} 音频内容校验失败 | 相似度={similarity:.2f} < 阈值 | "
        f"ASR='{asr_text}' | 期望='{expected_text[:50]}'"
    )

