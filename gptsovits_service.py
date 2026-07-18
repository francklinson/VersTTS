#!/usr/bin/env python3
"""
GPT-SoVITS 独立服务
使用 transformers 4.51.3 (lib/transformers4)，运行在独立端口上
启动方式: nohup python gptsovits_service.py > logs/gptsovits_service.log 2>&1 &

日志文件: logs/gptsovits_service.log
"""

import sys
import os

# 在导入任何模块之前，设置 transformers 4.x 路径
TRANSFORMERS4_PATH = os.path.join(os.path.dirname(__file__), "lib", "transformers4")
sys.path.insert(0, TRANSFORMERS4_PATH)

import time
import traceback
import asyncio
import torch
import soundfile as sf
import uvicorn
from fastapi import FastAPI, Form, HTTPException
from contextlib import asynccontextmanager
from typing import Optional
import logging
from logging.handlers import RotatingFileHandler

# ========== 日志配置 ==========
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(PROJECT_ROOT, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

GPTSOVITS_LOG = os.path.join(LOG_DIR, 'gptsovits_service.log')

DETAILED_FORMATTER = logging.Formatter(
    '%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger("gptsovits_service")
logger.setLevel(logging.INFO)
logger.handlers = []

file_handler = RotatingFileHandler(
    GPTSOVITS_LOG,
    maxBytes=10 * 1024 * 1024,  # 10MB
    backupCount=3,
    encoding='utf-8'
)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(DETAILED_FORMATTER)
logger.addHandler(file_handler)
logger.propagate = False

# ========== 路径配置 ==========
ALGORITHMS_PATH = os.path.join(PROJECT_ROOT, "algorithms", "GPT-SoVITS")
GPTSOVITS_MODULE = os.path.join(ALGORITHMS_PATH, "GPT_SoVITS")

# 添加 GPT-SoVITS 到 sys.path
for p in [ALGORITHMS_PATH, GPTSOVITS_MODULE]:
    if p not in sys.path:
        sys.path.insert(0, p)

# 添加 eres2net 到 sys.path（sv.py 需要 ERes2NetV2 模块）
ERES2NET_PATH = os.path.join(GPTSOVITS_MODULE, "eres2net")
if ERES2NET_PATH not in sys.path:
    sys.path.insert(0, ERES2NET_PATH)

# 创建 fast_langdetect 缓存目录（langsegmenter.py 指定了该路径作为 cache_dir）
# langsegmenter.py 使用 Path(__file__).parent.parent.parent 定位，即 GPTSOVITS_MODULE
FAST_LANGDETECT_CACHE = os.path.join(GPTSOVITS_MODULE, "pretrained_models", "fast_langdetect")
os.makedirs(FAST_LANGDETECT_CACHE, exist_ok=True)

# 设置环境变量
os.environ["bert_path"] = os.path.join(
    PROJECT_ROOT, "models", "GPT-SoVITS", "chinese-roberta-wwm-ext-large"
)
# 注意：不要预先创建 G2PWModel 目录，否则 download_and_decompress 会跳过下载

# 模型路径（确保绝对路径，避免 chdir 后路径错误）
MODELS_DIR = os.environ.get("MODELS_DIR", os.path.join(PROJECT_ROOT, "models"))
if not os.path.isabs(MODELS_DIR):
    MODELS_DIR = os.path.join(PROJECT_ROOT, MODELS_DIR)
MODEL_PATH = os.path.join(MODELS_DIR, "GPT-SoVITS")

# 默认版本
DEFAULT_VERSION = os.environ.get("GPTSOVITS_VERSION", "v2")

# 延迟导入
pipeline = None
current_version = None  # 当前已加载的版本
last_used_time = None  # 最后使用时间
_gpu_id = None  # 当前服务使用的 GPU ID
_main_service_url = None  # 主服务地址，用于 OOM 驱逐

# ========== 空闲超时配置 ==========
IDLE_TIMEOUT = int(os.environ.get("IDLE_TIMEOUT", "300"))  # 默认 5 分钟
HEARTBEAT_INTERVAL = int(os.environ.get("HEARTBEAT_INTERVAL", "60"))  # 心跳间隔秒
_idle_check_task = None


# ========== 各版本模型路径映射 ==========
VERSION_MODEL_MAP = {
    "v1": {
        "t2s_weights_path": "s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt",
        "vits_weights_path": "s2G488k.pth",
    },
    "v2": {
        "t2s_weights_path": os.path.join("gsv-v2final-pretrained", "s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"),
        "vits_weights_path": os.path.join("gsv-v2final-pretrained", "s2G2333k.pth"),
    },
    "v2Pro": {
        "t2s_weights_path": "s1v3.ckpt",
        "vits_weights_path": os.path.join("v2Pro", "s2Gv2Pro.pth"),
    },
    "v2ProPlus": {
        "t2s_weights_path": "s1v3.ckpt",
        "vits_weights_path": os.path.join("v2Pro", "s2Gv2ProPlus.pth"),
    },
    "v3": {
        "t2s_weights_path": "s1v3.ckpt",
        "vits_weights_path": "s2Gv3.pth",
    },
    "v4": {
        "t2s_weights_path": "s1v3.ckpt",
        "vits_weights_path": os.path.join("gsv-v4-pretrained", "s2Gv4.pth"),
    },
}

# SV 模型路径（v2Pro/v2ProPlus/v3/v4 需要）
SV_MODEL_PATH = os.path.join(MODEL_PATH, "sv", "pretrained_eres2netv2w24s4ep4.ckpt")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global _idle_check_task, _gpu_id, _main_service_url

    logger.info("=" * 60)
    logger.info("【GPT-SoVITS服务】启动中...")
    logger.info(f"【日志文件】{GPTSOVITS_LOG}")
    logger.info(f"【空闲超时】{IDLE_TIMEOUT}s | 【心跳间隔】{HEARTBEAT_INTERVAL}s")
    logger.info("=" * 60)

    # 获取 GPU ID
    _gpu_id = os.environ.get("GPU_ID", "0")
    # 主服务地址
    main_host = os.environ.get("MAIN_HOST", "127.0.0.1")
    main_port = os.environ.get("MAIN_PORT", "8000")
    main_scheme = os.environ.get("MAIN_SCHEME", "https")
    _main_service_url = f"{main_scheme}://{main_host}:{main_port}"

    # 注册到主服务（重试3次，间隔2秒）
    for attempt in range(3):
        if _register_to_main_service():
            break
        await asyncio.sleep(2)

    # 启动空闲检查定时器
    _idle_check_task = asyncio.create_task(_idle_check_loop())

    yield

    # 停止定时器
    if _idle_check_task:
        _idle_check_task.cancel()

    # 卸载模型
    unload_model()

    # 从主服务注销
    _unregister_from_main_service()

    logger.info("【GPT-SoVITS服务】已停止")


app = FastAPI(title="GPT-SoVITS 独立服务", lifespan=lifespan)


def unload_model():
    """卸载模型，释放显存"""
    global pipeline, current_version, last_used_time
    if pipeline is None:
        return
    logger.info(f"【模型卸载】正在卸载模型 (版本: {current_version})...")
    pipeline = None
    current_version = None
    last_used_time = None
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("【模型卸载】GPU缓存已清理")


def _touch_last_used():
    """更新最后使用时间"""
    global last_used_time
    last_used_time = time.time()


async def _idle_check_loop():
    """后台定时检查空闲超时"""
    while True:
        await asyncio.sleep(HEARTBEAT_INTERVAL)
        if pipeline is not None and last_used_time is not None:
            idle_seconds = time.time() - last_used_time
            if idle_seconds > IDLE_TIMEOUT:
                logger.info(f"【空闲超时】模型已空闲 {idle_seconds:.0f}s > {IDLE_TIMEOUT}s，自动卸载")
                unload_model()
        # 心跳上报
        _heartbeat()


def _register_to_main_service():
    """向主服务注册，返回是否成功"""
    try:
        import requests
        port = int(os.environ.get("GPTSOVITS_PORT", "8004"))
        host = os.environ.get("GPTSOVITS_HOST", "127.0.0.1")
        requests.post(
            f"{_main_service_url}/services/register",
            json={
                "service_id": "gptsovits",
                "port": port,
                "host": host,
                "gpu_id": _gpu_id,
            },
            timeout=5,
            verify=False,
        )
        logger.info(f"【服务注册】已注册到主服务 (GPU: {_gpu_id})")
        return True
    except Exception as e:
        logger.warning(f"【服务注册】注册失败: {e}")
        return False


def _unregister_from_main_service():
    """从主服务注销"""
    try:
        import requests
        requests.post(
            f"{_main_service_url}/services/unregister",
            json={"service_id": "gptsovits"},
            timeout=5,
            verify=False,
        )
        logger.info("【服务注销】已从主服务注销")
    except Exception as e:
        logger.warning(f"【服务注销】注销失败: {e}")


def _heartbeat():
    """向主服务上报心跳"""
    try:
        import requests
        vram_mb = 0
        if pipeline is not None and torch.cuda.is_available():
            vram_mb = torch.cuda.memory_allocated() // (1024 * 1024)
        requests.post(
            f"{_main_service_url}/services/heartbeat",
            json={
                "service_id": "gptsovits",
                "model_loaded": pipeline is not None,
                "current_version": current_version,
                "vram_used_mb": vram_mb,
                "last_used_time": last_used_time,
                "gpu_id": _gpu_id,
                "host": os.environ.get("GPTSOVITS_HOST", "127.0.0.1"),
                "port": int(os.environ.get("GPTSOVITS_PORT", "8004")),
            },
            timeout=5,
            verify=False,
        )
    except Exception:
        pass  # 心跳失败不影响服务


def _request_eviction_from_main_service(needed_mb: int):
    """OOM 时请求主服务驱逐同 GPU 上其他模型"""
    try:
        import requests
        logger.info(f"【OOM驱逐】请求主服务释放 {needed_mb}MB 显存 (GPU: {_gpu_id})")
        resp = requests.post(
            f"{_main_service_url}/services/evict",
            json={"gpu_id": _gpu_id, "exclude_service": "gptsovits", "needed_mb": needed_mb},
            timeout=30,
            verify=False,
        )
        if resp.status_code == 200:
            result = resp.json()
            evicted = result.get("evicted_services", [])
            logger.info(f"【OOM驱逐】主服务已驱逐: {evicted}")
            return True
        else:
            logger.warning(f"【OOM驱逐】主服务返回: {resp.status_code}")
            return False
    except Exception as e:
        logger.warning(f"【OOM驱逐】请求失败: {e}")
        return False


def _build_version_configs(target_version: str) -> dict:
    """构建传入 TTS_Config 的完整配置字典，包含 custom 和所有版本的绝对路径。

    TTS_Config.__init__ 的逻辑：
    1. configs_.update(configs) → 合并传入的 dict
    2. self.configs = configs_.get("custom", configs_["v2"]) → 优先取 custom
    3. init_vits_weights 读取 self.configs.default_configs[model_version] → 需要更新所有版本

    因此必须：
    - 传入 "custom" 键，包含目标版本的所有路径
    - 同时传入所有版本键，确保 default_configs 都有绝对路径
    """
    bert_path = os.path.join(MODEL_PATH, "chinese-roberta-wwm-ext-large")
    cnhubert_path = os.path.join(MODEL_PATH, "chinese-hubert-base")

    # custom: 目标版本的完整配置（TTS_Config 优先使用 custom）
    version_paths = VERSION_MODEL_MAP.get(target_version, {})
    custom = {
        "version": target_version,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "is_half": torch.cuda.is_available(),
        "bert_base_path": bert_path,
        "cnhuhbert_base_path": cnhubert_path,
        "t2s_weights_path": os.path.join(MODEL_PATH, version_paths["t2s_weights_path"]),
        "vits_weights_path": os.path.join(MODEL_PATH, version_paths["vits_weights_path"]),
    }

    # 所有版本都用绝对路径，确保 default_configs[model_version] 也正确
    all_versions = {}
    for ver, paths in VERSION_MODEL_MAP.items():
        all_versions[ver] = {
            "version": ver,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "is_half": torch.cuda.is_available(),
            "bert_base_path": bert_path,
            "cnhuhbert_base_path": cnhubert_path,
            "t2s_weights_path": os.path.join(MODEL_PATH, paths["t2s_weights_path"]),
            "vits_weights_path": os.path.join(MODEL_PATH, paths["vits_weights_path"]),
        }

    # 合并：custom + 所有版本
    result = {"custom": custom}
    result.update(all_versions)
    return result


def _ensure_ref_audio_duration(ref_audio_path: str, prompt_text: str = "") -> tuple:
    """确保参考音频时长在 3~10 秒范围内，并同步调整参考文本。

    GPT-SoVITS 要求参考音频在 3~10 秒之间，否则会抛出 OSError。
    - 超过 10 秒：用 ASR 时间戳在词边界精准切分，取最接近10秒的切分点
    - 不足 3 秒：给出明确提示
    - 参考文本为空：自动用 ASR 识别

    Returns:
        (处理后的音频路径, 调整后的参考文本)
    """
    MIN_SEC, MAX_SEC = 3, 10
    try:
        info = sf.info(ref_audio_path)
        duration = info.duration
    except Exception:
        return ref_audio_path, prompt_text

    # 参考文本为空时，自动用 ASR 识别
    if not prompt_text and duration >= MIN_SEC:
        asr_text = _asr_transcribe(ref_audio_path)
        if asr_text:
            logger.info(f"【参考音频】参考文本为空，ASR自动识别: '{asr_text}'")
            prompt_text = asr_text

    if MIN_SEC <= duration <= MAX_SEC:
        return ref_audio_path, prompt_text

    audio_data, sr = sf.read(ref_audio_path)

    if duration > MAX_SEC:
        # 用 ASR 时间戳在词边界精准切分
        cut_sample = _asr_find_cut_point(ref_audio_path, MAX_SEC, sr, len(audio_data))
        if cut_sample and cut_sample < len(audio_data):
            cut_sec = cut_sample / sr
            audio_data = audio_data[:cut_sample]
            # 对裁剪后的音频重新 ASR 识别参考文本
            tmp_path = f"/tmp/gptsovits_ref_{int(time.time()*1000)}.wav"
            sf.write(tmp_path, audio_data, sr)
            asr_text = _asr_transcribe(tmp_path)
            if asr_text:
                logger.info(f"【参考音频】{duration:.1f}s > {MAX_SEC}s，时间戳精准切分至 {cut_sec:.2f}s，ASR识别参考文本: '{asr_text}'")
                prompt_text = asr_text
            else:
                # ASR 失败，按比例截断
                ratio = cut_sec / duration
                adjusted_text = _truncate_text_at_sentence_boundary(prompt_text, int(len(prompt_text) * ratio)) if prompt_text else prompt_text
                logger.info(f"【参考音频】{duration:.1f}s > {MAX_SEC}s，切分至 {cut_sec:.2f}s，ASR失败，截断参考文本: '{prompt_text}' -> '{adjusted_text}'")
                prompt_text = adjusted_text
            return tmp_path, prompt_text

        # ASR 时间戳不可用，回退到截取前 MAX_SEC 秒
        max_samples = int(MAX_SEC * sr)
        audio_data = audio_data[:max_samples]
        tmp_path = f"/tmp/gptsovits_ref_{int(time.time()*1000)}.wav"
        sf.write(tmp_path, audio_data, sr)
        asr_text = _asr_transcribe(tmp_path)
        if asr_text:
            logger.info(f"【参考音频】{duration:.1f}s > {MAX_SEC}s，截取前 {MAX_SEC}s，ASR识别参考文本: '{prompt_text}' -> '{asr_text}'")
            prompt_text = asr_text
        else:
            ratio = MAX_SEC / duration
            adjusted_text = _truncate_text_at_sentence_boundary(prompt_text, int(len(prompt_text) * ratio)) if prompt_text else prompt_text
            logger.info(f"【参考音频】{duration:.1f}s > {MAX_SEC}s，截取前 {MAX_SEC}s，ASR失败，截断参考文本: '{prompt_text}' -> '{adjusted_text}'")
            prompt_text = adjusted_text
        return tmp_path, prompt_text
    else:
        raise HTTPException(
            status_code=400,
            detail=f"参考音频仅 {duration:.1f} 秒，GPT-SoVITS 至少需要 3 秒参考音频才能保证克隆质量，请更换更长的参考音频"
        )


# ========== ASR（懒加载，仅参考音频>10s时使用） ==========
_asr_model = None
WENET_PATH = os.path.join(PROJECT_ROOT, "lib")


def _get_asr_model():
    """懒加载 wenet ASR 模型"""
    global _asr_model
    if _asr_model is None:
        if WENET_PATH not in sys.path:
            sys.path.insert(0, WENET_PATH)
        from wenet.cli.model import load_model
        logger.info("【ASR】加载 wenetspeech 模型...")
        _asr_model = load_model("wenetspeech", device="cpu")
    return _asr_model


def _asr_transcribe(audio_path: str) -> str:
    """用 wenet (wenetspeech) 识别音频，返回识别文本。失败返回空字符串。"""
    try:
        model = _get_asr_model()
        result = model.transcribe(audio_path)
        text = result.text.strip() if result and result.text else ""
        if text:
            logger.info(f"【ASR】识别结果: '{text}'")
        return text
    except Exception as e:
        logger.warning(f"【ASR】识别失败: {e}")
        return ""


def _asr_find_cut_point(audio_path: str, target_sec: float, sr: int, total_samples: int) -> int:
    """用 wenet ASR 时间戳找到最接近 target_sec 的词边界切分点（样本数）。

    wenet 的 times 是 subsampled 帧索引，不是毫秒。
    通过比例换算：times[-1] 对应音频总时长，目标切分点 = times[-1] * (target_sec / duration)。

    返回切分点的样本数，失败返回 None。
    """
    try:
        model = _get_asr_model()
        result = model.transcribe(audio_path)
        if not result or not result.times or not result.tokens or len(result.times) < 2:
            return None

        times = result.times
        # 用比例换算：times[-1] ≈ 音频总时长
        audio_duration = total_samples / sr
        ratio = target_sec / audio_duration
        target_times_val = times[-1] * ratio

        # 找到不超过 target_times_val 的最大 times 索引
        cut_idx = 0
        for i, t in enumerate(times):
            if t <= target_times_val:
                cut_idx = i
            else:
                break

        if cut_idx == 0:
            return None

        # 将 times[cut_idx] 转为实际秒数，再转为样本数
        cut_sec = (times[cut_idx] / times[-1]) * audio_duration
        cut_sample = int(cut_sec * sr)
        logger.info(f"【ASR时间戳】目标{target_sec}s，切分点: {cut_sec:.2f}s (token {cut_idx}/{len(times)})")
        return min(cut_sample, total_samples)
    except Exception as e:
        logger.warning(f"【ASR时间戳】获取切分点失败: {e}")
        return None


def _truncate_text_at_sentence_boundary(text: str, target_len: int) -> str:
    """在句子边界处截断文本，尽量接近 target_len。

    优先在句号/感叹号/问号处断开，其次逗号/顿号/分号。
    允许少量超出 target_len（最多20%），超出太多则找更近的短语边界。
    """
    if target_len >= len(text):
        return text

    # 句子结束标点（优先在此断开）
    sentence_ends = '。！？'
    # 短语分隔标点（其次在此断开）
    phrase_ends = '，、；,;'
    max_overshoot = max(5, int(target_len * 0.2))  # 最多超出20%

    # 从 target_len 往后找最近的句子边界（不超过 max_overshoot）
    for i in range(target_len, min(target_len + max_overshoot + 1, len(text))):
        if text[i] in sentence_ends:
            return text[:i + 1]

    # 超出太多，退而求其次找短语边界
    for i in range(target_len, min(target_len + max_overshoot + 1, len(text))):
        if text[i] in phrase_ends:
            return text[:i + 1]

    # 往前找句子边界
    for i in range(target_len, 0, -1):
        if text[i] in sentence_ends:
            return text[:i + 1]

    # 往前找短语边界
    for i in range(target_len, 0, -1):
        if text[i] in phrase_ends:
            return text[:i + 1]

    # 都找不到，直接截断
    return text[:target_len]


def load_model(version=None):
    """加载 GPT-SoVITS 模型，支持版本切换，OOM 时请求主服务驱逐"""
    global pipeline, current_version

    target_version = version or DEFAULT_VERSION

    # 如果模型已加载且版本一致，无需重新加载
    if pipeline is not None and current_version == target_version:
        _touch_last_used()
        return

    # 如果模型已加载但版本不同，先卸载
    if pipeline is not None and current_version != target_version:
        logger.info(f"【版本切换】{current_version} -> {target_version}，正在卸载旧模型...")
        unload_model()

    # 关键：在 import TTS 之前先 chdir 到 ALGORITHMS_PATH
    original_cwd = os.getcwd()
    os.chdir(ALGORITHMS_PATH)

    try:
        from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config

        logger.info(f"【模型加载】模型路径: {MODEL_PATH} | 目标版本: {target_version}")
        start_time = time.time()

        # 构建完整配置字典，传入 TTS_Config 而非无参构造后 setattr
        # 注意：传入 dict 时 TTS_Config 不会读取 YAML 文件，无需删除
        configs_dict = _build_version_configs(target_version)
        tts_config = TTS_Config(configs_dict)

        # 验证路径是否正确
        logger.info(f"【模型加载】t2s_path: {tts_config.t2s_weights_path}")
        logger.info(f"【模型加载】vits_path: {tts_config.vits_weights_path}")
        logger.info(f"【模型加载】bert_path: {tts_config.bert_base_path}")
        logger.info(f"【模型加载】cnhubert_path: {tts_config.cnhuhbert_base_path}")
        logger.info(f"【模型加载】version: {tts_config.version} | device: {tts_config.device} | is_half: {tts_config.is_half}")

        # 检查模型文件是否存在
        for name, path in [
            ("t2s", tts_config.t2s_weights_path),
            ("vits", tts_config.vits_weights_path),
            ("bert", tts_config.bert_base_path),
            ("cnhubert", tts_config.cnhuhbert_base_path),
        ]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"模型文件不存在: {name}={path}")

        # SV 模型路径
        if target_version in ("v2Pro", "v2ProPlus", "v3", "v4"):
            sv_dir = os.path.join(GPTSOVITS_MODULE, "pretrained_models", "sv")
            os.makedirs(sv_dir, exist_ok=True)
            sv_expected = os.path.join(sv_dir, "pretrained_eres2netv2w24s4ep4.ckpt")
            if not os.path.exists(sv_expected) and os.path.exists(SV_MODEL_PATH):
                import shutil
                shutil.copy2(SV_MODEL_PATH, sv_expected)

        # v3 bigvgan
        if target_version == "v3":
            bigvgan_src = os.path.join(MODEL_PATH, "models--nvidia--bigvgan_v2_24khz_100band_256x")
            bigvgan_dst = os.path.join(GPTSOVITS_MODULE, "pretrained_models", "models--nvidia--bigvgan_v2_24khz_100band_256x")
            if not os.path.exists(bigvgan_dst) and os.path.exists(bigvgan_src):
                import shutil
                shutil.copytree(bigvgan_src, bigvgan_dst)

        # v4 vocoder
        if target_version == "v4":
            v4_vocoder_src = os.path.join(MODEL_PATH, "gsv-v4-pretrained", "vocoder.pth")
            v4_vocoder_dst_dir = os.path.join(GPTSOVITS_MODULE, "pretrained_models", "gsv-v4-pretrained")
            os.makedirs(v4_vocoder_dst_dir, exist_ok=True)
            v4_vocoder_dst = os.path.join(v4_vocoder_dst_dir, "vocoder.pth")
            if not os.path.exists(v4_vocoder_dst) and os.path.exists(v4_vocoder_src):
                import shutil
                shutil.copy2(v4_vocoder_src, v4_vocoder_dst)

        # 带驱逐重试的加载逻辑
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

                pipeline = TTS(tts_config)
                current_version = target_version
                _touch_last_used()

                duration = time.time() - start_time
                logger.info(f"【模型加载】完成 | 版本: {tts_config.version} | 耗时: {duration:.2f}s")
                return

            except RuntimeError as e:
                if "CUDA" in str(e) or "out of memory" in str(e).lower():
                    if attempt < max_retries - 1:
                        logger.warning(f"【模型加载】GPU OOM，尝试驱逐其他模型... ({attempt + 1}/{max_retries})")
                        _request_eviction_from_main_service(needed_mb=3000)
                        time.sleep(3)
                    else:
                        logger.error(f"【模型加载】OOM，驱逐后仍无法加载: {str(e)}")
                        raise
                else:
                    logger.error(f"【模型加载】失败: {str(e)}")
                    raise

        raise RuntimeError(f"模型加载失败，已重试 {max_retries} 次")

    finally:
        os.chdir(original_cwd)


@app.get("/health")
async def health():
    """健康检查"""
    return {
        "status": "ok",
        "model_loaded": pipeline is not None,
        "current_version": current_version,
        "available_versions": list(VERSION_MODEL_MAP.keys()),
        "last_used_time": last_used_time,
        "idle_timeout": IDLE_TIMEOUT,
        "gpu_id": _gpu_id,
    }


@app.post("/asr")
async def asr(audio_path: str = Form(...)):
    """ASR 识别音频，返回文本。用于前端上传参考音频时自动识别参考文本。"""
    if not os.path.exists(audio_path):
        raise HTTPException(status_code=400, detail=f"音频文件不存在: {audio_path}")
    text = _asr_transcribe(audio_path)
    if not text:
        raise HTTPException(status_code=500, detail="ASR 识别失败")
    return {"success": True, "text": text}


@app.get("/versions")
async def versions():
    """列出所有可用版本及其模型路径"""
    result = {}
    for ver, paths in VERSION_MODEL_MAP.items():
        result[ver] = {
            k: os.path.join(MODEL_PATH, v) for k, v in paths.items()
        }
        result[ver]["available"] = all(
            os.path.exists(os.path.join(MODEL_PATH, v)) for v in paths.values()
        )
    return result


@app.post("/model/load")
async def model_load(version: str = Form("v2")):
    """手动加载模型"""
    valid_versions = list(VERSION_MODEL_MAP.keys())
    if version not in valid_versions:
        raise HTTPException(status_code=400, detail=f"不支持的版本: {version}，支持: {valid_versions}")
    try:
        load_model(version=version)
        return {"success": True, "version": current_version}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/model/unload")
async def model_unload():
    """手动卸载模型，释放显存"""
    if pipeline is None:
        return {"success": True, "message": "模型未加载"}
    unload_model()
    return {"success": True, "message": "模型已卸载"}


@app.get("/model/status")
async def model_status():
    """获取模型状态"""
    vram_mb = 0
    if pipeline is not None and torch.cuda.is_available():
        vram_mb = torch.cuda.memory_allocated() // (1024 * 1024)
    idle_seconds = time.time() - last_used_time if last_used_time else None
    return {
        "service_id": "gptsovits",
        "model_loaded": pipeline is not None,
        "current_version": current_version,
        "vram_used_mb": vram_mb,
        "last_used_time": last_used_time,
        "idle_seconds": int(idle_seconds) if idle_seconds is not None else None,
        "idle_timeout": IDLE_TIMEOUT,
        "gpu_id": _gpu_id,
    }


@app.post("/tts")
async def tts(
    text: str = Form(...),
    text_lang: str = Form("zh"),
    prompt_text: str = Form(...),
    prompt_lang: str = Form("zh"),
    ref_audio_path: str = Form(...),
    top_k: int = Form(15),
    top_p: float = Form(1.0),
    temperature: float = Form(1.0),
    text_split_method: str = Form("cut5"),
    batch_size: int = Form(1),
    speed_factor: float = Form(1.0),
    version: str = Form("v2"),
    output_format: str = Form("url")
):
    """GPT-SoVITS TTS 合成"""
    # 验证版本参数
    valid_versions = list(VERSION_MODEL_MAP.keys())
    if version not in valid_versions:
        raise HTTPException(status_code=400, detail=f"不支持的版本: {version}，支持: {valid_versions}")

    load_model(version=version)
    _touch_last_used()

    start_time = time.time()
    logger.info(f"【TTS请求】文本: {text[:50]}... | 版本: {version} | 参考音频: {ref_audio_path}")

    try:
        # 验证参考音频
        if not ref_audio_path or not os.path.exists(ref_audio_path):
            raise HTTPException(status_code=400, detail=f"参考音频不存在: {ref_audio_path}")

        # 自动裁剪参考音频至 3~10 秒范围，同步调整参考文本
        ref_audio_path, prompt_text = _ensure_ref_audio_duration(ref_audio_path, prompt_text)

        # 切换工作目录
        original_cwd = os.getcwd()
        os.chdir(ALGORITHMS_PATH)

        try:
            # 构建请求参数
            req = {
                "text": text,
                "text_lang": text_lang.lower(),
                "ref_audio_path": ref_audio_path,
                "prompt_text": prompt_text,
                "prompt_lang": prompt_lang.lower(),
                "top_k": top_k,
                "top_p": top_p,
                "temperature": temperature,
                "text_split_method": text_split_method,
                "batch_size": batch_size,
                "speed_factor": speed_factor,
                "media_type": "wav",
                "streaming_mode": False,
                "parallel_infer": True,
            }

            # 执行推理
            tts_generator = pipeline.run(req)
            sr, audio_data = next(tts_generator)

        finally:
            os.chdir(original_cwd)

        # 保存临时文件
        timestamp = int(time.time() * 1000)
        output_path = f"/tmp/gptsovits_{timestamp}.wav"
        sf.write(output_path, audio_data, sr)

        duration = time.time() - start_time
        audio_duration = len(audio_data) / sr if sr > 0 else 0
        logger.info(f"【TTS完成】音频路径: {output_path} | 采样率: {sr} | 音频时长: {audio_duration:.1f}s | 耗时: {duration:.2f}s")

        # 清理显存
        if torch.cuda.is_available():
            del tts_generator
            torch.cuda.empty_cache()

        return {"success": True, "audio_path": output_path, "sample_rate": sr}

    except HTTPException:
        raise
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"【TTS错误】异常类型: {type(e).__name__} | 错误: {str(e)} | 耗时: {duration:.2f}s")
        logger.error(f"【错误堆栈】\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {str(e)}")


if __name__ == "__main__":
    port = int(os.environ.get("GPTSOVITS_PORT", 8004))
    host = os.environ.get("GPTSOVITS_HOST", "127.0.0.1")
    logger.info(f"【服务启动】地址: {host}:{port}")
    uvicorn.run(app, host=host, port=port)
