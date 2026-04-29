import os
import sys
import json
import torch
import soundfile as sf
import numpy as np
from time import time as ttime

# 添加当前目录和 GPT_SoVITS 目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "GPT_SoVITS"))

# 设置环境变量
os.environ['version'] = 'v2'

# 导入必要的模块
from GPT_SoVITS.module.models import Generator, SynthesizerTrn, SynthesizerTrnV3
from GPT_SoVITS.AR.models.t2s_lightning_module import Text2SemanticLightningModule
from GPT_SoVITS.feature_extractor import cnhubert
from transformers import AutoModelForMaskedLM, AutoTokenizer
from GPT_SoVITS.text import cleaned_text_to_sequence
from GPT_SoVITS.text.cleaner import clean_text
from tools.i18n.i18n import I18nAuto
from GPT_SoVITS.text.LangSegmenter import LangSegmenter
from GPT_SoVITS.module.mel_processing import mel_spectrogram_torch, spectrogram_torch

# 初始化 i18n
i18n = I18nAuto()

# 设置设备
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 设置模型路径
cnhubert_base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "GPT_SoVITS", "pretrained_models", "chinese-hubert-base")
bert_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "GPT_SoVITS", "pretrained_models", "chinese-roberta-wwm-ext-large")
cnhubert.cnhubert_base_path = cnhubert_base_path

# 全局变量
GPT_model = None
ssl_model = None
vq_model = None
bigvgan_model = None
hifigan_model = None
sv_cn_model = None
tokenizer = None
bert_model = None

# 配置参数
hps = None
dtype = torch.float32  # 使用 float32 以避免类型不匹配的问题
model_version = "v3"  # 默认使用 v3 版本
v3v4set = {"v3", "v4"}

# 分割符号
splits = {
    "，", "。", "？", "！", ",", ".", "?", "!", "~", ":", "：", "—", "…"
}

# 语言映射
dict_language = {
    "中文": "zh",
    "英文": "en",
    "日文": "ja",
    "中英混合": "zh",
    "日英混合": "ja",
    "多语种混合": "auto"
}

# 初始化函数
def initialize():
    """初始化 GPT-SoVITS 模型"""
    global GPT_model, SoVITS_model, tokenizer, bert_model, ssl_model, vq_model, hps
    
    print("Initializing GPT-SoVITS models...")
    
    # 加载 VQ/GPT 模型配置和权重
    vq_gpt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "GPT_SoVITS", "text", "XXXXRT", "GPT-SoVITS-Pretrained", "pretrained_models", "s2Gv3.pth")
    dict_s2 = torch.load(vq_gpt_path, map_location=device, weights_only=False)
    hps = dict_s2["config"]
    
    # 将配置转换为 DictToAttrRecursive 对象
    class DictToAttrRecursive(dict):
        def __init__(self, input_dict):
            super().__init__(input_dict)
            for key, value in input_dict.items():
                if isinstance(value, dict):
                    value = DictToAttrRecursive(value)
                self[key] = value
                setattr(self, key, value)
    
    hps = DictToAttrRecursive(hps)
    hps.model.semantic_frame_rate = "25hz"
    hps.model.version = "v3"
    
    # 加载 tokenizer 和 bert 模型
    print("Loading tokenizer and BERT model...")
    global tokenizer, bert_model
    tokenizer = AutoTokenizer.from_pretrained(bert_path)
    bert_model = AutoModelForMaskedLM.from_pretrained(bert_path)
    bert_model.eval()
    bert_model.to(device)
    
    # 加载 SSL 模型
    print("Loading SSL model...")
    from GPT_SoVITS.feature_extractor.cnhubert import CNHubert
    ssl_model = CNHubert()
    ssl_model.model.to(device)
    
    # 加载 VQ 模型（实际上是 SynthesizerTrnV3）
    print("Loading VQ model...")
    from GPT_SoVITS.module.models import SynthesizerTrnV3
    # 使用与原始项目一致的参数
    vq_model = SynthesizerTrnV3(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model,
    )
    # 加载 VQ 模型权重
    vq_model = vq_model.to(device)
    vq_model.eval()
    vq_model.load_state_dict(dict_s2["weight"], strict=False)
    
    # 加载 GPT 模型
    print("Loading GPT model...")
    gpt_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "GPT_SoVITS", "text", "XXXXRT", "GPT-SoVITS-Pretrained", "pretrained_models", "s1v3.ckpt")
    checkpoint = torch.load(gpt_model_path, map_location=device, weights_only=False)
    
    # 初始化模型
    from GPT_SoVITS.AR.models.t2s_model import Text2SemanticDecoder
    # 创建模型配置（需要包装在 {"model": ...} 中）
    model_config = {"model": checkpoint["config"]["model"]}
    GPT_model = Text2SemanticDecoder(model_config)
    # 加载权重
    GPT_model.load_state_dict(checkpoint["weight"], strict=False)
    GPT_model.eval()
    GPT_model.to(device)
    
    # 加载 BigVGAN 模型
    print("Loading BigVGAN model...")
    global bigvgan_model
    from GPT_SoVITS.BigVGAN.bigvgan import BigVGAN
    bigvgan_model = BigVGAN.from_pretrained(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "GPT_SoVITS", "text", "XXXXRT", "GPT-SoVITS-Pretrained", "pretrained_models", "models--nvidia--bigvgan_v2_24khz_100band_256x"),
        use_cuda_kernel=False,
    )
    # 移除权重归一化
    bigvgan_model.remove_weight_norm()
    bigvgan_model.eval()
    bigvgan_model.to(device)
    
    print("GPT-SoVITS initialization completed!")

# 文本处理函数
def clean_text_inf(text, language, version):
    """清理文本并转换为音素"""
    language = language.replace("all_", "")
    phones, word2ph, norm_text = clean_text(text, language, version)
    phones = cleaned_text_to_sequence(phones, version)
    return phones, word2ph, norm_text

# 获取 BERT 特征
def get_bert_feature(text, word2ph):
    """获取 BERT 特征"""
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        for i in inputs:
            inputs[i] = inputs[i].to(device)
        res = bert_model(**inputs, output_hidden_states=True)
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
    assert len(word2ph) == len(text)
    phone_level_feature = []
    for i in range(len(word2ph)):
        repeat_feature = res[i].repeat(word2ph[i], 1)
        phone_level_feature.append(repeat_feature)
    phone_level_feature = torch.cat(phone_level_feature, dim=0)
    return phone_level_feature.T

# 获取 BERT 特征
def get_bert_inf(phones, word2ph, norm_text, language):
    """获取 BERT 特征"""
    language = language.replace("all_", "")
    if language == "zh":
        bert = get_bert_feature(norm_text, word2ph).to(device)
    else:
        bert = torch.zeros(
            (1024, len(phones)),
            dtype=dtype,
        ).to(device)
    return bert

# 获取音素和 BERT 特征
def get_phones_and_bert(text, language, version, final=False):
    """获取音素和 BERT 特征"""
    text = text.replace("  ", " ")
    textlist = []
    langlist = []
    
    if language == "all_zh":
        for tmp in LangSegmenter.getTexts(text, "zh"):
            langlist.append(tmp["lang"])
            textlist.append(tmp["text"])
    elif language == "en":
        langlist.append("en")
        textlist.append(text)
    elif language == "auto":
        for tmp in LangSegmenter.getTexts(text):
            langlist.append(tmp["lang"])
            textlist.append(tmp["text"])
    else:
        for tmp in LangSegmenter.getTexts(text):
            if langlist:
                if (tmp["lang"] == "en" and langlist[-1] == "en") or (tmp["lang"] != "en" and langlist[-1] != "en"):
                    textlist[-1] += tmp["text"]
                    continue
            if tmp["lang"] == "en":
                langlist.append(tmp["lang"])
            else:
                langlist.append(language)
            textlist.append(tmp["text"])
    
    phones_list = []
    bert_list = []
    norm_text_list = []
    
    for i in range(len(textlist)):
        lang = langlist[i]
        phones, word2ph, norm_text = clean_text_inf(textlist[i], lang, version)
        bert = get_bert_inf(phones, word2ph, norm_text, lang)
        phones_list.append(phones)
        norm_text_list.append(norm_text)
        bert_list.append(bert)
    
    bert = torch.cat(bert_list, dim=1)
    phones = sum(phones_list, [])
    norm_text = "".join(norm_text_list)
    
    if not final and len(phones) < 6:
        return get_phones_and_bert("." + text, language, version, final=True)
    
    return phones, bert.to(dtype), norm_text

# 文本切分函数
def split_text(text, how_to_cut="凑四句一切"):
    """切分文本"""
    def cut1(inp):
        inp = inp.strip("\n")
        inps = []
        i_split_head = i_split_tail = 0
        len_text = len(inp)
        while i_split_head < len_text:
            if inp[i_split_head] in splits:
                i_split_head += 1
                inps.append(inp[i_split_tail:i_split_head])
                i_split_tail = i_split_head
            else:
                i_split_head += 1
        split_idx = list(range(0, len(inps), 4))
        split_idx[-1] = None
        if len(split_idx) > 1:
            opts = []
            for idx in range(len(split_idx) - 1):
                opts.append("".join(inps[split_idx[idx] : split_idx[idx + 1]]))
        else:
            opts = [inp]
        opts = [item for item in opts if not set(item).issubset(splits)]
        return "\n".join(opts)
    
    if how_to_cut == "凑四句一切":
        text = cut1(text)
    
    while "\n\n" in text:
        text = text.replace("\n\n", "\n")
    
    texts = text.split("\n")
    texts = [t for t in texts if t.strip()]
    return texts

# 合并短文本
def merge_short_text_in_array(texts, threshold):
    """合并短文本"""
    if len(texts) < 2:
        return texts
    result = []
    text = ""
    for ele in texts:
        text += ele
        if len(text) >= threshold:
            result.append(text)
            text = ""
    if len(text) > 0:
        if len(result) == 0:
            result.append(text)
        else:
            result[-1] += text
    return result

# 核心推理函数
def get_tts_wav(
    ref_wav_path,
    prompt_text,
    prompt_language,
    text,
    text_language,
    how_to_cut="凑四句一切",
    top_k=20,
    top_p=0.6,
    temperature=0.6,
    ref_free=False,
    speed=1,
    sample_steps=8,
    if_sr=False,
    pause_second=0.3,
):
    """
    生成 TTS 音频
    
    Args:
        ref_wav_path: 参考音频路径
        prompt_text: 参考音频的文本
        prompt_language: 参考音频的语言
        text: 需要合成的文本
        text_language: 需要合成的语言
        how_to_cut: 文本切分方式
        top_k: GPT 采样参数
        top_p: GPT 采样参数
        temperature: GPT 采样参数
        ref_free: 是否使用无参考文本模式
        speed: 语速
        sample_steps: 采样步数
        if_sr: 是否使用超分
        pause_second: 句间停顿秒数
    
    Returns:
        tuple: (采样率, 音频数据)
    """
    global GPT_model, ssl_model, vq_model, bigvgan_model, hps
    
    if not GPT_model:
        initialize()
    
    # 检查输入参数
    if not ref_wav_path:
        raise ValueError("请提供参考音频路径")
    if not text:
        raise ValueError("请提供需要合成的文本")
    
    # 处理参考文本
    if not prompt_text or len(prompt_text) == 0:
        ref_free = True
    if model_version in v3v4set:
        ref_free = False  # v3/v4 暂不支持无参考文本模式
    
    # 转换语言
    prompt_language = dict_language.get(prompt_language, "zh")
    text_language = dict_language.get(text_language, "zh")
    
    # 处理参考文本
    if not ref_free:
        prompt_text = prompt_text.strip("\n")
        if not prompt_text:
            # 如果 prompt_text 为空，使用默认文本
            prompt_text = "您好，这是一段参考音频。"
        if prompt_text and prompt_text[-1] not in splits:
            prompt_text += "。" if prompt_language != "en" else "."
    
    # 处理目标文本
    text = text.strip("\n")
    
    # 生成零音频（用于句间停顿）
    zero_wav = np.zeros(
        int(hps["data"]["sampling_rate"] * pause_second),
        dtype=np.float16 if dtype == torch.float16 else np.float32,
    )
    zero_wav_torch = torch.from_numpy(zero_wav)
    zero_wav_torch = zero_wav_torch.to(dtype).to(device)
    
    # 提取参考音频特征
    import librosa
    import torchaudio
    
    if not ref_free:
        with torch.no_grad():
            # 加载参考音频
            wav16k, sr = librosa.load(ref_wav_path, sr=16000)
            if wav16k.shape[0] > 160000 or wav16k.shape[0] < 48000:
                raise ValueError("参考音频长度应在 3~10 秒之间")
            
            # 转换为张量
            wav16k = torch.from_numpy(wav16k)
            wav16k = wav16k.to(torch.float32).to(device)  # 使用 float32 以匹配模型权重类型
            wav16k = torch.cat([wav16k, zero_wav_torch.to(torch.float32)])
            
            # 提取 SSL 特征
            ssl_content = ssl_model.model(wav16k.unsqueeze(0))["last_hidden_state"].transpose(1, 2)
            # 转换为 float32 以匹配 VQ 模型的偏置类型
            ssl_content = ssl_content.to(torch.float32)
            
            # 提取 VQ 特征
            codes = vq_model.extract_latent(ssl_content)
            prompt_semantic = codes[0, 0]
            prompt = prompt_semantic.unsqueeze(0).to(device)
    
    # 切分文本
    text = split_text(text, how_to_cut)
    texts = merge_short_text_in_array(text, 5)
    
    # 生成音频
    audio_opt = []
    
    if not ref_free:
        # 获取参考文本的音素和 BERT 特征
        phones1, bert1, norm_text1 = get_phones_and_bert(prompt_text, prompt_language, model_version)
    
    for i_text, text in enumerate(texts):
        if len(text.strip()) == 0:
            continue
        
        # 确保文本以标点结尾
        if text[-1] not in splits:
            text += "。" if text_language != "en" else "."
        
        # 获取目标文本的音素和 BERT 特征
        phones2, bert2, norm_text2 = get_phones_and_bert(text, text_language, model_version)
        
        # 合并特征
        if not ref_free:
            bert = torch.cat([bert1, bert2], 1)
            all_phoneme_ids = torch.LongTensor(phones1 + phones2).to(device).unsqueeze(0)
        else:
            bert = bert2
            all_phoneme_ids = torch.LongTensor(phones2).to(device).unsqueeze(0)
        
        # 准备输入
        bert = bert.to(device).unsqueeze(0)
        all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(device)
        
        # 生成语义特征
        with torch.no_grad():
            pred_semantic, idx = GPT_model.infer_panel(
                all_phoneme_ids,
                all_phoneme_len,
                None if ref_free else prompt,
                bert,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                early_stop_num=25 * 30,  # 25hz * 30s = 750 (对应 30 秒音频)
            )
            pred_semantic = pred_semantic[:, -idx:].unsqueeze(0)
        
        # 生成音频
        if model_version in v3v4set:
            # v3/v4 版本的处理
            from module.mel_processing import mel_spectrogram_torch
            
            # 加载参考音频
            refer, audio_tensor = get_spepc(hps, ref_wav_path, dtype, device)
            
            # 准备音素
            phoneme_ids0 = torch.LongTensor(phones1).to(device).unsqueeze(0)
            phoneme_ids1 = torch.LongTensor(phones2).to(device).unsqueeze(0)
            
            # 解码
            fea_ref, ge = vq_model.decode_encp(prompt.unsqueeze(0), phoneme_ids0, refer)
            
            # 加载参考音频用于 Mel 提取
            ref_audio, sr = torchaudio.load(ref_wav_path)
            ref_audio = ref_audio.to(device).float()
            if ref_audio.shape[0] == 2:
                ref_audio = ref_audio.mean(0).unsqueeze(0)
            
            # 重采样
            tgt_sr = 24000 if model_version == "v3" else 32000
            if sr != tgt_sr:
                from torchaudio.transforms import Resample
                resampler = Resample(sr, tgt_sr).to(device)
                ref_audio = resampler(ref_audio)
            
            # 提取 Mel 频谱
            if model_version == "v3":
                mel2 = mel_spectrogram_torch(
                    ref_audio,
                    n_fft=1024,
                    win_size=1024,
                    hop_size=256,
                    num_mels=100,
                    sampling_rate=24000,
                    fmin=0,
                    fmax=None,
                    center=False,
                )
            else:
                mel2 = mel_spectrogram_torch(
                    ref_audio,
                    n_fft=1280,
                    win_size=1280,
                    hop_size=320,
                    num_mels=100,
                    sampling_rate=32000,
                    fmin=0,
                    fmax=None,
                    center=False,
                )
            
            # 归一化
            spec_min = -12
            spec_max = 2
            mel2 = (mel2 - spec_min) / (spec_max - spec_min) * 2 - 1
            
            # 处理长度
            T_min = min(mel2.shape[2], fea_ref.shape[2])
            mel2 = mel2[:, :, :T_min]
            fea_ref = fea_ref[:, :, :T_min]
            
            # 处理长音频
            Tref = 468 if model_version == "v3" else 500
            Tchunk = 934 if model_version == "v3" else 1000
            if T_min > Tref:
                mel2 = mel2[:, :, -Tref:]
                fea_ref = fea_ref[:, :, -Tref:]
                T_min = Tref
            
            chunk_len = Tchunk - T_min
            mel2 = mel2.to(dtype)
            
            # 解码目标语义特征
            fea_todo, ge = vq_model.decode_encp(pred_semantic, phoneme_ids1, refer, ge, speed)
            
            # 分块处理
            cfm_resss = []
            idx = 0
            while True:
                fea_todo_chunk = fea_todo[:, :, idx : idx + chunk_len]
                if fea_todo_chunk.shape[-1] == 0:
                    break
                idx += chunk_len
                
                # 拼接参考特征和目标特征
                fea = torch.cat([fea_ref, fea_todo_chunk], 2).transpose(2, 1)
                
                # 生成 Mel 频谱
                cfm_res = vq_model.cfm.inference(
                    fea, 
                    torch.LongTensor([fea.size(1)]).to(fea.device), 
                    mel2, 
                    sample_steps, 
                    inference_cfg_rate=0
                )
                
                # 截取生成的部分
                cfm_res = cfm_res[:, :, mel2.shape[2] :]
                
                # 更新参考特征
                mel2 = cfm_res[:, :, -T_min:]
                fea_ref = fea_todo_chunk[:, :, -T_min:]
                
                cfm_resss.append(cfm_res)
            
            # 合并结果
            cfm_res = torch.cat(cfm_resss, 2)
            
            # 反归一化
            cfm_res = (cfm_res + 1) / 2 * (spec_max - spec_min) + spec_min
            
            # 使用 BigVGAN 生成音频
            with torch.inference_mode():
                wav_gen = bigvgan_model(cfm_res)
                audio = wav_gen[0][0]
        else:
            # v1/v2 版本的处理
            refers, audio_tensor = get_spepc(hps, ref_wav_path, dtype, device)
            refers = [refers]
            
            # 解码
            audio = vq_model.decode(
                pred_semantic, 
                torch.LongTensor(phones2).to(device).unsqueeze(0), 
                refers, 
                speed=speed
            )[0][0]
        
        # 归一化音频
        max_audio = torch.abs(audio).max()
        if max_audio > 1:
            audio = audio / max_audio
        
        # 添加到结果列表
        audio_opt.append(audio)
        audio_opt.append(zero_wav_torch)
    
    # 合并音频
    audio_opt = torch.cat(audio_opt, 0)
    
    # 设置输出采样率
    if model_version in {"v1", "v2", "v2Pro", "v2ProPlus"}:
        opt_sr = 32000
    elif model_version == "v3":
        opt_sr = 24000
    else:
        opt_sr = 48000  # v4
    
    # 超分处理
    if if_sr and opt_sr == 24000:
        try:
            from tools.audio_sr import AP_BWE
            sr_model = AP_BWE(device, type('DictToAttrRecursive', (), {}))
            audio_opt, opt_sr = sr_model(audio_opt.unsqueeze(0), opt_sr)
            max_audio = np.abs(audio_opt).max()
            if max_audio > 1:
                audio_opt /= max_audio
        except Exception as e:
            print(f"超分处理失败: {e}")
    
    # 转换为 numpy 数组
    if isinstance(audio_opt, torch.Tensor):
        audio_opt = audio_opt.cpu().detach().numpy()
    
    return opt_sr, audio_opt

# 辅助函数：获取频谱
def get_spepc(hps, filename, dtype, device, is_v2pro=False):
    """获取音频的频谱特征"""
    import torchaudio
    from module.mel_processing import spectrogram_torch
    
    sr1 = int(hps["data"]["sampling_rate"])
    audio, sr0 = torchaudio.load(filename)
    
    # 重采样
    if sr0 != sr1:
        audio = audio.to(device)
        if audio.shape[0] == 2:
            audio = audio.mean(0).unsqueeze(0)
        from torchaudio.transforms import Resample
        resampler = Resample(sr0, sr1).to(device)
        audio = resampler(audio)
    else:
        audio = audio.to(device)
        if audio.shape[0] == 2:
            audio = audio.mean(0).unsqueeze(0)
    
    # 归一化
    maxx = audio.abs().max()
    if maxx > 1:
        audio /= min(2, maxx)
    
    # 提取频谱
    spec = spectrogram_torch(
        audio,
        hps["data"]["filter_length"],
        hps["data"]["sampling_rate"],
        hps["data"]["hop_length"],
        hps["data"]["win_length"],
        center=False,
    )
    spec = spec.to(dtype)
    
    # 处理 v2Pro
    if is_v2pro:
        from torchaudio.transforms import Resample
        resampler = Resample(sr1, 16000).to(device)
        audio = resampler(audio).to(dtype)
    
    return spec, audio

# 简化的接口函数
def generate_audio(
    ref_wav_path,
    text,
    prompt_text="",
    prompt_language="中文",
    text_language="中文",
    output_path=None,
    **kwargs
):
    """
    简化的音频生成接口
    
    Args:
        ref_wav_path: 参考音频路径
        text: 需要合成的文本
        prompt_text: 参考音频的文本（可选）
        prompt_language: 参考音频的语言（可选）
        text_language: 需要合成的语言（可选）
        output_path: 输出音频路径（可选）
        **kwargs: 其他参数
    
    Returns:
        tuple: (采样率, 音频数据) 如果指定了 output_path，则返回 None
    """
    # 生成音频
    sr, audio = get_tts_wav(
        ref_wav_path=ref_wav_path,
        prompt_text=prompt_text,
        prompt_language=prompt_language,
        text=text,
        text_language=text_language,
        **kwargs
    )
    
    # 保存音频
    if output_path:
        sf.write(output_path, audio, sr)
        print(f"音频已保存到: {output_path}")
        return None
    else:
        return sr, audio

if __name__ == "__main__":
    # 测试代码
    ref_wav = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output", "case3_promptBatch_synBatch_promptThenGen_xvec_only_1.wav")
    text = "您好，我是GPT-SoVITS，一个强大的文本转语音模型。"
    output = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output", "gpt_sovits_api_test.wav")
    
    print("生成音频中...")
    generate_audio(
        ref_wav_path=ref_wav,
        text=text,
        output_path=output,
        top_k=50,
        top_p=0.7,
        temperature=0.7
    )
    print("音频生成完成！")
