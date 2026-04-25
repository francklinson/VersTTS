# VersTTS 项目工作记录

> 创建时间: 2025-04-25
> 项目路径: `/home/zhouchenghao/PycharmProjects/VersTTS`

---

## 1. 项目概述

本项目包含6个TTS（Text-to-Speech）语音合成项目，用于测试和验证各项目的运行状态。

### 项目列表

| 项目 | 路径 | 状态 |
|------|------|------|
| Qwen3-TTS | `Qwen3-TTS/` | ✅ 可用 |
| CosyVoice | `CosyVoice/` | ✅ 可用 |
| ChatTTS | `ChatTTS/` | ✅ 可用 |
| F5-TTS | `F5-TTS/` | ✅ 可用 |
| OpenVoice | `OpenVoice/` | ✅ 可用 |
| GPT-SoVITS | `GPT-SoVITS/` | ✅ 可用 |

---

## 2. 环境配置

### 2.1 虚拟环境

- 虚拟环境路径: `/home/zhouchenghao/PycharmProjects/VersTTS/.venv`
- Python 版本: 3.12

### 2.2 公共依赖

已安装的依赖包:

```
conformer==0.3.2
diffusers==0.29.0
lightning==2.2.4
gdown==5.1.0
wget==3.2
pyworld==0.3.4
eng-to-ipa
wavmark
```

---

## 3. 各项目详细配置

### 3.1 Qwen3-TTS

**状态**: ✅ 完全可用

**测试命令**:
```bash
cd Qwen3-TTS
source ../.venv/bin/activate
python examples/test_model_12hz_base.py
```

**输出**: 生成12个测试音频文件

---

### 3.2 CosyVoice

**状态**: ✅ 完全可用

**初始化步骤**:
```bash
cd CosyVoice
git submodule update --init --recursive
```

**额外依赖**:
```bash
pip install conformer==0.3.2 diffusers==0.29.0 lightning==0.10.0 gdown==5.1.0 wget==3.2 inflect==7.3.1 pyworld==0.3.4
```

**测试命令**:
```python
import sys
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import AutoModel
import torchaudio

cosyvoice = AutoModel(model_dir='models/iic/CosyVoice-300M-SFT', fp16=True)
for i, j in enumerate(cosyvoice.inference_sft('你好，欢迎使用语音合成系统。', '中文女', stream=False)):
    torchaudio.save('cosyvoice_output.wav', j['tts_speech'], cosyvoice.sample_rate)
```

---

### 3.3 ChatTTS

**状态**: ✅ 完全可用

**模型路径**: `ChatTTS/models/`

**测试命令**:
```python
import torch
import ChatTTS
import soundfile as sf

chat = ChatTTS.Chat()
chat.load(source='local', custom_path='./models', device='cuda')

texts = ['你好，欢迎使用语音合成系统。']
with torch.no_grad():
    wavs = chat.infer(texts, use_decoder=True)

wav = wavs[0].cpu().numpy() if hasattr(wavs[0], 'cpu') else wavs[0]
sf.write('chattts_output.wav', wav, 24000)
```

---

### 3.4 F5-TTS

**状态**: ✅ 完全可用

**修复内容**:
1. 从 `ChatTTS/models/asset/Vocos.pt` 复制 Vocos 模型到 `checkpoints/vocos-mel-24khz/`
2. 创建 `config.yaml` 配置文件

**模型目录**:
```
F5-TTS/checkpoints/vocos-mel-24khz/
├── config.yaml
└── pytorch_model.bin
```

**测试命令**:
```bash
python -m f5_tts.infer.infer_cli \
    -r <ref_audio.wav> \
    -s "<source_text>" \
    -t "<target_text>" \
    -o ./tests \
    --no_legacy_text \
    --load_vocoder_from_local \
    --vocoder_name vocos
```

---

### 3.5 OpenVoice

**状态**: ✅ 完全可用

**修复内容**:
- 修改 `openvoice/api.py` 中 `ToneColorConverter` 类的 `enable_watermark` 参数传递方式

**代码修改** (`openvoice/api.py` 第100-115行):
```python
class ToneColorConverter(OpenVoiceBaseClass):
    def __init__(self, *args, **kwargs):
        enable_watermark = kwargs.pop('enable_watermark', True)
        super().__init__(*args, **kwargs)

        if enable_watermark:
            import wavmark
            self.watermark_model = wavmark.load_model().to(self.device)
        else:
            self.watermark_model = None
        self.version = getattr(self.hps, '_version_', "v1")
```

**测试命令**:
```python
from openvoice.api import BaseSpeakerTTS, ToneColorConverter

en_base_speaker_tts = BaseSpeakerTTS('checkpoints_v1/checkpoints/base_speakers/EN/config.json', device='cuda')
en_base_speaker_tts.load_ckpt('checkpoints_v1/checkpoints/base_speakers/EN/checkpoint.pth')
tone_color_converter = ToneColorConverter('checkpoints_v1/checkpoints/converter/config.json', device='cuda', enable_watermark=False)

src_path = 'tests/tmp.wav'
en_base_speaker_tts.tts('Hello, this is a test.', src_path, speaker='default', language='English')
```

---

### 3.6 GPT-SoVITS

**状态**: ✅ 完全可用

#### 3.6.1 代码修复

**修复1**: `inference_webui.py` 第384行，处理缺失的 `max_sec` 配置
```python
max_sec = config["data"].get("max_sec", 30)
```

**修复2**: `AR/models/t2s_model.py` 第263-274行，处理配置键名兼容
```python
self.model_dim = model_config.get("hidden_dim", model_config.get("hidden_channels", 512))
self.embedding_dim = model_config.get("embedding_dim", model_config.get("hidden_channels", 512))
self.num_head = model_config.get("head", 8)
self.num_layers = model_config.get("n_layer", 12)
self.vocab_size = model_config.get("vocab_size", 1025)
self.phoneme_vocab_size = model_config.get("phoneme_vocab_size", 1024)
self.p_dropout = model_config.get("dropout", 0.1)
self.EOS = model_config.get("EOS", self.vocab_size - 1)
```

#### 3.6.2 模型文件

**预训练模型** (来源: `lj1995/GPT-SoVITS` on HuggingFace):

| 文件 | 说明 | 大小 |
|------|------|------|
| `s1v3.ckpt` | GPT/Text2Semantic 模型 | 148MB |
| `s2Gv3.pth` | SoVITS 声码器模型 | 733MB |

**预训练模型路径**:
```
GPT-SoVITS/GPT_SoVITS/pretrained_models/
├── s1v3.ckpt
├── s2Gv3.pth
├── chinese-hubert-base/
├── chinese-roberta-wwm-ext-large/
├── fast_langdetect/lid.176.bin
└── G2PWModel/  (用于中文文本处理)
```

**G2PWModel**:
- 来源: ModelScope `XXXXRT/GPT-SoVITS-Pretrained`
- 路径: `GPT_SoVITS/text/G2PWModel/`

#### 3.6.3 测试命令

```python
import sys
sys.path.insert(0, 'GPT_SoVITS')
from GPT_SoVITS.inference_webui import change_gpt_weights, change_sovits_weights, get_tts_wav
from scipy.io import wavfile

change_gpt_weights(gpt_path='GPT_SoVITS/pretrained_models/s1v3.ckpt')
change_sovits_weights(sovits_path='GPT_SoVITS/pretrained_models/s2Gv3.pth')

result = list(get_tts_wav(
    ref_wav_path='<ref_audio.wav>',
    prompt_text='<prompt_text>',
    prompt_language='英文',
    text='<target_text>',
    text_language='英文',
    ref_free=False,
))

if result and len(result) > 0:
    idx, wav = result[0]
    wavfile.write('output.wav', 24000, wav)
```

**使用注意**:
- 需要设置 `PYTHONPATH=GPT_SoVITS`
- 保存音频使用 `scipy.io.wavfile.write` 而非 `torchaudio.save` (避免codec错误)

---

## 4. 输出音频

所有项目生成的测试音频已保存至 `/home/zhouchenghao/PycharmProjects/VersTTS/output/`

### 4.1 主要输出文件

| 项目 | 文件名 | 大小 |
|------|--------|------|
| Qwen3-TTS | `qwen3tts_output.wav` | 242KB |
| CosyVoice | `cosyvoice_output.wav` | 90KB |
| ChatTTS | `chattts_output.wav` | 116KB |
| F5-TTS | `f5tts_output.wav` | 196KB |
| OpenVoice | `openvoice_output.wav` | 108KB |
| GPT-SoVITS | `gpt_sovits_output.wav` | 206KB |

### 4.2 完整文件列表

```
output/
├── qwen3tts_output.wav
├── cosyvoice_output.wav
├── chattts_output.wav
├── f5tts_output.wav
├── openvoice_output.wav
├── gpt_sovits_output.wav
├── case1_promptSingle_synSingle_direct_icl_0.wav
├── case1_promptSingle_synSingle_direct_xvec_only_0.wav
├── case1_promptSingle_synSingle_promptThenGen_icl_0.wav
├── case1_promptSingle_synSingle_promptThenGen_xvec_only_0.wav
├── case2_promptSingle_synBatch_direct_icl_0.wav
├── case2_promptSingle_synBatch_direct_icl_1.wav
├── case2_promptSingle_synBatch_direct_xvec_only_0.wav
├── case2_promptSingle_synBatch_direct_xvec_only_1.wav
├── case2_promptSingle_synBatch_promptThenGen_icl_0.wav
├── case2_promptSingle_synBatch_promptThenGen_icl_1.wav
├── case2_promptSingle_synBatch_promptThenGen_xvec_only_0.wav
├── case2_promptSingle_synBatch_promptThenGen_xvec_only_1.wav
├── case3_promptBatch_synBatch_direct_icl_0.wav
├── case3_promptBatch_synBatch_direct_icl_1.wav
├── case3_promptBatch_synBatch_direct_xvec_only_0.wav
├── case3_promptBatch_synBatch_direct_xvec_only_1.wav
├── case3_promptBatch_synBatch_promptThenGen_icl_0.wav
├── case3_promptBatch_synBatch_promptThenGen_icl_1.wav
├── case3_promptBatch_synBatch_promptThenGen_xvec_only_0.wav
└── case3_promptBatch_synBatch_promptThenGen_xvec_only_1.wav
```

---

## 5. 共享模型目录

为避免重复下载，创建了共享模型目录 `/home/zhouchenghao/PycharmProjects/VersTTS/models/`

```
models/
└── vocos-mel-24khz/
    ├── config.yaml
    └── pytorch_model.bin
```

---

## 6. 待解决问题

### 6.1 GPT-SoVITS NLTK 依赖

GPT-SoVITS 在推理时需要 NLTK 的 `averaged_perceptron_tagger_eng` 模型，但因网络限制无法自动下载。

**临时解决方案**: 英文文本推理不受影响，中文文本推理需要额外配置。

**手动下载方式**:
```python
import nltk
nltk.download('averaged_perceptron_tagger_eng')
```

---

## 7. 使用建议

### 7.1 统一环境激活

所有项目共用同一个虚拟环境:

```bash
source /home/zhouchenghao/PycharmProjects/VersTTS/.venv/bin/activate
```

### 7.2 项目特定环境变量

**GPT-SoVITS**:
```bash
cd GPT-SoVITS
PYTHONPATH=GPT_SoVITS python <script.py>
```

---

## 8. 修改文件清单

| 文件 | 修改内容 |
|------|----------|
| `OpenVoice/openvoice/api.py` | 修复 `ToneColorConverter.enable_watermark` 参数传递 |
| `GPT-SoVITS/GPT_SoVITS/inference_webui.py` | 修复 `max_sec` 配置读取 |
| `GPT-SoVITS/GPT_SoVITS/AR/models/t2s_model.py` | 修复配置键名兼容性 |

---

*本记录由 Trae AI Assistant 自动生成*
