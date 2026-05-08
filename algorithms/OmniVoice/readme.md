# OmniVoice 集成说明

## 项目概述

OmniVoice 是由 k2-fsa (Next-gen Kaldi) 开发的先进多语言零样本文本到语音 (TTS) 模型，支持超过 600 种语言。基于扩散语言模型架构，具有高速推理能力和卓越的语音克隆质量。

- **官方仓库**: https://github.com/k2-fsa/OmniVoice
- **HuggingFace**: https://huggingface.co/k2-fsa/OmniVoice
- **论文**: https://arxiv.org/abs/2604.00688

## 核心特性

### 1. 超广泛语言支持
- **600+ 种语言**: 零样本 TTS 模型中最广泛的语言覆盖
- 支持中文（含多种方言）、英语、日语、韩语等主流语言
- 支持众多低资源语言

### 2. 三种生成模式

#### 声音克隆 (Voice Cloning)
- 通过短参考音频克隆说话人音色
- 支持自动转录（使用 Whisper ASR）
- 建议参考音频时长：3-10 秒
- 跨语言克隆时会产生参考音频语言的口音

#### 声音设计 (Voice Design)
- 通过属性描述生成音色，无需参考音频
- 支持属性：
  - **性别**: male, female
  - **年龄**: child, teenager, young adult, middle-aged, elderly
  - **音调**: very low pitch, low pitch, moderate pitch, high pitch, very high pitch
  - **风格**: whisper（耳语）
  - **英语口音**: american, british, australian, canadian, indian, chinese, korean, japanese, portuguese, russian
  - **中文方言**: 四川话、东北话、河南话、陕西话、贵州话、云南话、桂林话、济南话、石家庄话、甘肃话、宁夏话、青岛话

#### 随机音色 (Auto Voice)
- 自动生成随机音色
- 适用于不需要特定音色的场景

### 3. 高速推理
- **RTF 低至 0.025**: 40 倍实时速度
- 支持 16 步快速推理或 32 步高质量推理
- 扩散语言模型架构优化

### 4. 精细控制
- **语速控制**: speed 参数（>1.0 更快，<1.0 更慢）
- **时长控制**: duration 参数固定输出长度
- **非语言符号**: 支持 [laughter], [sigh], [confirmation-en] 等表情符号
- **发音校正**: 支持拼音（中文）和 CMU 音标（英文）

## VersTTS 集成说明

### 后端 API

**端点**: `POST /tts/omnivoice`

**参数**:
- `text` (required): 要合成的文本
- `mode`: 生成模式 (`auto_voice`, `voice_clone`, `voice_design`)
- `clone_speaker_id`: 克隆说话人ID（voice_clone 模式）
- `voice_design_prompt`: 声音设计描述（voice_design 模式）
- `num_steps`: 扩散步数（16 或 32）
- `speed`: 语速因子（默认 1.0）
- `output_format`: 输出格式 (`url` 或 `base64`)

**响应**:
```json
{
  "success": true,
  "message": "合成成功",
  "audio_url": "/audio/omnivoice_20250101_120000.wav",
  "sample_rate": 24000
}
```

### 支持的功能

| 功能 | 支持状态 | 说明 |
|------|---------|------|
| 随机音色 | ✅ | 自动生成随机音色 |
| 声音克隆 | ✅ | 通过说话人ID使用说话人管理模块的音频 |
| 声音设计 | ✅ | 通过属性描述生成音色 |
| 语速控制 | ✅ | speed 参数 |
| 扩散步数 | ✅ | 16/32 步可选 |
| 非语言符号 | ✅ | [laughter] 等 |
| 发音校正 | ✅ | 拼音/音标支持 |

### 说话人模块适配

OmniVoice 的声音克隆模式已适配 VersTTS 的说话人管理模块：
- 从说话人管理模块选择说话人
- 自动获取说话人音频路径和参考文本
- 如果参考文本为空，自动使用 Whisper ASR 转录

### 前端页面

**路径**: `/pages/omnivoice.html`

**功能**:
- 三种生成模式切换（随机音色/声音克隆/声音设计）
- 说话人选择下拉框（声音克隆模式）
- 声音设计属性快速选择器
- 扩散步数和语速参数调整

## 模型信息

- **模型大小**: 约 2-3GB（取决于具体配置）
- **采样率**: 24kHz
- **模型架构**: 扩散语言模型
- **依赖**: transformers>=5.3.0, torch>=2.4

## 使用示例

### 随机音色
```python
from omnivoice import OmniVoice
import torch

model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cuda:0")
audio = model.generate(text="你好，这是OmniVoice的测试。")
```

### 声音克隆
```python
audio = model.generate(
    text="你好，这是克隆的声音。",
    ref_audio="reference.wav",
    ref_text="参考音频的文本内容"  # 可选，不提供则自动转录
)
```

### 声音设计
```python
audio = model.generate(
    text="你好，这是设计的音色。",
    instruct="female, young adult, high pitch, 四川话"
)
```

## 注意事项

1. **transformers 版本**: 需要 >=5.3.0
2. **显存需求**: 建议至少 8GB GPU 显存
3. **音频时长**: 参考音频建议 3-10 秒，过长会降低质量
4. **跨语言克隆**: 会产生参考音频语言的口音
5. **声音设计**: 仅在中英文数据上训练，对其他语言可能不稳定

## 局限性

1. 声音设计功能仅针对中英文优化
2. 跨语言克隆会产生口音
3. 某些属性组合可能效果不佳
4. 模型文件较大，首次加载时间较长

## 许可证

Apache-2.0 License

## 免责声明

严禁将本模型用于未经授权的声音克隆、声音模仿、欺诈、诈骗或其他违法或不道德活动。
