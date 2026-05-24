# VoxCPM 算法分析与验证

## 项目信息
- **项目地址**: https://github.com/OpenBMB/VoxCPM
- **官方文档**: https://voxcpm.readthedocs.io/en/latest/
- **HuggingFace**: https://huggingface.co/openbmb/VoxCPM2
- **许可证**: Apache-2.0 (可商用)

## 核心思路

VoxCPM是一个**无tokenizer**的文本转语音(TTS)系统，采用端到端的**扩散自回归架构**，直接生成连续语音表示，绕过了传统的离散tokenization步骤。

### 技术架构
1. **MiniCPM-4 主干网络**: 基于2B参数的大语言模型
2. **AudioVAE V2**: 非对称编解码设计，支持16kHz输入和48kHz输出
3. **扩散自回归生成**: 结合扩散模型和自回归生成的优势
4. **流匹配(Flow Matching)**: 用于高质量音频合成

### 核心创新
- **无Tokenizer设计**: 直接生成连续表示，避免离散化带来的信息损失
- **上下文感知合成**: 自动从文本内容推断适当的韵律和表达
- **内置超分辨率**: 无需外部上采样器即可输出48kHz音频

## 支持的TTS能力

### ✅ 支持的功能
1. **文本转语音 (TTS)**
   - 支持30种语言输入
   - 无需语言标签，自动识别
   - 上下文感知韵律生成

2. **声音设计 (Voice Design)**
   - 通过自然语言描述创建全新音色
   - 支持控制性别、年龄、语调、情感、语速等
   - 无需参考音频

3. **可控声音克隆 (Controllable Cloning)**
   - 从短参考音频克隆音色
   - 支持风格指导调整语速、情感和表达
   - 保持原始音色特征

4. **极致克隆 (Ultimate Cloning)**
   - 提供参考音频和对应转录文本
   - 从参考音频继续生成
   - 精确复现每个声音细节(音色、节奏、情感、风格)

5. **实时流式生成**
   - RTF低至~0.3 (NVIDIA RTX 4090)
   - 支持vLLM加速 (~0.13 RTF)
   - OpenAI兼容API

### ❌ 不支持的功能
- 不支持SSML标记
- 不支持音素级别的精确控制
- 不支持下说话人分离(需要外部处理)

## 支持的语音克隆模式

| 模式 | 需要参考音频 | 需要参考文本 | 特点 |
|------|-------------|-------------|------|
| 基础TTS | ❌ | ❌ | 使用默认或设计音色 |
| 声音设计 | ❌ | ❌ | 通过描述创建音色 |
| 可控克隆 | ✅ | ❌ | 克隆音色+风格控制 |
| 极致克隆 | ✅ | ✅ | 最高相似度克隆 |

## 支持的语言 (30种)

阿拉伯语、缅甸语、中文、丹麦语、荷兰语、英语、芬兰语、法语、德语、希腊语、希伯来语、印地语、印尼语、意大利语、日语、高棉语、韩语、老挝语、马来语、挪威语、波兰语、葡萄牙语、俄语、西班牙语、斯瓦希里语、瑞典语、他加禄语、泰语、土耳其语、越南语

### 中文方言支持
四川话、粤语、吴语、东北话、河南话、陕西话、山东话、天津话、闽南话

## 模型版本

| 版本 | 参数 | 特点 | 推荐场景 |
|------|------|------|----------|
| VoxCPM2 | 2B | 最新版本，30语言，48kHz输出 | 生产环境 |
| VoxCPM1.5 | - | SFT & LoRA微调支持 | 自定义训练 |
| VoxCPM-0.5B | 0.5B | 轻量级版本 | 资源受限环境 |

## 性能指标

- **实时因子(RTF)**: ~0.3 (RTX 4090), ~0.13 (vLLM加速)
- **输出采样率**: 48kHz
- **输入参考音频**: 16kHz
- **支持的最大文本长度**: 取决于GPU显存

## 用法示例

### 基础TTS
```python
from voxcpm import VoxCPM
import soundfile as sf

model = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False)
wav = model.generate(
    text="VoxCPM2支持高质量的多语言语音合成。",
    cfg_value=2.0,
    inference_timesteps=10,
)
sf.write("output.wav", wav, model.tts_model.sample_rate)
```

### 声音设计
```python
wav = model.generate(
    text="(A young woman, gentle and sweet voice)你好，欢迎使用VoxCPM2！",
    cfg_value=2.0,
    inference_timesteps=10,
)
```

### 声音克隆
```python
wav = model.generate(
    text="这是使用VoxCPM2克隆的声音。",
    reference_wav_path="path/to/voice.wav",
    cfg_value=2.0,
    inference_timesteps=10,
)
```

### 极致克隆
```python
wav = model.generate(
    text="这是极致克隆的演示。",
    prompt_wav_path="path/to/voice.wav",
    prompt_text="参考音频的转录文本。",
    reference_wav_path="path/to/voice.wav",
)
```

## 局限性

1. **计算资源需求**: 2B模型需要较大显存(建议16GB+)
2. **推理速度**: 相比轻量级模型较慢，但支持流式生成
3. **语言支持**: 虽然支持30种语言，但某些低资源语言效果可能不如高资源语言
4. **长文本处理**: 超长文本可能需要分段处理
5. **情感控制**: 通过文本描述控制，不如参数化控制精确

## 最佳实践

1. **参考音频选择**: 使用3-10秒清晰、无噪音的音频
2. **CFG值调整**: 默认2.0，增大增强提示遵循，减小增加多样性
3. **推理步数**: 默认10步，质量要求不高可减少以提速
4. **文本预处理**: 确保输入文本清晰、无特殊字符

## 部署建议

- **GPU**: 建议使用NVIDIA RTX 4090或同等性能显卡
- **显存**: 至少16GB显存用于2B模型
- **CUDA**: 需要CUDA 12.0+
- **优化**: 生产环境建议使用vLLM加速

## 相关资源

- **在线演示**: https://huggingface.co/spaces/OpenBMB/VoxCPM-Demo
- **音频样本**: https://openbmb.github.io/voxcpm2-demopage/
- **技术报告**: https://arxiv.org/abs/2509.24650

---

## VersTTS 集成说明

### 后端API适配

VersTTS为VoxCPM提供了统一的后端API接口，支持以下功能：

#### API端点
```
POST /tts/voxcpm
```

#### 请求参数
| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| text | string | 是 | 要合成的文本 |
| mode | string | 否 | 模式：base/voice_design/clone/ultimate_clone，默认base |
| clone_speaker_id | string | 否 | 说话人ID（clone/ultimate_clone模式时使用） |
| ref_text | string | 否 | 参考文本（ultimate_clone模式时使用） |
| voice_design_prompt | string | 否 | 音色描述（voice_design模式时使用） |
| cfg_value | float | 否 | CFG值，默认2.0 |
| inference_timesteps | int | 否 | 推理步数，默认10 |

#### 模式说明
1. **base模式**: 基础TTS生成，不使用参考音频
2. **voice_design模式**: 使用自然语言描述创建音色
3. **clone模式**: 使用说话人管理模块中的人声进行声音克隆
4. **ultimate_clone模式**: 使用说话人管理模块中的人声+参考文本进行极致克隆

#### 集成特点
- **说话人管理**: 所有克隆模式均使用统一的说话人管理模块，无需单独上传参考音频
- **自动匹配**: 后端自动根据clone_speaker_id获取对应的音频路径和参考文本
- **参数透传**: CFG值、推理步数等参数直接透传给VoxCPM模型

### 前端页面适配

前端页面已适配说话人管理模块：
- **说话人选择**: clone和ultimate_clone模式使用下拉框选择说话人
- **自动加载**: 页面加载时自动获取已保存的说话人列表
- **信息展示**: 显示选中说话人的名称、创建时间和参考文本预览

### 模型加载

```python
from voxcpm import VoxCPM

# VersTTS使用以下方式加载模型
model = VoxCPM.from_pretrained(
    "openbmb/VoxCPM2",
    load_denoiser=False,
)
```

### 使用示例

#### 基础生成
```python
audio_data = model.generate(
    text="你好，这是VoxCPM的演示。",
    cfg_value=2.0,
    inference_timesteps=10,
)
```

#### 使用说话人克隆（VersTTS集成方式）
```python
# 后端根据clone_speaker_id自动获取参考音频路径
audio_data = model.generate(
    text="使用选中说话人的声音生成这段文本。",
    reference_wav_path=speaker_audio_path,  # 从说话人管理模块获取
    cfg_value=2.0,
    inference_timesteps=10,
)
```

#### 极致克隆（VersTTS集成方式）
```python
# 后端根据clone_speaker_id自动获取参考音频路径和参考文本
audio_data = model.generate(
    text="这是极致克隆的演示。",
    prompt_wav_path=speaker_audio_path,     # 从说话人管理模块获取
    prompt_text=speaker_reference_text,      # 从说话人管理模块获取
    reference_wav_path=speaker_audio_path,   # 可选，用于提高相似度
)
```
