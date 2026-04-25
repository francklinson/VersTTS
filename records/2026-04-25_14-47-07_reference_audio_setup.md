# 参考人声音频管理系统搭建记录

## 日期
2026-04-25

## 任务
1. 从公开资源渠道下载参考人声音频(要求声音纯净,没有背景噪声干扰,以儿童中学生为主)
2. 内置提供参考人声(人声标签),不同项目之间能否共用这些人声特征?

## 完成内容

### 1. 创建参考音频管理框架

**目录结构:**
```
reference_audio/
├── README.md              # 参考音频管理指南
├── metadata.json          # 元数据配置
├── children/              # 儿童声音(3-12岁)
├── teenagers/             # 中学生声音(13-18岁)
└── adults/                # 成人声音(18岁以上)
```

### 2. 元数据系统

**文件:** `reference_audio/metadata.json`

**包含内容:**
- 版本信息和描述
- 分类定义(children/teenagers/adults)
- 各分类的技术要求:
  - 最小/最大时长: 5-30秒
  - 采样率: 22050Hz
  - 声道: 单声道
- 模型兼容性矩阵:

| 模型 | 支持参考音频 | 说明 |
|------|-------------|------|
| ChatTTS | ❌ | 使用随机说话人 |
| CosyVoice | ✅ | zero_shot/cross_lingual模式 |
| F5-TTS | ✅ | 需要参考音频+文本 |
| Qwen3-TTS | ✅ | voice_clone模式 |
| OpenVoice | ✅ | 可选，使用音色转换 |
| GPT-SoVITS | ✅ | 必须提供参考音频+文本 |

### 3. 音频收集规范

**技术规范:**
- 采样率: 22050Hz 或 44100Hz
- 声道: 单声道
- 位深度: 16-bit
- 时长: 5-30秒
- 格式: WAV

**质量要求:**
- ✅ 声音清晰，无明显背景噪音
- ✅ 无混响/回声
- ✅ 无背景音乐
- ✅ 无其他说话人声音
- ✅ 发音标准，语速正常

### 4. 合法来源建议

**公开数据集:**
1. **Common Voice** (Mozilla)
   - 许可: CC0 (公有领域)
   - 网址: https://commonvoice.mozilla.org/
   - 需要筛选儿童和青少年音频

2. **AIShell**
   - 部分免费的中文语音数据集
   - 网址: http://www.aishelltech.com/

3. **THCHS-30**
   - 清华大学中文语音数据集
   - 免费用于学术研究

**自建录音(推荐):**
- 获得本人/监护人授权
- 签署语音使用授权书
- 最安全可靠的方式

### 5. 音频验证工具

**文件:** `tools/validate_references.py`

**功能:**
- 检查音频格式(WAV)
- 验证采样率(建议22050Hz)
- 验证声道(建议单声道)
- 检查时长(5-30秒)
- 检测音量水平
- 估计信噪比
- 检测静音比例

**使用方法:**
```bash
# 验证整个目录
python tools/validate_references.py reference_audio

# 验证单个文件
python tools/validate_references.py reference_audio/children/child_001.wav

# 输出JSON格式
python tools/validate_references.py reference_audio --json
```

### 6. 人声兼容性分析

**不同项目间的人声特征能否共用?**

| 模型 | 人声特征格式 | 能否共用 |
|------|-------------|---------|
| ChatTTS | Speaker Embedding | ❌ 不支持参考音频 |
| CosyVoice | Speaker Embedding | ⚠️ 仅同模型共用 |
| F5-TTS | Reference Audio | ✅ 可共用参考音频 |
| Qwen3-TTS | Reference Audio | ✅ 可共用参考音频 |
| OpenVoice | Speaker Embedding (SE) | ⚠️ 仅同模型共用 |
| GPT-SoVITS | Reference Audio + Text | ✅ 可共用参考音频 |

**结论:**
- 使用原始参考音频(WAV文件)的方式具有最好的兼容性
- 需要Speaker Embedding的模型之间特征不通用
- 建议统一使用**参考音频文件**作为人声标签

## 注意事项

⚠️ **版权警告**
- 请勿使用未经授权的商业音频
- 儿童音频需获得监护人书面同意
- 建议优先使用自建录音

⚠️ **隐私保护**
- 妥善保管参考音频文件
- 不要公开分享包含个人信息的音频
- 遵守相关法律法规

## 后续计划

1. **收集参考音频样本**
   - 儿童声音样本: 目标10-20个
   - 中学生声音样本: 目标10-20个

2. **人声标签系统**
   - 创建前端人声选择组件
   - 实现人声预览功能
   - 添加人声搜索/筛选

3. **兼容性测试**
   - 测试同一参考音频在不同模型的效果
   - 评估声音克隆质量
   - 建立人声质量评价体系

## 元数据示例

```json
{
  "id": "child_female_001",
  "filename": "children/child_female_001.wav",
  "category": "children",
  "gender": "female",
  "age_group": "8-10",
  "language": "zh",
  "text": "今天天气真好，我想去公园玩。",
  "duration": 5.2,
  "sample_rate": 22050,
  "source": "自建录音",
  "license": "授权使用",
  "tags": ["清晰", "标准普通话", "开心"],
  "compatible_models": {
    "chattts": false,
    "cosyvoice": true,
    "f5tts": true,
    "qwen3tts": true,
    "openvoice": true,
    "gptsovits": true
  }
}
```
