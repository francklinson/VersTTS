# 参考人声音频管理

## 目录结构

```
reference_audio/
├── README.md              # 本文件
├── metadata.json          # 参考音频元数据
├── download_tool.py       # 音频下载工具(待开发)
├── children/              # 儿童声音
│   ├── child_001.wav
│   └── ...
├── teenagers/             # 中学生声音
│   ├── teen_001.wav
│   └── ...
└── adults/                # 成人声音(可选)
    ├── adult_001.wav
    └── ...
```

## 音频收集规范

### 音频质量要求
1. **采样率**: 22050Hz 或 44100Hz
2. **声道**: 单声道(Mono)
3. **位深度**: 16-bit
4. **时长**: 5-30秒
5. **格式**: WAV

### 声音质量要求
1. ✅ 声音清晰，无明显背景噪音
2. ✅ 无混响/回声
3. ✅ 无背景音乐
4. ✅ 无其他说话人声音
5. ✅ 发音标准，语速正常
6. ❌ 避免咳嗽、笑声等杂音
7. ❌ 避免过度情感化的表达

### 内容要求
1. 中文文本优先
2. 内容健康积极
3. 避免敏感词汇
4. 建议使用日常对话内容

## 合法来源建议

### 可考虑的公开数据源
1. **Common Voice** (Mozilla)
   - 网址: https://commonvoice.mozilla.org/
   - 许可: CC0 (公有领域)
   - 说明: 众包语音数据集，需筛选儿童和青少年音频

2. **AIShell**
   - 网址: http://www.aishelltech.com/
   - 说明: 中文语音数据集，部分免费

3. **THCHS-30**
   - 说明: 清华大学中文语音数据集
   - 免费用于学术研究

4. **自建录音**
   - 获得本人/监护人授权
   - 签署语音使用授权书
   - 最安全可靠的方式

## 元数据格式

每个参考音频需要记录以下信息:

```json
{
  "id": "child_001",
  "filename": "children/child_001.wav",
  "category": "children",  // children/teenagers/adults
  "gender": "female",      // male/female
  "age_group": "8-10",     // 年龄段
  "language": "zh",
  "text": "这是一段示例文本",
  "duration": 5.2,
  "sample_rate": 22050,
  "source": "自建录音",
  "license": "授权使用",
  "tags": ["清晰", "标准普通话"],
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

## 使用说明

### 添加新的参考音频

1. 将音频文件放入对应目录(children/teenagers/)
2. 更新 metadata.json
3. 运行验证工具: `python validate_references.py`

### 在不同模型中使用

| 模型 | 使用方法 |
|------|---------|
| CosyVoice | 上传参考音频，选择zero_shot模式 |
| F5-TTS | 上传参考音频和对应文本 |
| Qwen3-TTS | 上传参考音频，选择voice_clone模式 |
| OpenVoice | 上传参考音频进行音色转换 |
| GPT-SoVITS | 上传参考音频和对应文本 |

## 注意事项

⚠️ **版权警告**
- 请勿使用未经授权的商业音频
- 请勿使用受版权保护的音乐/影视片段
- 儿童音频需获得监护人书面同意

⚠️ **隐私保护**
- 妥善保管参考音频文件
- 不要公开分享包含个人信息的音频
- 遵守相关法律法规

## 开发计划

- [x] 创建音频下载工具脚本
- [x] 实现音频质量检测工具
- [x] 创建音频格式转换工具
- [x] 添加人声标签管理系统
- [x] 测试不同模型间的兼容性
- [ ] 从公开数据集下载更多参考音频

## 下载工具

已创建下载/处理工具: `tools/download_reference_voices.py`

### 查看可用公开数据集
```bash
python tools/download_reference_voices.py --list
```

### 处理下载的音频
```bash
python tools/download_reference_voices.py \
  --process /path/to/downloaded/audio.wav \
  --category children \
  --name "儿童声音-01" \
  --gender female \
  --age "8-10" \
  --text "你好，这是参考音频的文本内容。"
```

## 当前参考音频来源

### 示例音频（4个）
这些音频是从项目现有TTS输出整理的示例数据：

| 人声ID | 分类 | 性别 | 来源 |
|--------|------|------|------|
| voice_child_001 | 儿童 | 女 | 项目音频 |
| voice_teen_001 | 中学生 | 男 | 项目音频 |
| voice_male_001 | 成人 | 男 | 项目音频 |
| voice_female_001 | 成人 | 女 | 项目音频 |

### 公开数据集音频（待下载）
推荐从以下公开数据集获取更多音频：

| 数据集 | 许可 | 特点 |
|--------|------|------|
| Common Voice | CC0 | 众包数据，含多年龄段 |
| AIShell | 学术免费 | 中文语音数据集 |
| THCHS-30 | Apache-2.0 | 清华大学中文语音 |

### 目标数量
- 儿童声音：目标10-20个（当前1个）
- 中学生声音：目标10-20个（当前1个）
- 成人声音：目标5-10个（当前2个）
