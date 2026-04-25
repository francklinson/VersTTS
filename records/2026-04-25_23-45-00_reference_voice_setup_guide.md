# 内置参考人声准备指南

**时间**: 2026-04-25 23:45:00  
**任务**: 创建内置参考人声的准备文档和目录结构  
**状态**: ✅ 已完成

---

## 1. 概述

内置参考人声是预存在服务器上的参考音频文件，供用户在进行声音克隆时直接选择使用，无需手动上传音频。

---

## 2. 目录结构

已创建的目录结构：

```
reference_audio/                    # 参考人声根目录
├── metadata.json                   # 元数据配置文件（当前为空）
├── metadata.example.json           # 元数据示例文件（含8个示例）
├── README.md                       # 详细准备指南
├── adults/                         # 成年人声音分类
├── teenagers/                      # 青少年声音分类
└── children/                       # 儿童声音分类
```

---

## 3. 音频文件要求

### 格式要求
| 参数 | 要求 |
|------|------|
| **格式** | WAV 或 MP3 |
| **采样率** | 22050Hz 或 44100Hz |
| **声道** | 单声道（Mono） |
| **位深度** | 16bit |

### 内容要求
| 参数 | 要求 |
|------|------|
| **时长** | 5-15秒（推荐8-10秒） |
| **清晰度** | 无噪音、无混响、无背景音 |
| **语速** | 正常语速 |
| **音量** | 适中 |

---

## 4. 准备步骤

### 步骤1：录制音频
- 选择安静环境
- 使用质量较好的麦克风
- 距离麦克风15-20厘米
- 录制5-15秒的清晰音频

### 步骤2：音频处理（可选）
```bash
# 降噪处理
ffmpeg -i input.wav -af "highpass=f=200,lowpass=f=4000" output_clean.wav

# 标准化音量
ffmpeg -i input.wav -af "loudnorm=I=-16:TP=-1.5:LRA=11" output_normalized.wav
```

### 步骤3：放置音频文件
将处理好的音频文件放入对应分类目录：
- `reference_audio/adults/` - 成年人声
- `reference_audio/teenagers/` - 青少年声
- `reference_audio/children/` - 儿童声

### 步骤4：编辑 metadata.json
参考 `metadata.example.json` 格式，为每个音频文件添加元数据：

```json
{
  "id": "adult_male_01",
  "name": "成年男声-沉稳",
  "filename": "male_01.wav",
  "category": "adults",
  "gender": "male",
  "duration": 8.5,
  "text": "欢迎使用语音合成系统...",
  "description": "沉稳的中年男声",
  "language": "zh",
  "compatible_models": {
    "qwen3tts": true,
    "cosyvoice": true,
    "f5tts": true,
    "openvoice": true,
    "gptsovits": true
  }
}
```

### 步骤5：重启服务
修改完成后，重启后端服务使配置生效。

---

## 5. 验证

访问 API 验证配置：
```bash
curl http://localhost:8000/reference_voices
```

应返回参考人声列表。

---

## 6. 建议配置

### 推荐内置人声组合

| 分类 | 数量 | 建议类型 |
|------|------|----------|
| 成年人 | 4个 | 男声沉稳、男声阳光、女声温柔、女声清脆 |
| 青少年 | 2个 | 少年声、少女声 |
| 儿童 | 2个 | 男童声、女童声 |

**总计**: 8个内置人声，覆盖主要使用场景。

---

## 7. 注意事项

1. **版权问题**: 确保音频有合法授权
2. **隐私保护**: 不要使用真实个人信息
3. **质量控制**: 定期检查和更新低质量音频
4. **备份**: 定期备份 reference_audio 目录

---

## 8. 相关文件

- 📄 `reference_audio/README.md` - 详细准备指南
- 📄 `reference_audio/metadata.json` - 当前元数据（空）
- 📄 `reference_audio/metadata.example.json` - 示例元数据（8个示例）
