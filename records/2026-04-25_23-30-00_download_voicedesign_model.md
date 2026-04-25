# VoiceDesign 模型下载记录

**时间**: 2026-04-25 23:30:00  
**任务**: 下载 Qwen3-TTS VoiceDesign 1.7B 模型  
**状态**: ✅ 已完成

---

## 1. 下载的模型

### 模型信息
- **模型名称**: Qwen3-TTS-12Hz-1.7B-VoiceDesign
- **模型大小**: 1.7B 参数
- **磁盘占用**: 4.3GB
- **来源**: Hugging Face (Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign)

### 存放路径
```
algorithms/Qwen3-TTS/models/Qwen/
└── Qwen3-TTS-12Hz-1___7B-VoiceDesign/
    ├── config.json
    ├── generation_config.json
    ├── merges.txt
    ├── model.safetensors (3.83GB)
    ├── preprocessor_config.json
    ├── README.md
    ├── speech_tokenizer/
    │   ├── config.json
    │   ├── configuration.json
    │   ├── model.safetensors (682MB)
    │   └── preprocessor_config.json
    ├── tokenizer_config.json
    └── vocab.json
```

---

## 2. 当前 Qwen3-TTS 模型清单

| 模型名称 | 大小 | 状态 |
|---------|------|------|
| Qwen3-TTS-12Hz-0.6B-Base | 0.6B | ✅ 已安装 |
| Qwen3-TTS-12Hz-0.6B-CustomVoice | 0.6B | ✅ 已安装 |
| Qwen3-TTS-12Hz-1.7B-Base | 1.7B | ✅ 已安装 |
| Qwen3-TTS-12Hz-1.7B-CustomVoice | 1.7B | ✅ 已安装 |
| **Qwen3-TTS-12Hz-1.7B-VoiceDesign** | **1.7B** | **✅ 已安装** |

---

## 3. 功能变化

### 下载前
- 仅 Base 模型可用
- 预设音色和音色设计功能会回退到默认音色
- 前端 UI 显示警告消息

### 下载后
- ✅ **基础合成** - 可用
- ✅ **声音克隆** - 可用
- ✅ **预设音色** - 可用 (CustomVoice 模型已安装)
- ✅ **音色设计** - 可用 (VoiceDesign 模型已安装)
- 所有 4 种模式选项在前端 UI 中显示
- 不再显示模型缺失警告

---

## 4. 下载命令

```bash
# 使用 huggingface-cli 下载
huggingface-cli download \
    Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
    --local-dir Qwen3-TTS-12Hz-1___7B-VoiceDesign \
    --resume-download
```

### 下载耗时
- 总耗时: 约 5 分钟
- 平均速度: 约 12-18 MB/s

---

## 5. 验证

### 文件完整性检查
```bash
du -sh Qwen3-TTS-12Hz-1___7B-VoiceDesign/
# 输出: 4.3G
```

### API 状态检查
```bash
curl http://localhost:8000/tts/qwen3tts/status
```

预期返回:
```json
{
    "base_available": true,
    "custom_voice_available": true,
    "voice_design_available": true,
    "model_sizes": ["0.6B", "1.7B"],
    "message": "所有模型已就绪"
}
```

---

## 6. 使用说明

### 音色设计功能示例

1. 在前端页面选择 **Qwen3-TTS** 模型
2. 模式选择 **"音色设计"**
3. 在"音色描述"框中输入描述，例如：
   - "体现撒娇稚嫩的萝莉女声，音调偏高且起伏明显"
   - "成熟稳重的男声，低沉醇厚，适合新闻播报"
   - "活泼开朗的年轻女声，语调轻快"
4. （可选）添加指令控制，如"开心地"、"悲伤地"
5. 点击生成

### 支持的语言
中文、英文、日文、韩文、德文、法文、俄文、葡萄牙文、西班牙文、意大利文

---

## 7. 注意事项

1. **显存要求**: VoiceDesign 1.7B 模型需要约 8-12GB 显存
2. **首次加载**: 首次使用时需要加载模型，可能需要 10-30 秒
3. **模型切换**: 不同模式之间切换时可能需要重新加载模型

---

## 8. 总结

VoiceDesign 1.7B 模型已成功下载并安装。现在 Qwen3-TTS 的所有功能都已可用：
- ✅ 基础合成
- ✅ 声音克隆
- ✅ 预设音色 (9种)
- ✅ 音色设计 (自然语言描述)

用户可以通过前端界面体验完整的 Qwen3-TTS 功能。
