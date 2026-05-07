# CosyVoice Instruct 模式修复

**操作时间**: 2026-05-06 21:15:00

**问题描述**: 调用 instruct 模式时报错 `CosyVoice错误: '中文女'`

**问题原因**: 
- `inference_instruct` 方法依赖 `spk2info.pt` 文件中的预设音色
- Fun-CosyVoice3-0.5B 模型不包含 `spk2info.pt` 文件
- 因此无法使用预设音色（如"中文女"、"中文男"）

**解决方案**: 
- 改用 `inference_instruct2` 方法
- 该方法基于参考音频进行指令控制，不需要预设音色

---

## 修改详情

### 修改文件
- **文件路径**: `backend/routers/tts/cosyvoice.py`

### 核心修改

**修改前**:
```python
# 使用 inference_instruct 方法
model_output = cosyvoice.inference_instruct(text, speaker_desc, instruct_text, stream=False)
```

**修改后**:
```python
# 格式化指令文本
formatted_instruct = f"You are a helpful assistant.<|endofprompt|>{instruct_text}"
# 使用 inference_instruct2 方法
model_output = cosyvoice.inference_instruct2(text, formatted_instruct, prompt_wav_path, stream=False)
```

### 方法区别

| 方法 | 说明 | 需要 | 适用模型 |
|------|------|------|----------|
| `inference_instruct` | 基于预设音色 | `spk2info.pt` + 音色ID | CosyVoice 1.0/2.0 |
| `inference_instruct2` | 基于参考音频 | 参考音频文件 | CosyVoice 3.0 |

### 请求参数

**Instruct 模式**:
- `mode`: "instruct"
- `text`: 要合成的文本
- `instruct_text`: 指令文本（如"用四川话说"、"用开心的语气说"）
- `clone_speaker_id`: 参考说话人ID（从说话人管理中选择）
- `prompt_wav`: （可选）直接上传参考音频

---

**操作状态**: 已完成
