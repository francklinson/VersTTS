# Qwen3-TTS 模型回退机制修复记录

**时间**: 2026-04-25 23:10:00  
**问题**: 当 VoiceDesign/CustomVoice 模型不存在时，虽然回退到 Base 模型，但调用 generate_voice_design/generate_custom_voice 方法时抛出 ValueError  
**状态**: ✅ 已修复

---

## 1. 问题描述

### 错误日志
```
ValueError: model with 
tokenizer_type: qwen3_tts_tokenizer_12hz
tts_model_size: 0b6
tts_model_type: base
does not support generate_voice_design, Please check Model Card or Readme for more details.
```

### 原因分析
1. 用户选择 `voice_design` 或 `custom_voice` 模式
2. 系统尝试加载对应的 VoiceDesign/CustomVoice 模型
3. 模型文件不存在，回退到加载 Base 模型
4. 代码使用 `hasattr()` 检查模型是否有对应方法
5. Base 模型类确实有这个属性，所以检查通过
6. 但实际调用方法时，Base 模型内部抛出 ValueError

### 问题代码
```python
# 原有代码逻辑
if hasattr(tts, 'generate_voice_design'):  # Base 模型也有这个方法，返回 True
    wavs, sr = tts.generate_voice_design(...)  # 但调用时抛出 ValueError
```

---

## 2. 修复方案

修改 `backend/api_server.py` 中的错误处理逻辑，使用 try-except 捕获 ValueError：

```python
# 修复后的代码
voice_design_success = False
try:
    if hasattr(tts, 'generate_voice_design'):
        wavs, sr = tts.generate_voice_design(
            text=text,
            language="Auto",
            instruct=voice_design_prompt
        )
        wav = wavs[0] if isinstance(wavs, list) else wavs
        voice_design_success = True
        logger.info("使用 VoiceDesign 模型生成成功")
except ValueError as e:
    if "does not support generate_voice_design" in str(e):
        logger.warning(f"VoiceDesign 模型不支持该方法: {e}")
    else:
        raise

if not voice_design_success:
    # 回退到 Base 模型的 voice_clone
    logger.warning(f"当前模型不支持 generate_voice_design，回退到 Base 模型使用默认音色")
    tts_base = get_qwen3tts_model(model_size, "Base")
    ...
```

---

## 3. 修改的文件

### `backend/api_server.py`

- **Custom Voice 模式处理** (行 1412-1442)
  - 添加 try-except 捕获 ValueError
  - 使用 `custom_voice_success` 标志位跟踪执行状态
  - 失败时回退到 Base 模型的 voice_clone

- **Voice Design 模式处理** (行 1444-1475)
  - 添加 try-except 捕获 ValueError
  - 使用 `voice_design_success` 标志位跟踪执行状态
  - 失败时回退到 Base 模型的 voice_clone

---

## 4. 修复后的行为

### 场景 1: 模型文件存在
1. 加载指定的 CustomVoice/VoiceDesign 模型
2. 调用对应方法生成语音
3. 返回生成的音频

### 场景 2: 模型文件不存在
1. 尝试加载指定的模型 → 失败
2. 加载 Base 模型作为回退
3. 尝试调用对应方法 → 抛出 ValueError
4. 捕获异常，回退到 voice_clone 方法
5. 使用默认参考音频生成语音
6. 返回生成的音频（使用默认音色）

---

## 5. 用户提示

虽然修复后系统可以正常工作，但建议用户：
1. 下载所需的 CustomVoice 和 VoiceDesign 模型以获得最佳体验
2. 模型文件应放置在 `algorithms/Qwen3-TTS/models/Qwen/` 目录下
3. 需要的模型：
   - `Qwen3-TTS-12Hz-1.7B-CustomVoice` 或 `Qwen3-TTS-12Hz-0.6B-CustomVoice`
   - `Qwen3-TTS-12Hz-1.7B-VoiceDesign`

---

## 6. 后续优化建议

1. **前端提示**: 当回退发生时，前端可以显示提示信息，告知用户正在使用默认音色
2. **模型状态检测**: 启动时检测模型文件是否存在，提供模型下载指引
3. **模型管理界面**: 添加模型下载/管理功能，方便用户获取所需模型
