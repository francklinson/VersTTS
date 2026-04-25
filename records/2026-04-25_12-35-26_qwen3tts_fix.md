# Qwen3-TTS API 修复记录

## 日期
2026-04-25

## 问题描述
Qwen3-TTS API 调用时出现错误：
1. 模型路径格式错误：`Repo id must be in the form 'repo_name' or 'namespace/repo_name'`
2. 方法不存在：`'Qwen3TTSModel' object has no attribute 'generate'`

## 修复内容

### 1. 模型路径修复
**问题**: 路径中的版本号格式错误
**修复**: 使用正确的路径映射
```python
size_map = {
    "0.6B": "0___6B",
    "1.7B": "1___7B"
}
size_str = size_map.get(model_size, model_size.replace('.', '___'))
model_path = os.path.join(PROJECT_ROOT, "Qwen3-TTS", "models", "Qwen", f"Qwen3-TTS-12Hz-{size_str}-Base")
```

### 2. 生成方法修复
**问题**: Qwen3TTSModel 没有 `generate` 方法
**分析**: Qwen3TTSModel 只支持以下方法：
- `generate_voice_clone` - 语音克隆（需要参考音频）
- `generate_custom_voice` - 自定义声音
- `generate_voice_design` - 声音设计

**修复**: 基础模式使用默认参考音频进行 voice_clone
```python
# 基础模式：使用默认参考音频进行voice_clone
default_ref_audio = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone_1.wav"
default_ref_text = "甚至出现交易几乎停滞的情况。"

wavs, sr = tts.generate_voice_clone(
    text=text,
    language="Auto",
    ref_audio=default_ref_audio,
    ref_text=default_ref_text,
    x_vector_only_mode=True  # 只使用音色特征
)
wav = wavs[0] if isinstance(wavs, list) else wavs
```

## 测试结果

### API调用
```bash
curl -X POST "http://localhost:8000/tts/qwen3tts" \
  -F "text=你好，这是Qwen3-TTS的修复测试。" \
  -F "model_size=0.6B" \
  -F "mode=base"
```

### 响应
```json
{
  "success": true,
  "message": "合成成功",
  "audio_url": "/audio/tts_20260425_123456_151031.wav",
  "sample_rate": 24000
}
```

### 生成的音频文件
- 文件：`tts_20260425_123456_151031.wav`
- 大小：218K
- 采样率：24000Hz

## 总结

✅ Qwen3-TTS API 修复完成
✅ 基础模式使用默认参考音频成功生成语音
✅ 声音克隆模式（上传参考音频）正常工作

## 注意事项

1. Qwen3-TTS 必须通过 voice_clone 方式使用，需要参考音频
2. 基础模式使用官方默认参考音频URL
3. 如需使用自定义声音，请上传参考音频并使用 `mode=voice_clone`
