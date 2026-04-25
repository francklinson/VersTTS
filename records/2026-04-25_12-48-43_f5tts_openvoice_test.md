# F5-TTS / OpenVoice 接口测试记录

## 日期
2026-04-25

## 测试目标
验证 F5-TTS 和 OpenVoice 在无参考音频情况下是否能正常生成语音。

## 解决方案

### F5-TTS
- **默认参考音频（中文）**: `F5-TTS/src/f5_tts/infer/examples/basic/basic_ref_zh.wav`
- **默认参考音频（英文）**: `F5-TTS/src/f5_tts/infer/examples/basic/basic_ref_en.wav`
- **默认参考文本（中文）**: "在一无所知中，梦里的一天结束了，一个新的轮回便会开始。"
- **默认参考文本（英文）**: "Some call me nature, others call me mother nature."

### OpenVoice
- OpenVoice 本身支持无参考音频模式，使用内置的基础音色
- 可选使用 `OpenVoice/resources/example_reference.mp3` 作为默认参考

## 测试结果

### F5-TTS 测试
```bash
curl -X POST "http://localhost:8000/tts/f5tts" \
  -F "text=你好，这是F5-TTS的测试。"
```

响应：
```json
{
  "success": true,
  "message": "合成成功",
  "audio_url": "/audio/tts_20260425_124755_501323.wav",
  "sample_rate": 24000
}
```

生成的音频文件：
- 文件：`tts_20260425_124755_501323.wav`
- 采样率：24000Hz

### OpenVoice 测试
```bash
curl -X POST "http://localhost:8000/tts/openvoice" \
  -F "text=你好，这是OpenVoice的测试。" \
  -F "language=zh"
```

响应：
```json
{
  "success": true,
  "message": "合成成功",
  "audio_url": "/audio/openvoice_20260425_124809_724415.wav",
  "sample_rate": 22050
}
```

生成的音频文件：
- 文件：`openvoice_20260425_124809_724415.wav`
- 大小：138K
- 采样率：22050Hz

## 所有模型测试状态

| 模型 | 状态 | 无需参考音频 | 备注 |
|------|------|-------------|------|
| **ChatTTS** | ✅ 可用 | ✅ 支持 | 内置随机说话人 |
| **CosyVoice** | ✅ 可用 | ✅ 支持 | 预训练音色 |
| **F5-TTS** | ✅ 可用 | ✅ 支持 | 使用默认参考音频 |
| **Qwen3-TTS** | ✅ 可用 | ✅ 支持 | 使用默认参考音频 |
| **OpenVoice** | ✅ 可用 | ✅ 支持 | 基础音色 + 可选参考音频 |

## 总结

✅ **所有5个TTS模型API均已验证可用**

- 无需参考音频即可直接生成语音
- 支持中文语音合成
- 支持上传参考音频进行声音克隆
