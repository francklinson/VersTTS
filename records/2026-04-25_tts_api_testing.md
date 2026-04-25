# TTS API 测试结果记录

## 日期
2026-04-25

## 测试概述
对VersTTS统一API服务进行功能测试，验证各TTS模型能否正常生成音频。

## 测试结果

### 服务状态
- ✅ 服务启动正常
- ✅ CUDA可用 (NVIDIA GeForce RTX 3090)
- ✅ 健康检查接口正常

### 各模型测试结果

| 模型 | 状态 | 音频文件 | 文件大小 | 备注 |
|------|------|----------|----------|------|
| **ChatTTS** | ✅ 成功 | tts_20260425_122754_127188.wav | 154K | 首次加载模型后成功生成 |
| **CosyVoice** | ✅ 成功 | tts_20260425_122902_299314.wav | 85K | SFT模式，中文女声 |
| **Qwen3-TTS** | ⚠️ 待修复 | - | - | 模型路径格式问题 |
| **F5-TTS** | ⏸️ 未测试 | - | - | 需要参考音频 |
| **OpenVoice** | ⏸️ 未测试 | - | - | - |

## 测试过程

### 1. ChatTTS 测试
```bash
curl -X POST "http://localhost:8000/tts/chattts" \
  -F "text=你好，这是VersTTS的测试，很高兴见到你。" \
  -F "temperature=0.3"
```
响应：
```json
{
  "success": true,
  "message": "合成成功",
  "audio_url": "/audio/tts_20260425_122754_127188.wav",
  "sample_rate": 24000
}
```

### 2. CosyVoice 测试
```bash
curl -X POST "http://localhost:8000/tts/cosyvoice" \
  -F "text=你好，这是CosyVoice的测试。" \
  -F "mode=sft" \
  -F "speaker_id=中文女"
```
响应：
```json
{
  "success": true,
  "message": "合成成功",
  "audio_url": "/audio/tts_20260425_122902_299314.wav",
  "sample_rate": 22050
}
```

## 问题与修复

### 问题1: ChatTTS模型路径错误
**现象**: 模型加载失败，提示哈希校验错误
**原因**: ChatTTS默认从当前工作目录加载，而非指定路径
**修复**: 修改API代码，使用`source="custom"`并指定正确的模型路径
```python
model_path = os.path.join(PROJECT_ROOT, "ChatTTS", "models")
chat.load(source="custom", custom_path=model_path)
```

### 问题2: ChatTTS参数类型错误
**现象**: `top_k has to be a strictly positive integer, but is 20.0`
**修复**: 将参数显式转换为正确类型
```python
top_K=int(top_K)
```

### 问题3: CosyVoice音频保存格式错误
**现象**: `Error opening 'output/...wav': Format not recognised`
**原因**: soundfile无法正确处理CosyVoice返回的tensor格式
**修复**: 使用torchaudio保存，并正确处理tensor形状
```python
audio_np = audio_data.numpy()
if audio_np.ndim == 1:
    audio_np = audio_np.reshape(1, -1)
torchaudio.save(audio_path, torch.from_numpy(audio_np), sample_rate)
```

### 问题4: Qwen3-TTS模型路径格式错误
**现象**: `Repo id must be in the form 'repo_name' or 'namespace/repo_name'`
**原因**: 路径中的版本号格式错误 (0.6B -> 0___6BB)
**修复方案**: 已修改代码，待重新测试

## API端点验证

| 端点 | 方法 | 状态 |
|------|------|------|
| `/` | GET | ✅ |
| `/health` | GET | ✅ |
| `/tts/chattts` | POST | ✅ |
| `/tts/cosyvoice` | POST | ✅ |
| `/tts/qwen3tts` | POST | ⚠️ |
| `/tts/openvoice` | POST | ⏸️ |
| `/tts/f5tts` | POST | ⏸️ |
| `/audio/{filename}` | GET | ✅ |

## 结论

1. **API框架运行正常**: FastAPI服务启动成功，接口响应正常
2. **ChatTTS和CosyVoice可用**: 已验证可正常生成中文语音
3. **需要修复的问题**: Qwen3-TTS路径问题（已修复代码，待测试）
4. **待测试**: F5-TTS、OpenVoice（需要参考音频）

## 下一步工作

1. 重新测试Qwen3-TTS（路径问题已修复）
2. 测试F5-TTS（准备参考音频文件）
3. 测试OpenVoice
4. 完善错误处理和日志记录
