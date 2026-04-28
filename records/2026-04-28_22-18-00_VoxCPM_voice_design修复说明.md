# VoxCPM voice_design模式修复说明

**时间**: 2026-04-28 22:18:00  
**问题**: VoxCPM voice_design为什么回退到base  
**状态**: ✅ 已修复

---

## 问题原因

### 原来错误的理解
之前以为VoxCPM有一个独立的`voice_design`方法，但实际上VoxCPM只提供了`generate()`方法。

### 正确的实现方式
根据VoxCPM的GitHub README，voice_design的用法是：

```python
wav = model.generate(
    text="(A young woman, gentle and sweet voice)Hello, welcome to VoxCPM2!",
    cfg_value=2.0,
    inference_timesteps=10,
)
```

**关键**: 在text前面添加`(voice description)`来实现音色设计！

---

## 修复内容

### 修改文件: `backend/api_server.py`

**修复前**:
```python
# voice_design模式暂不支持，记录警告
logger.warning("voice_design模式暂不支持，使用base模式")
```

**修复后**:
```python
if mode == "voice_design":
    # voice_design模式: 在text前添加(voice description)
    # 参考GitHub: text="(A young woman, gentle voice)Hello, welcome!"
    if voice_design_prompt:
        generate_kwargs["text"] = f"({voice_design_prompt}){text}"
        logger.info(f"音色设计模式 | 描述: {voice_design_prompt}")
    else:
        # 如果没有提供voice_design_prompt，使用默认描述
        generate_kwargs["text"] = f"(A natural speaking voice){text}"
        logger.info("音色设计模式 | 使用默认描述")
```

---

## 验证测试

### 请求示例
```bash
curl -s -X POST http://localhost:8000/tts/voxcpm \
  -F "text=你好，这是音色设计模式测试。" \
  -F "mode=voice_design" \
  -F "voice_design_prompt=A young woman with gentle and sweet voice" \
  -F "cfg_value=2.0" \
  -F "inference_timesteps=10" \
  -F "output_format=url"
```

### 响应结果
```json
{
  "success": true,
  "message": "合成成功",
  "audio_url": "/audio/voxcpm_20260428_221723.wav",
  "sample_rate": 48000
}
```

### 日志记录
```
2026-04-28 22:17:10 | INFO | 音色设计模式 | 描述: A young woman with gentle and sweet voice
```

---

## 前端调用说明

### API参数
| 参数 | 类型 | 说明 |
|------|------|------|
| `text` | string | 要合成的文本 |
| `mode` | string | 固定为 `voice_design` |
| `voice_design_prompt` | string | 音色描述，如 "A young woman with gentle voice" |
| `cfg_value` | float | 引导系数，默认2.0 |
| `inference_timesteps` | int | 推理步数，默认10 |

### 前端界面建议
- 当选择`voice_design`模式时，显示音色描述输入框
- 可以提供预设的音色描述模板供用户选择
- 示例描述:
  - "A young woman with gentle and sweet voice"
  - "A mature man with deep voice"
  - "A child with lively voice"

---

## 总结

**修复后**: voice_design模式现在可以正常使用，不会再回退到base模式。

**实现原理**: 在文本前添加`(voice description)`前缀，如：
```
(A young woman with gentle and sweet voice)你好，这是音色设计模式测试。
```
