# CosyVoice Instruct 文本格式修复

**操作时间**: 2026-05-06 21:25:00

**问题描述**: 指令文本出现在了生成的音频中

**问题原因**: 
- `inference_instruct2` 方法的指令文本需要使用 `<|endofprompt|>` 标记来分隔指令和实际内容
- 错误的格式: `f"You are a helpful assistant.<|endofprompt|>{instruct_text}"`
- 这导致 `{instruct_text}` 出现在 `<|endofprompt|>` 之后，被模型当作要合成的内容

**正确格式**:
```python
f"You are a helpful assistant.{instruct_text}<|endofprompt|>"
```

`<|endofprompt|>` 必须在指令文本的**末尾**，用于告诉模型前面的内容是指令，不要读出来。

---

## 修改详情

### 修改文件
- **文件路径**: `backend/routers/tts/cosyvoice.py`

### 核心修改

**修改前**:
```python
formatted_instruct = f"You are a helpful assistant.<|endofprompt|>{instruct_text}"
```

**修改后**:
```python
# 注意: <|endofprompt|> 必须在指令文本末尾，用于分隔指令和实际内容
formatted_instruct = f"You are a helpful assistant.{instruct_text}<|endofprompt|>"
system_logger.info(f"Instruct模式 - 原文: {text[:50]}... 指令: {formatted_instruct}")
```

### 格式说明

| 格式 | 结果 |
|------|------|
| `...<|endofprompt|>用四川话说` | ❌ 指令被读出来 |
| `...用四川话说<|endofprompt|>` | ✅ 指令作为控制信号 |

`<|endofprompt|>` 是特殊分隔标记，告诉模型前面的内容只是指令，不要朗读。

---

**操作状态**: 已完成
