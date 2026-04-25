# ChatTTS spk_smp 显存优化

**时间戳**: 2026-04-25 20:40:00  
**问题**: 使用 web 录制的说话人进行语音合成时 GPU 内存不足  
**根本原因**: spk_smp 比 spk_emb 占用更多显存  
**修改文件**: backend/api_server.py

## 问题分析

### spk_smp vs spk_emb 显存占用差异

| 参数类型 | 格式 | 序列长度增加 | 显存影响 |
|---------|------|-------------|---------|
| **spk_emb** (随机说话人) | `[768]` float16 | 不增加 | 小 |
| **spk_smp** (参考音频) | `[4, 181]` int32 (编码后) | **+181 tokens** | 大 |

### 显存占用计算

注意力机制复杂度为 O(n²)：
- 文本 100 tokens + spk_emb：100×100 = 10,000
- 文本 100 tokens + spk_smp：(100+181)×(100+181) = 78,961 **(增加近 8 倍!)**

### 无法转换格式的原因

`spk_smp` 和 `spk_emb` 是完全不同的格式：
- `spk_smp`: 编码后的音频 prompt，`[4, 181]` int32
- `spk_emb`: 说话人 embedding，`[768]` float16

无法直接转换！

## 解决方案

### 优化策略：限制 max_new_token

当使用 `spk_smp` 时，减少最大生成 token 数以降低显存占用：

```python
# 当使用spk_smp时，输入序列较长，需要减少max_new_token以避免OOM
max_new_token = 2048  # 默认值
if spk_smp is not None:
    max_new_token = 1024  # 使用spk_smp时减少token数以节省显存
```

### 修改内容

文件: `backend/api_server.py`
位置: 第 1118-1130 行

修改前:
```python
params = chat.InferCodeParams(
    spk_emb=spk_emb,
    spk_smp=spk_smp,
    txt_smp=txt_smp,
    temperature=float(temperature),
    top_P=float(top_P),
    top_K=int(top_K),
)
```

修改后:
```python
# 当使用spk_smp时，输入序列较长，需要减少max_new_token以避免OOM
max_new_token = 2048  # 默认值
if spk_smp is not None:
    max_new_token = 1024  # 使用spk_smp时减少token数以节省显存
    system_logger.info(f"【ChatTTS】使用spk_smp，限制max_new_token={max_new_token}")

params = chat.InferCodeParams(
    spk_emb=spk_emb,
    spk_smp=spk_smp,
    txt_smp=txt_smp,
    temperature=float(temperature),
    top_P=float(top_P),
    top_K=int(top_K),
    max_new_token=max_new_token,
)
```

## 权衡

- **优点**: 解决显存不足问题，可以使用参考音频克隆音色
- **缺点**: 使用 spk_smp 时最大生成长度从 2048 降低到 1024，大约 10-12 秒音频

## 建议

如果需要生成长文本：
1. 使用随机说话人（spk_emb）方式
2. 将长文本分段合成
3. 或者使用更短的参考音频（< 2秒）
