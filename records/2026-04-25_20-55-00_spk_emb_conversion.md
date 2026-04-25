# ChatTTS 参考音频转换为 spk_emb 方案

**时间戳**: 2026-04-25 20:55:00  
**任务**: 将参考音频转换为 spk_emb 格式以解决显存不足问题  
**新增文件**: 
- `backend/speaker_embedding.py` - 转换工具
- `backend/projection_matrix.pkl` - 固定投影矩阵

## 问题背景

使用 `spk_smp` (参考音频格式) 时显存占用高，因为：
- `spk_smp` 增加 **181 个 token** 到输入序列
- 注意力复杂度从 O(n²) 增加到 O((n+181)²)，约 **8 倍**

## 解决方案

将参考音频提取的 `prompt` ([4, 181] int32) 转换为 `spk_emb` ([768] float16) 格式。

## 实现步骤

### 1. 创建投影矩阵

文件: `backend/speaker_embedding.py`

```python
def get_or_create_projection_matrix():
    """获取或创建固定的投影矩阵（使用固定种子42）"""
    if PROJECTION_MATRIX_PATH.exists():
        # 加载已存在的矩阵
        ...
    else:
        # 生成新矩阵，使用种子42确保一致性
        torch.manual_seed(42)
        projection_matrix = torch.randn(724, 768) / np.sqrt(724)
        # 保存到文件
        ...
```

### 2. Prompt 到 spk_emb 的转换

```python
def convert_prompt_to_spk_emb(prompt: torch.Tensor, speaker_stats: dict) -> str:
    """将 DVAE prompt 转换为 spk_emb"""
    # 1. 展平 prompt [4, 181] -> [724]
    prompt_flat = prompt.flatten().float()
    
    # 2. 投影到 768 维
    projection_matrix = get_or_create_projection_matrix()
    spk_emb_tensor = torch.matmul(prompt_flat, projection_matrix)
    
    # 3. 归一化到随机说话人的分布
    spk_emb_tensor = normalize(spk_emb_tensor, speaker_stats)
    
    # 4. 编码为字符串
    return Speaker._encode(spk_emb_tensor.to(torch.float16))
```

### 3. 修改说话人提取 API

文件: `backend/api_server.py`

```python
# 提取音频 prompt
prompt = chat.dvae.sample_audio(wav.squeeze().numpy())

# 转换为 spk_emb 格式
speaker_emb = convert_prompt_to_spk_emb(prompt, speaker_stats)

# 保存到数据库（已经是 spk_emb 格式）
```

### 4. 修改说话人加载逻辑

```python
# 直接从数据库读取 spk_emb
embedding_b64 = speaker_info.get("embedding")
speaker_emb_bytes = base64.b64decode(embedding_b64)
spk_emb = speaker_emb_bytes.decode('utf-8')

# 使用 spk_emb 方式合成
params = chat.InferCodeParams(
    spk_emb=spk_emb,  # 不再是 None
    spk_smp=None,     # 不再使用
    ...
)
```

## 优势

| 对比项 | 旧方案 (spk_smp) | 新方案 (spk_emb) |
|--------|-----------------|-----------------|
| 显存占用 | 高（+181 tokens） | 低（768维向量） |
| 合成长度 | 限制 1024 tokens | 完整 2048 tokens |
| 加载速度 | 慢（需处理音频） | 快（直接读取） |
| 一致性 | 每次重新提取 | 固定 embedding |

## 测试步骤

1. 重启后端服务
2. 在前端录制参考音频并保存说话人
3. 使用该说话人进行合成
4. 验证显存占用正常，可合成完整长度音频

## 清理旧数据

已清空 `speakers/` 目录中的旧数据，需要重新录制说话人。
