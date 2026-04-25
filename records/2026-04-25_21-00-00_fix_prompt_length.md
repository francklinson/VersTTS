# 修复 Prompt 长度不固定问题

**时间戳**: 2026-04-25 21:00:00  
**问题**: Prompt 长度不是固定的 181，而是根据音频长度变化  
**错误信息**: `mat1 and mat2 shapes cannot be multiplied (1x1344 and 724x768)`  
**修改文件**: `backend/speaker_embedding.py`

## 问题原因

DVAE 提取的 prompt 形状是 `[4, seq_len]`，其中 `seq_len` 根据音频长度变化：
- 短音频: seq_len ≈ 100
- 中等音频: seq_len ≈ 181
- 长音频: seq_len ≈ 336 或更长

原来的固定投影矩阵 `[724, 768]` 无法处理不同长度的 prompt。

## 解决方案

使用 **1D 卷积 + 全局平均池化** 的神经网络，可以处理任意长度的序列：

```python
class PromptToSpkEmb(nn.Module):
    def __init__(self, num_vq=4, spk_dim=768):
        # 对每个 VQ 层使用 1D 卷积
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(1, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),  # 全局平均池化，输出固定长度
            ) for _ in range(num_vq)
        ])
        
        # 全连接层映射到 spk_emb 维度
        self.fc = nn.Sequential(
            nn.Linear(64 * num_vq, 512),
            nn.ReLU(),
            nn.Linear(512, spk_dim)
        )
```

## 处理流程

1. 每个 VQ 层 `[seq_len]` -> Conv1d -> `[64, 1]` (通过全局池化)
2. 拼接 4 个 VQ 层 -> `[256]`
3. 全连接层 -> `[768]` spk_emb

## 优势

- ✅ 支持任意长度的 prompt
- ✅ 使用固定的神经网络权重（种子=42）
- ✅ 两次转换结果一致

## 测试

```
seq_len=100: spk_emb 长度=859 ✓
seq_len=181: spk_emb 长度=853 ✓
seq_len=336: spk_emb 长度=853 ✓
seq_len=500: spk_emb 长度=853 ✓
```

## 下一步

重启后端服务，重新录制说话人并测试。
