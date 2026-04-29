# IndexTTS2 修复 Wav2Vec2BertModel 离线加载问题

**时间**: 2026-04-30 00:15:00  
**任务**: 修复 IndexTTS2 在离线模式下 Wav2Vec2BertModel 仍然尝试访问 HuggingFace 的问题

---

## 问题描述

在离线部署测试中，发现即使已经将 w2v-bert-2.0 模型复制到本地，IndexTTS2 仍然报错：

```
We couldn't connect to 'https://huggingface.co' to load the files, 
and couldn't find them in the cached files.
```

日志显示：
```
>> Loading w2v-bert-2.0 from local path: /home/.../algorithms/IndexTTS/checkpoints/w2v-bert-2.0
【模型加载】IndexTTS2 | 状态: 失败 | We couldn't connect to 'https://huggingface.co'...
```

这表明虽然 SeamlessM4TFeatureExtractor 成功从本地加载，但还有其他组件在尝试访问 HuggingFace。

---

## 问题定位

经过代码审查，发现 `indextts/utils/maskgct_utils.py` 中的 `build_semantic_model` 函数直接硬编码了 HuggingFace 模型名称：

```python
def build_semantic_model(path_='./models/tts/maskgct/ckpt/wav2vec2bert_stats.pt'):
    semantic_model = Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0")
    ...
```

而 `infer_v2.py` 调用该函数时：
```python
self.semantic_model, self.semantic_mean, self.semantic_std = build_semantic_model(
    os.path.join(self.model_dir, self.cfg.w2v_stat))
```

没有传入本地模型路径，导致 `Wav2Vec2BertModel` 仍然尝试从 HuggingFace 下载。

---

## 修复内容

### 1. 修改 `algorithms/IndexTTS/indextts/utils/maskgct_utils.py`

**修改前：**
```python
def build_semantic_model(path_='./models/tts/maskgct/ckpt/wav2vec2bert_stats.pt'):
    semantic_model = Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0")
    semantic_model.eval()
    stat_mean_var = torch.load(path_)
    semantic_mean = stat_mean_var["mean"]
    semantic_std = torch.sqrt(stat_mean_var["var"])
    return semantic_model, semantic_mean, semantic_std
```

**修改后：**
```python
def build_semantic_model(path_='./models/tts/maskgct/ckpt/wav2vec2bert_stats.pt', w2v_bert_local_path=None):
    """
    构建语义模型
    Args:
        path_: w2v-bert 统计文件路径
        w2v_bert_local_path: 本地 w2v-bert-2.0 模型路径，如果提供则优先使用
    """
    import os
    
    if w2v_bert_local_path and os.path.exists(w2v_bert_local_path):
        print(f">> Loading Wav2Vec2BertModel from local path: {w2v_bert_local_path}")
        semantic_model = Wav2Vec2BertModel.from_pretrained(w2v_bert_local_path, local_files_only=True)
    else:
        # 检查是否离线模式
        is_offline = os.environ.get('TRANSFORMERS_OFFLINE') == '1' or os.environ.get('HF_HUB_OFFLINE') == '1'
        if is_offline:
            raise FileNotFoundError(f"Offline mode: w2v-bert-2.0 model not found at {w2v_bert_local_path}")
        print(">> Loading Wav2Vec2BertModel from HuggingFace: facebook/w2v-bert-2.0")
        semantic_model = Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0")
    
    semantic_model.eval()
    stat_mean_var = torch.load(path_)
    semantic_mean = stat_mean_var["mean"]
    semantic_std = torch.sqrt(stat_mean_var["var"])
    return semantic_model, semantic_mean, semantic_std
```

### 2. 修改 `algorithms/IndexTTS/indextts/infer_v2.py`

**修改前：**
```python
self.semantic_model, self.semantic_mean, self.semantic_std = build_semantic_model(
    os.path.join(self.model_dir, self.cfg.w2v_stat))
```

**修改后：**
```python
# 加载语义模型，传入本地 w2v-bert-2.0 路径
w2v_bert_local = os.path.join(self.model_dir, "w2v-bert-2.0")
self.semantic_model, self.semantic_mean, self.semantic_std = build_semantic_model(
    os.path.join(self.model_dir, self.cfg.w2v_stat),
    w2v_bert_local_path=w2v_bert_local if os.path.exists(w2v_bert_local) else None
)
```

---

## 修复后 IndexTTS2 离线模式支持的模型

| 模型 | 类型 | 本地路径 | 状态 |
|------|------|---------|------|
| w2v-bert-2.0 | SeamlessM4TFeatureExtractor | `checkpoints/w2v-bert-2.0/` | ✅ 已支持 |
| w2v-bert-2.0 | Wav2Vec2BertModel | `checkpoints/w2v-bert-2.0/` | ✅ 已修复 |
| semantic_codec | 权重文件 | `checkpoints/semantic_codec/model.safetensors` | ✅ 已支持 |
| campplus | 权重文件 | `checkpoints/campplus_cn_common.bin` | ✅ 已支持 |
| bigvgan | 声码器 | `checkpoints/bigvgan/` | ✅ 已支持 |

---

## 离线部署验证步骤

1. **确认模型文件已复制**
   ```bash
   python check_models.py
   ```
   确保 IndexTTS 的 HuggingFace 模型检查显示全部通过。

2. **使用离线模式启动服务**
   ```bash
   ./start_server.sh start --offline
   ```

3. **测试 IndexTTS 合成功能**
   - 访问前端页面
   - 选择 IndexTTS 算法
   - 输入文本并生成语音
   - 验证是否成功生成音频文件

---

## 文件变更清单

| 文件路径 | 变更类型 | 说明 |
|---------|---------|------|
| `algorithms/IndexTTS/indextts/utils/maskgct_utils.py` | 修改 | 添加 `w2v_bert_local_path` 参数支持本地模型加载 |
| `algorithms/IndexTTS/indextts/infer_v2.py` | 修改 | 调用 `build_semantic_model` 时传入本地模型路径 |
| `records/2026-04-30_00-15-00_IndexTTS2_修复Wav2Vec2BertModel离线加载.md` | 新增 | 本工作记录 |

---

## 注意事项

1. **模型路径一致性**：确保 `checkpoints/w2v-bert-2.0/` 目录包含完整的模型文件（config.json、pytorch_model.bin 等）。

2. **离线模式检测**：代码通过检查 `TRANSFORMERS_OFFLINE` 或 `HF_HUB_OFFLINE` 环境变量来判断是否处于离线模式。

3. **错误提示**：如果离线模式下找不到本地模型，会抛出 `FileNotFoundError` 并提供清晰的错误信息。

---

## 相关记录

- [2026-04-30_00-00-00_IndexTTS2_离线模式修复.md](./2026-04-30_00-00-00_IndexTTS2_离线模式修复.md) - IndexTTS2 初始离线模式修复
- [2026-04-29_23-45-00_离线部署配置完成.md](./2026-04-29_23-45-00_离线部署配置完成.md) - 项目整体离线部署配置

---

**工作完成时间**: 2026-04-30 00:15:00  
**状态**: ✅ 已完成
