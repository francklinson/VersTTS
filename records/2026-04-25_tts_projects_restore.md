# VersTTS 项目工作记录

> 创建时间: 2026-04-25
> 项目路径: `/home/zhouchenghao/PycharmProjects/VersTTS`

---

## 1. 工作概述

本次工作主要针对 GPT-SoVITS 项目进行还原操作，将之前修改的代码恢复到原始状态，同时保留已下载的预训练模型。

### 工作内容

| 项目 | 操作 | 状态 |
|------|------|------|
| GPT-SoVITS | 代码还原 | ✅ 完成 |
| GPT-SoVITS | 模型保留 | ✅ 完成 |

---

## 2. 环境配置

### 2.1 虚拟环境

- 虚拟环境路径: `/home/zhouchenghao/PycharmProjects/VersTTS/.venv`
- Python 版本: 3.12

### 2.2 依赖状态

- 所有依赖已保持不变
- 预训练模型已保留

---

## 3. GPT-SoVITS 还原操作

### 3.1 还原的文件

| 文件 | 还原操作 | 状态 |
|------|----------|------|
| `GPT_SoVITS/AR/models/t2s_model.py` | 还原到原始状态 | ✅ 完成 |
| `GPT_SoVITS/inference_webui.py` | 还原到原始状态 | ✅ 完成 |
| `GPT_SoVITS/module/core_vq.py` | 还原到原始状态 | ✅ 完成 |
| `GPT_SoVITS/text/chinese2.py` | 还原到原始状态 | ✅ 完成 |

### 3.2 保留的文件

**预训练模型文件**:

| 文件 | 路径 | 大小 | 状态 |
|------|------|------|------|
| `s1v3.ckpt` | `GPT-SoVITS/s1v3.ckpt` | 148MB | ✅ 保留 |
| `s2Gv3.pth` | `GPT-SoVITS/GPT_SoVITS/text/XXXXRT/GPT-SoVITS-Pretrained/pretrained_models/s2Gv3.pth` | 733MB | ✅ 保留 |
| `chinese-hubert-base` | `GPT-SoVITS/GPT_SoVITS/text/XXXXRT/GPT-SoVITS-Pretrained/pretrained_models/chinese-hubert-base/` | - | ✅ 保留 |
| `chinese-roberta-wwm-ext-large` | `GPT-SoVITS/GPT_SoVITS/text/XXXXRT/GPT-SoVITS-Pretrained/pretrained_models/chinese-roberta-wwm-ext-large/` | - | ✅ 保留 |
| `bigvgan_generator.pt` | `GPT-SoVITS/GPT_SoVITS/text/XXXXRT/GPT-SoVITS-Pretrained/pretrained_models/models--nvidia--bigvgan_v2_24khz_100band_256x/bigvgan_generator.pt` | - | ✅ 保留 |
| `G2PWModel` | `GPT-SoVITS/GPT_SoVITS/G2PWModel/` | - | ✅ 保留 |

### 3.3 还原命令

使用 git 命令将修改的文件还原到原始状态:

```bash
cd /home/zhouchenghao/PycharmProjects/VersTTS/GPT-SoVITS
git checkout HEAD -- GPT_SoVITS/AR/models/t2s_model.py GPT_SoVITS/inference_webui.py GPT_SoVITS/module/core_vq.py GPT_SoVITS/text/chinese2.py
```

---

## 4. 项目状态

### 4.1 GPT-SoVITS 状态

| 项目 | 状态 | 说明 |
|------|------|------|
| GPT-SoVITS | ✅ 原始状态 | 代码已还原，模型已保留 |
| Qwen3-TTS | ✅ 可用 | 无变更 |
| CosyVoice | ✅ 可用 | 无变更 |
| ChatTTS | ✅ 可用 | 无变更 |
| F5-TTS | ✅ 可用 | 无变更 |
| OpenVoice | ✅ 可用 | 无变更 |

### 4.2 项目目录结构

**GPT-SoVITS 主要目录结构**:

```
GPT-SoVITS/
├── GPT_SoVITS/  # 核心代码（已还原）
├── GPT_SoVITS/text/XXXXRT/  # 预训练模型（已保留）
├── GPT_SoVITS/G2PWModel/  # 中文文本处理模型（已保留）
├── s1v3.ckpt  # GPT 模型（已保留）
├── webui.py  # Web UI 入口
└── inference_webui.py  # 推理代码（已还原）
```

---

## 5. 后续建议

### 5.1 运行 GPT-SoVITS

要运行还原后的 GPT-SoVITS，可以使用以下命令:

```bash
cd /home/zhouchenghao/PycharmProjects/VersTTS/GPT-SoVITS
source ../.venv/bin/activate
python webui.py
```

### 5.2 注意事项

- 所有预训练模型已保留，无需重新下载
- 代码已还原到原始状态，与 GitHub 仓库版本一致
- 若需要 API 接口功能，建议使用项目自带的 `api.py` 或 `api_v2.py`

---

## 6. 总结

本次操作成功完成了以下任务:

1. **代码还原**: 将 GPT-SoVITS 的修改文件还原到原始状态
2. **模型保留**: 保留了所有已下载的预训练模型文件
3. **状态确认**: 验证了所有 TTS 项目的可用状态

项目现在已恢复到原始的 GitHub 仓库状态，同时保留了所有必要的预训练模型，确保可以正常运行。

---

*本记录由 Trae AI Assistant 自动生成*