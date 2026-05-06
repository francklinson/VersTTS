# VersTTS 离线部署指南

本文档指导如何在**无外网环境**（无法访问 HuggingFace、ModelScope 等外部资源）的服务器上部署 VersTTS 项目。

## 目录

- [概述](#概述)
- [环境要求](#环境要求)
- [离线部署步骤](#离线部署步骤)
- [模型文件准备](#模型文件准备)
- [启动服务](#启动服务)
- [故障排查](#故障排查)

## 概述

VersTTS 项目默认依赖以下外部资源：

- **HuggingFace Hub**: 部分模型使用 `from_pretrained()` 从 HuggingFace 下载
- **ModelScope**: VoxCPM 使用 ModelScope 下载降噪模型
- **PyTorch Hub**: 可能用于下载预训练模型

离线部署通过以下方式解决依赖问题：

1. 设置环境变量禁用所有在线访问
2. 使用本地模型路径加载
3. 预先将所有模型文件下载到服务器

## 环境要求

### 硬件要求

- GPU: NVIDIA GPU (推荐 16GB+ 显存)
- 内存: 32GB+
- 磁盘: 100GB+ (模型文件约占用 60-80GB)

### 软件要求

- OS: Linux (Ubuntu 20.04+ 推荐)
- Python: 3.10+
- CUDA: 11.8+ 或 12.1+
- 虚拟环境: 项目使用 `.venv` 虚拟环境

## 离线部署步骤

### 步骤 1: 准备模型文件

在有网络的环境中，下载所有模型文件并打包：

```bash
# 1. 下载各算法模型
# ChatTTS
cd algorithms/ChatTTS
# 按项目 README 下载模型到 models/ 目录

# CosyVoice
cd algorithms/CosyVoice
# 按项目 README 下载模型到 models/ 目录

# F5-TTS
cd algorithms/F5-TTS
# 按项目 README 下载模型

# GPT-SoVITS
cd algorithms/GPT-SoVITS
python download_models.py

# OpenVoice
cd algorithms/OpenVoice
# 按项目 README 下载 checkpoints_v1 和 checkpoints_v2

# Qwen3-TTS
cd algorithms/Qwen3-TTS
# 按项目 README 下载模型到 models/ 目录

# VoxCPM
cd algorithms/VoxCPM
# 按项目 README 下载模型到 models/VoxCPM2/ 目录

# IndexTTS
cd algorithms/IndexTTS
# 按项目 README 下载模型到 checkpoints/ 目录

# FireRedTTS2
cd algorithms/FireRedTTS2
# 按项目 README 下载模型到 pretrained_models/FireRedTTS2/ 目录
```

### 步骤 2: 打包传输

将模型文件打包并传输到目标服务器：

```bash
# 打包模型文件
tar -czvf verstts_models.tar.gz \
    algorithms/ChatTTS/models \
    algorithms/CosyVoice/models \
    algorithms/F5-TTS/models \
    algorithms/GPT-SoVITS/GPT_SoVITS/pretrained_models \
    algorithms/OpenVoice/checkpoints_v1 \
    algorithms/OpenVoice/checkpoints_v2 \
    algorithms/Qwen3-TTS/models \
    algorithms/VoxCPM/models \
    algorithms/IndexTTS/checkpoints \
    algorithms/FireRedTTS2/pretrained_models

# 传输到目标服务器
scp verstts_models.tar.gz user@target-server:/path/to/VersTTS/
```

### 步骤 3: 解压模型

在目标服务器上解压模型文件：

```bash
cd /path/to/VersTTS
tar -xzvf verstts_models.tar.gz
```

### 步骤 4: 配置环境

确保 `.env.offline` 文件已存在（已包含在项目中）：

```bash
# 检查环境变量配置文件
cat .env.offline
```

主要环境变量说明：

```bash
# 禁用 HuggingFace 在线访问
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_DOWNLOADS=1

# 设置本地缓存目录
export HF_HOME="/home/zhouchenghao/PycharmProjects/VersTTS/models/hf_cache"
export TRANSFORMERS_CACHE="/home/zhouchenghao/PycharmProjects/VersTTS/models/transformers_cache"
```

### 步骤 5: 验证模型路径

运行模型路径检查脚本：

```bash
# 激活虚拟环境
source .venv/bin/activate

# 运行检查脚本
python check_models.py
```

## 模型文件准备

### 必需的模型目录结构

```
/home/zhouchenghao/PycharmProjects/VersTTS/
├── algorithms/
│   ├── ChatTTS/
│   │   └── models/              # ChatTTS 模型
│   ├── CosyVoice/
│   │   └── models/              # CosyVoice 模型
│   ├── F5-TTS/
│   │   └── models/              # F5-TTS 模型
│   ├── GPT-SoVITS/
│   │   └── GPT_SoVITS/
│   │       └── pretrained_models/   # GPT-SoVITS 预训练模型
│   ├── OpenVoice/
│   │   ├── checkpoints_v1/      # OpenVoice V1 模型
│   │   └── checkpoints_v2/      # OpenVoice V2 模型
│   ├── Qwen3-TTS/
│   │   └── models/              # Qwen3-TTS 模型
│   ├── VoxCPM/
│   │   └── models/
│   │       └── VoxCPM2/         # VoxCPM2 模型
│   ├── IndexTTS/
│   │   └── checkpoints/         # IndexTTS 模型
│   └── FireRedTTS2/
│       └── pretrained_models/
│           └── FireRedTTS2/     # FireRedTTS2 模型
└── models/                      # 可选：HuggingFace 缓存目录
    ├── hf_cache/
    ├── transformers_cache/
    └── torch_cache/
```

### 各算法模型下载指南

#### ChatTTS

```bash
cd algorithms/ChatTTS
# 参考项目 README 使用 huggingface-cli 下载
# 或使用 ModelScope 镜像
```

#### CosyVoice

```bash
cd algorithms/CosyVoice
# 参考项目 README 下载模型
# 安装 modelscope: pip install modelscope
# 运行下载脚本
```

#### GPT-SoVITS

```bash
cd algorithms/GPT-SoVITS
python download_models.py
```

#### OpenVoice

```bash
# V1 模型
cd algorithms/OpenVoice
git clone https://github.com/myshell-ai/OpenVoice.git temp
cp -r temp/checkpoints checkpoints_v1
rm -rf temp

# V2 模型
git clone https://github.com/myshell-ai/OpenVoiceV2.git temp
cp -r temp/checkpoints_v2 checkpoints_v2
rm -rf temp
```

#### Qwen3-TTS

```bash
cd algorithms/Qwen3-TTS
# 使用 huggingface-cli 下载模型
huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-Base --local-dir models/Qwen/Qwen3-TTS-12Hz-1.7B-Base
```

#### VoxCPM

```bash
cd algorithms/VoxCPM
# 使用 huggingface-cli 下载模型
huggingface-cli download openbmb/VoxCPM2 --local-dir models/VoxCPM2
```

#### IndexTTS

```bash
cd algorithms/IndexTTS
# 参考项目 README 下载模型到 checkpoints/ 目录
```

#### FireRedTTS2

```bash
cd algorithms/FireRedTTS2
# 参考项目 README 下载模型到 pretrained_models/FireRedTTS2/ 目录
```

## 启动服务

### 离线模式启动

```bash
# 使用离线模式启动
./start_server.sh start --offline

# 或指定端口
./start_server.sh start --offline --port 8080
```

### 普通模式启动（需要网络）

```bash
# 普通模式（如果服务器有网络访问）
./start_server.sh start
```

### 检查服务状态

```bash
./start_server.sh status
```

### 停止服务

```bash
./start_server.sh stop
```

## 故障排查

### 问题 1: "Connection error" 或无法访问 HuggingFace

**症状**: 启动时提示网络连接错误

**解决**: 确保使用 `--offline` 参数启动，并检查模型文件是否存在

```bash
# 检查模型文件
ls -la algorithms/VoxCPM/models/VoxCPM2/
ls -la algorithms/IndexTTS/checkpoints/
```

### 问题 2: "Model not found" 或找不到模型

**症状**: 提示模型文件不存在

**解决**: 
1. 运行检查脚本确认模型路径
2. 检查 `.env.offline` 中的路径配置
3. 确认模型文件已正确解压

```bash
python check_models.py
```

### 问题 3: "Permission denied"

**症状**: 权限错误

**解决**:
```bash
chmod +x start_server.sh
checkmod +x check_models.py
```

### 问题 4: CUDA 错误

**症状**: CUDA out of memory 或其他 CUDA 错误

**解决**:
1. 检查 GPU 状态: `nvidia-smi`
2. 减少 batch size 或同时加载的模型数量
3. 检查 CUDA 版本兼容性

### 问题 5: 特定算法加载失败

**症状**: 某个 TTS 算法无法加载

**解决**:
1. 检查该算法的模型文件完整性
2. 查看日志: `tail -f logs/server.log`
3. 单独测试该算法: `python test_scripts/test_<algorithm>.py`

## 附录

### 环境变量完整列表

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `HF_HUB_OFFLINE` | 禁用 HuggingFace Hub 在线访问 | `1` |
| `HF_DATASETS_OFFLINE` | 禁用 HuggingFace Datasets 在线访问 | `1` |
| `TRANSFORMERS_OFFLINE` | 禁用 Transformers 在线下载 | `1` |
| `HF_HUB_DISABLE_DOWNLOADS` | 禁用 HuggingFace 下载 | `1` |
| `HF_HOME` | HuggingFace 缓存根目录 | `项目目录/models/hf_cache` |
| `TRANSFORMERS_CACHE` | Transformers 缓存目录 | `项目目录/models/transformers_cache` |
| `TORCH_HOME` | PyTorch Hub 缓存目录 | `项目目录/models/torch_cache` |

### 相关脚本

- `start_server.sh`: 服务管理脚本（支持 `--offline` 参数）
- `check_models.py`: 模型路径检查脚本
- `.env.offline`: 离线环境变量配置文件

### 联系支持

如有问题，请检查：
1. 项目日志: `logs/server.log`
2. 各算法目录下的 `readme.md` 文件
3. 项目 README.md
