# VersTTS - 统一语音合成平台

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/CUDA-支持-green.svg" alt="CUDA Support">
  <img src="https://img.shields.io/badge/支持-10种TTS算法-orange.svg" alt="10 TTS Algorithms">
  <img src="https://img.shields.io/badge/许可-Apache%202.0-yellow.svg" alt="License">
</p>

VersTTS 是一个统一的语音合成（Text-to-Speech）平台，集成了业界领先的 10 种开源 TTS 算法，提供统一的后端 API 服务和美观的前端交互界面，支持音色克隆、预设音色、语音设计等多种语音合成能力。

---

## 📑 目录

- [项目简介](#-项目简介)
- [支持的TTS算法](#-支持的tts算法)
- [项目结构](#-项目结构)
- [快速开始](#-快速开始)
- [安装部署](#-安装部署)
- [使用指南](#-使用指南)
- [API文档](#-api文档)
- [算法分析](#-算法分析)
- [前端功能](#-前端功能)
- [开发记录](#-开发记录)
- [常见问题](#-常见问题)

---

## 🎯 项目简介

### 核心功能

- **多算法集成**: 一键切换 10 种主流 TTS 算法（3种前端可用）
- **独立服务架构**: 主服务 + OmniVoice/CosyVoice 独立服务，解决依赖冲突
- **详细日志记录**: 所有 TTS 服务记录完整的模型信息和请求参数
- **音色克隆**: 支持使用参考音频进行声音克隆
- **预设音色**: 提供多种优质预设人声
- **语音设计**: 通过自然语言描述生成特定音色
- **批量处理**: 支持批量文本上传和音频下载
- **Web界面**: 美观的现代化 Web 交互界面
- **统一API**: RESTful API 设计，易于集成
- **说话人管理**: 统一管理参考人声，多算法共享音色库
- **并发控制**: GPU 资源锁 + 速率限制 + 任务队列

### 技术特点

- 基于 FastAPI 的高性能后端服务
- 支持 CUDA 加速推理
- 模型文件本地化管理（非 .cache 目录）
- 完善的日志系统
- 内置登录认证（默认: admin / tp123456）
- 内存泄漏防护与显存自动释放
- 多用户并发访问支持

---

## 🎙️ 支持的TTS算法

### 当前前端可用算法

| 算法 | 能力 | 音色克隆 | 预设音色 | 指令控制 | 流式 | 中文效果 | 服务类型 | 状态 |
|------|------|----------|----------|----------|------|----------|----------|------|
| **Qwen3-TTS** | 全功能 | ✅ (3秒) | 9种 | ✅ | ✅ (97ms) | 优秀 | 主服务 | 🟢 前端可用 |
| **VoxCPM** | 全功能 | ✅ | 9种 | ✅ | ❌ | 优秀 | 主服务 | 🟢 前端可用 |
| **OmniVoice** | 全功能 | ✅ | 12种方言 | ✅ | ❌ | 优秀 | 独立服务 | 🟢 前端可用 |
| **CosyVoice** | 全功能 | ✅ | 18+方言 | ✅ | ❌ | 优秀 | 独立服务 | 🟢 前端可用 |

### 后端API可用算法（前端已隐藏）

| 算法 | 能力 | 音色克隆 | 预设音色 | 状态 |
|------|------|----------|----------|------|
| **GPT-SoVITS** | 克隆专用 | ✅ 优秀 | ❌ | 🔴 前端已屏蔽 |
| **ChatTTS** | 对话优化 | ✅ | ❌ | 🔴 前端已屏蔽 |
| **F5-TTS** | 克隆专用 | ✅ | ❌ | 🔴 前端已屏蔽 |
| **OpenVoice** | 克隆专用 | ✅ | ❌ | 🔴 前端已屏蔽 |
| **IndexTTS** | 全功能 | ✅ | ❌ | 🔴 前端已屏蔽 |
| **FireRedTTS2** | 对话优化 | ✅ | ❌ | 🔴 前端已屏蔽 |

> **说明**: 被屏蔽的算法仍可通过后端 API 调用，前端界面暂时隐藏以优化用户体验。

### 算法选择建议

- **需要流式低延迟**: 选择 Qwen3-TTS
- **需要音色设计**: 选择 OmniVoice (12种方言)
- **最佳克隆效果**: 选择 VoxCPM
- **方言支持**: 
  - CosyVoice: 18+种中文方言（粤语、闽南语、四川话、东北话、河南话、陕西话、山东话、上海话、天津话等）
  - OmniVoice: 12种方言（四川话、东北话、河南话、陕西话、云南话、贵州话、桂林话、甘肃话、宁夏话、济南话、青岛话、石家庄话）
  - Qwen3-TTS: 北京话、四川话
  - VoxCPM: 30种语言+9种方言

---

## 📁 项目结构

```
VersTTS/
├── algorithms/              # TTS算法目录
│   ├── ChatTTS/            # ChatTTS 项目
│   ├── CosyVoice/          # CosyVoice 项目
│   ├── F5-TTS/             # F5-TTS 项目
│   ├── FireRedTTS2/        # FireRedTTS2 项目
│   ├── GPT-SoVITS/         # GPT-SoVITS 项目
│   ├── IndexTTS/           # IndexTTS 项目
│   ├── OmniVoice/          # OmniVoice 项目
│   ├── OpenVoice/          # OpenVoice 项目
│   ├── Qwen3-TTS/          # Qwen3-TTS 项目
│   └── VoxCPM/             # VoxCPM 项目
│
├── backend/                 # 后端服务
│   ├── api_server.py       # 统一API服务主文件
│   ├── core/               # 核心模块
│   │   ├── concurrency.py  # 并发控制
│   │   └── memory_utils.py # 内存管理
│   ├── routers/            # API路由
│   └── ...
│
├── frontend/                # 前端界面
│   ├── login.html          # 登录页面
│   ├── index.html          # 主页面
│   └── pages/              # 各算法页面
│
├── lib/                     # 独立依赖库
│   ├── transformers5/      # OmniVoice 专用 transformers 5.x
│   └── transformers4/      # CosyVoice 专用 transformers 4.51.3
│
├── models/                  # 模型文件存放目录
│   ├── FunAudioLLM/        # CosyVoice 模型 (~6GB)
│   ├── OmniVoice/          # OmniVoice 模型 (~2.3GB)
│   ├── Qwen3-TTS/          # Qwen3-TTS 模型 (~18GB)
│   └── VoxCPM/             # VoxCPM 模型 (~4.7GB)
│
├── cosyvoice_service.py     # CosyVoice 独立服务脚本
├── omnivoice_service.py     # OmniVoice 独立服务脚本
├── start_server.sh          # 统一启动脚本
├── test_scripts/            # 测试脚本
├── scripts/                 # 工具脚本
├── logs/                    # 日志文件目录
├── records/                 # 工作记录文档
├── outputs/                 # 音频输出目录
├── requirements.txt         # Python依赖
└── README.md               # 项目说明文档
```

---

## 🚀 快速开始

### 环境要求

- Python 3.10+
- CUDA 11.8+ (推荐)
- 16GB+ GPU 显存 (运行1.7B模型)
- 8GB+ 系统内存

### 1. 克隆项目

```bash
git clone <项目地址>
cd VersTTS
```

### 2. 创建虚拟环境

```bash
# 使用 conda (推荐)
conda create -n verstts python=3.10 -y
conda activate verstts

# 或使用 venv
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate  # Windows
```

### 3. 安装依赖

```bash
pip install -r requirements.txt

# 可选：安装 FlashAttention 2 以优化性能
pip install flash-attn --no-build-isolation
```

### 4. 下载模型文件

模型文件需要手动下载并存放在指定目录：

```bash
# Qwen3-TTS 模型下载示例
modelscope download --model Qwen/Qwen3-TTS-12Hz-1.7B-Base --local_dir ./models/Qwen3-TTS-12Hz-1.7B-Base

# VoxCPM 模型下载示例
modelscope download --model OpenBMB/VoxCPM-2B --local_dir ./models/VoxCPM-2B

# 其他模型请参考各算法目录下的 readme.md
```

### 5. 启动服务

```bash
# 使用启动脚本
./start_server.sh start                    # 启动主服务
./start_server.sh start --offline          # 离线模式启动（推荐）
./start_server.sh stop                     # 停止主服务
./start_server.sh restart                  # 重启主服务
./start_server.sh status                   # 查看主服务状态

# OmniVoice 独立服务（需要 transformers 5.x）
./start_server.sh start-omnivoice          # 启动 OmniVoice 独立服务
./start_server.sh start-omnivoice --offline  # 离线模式启动 OmniVoice
./start_server.sh stop-omnivoice           # 停止 OmniVoice 独立服务
./start_server.sh restart-omnivoice        # 重启 OmniVoice 独立服务
./start_server.sh status-omnivoice         # 查看 OmniVoice 服务状态

# CosyVoice 独立服务（需要 transformers 4.51.3）
./start_server.sh start-cosyvoice          # 启动 CosyVoice 独立服务
./start_server.sh start-cosyvoice --offline  # 离线模式启动 CosyVoice
./start_server.sh stop-cosyvoice           # 停止 CosyVoice 独立服务
./start_server.sh restart-cosyvoice        # 重启 CosyVoice 独立服务
./start_server.sh status-cosyvoice         # 查看 CosyVoice 服务状态

# 一键启动所有服务
./start_server.sh start-all                # 启动主服务 + 所有独立服务
./start_server.sh stop-all                 # 停止所有服务
./start_server.sh restart-all              # 重启所有服务

# 查看日志
tail -f logs/server.log                    # 主服务日志
tail -f logs/omnivoice_service.log         # OmniVoice 服务日志
tail -f logs/cosyvoice_service.log         # CosyVoice 服务日志

# 或手动启动
python backend/api_server.py               # 手动启动主服务
python omnivoice_service.py                # 手动启动 OmniVoice 服务
```

> **注意**: 
> - OmniVoice 需要 transformers 5.x，运行在独立进程（端口 8001）
> - CosyVoice 需要 transformers 4.51.3，运行在独立进程（端口 8002）
> - 其他 TTS 算法（Qwen3-TTS、VoxCPM 等）运行在主服务（端口 8000）
> 
> 三个服务相互独立，可以单独启停，通过 restart-all 可一键重启所有服务。

### 6. 访问系统

- 前端界面: http://localhost:8000/static/login.html
- 默认账号: `admin` / `tp123456`
- API文档: http://localhost:8000/docs

---

## 📖 安装部署

### 环境配置

所有配置项集中在 `start_server.sh` 脚本顶部，部署到服务器时直接修改脚本中的变量即可：

```bash
# 编辑启动脚本，修改配置项
nano start_server.sh
```

**可配置项说明（位于 start_server.sh 顶部）：**

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `HOST` | `0.0.0.0` | 主服务监听地址 |
| `PORT` | `8000` | 主服务端口 |
| `OMNIVOICE_HOST` | `127.0.0.1` | OmniVoice 服务地址 |
| `OMNIVOICE_PORT` | `8001` | OmniVoice 服务端口 |
| `COSYVOICE_HOST` | `127.0.0.1` | CosyVoice 服务地址 |
| `COSYVOICE_PORT` | `8002` | CosyVoice 服务端口 |
| `MAIN_GPU` | `0` | 主服务使用的 GPU |
| `OMNIVOICE_GPU` | `0` | OmniVoice 服务使用的 GPU |
| `COSYVOICE_GPU` | `0` | CosyVoice 服务使用的 GPU |
| `MODELS_DIR` | `models` | 模型文件目录 |
| `OUTPUTS_DIR` | `outputs` | 音频输出目录 |
| `LOGS_DIR` | `logs` | 日志目录 |
| `SPEAKERS_DIR` | `speakers` | 说话人数据库目录 |
| `TRANSFORMERS_OFFLINE` | `1` | 离线模式（1=启用） |
| `HF_HUB_OFFLINE` | `1` | HuggingFace 离线模式 |
| `LOG_LEVEL` | `INFO` | 日志级别 |

**服务器部署示例：**

```bash
# 编辑 start_server.sh，修改顶部配置变量
# 例如修改端口：
PORT="8888"
OMNIVOICE_PORT="9001"

# 修改模型目录到外部存储：
MODELS_DIR="/data/models"
OUTPUTS_DIR="/data/outputs"
```

### 详细安装步骤

#### 步骤1: 环境准备

```bash
# 检查Python版本
python --version  # 需要 3.10+

# 检查CUDA
nvidia-smi  # 确保CUDA可用
```

#### 步骤2: 安装基础依赖

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

#### 步骤3: 模型下载

各算法模型下载方式详见对应目录：

| 算法 | 模型下载文档 |
|------|-------------|
| ChatTTS | [algorithms/ChatTTS/readme.md](algorithms/ChatTTS/readme.md) |
| CosyVoice | [algorithms/CosyVoice/readme.md](algorithms/CosyVoice/readme.md) |
| F5-TTS | [algorithms/F5-TTS/readme.md](algorithms/F5-TTS/readme.md) |
| FireRedTTS2 | [algorithms/FireRedTTS2/readme.md](algorithms/FireRedTTS2/readme.md) |
| GPT-SoVITS | [algorithms/GPT-SoVITS/readme.md](algorithms/GPT-SoVITS/readme.md) |
| IndexTTS | [algorithms/IndexTTS/readme.md](algorithms/IndexTTS/readme.md) |
| OpenVoice | [algorithms/OpenVoice/readme.md](algorithms/OpenVoice/readme.md) |
| Qwen3-TTS | [algorithms/Qwen3-TTS/readme.md](algorithms/Qwen3-TTS/readme.md) |
| VoxCPM | [algorithms/VoxCPM/readme.md](algorithms/VoxCPM/readme.md) |

#### 步骤4: 验证安装

```bash
# 运行测试脚本
python test_scripts/test_all_tts.py
```

---

## 💡 使用指南

### Web界面使用

1. **登录系统**: 访问 `http://localhost:8000/static/login.html`，使用默认账号登录
2. **选择算法**: 在主页面选择想要使用的 TTS 算法
3. **输入文本**: 在文本框中输入要合成的内容
4. **配置参数**:
   - 选择预设音色（如支持）
   - 选择说话人进行克隆（如支持）
   - 调整生成参数
5. **生成语音**: 点击生成按钮，等待结果
6. **批量处理**: 支持批量上传文本文件，批量下载生成的音频

### API调用示例

```python
import requests

# API端点
url = "http://localhost:8000/tts/qwen3-tts"

# 请求参数
data = {
    "text": "你好，这是语音合成测试。",
    "speaker_id": "Vivian",  # 预设音色
    "mode": "sft"
}

# 发送请求
response = requests.post(url, json=data)

# 保存音频
with open("output.wav", "wb") as f:
    f.write(response.content)
```

### 音色克隆示例

```python
import requests

url = "http://localhost:8000/tts/voxcpm"

# 使用说话人ID进行克隆
data = {
    "text": "使用参考音频的声音说这句话。",
    "mode": "clone",
    "clone_speaker_id": "speaker_001"
}

response = requests.post(url, data=data)
```

### 批量TTS生成

```bash
# 使用批量TTS客户端
python scripts/batch_tts_client.py \
    --input scripts/examples/sample_texts.csv \
    --algorithm qwen3-tts \
    --output ./batch_output/
```

---

## 📚 API文档

### 主要API端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/` | GET | 服务状态检查 |
| `/tts/{algorithm}` | POST | 文本转语音（各算法端点） |
| `/algorithms` | GET | 获取支持的算法列表 |
| `/speakers` | GET/POST | 说话人管理 |
| `/system/gpu-memory` | GET | GPU显存状态 |
| `/concurrency/status` | GET | 并发状态查询 |

### 完整API文档

启动服务后访问: http://localhost:8000/docs

---

## 🔬 算法分析

各算法的详细分析文档已整理在对应目录：

| 算法 | 分析文档 | 核心特点 |
|------|----------|----------|
| **ChatTTS** | [algorithms/ChatTTS/readme.md](algorithms/ChatTTS/readme.md) | 扩散+GPT架构，对话优化 |
| **Qwen3-TTS** | [algorithms/Qwen3-TTS/readme.md](algorithms/Qwen3-TTS/readme.md) | 离散多码本LM，97ms流式延迟 |
| **CosyVoice** | [algorithms/CosyVoice/readme.md](algorithms/CosyVoice/readme.md) | 流匹配+LLM，多语言多方言支持 |
| **F5-TTS** | [algorithms/F5-TTS/readme.md](algorithms/F5-TTS/readme.md) | 流匹配+DiT，高效快速 |
| **OpenVoice** | [algorithms/OpenVoice/readme.md](algorithms/OpenVoice/readme.md) | VAE+VITS，音色风格解耦 |
| **GPT-SoVITS** | [algorithms/GPT-SoVITS/readme.md](algorithms/GPT-SoVITS/readme.md) | VQ+GPT+VITS，最佳克隆效果 |
| **VoxCPM** | [algorithms/VoxCPM/readme.md](algorithms/VoxCPM/readme.md) | 无Tokenizer扩散自回归，30语言支持 |
| **OmniVoice** | [algorithms/OmniVoice/readme.md](algorithms/OmniVoice/readme.md) | 扩散语言模型，600+语言，12种方言 |
| **CosyVoice** | [algorithms/CosyVoice/readme.md](algorithms/CosyVoice/readme.md) | 流匹配+LLM，18+中文方言支持 |
| **IndexTTS** | [algorithms/IndexTTS/readme.md](algorithms/IndexTTS/readme.md) | 自回归GPT架构，哔哩哔哩开源 |
| **FireRedTTS2** | [algorithms/FireRedTTS2/readme.md](algorithms/FireRedTTS2/readme.md) | 双Transformer，长对话优化 |

### 算法对比总结

```
┌─────────────┬────────────┬──────────┬──────────┬──────────┐
│   算法      │  架构类型   │ 克隆质量  │ 生成速度  │ 内存占用  │
├─────────────┼────────────┼──────────┼──────────┼──────────┤
│ OmniVoice   │ 扩散LM     │    ★★★★  │   ★★★★  │   高     │
│ Qwen3-TTS   │ 多码本LM   │    ★★★   │   ★★★   │   高     │
│ GPT-SoVITS  │ VQ+GPT     │    ★★★★  │   ★★    │   中     │
│ CosyVoice   │ 流匹配+LM  │    ★★★   │   ★★★   │   高     │
│ ChatTTS     │ 扩散+GPT   │    ★★    │   ★★    │   中     │
│ F5-TTS      │ 流匹配     │    ★★★   │   ★★★   │   低     │
│ OpenVoice   │ VAE+TTS    │    ★★    │   ★★★   │   低     │
│ VoxCPM      │ 扩散自回归 │    ★★★★  │   ★★    │   高     │
│ IndexTTS    │ GPT自回归  │    ★★★   │   ★★    │   中     │
│ FireRedTTS2 │ 双Transformer│  ★★★   │   ★★    │   高     │
└─────────────┴────────────┴──────────┴──────────┴──────────┘
```

### 独立服务说明

由于依赖冲突，以下算法运行在独立服务进程中：

| 服务 | 端口 | Transformers版本 | 说明 |
|------|------|------------------|------|
| 主服务 | 8000 | 4.57.3 | Qwen3-TTS, VoxCPM 等 |
| OmniVoice服务 | 8001 | 5.x | 需要最新版 transformers |
| CosyVoice服务 | 8002 | 4.51.3 | CosyVoice 3.0 专用版本 |

---

## 🖥️ 前端功能

### 当前前端可用算法功能

#### Qwen3-TTS

| 功能 | 状态 | 说明 |
|------|------|------|
| SFT模式(预设音色) | ✅ | 9种预设音色 |
| 声音克隆 | ✅ | 3秒参考音频克隆 |
| 音色设计 | ✅ | 自然语言描述生成音色 |
| 流式生成 | ✅ | 97ms低延迟 |
| 模型选择 | ✅ | 仅支持1.7B模型 |

#### VoxCPM

| 功能 | 状态 | 说明 |
|------|------|------|
| 基础生成 | ✅ | 默认音色生成 |
| 声音设计 | ✅ | 自然语言描述音色 |
| 声音克隆 | ✅ | Reference-only模式 |
| 极致克隆 | ✅ | Combined模式，需参考文本 |
| 控制指令 | ✅ | 支持语速/情绪控制 |
| 方言支持 | ✅ | 30种语言+9种方言

#### CosyVoice

| 功能 | 状态 | 说明 |
|------|------|------|
| Zero-shot克隆 | ✅ | 3-10秒参考音频克隆 |
| Instruct指令 | ✅ | 自然语言控制说话风格 |
| 方言支持 | ✅ | 18+种中文方言（粤语、闽南语、四川话等） |
| 跨语言克隆 | ✅ | 支持多语言音色迁移 |
| 服务类型 | 独立服务 | 端口8002，transformers 4.51.3 |

### 通用功能

- ✅ 算法选择卡片界面
- ✅ 文本输入与预览
- ✅ 说话人选择与管理
- ✅ 生成结果播放与下载
- ✅ 批量文本处理
- ✅ 登录认证
- ✅ 服务状态监控

---

## 🔧 系统架构

```
┌───────────────────────────────────────────────────────────────────────┐
│                              前端界面                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────────┐   │
│  │  登录页面    │  │  主应用页面  │  │      算法选择页面            │   │
│  └─────────────┘  └─────────────┘  └─────────────────────────────┘   │
└──────────────────────────┬────────────────────────────────────────────┘
                           │ HTTP/WebSocket
┌──────────────────────────▼────────────────────────────────────────────┐
│                      FastAPI 后端服务                                  │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │                    统一API接口层                                 │ │
│  │     /tts/*  /speakers  /system  /concurrency                    │ │
│  └────────────────────────┬────────────────────────────────────────┘ │
│                           │ 算法调度层 + 并发控制                      │
│  ┌───────────┬───────────┬───────────┬───────────┬───────────┐       │
│  │ ChatTTS   │ F5-TTS    │GPT-SoVITS │ OpenVoice │ VoxCPM    │       │
│  │ IndexTTS  │ FireRed   │ Qwen3-TTS │ 适配器    │ 适配器    │       │
│  │ 适配器    │ 适配器    │ 适配器    │           │           │       │
│  └─────┬─────┴─────┬─────┴─────┬─────┴─────┬─────┴─────┬─────┘       │
│        │           │           │           │           │             │
└────────┼───────────┼───────────┼───────────┼───────────┼─────────────┘
         │           │           │           │           │
    ┌────┴────┐ ┌────┴────┐ ┌────┴────┐ ┌────┴────┐ ┌────┴────┐
    │ ChatTTS │ │ F5-TTS  │ │GPT-SoVIT│ │OpenVoice│ │ VoxCPM  │
    │  模型   │ │  模型   │ │  模型   │ │  模型   │ │  模型   │
    └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘

┌───────────────────────────────────────────────────────────────────────┐
│                         独立服务进程                                   │
│  ┌──────────────────────────┐  ┌──────────────────────────┐          │
│  │   OmniVoice 服务          │  │   CosyVoice 服务          │          │
│  │   端口: 8001              │  │   端口: 8002              │          │
│  │   Transformers: 5.x       │  │   Transformers: 4.51.3    │          │
│  └───────────┬──────────────┘  └───────────┬──────────────┘          │
│              │                             │                         │
│         ┌────┴────┐                   ┌────┴────┐                    │
│         │OmniVoice│                   │CosyVoice│                    │
│         │  模型   │                   │  模型   │                    │
│         └─────────┘                   └─────────┘                    │
└───────────────────────────────────────────────────────────────────────┘
```

---

## 📝 开发记录

所有工作记录已按时间戳整理在 `records/` 目录：

```bash
# 查看最新记录
ls -lt records/ | head -20
```

主要记录类型：
- 项目结构整理记录
- 各算法部署调试记录
- API开发记录
- 前端开发记录
- 问题排查记录
- 功能验证记录

---

## ❓ 常见问题

### Q1: 为什么需要多个独立服务？

由于不同 TTS 算法依赖不同版本的 transformers 库，而 Python 无法在同一进程中加载多个版本的同一库：

| 服务 | Transformers版本 | 原因 |
|------|------------------|------|
| 主服务 | 4.57.3 | Qwen3-TTS、VoxCPM 等主流算法兼容版本 |
| OmniVoice服务 | 5.x | OmniVoice 需要最新版 transformers |
| CosyVoice服务 | 4.51.3 | CosyVoice 3.0 特定版本要求 |

因此采用独立进程架构，各服务通过 HTTP API 通信。

### Q2: 如何启动完整服务？

```bash
# 一键启动所有服务（推荐）
./start_server.sh start-all                # 启动主服务 + 所有独立服务
./start_server.sh stop-all                 # 停止所有服务
./start_server.sh restart-all              # 重启所有服务

# 单独控制主服务
./start_server.sh start                    # 启动主服务
./start_server.sh stop                     # 停止主服务
./start_server.sh restart                  # 重启主服务
./start_server.sh status                   # 查看主服务状态

# 单独控制 OmniVoice 服务
./start_server.sh start-omnivoice          # 启动 OmniVoice 服务
./start_server.sh stop-omnivoice           # 停止服务
./start_server.sh restart-omnivoice        # 重启服务
./start_server.sh status-omnivoice         # 查看状态

# 单独控制 CosyVoice 服务
./start_server.sh start-cosyvoice          # 启动 CosyVoice 服务
./start_server.sh stop-cosyvoice           # 停止服务
./start_server.sh restart-cosyvoice        # 重启服务
./start_server.sh status-cosyvoice         # 查看状态

# 查看日志
tail -f logs/server.log                    # 主服务日志
tail -f logs/omnivoice_service.log         # OmniVoice 服务日志
tail -f logs/cosyvoice_service.log         # CosyVoice 服务日志
```

### Q3: 模型文件存放位置

所有模型文件统一存放在 `models/` 目录下：

```
models/
├── FunAudioLLM/        # CosyVoice 模型 (~6GB)
│   └── Fun-CosyVoice3-0.5B-2512/
├── OmniVoice/          # OmniVoice 模型 (~2.3GB)
│   └── speechpi/
│       └── OmniVoice/
├── Qwen3-TTS/          # Qwen3-TTS 模型 (~18GB)
│   └── Qwen/
│       └── Qwen3-TTS-1.7B/
└── VoxCPM/             # VoxCPM 模型 (~4.7GB)
```

> **注意**: 模型文件较大，首次启动时会自动下载（需要网络连接），或使用离线模式需手动放置。

### Q4: 各算法支持哪些方言？

**CosyVoice**（18+种中文方言）：
- 粤语、闽南语、四川话、东北话、河南话
- 陕西话、山东话、上海话、天津话、山西话
- 宁夏话、甘肃话、客家话、湖南话、湖北话
- 河北话、安徽话、江苏话

**OmniVoice**（12种中文方言）：
- 四川话、东北话、河南话、陕西话、云南话
- 贵州话、桂林话、甘肃话、宁夏话、济南话
- 青岛话、石家庄话

**Qwen3-TTS**（2种中文方言）：
- 北京话、四川话

**VoxCPM**（30种语言+9种方言）

### Q5: 录音功能无法使用？

录音功能需要 HTTPS 安全连接或 localhost 访问。
- 使用 `http://localhost:8000` 访问（支持录音）
- 使用 `http://192.168.x.x:8000` 访问（不支持录音，浏览器安全限制）

### Q6: 如何查看详细的模型信息？

所有 TTS 服务都会在日志中详细记录模型信息：

```bash
# 查看各服务日志中的模型信息
tail -f logs/server.log | grep "模型信息"           # 主服务
tail -f logs/omnivoice_service.log | grep "模型信息" # OmniVoice
tail -f logs/cosyvoice_service.log | grep "模型信息" # CosyVoice
```

日志内容包括：模型名称、版本、路径、支持的语言/方言、采样率等。

---

## 📄 许可证

本项目采用 Apache 2.0 许可证。

各 TTS 算法遵循其原始开源许可证：
- ChatTTS: AGPL-3.0
- CosyVoice: Apache 2.0
- F5-TTS: MIT
- FireRedTTS2: Apache 2.0
- GPT-SoVITS: MIT
- IndexTTS: 自定义开源许可
- OpenVoice: MIT
- Qwen3-TTS: Apache 2.0
- VoxCPM: Apache 2.0
- OmniVoice: Apache 2.0

---

## 🤝 贡献指南

欢迎提交 Issue 和 PR！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

---

## 📞 联系方式

如有问题或建议，欢迎通过以下方式联系：

- 提交 GitHub Issue
- 查看工作记录: `records/` 目录

---

**更新时间**: 2026-05-10
