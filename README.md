# VersTTS - 统一语音合成平台

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/CUDA-支持-green.svg" alt="CUDA Support">
  <img src="https://img.shields.io/badge/支持-11种TTS算法-orange.svg" alt="11 TTS Algorithms">
  <img src="https://img.shields.io/badge/前端可用-6种算法-brightgreen.svg" alt="6 Frontend Available">
  <img src="https://img.shields.io/badge/许可-Apache%202.0-yellow.svg" alt="License">
</p>

VersTTS 是一个统一的语音合成（Text-to-Speech）平台，集成了业界领先的 11 种开源 TTS 算法，提供统一的后端 API 服务和美观的前端交互界面，支持音色克隆、预设音色、语音设计、方言合成等多种语音合成能力。

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
- [系统架构](#-系统架构)
- [开发记录](#-开发记录)
- [常见问题](#-常见问题)

---

## 🎯 项目简介

### 核心功能

- **多算法集成**: 集成 11 种主流 TTS 算法，6 种前端可用
- **独立服务架构**: 主服务 + 4 个独立服务（OmniVoice / CosyVoice / PilotTTS / GPT-SoVITS），解决依赖冲突
- **音色克隆**: 通过说话人管理模块统一管理参考人声，多算法共享音色库
- **预设音色**: 提供多种优质预设人声（Qwen3-TTS 9种、VoxCPM 多种等）
- **语音设计**: 通过自然语言描述生成特定音色（VoxCPM、OmniVoice、PilotTTS）
- **方言支持**: 多种中文方言合成能力（四川话、粤语、闽南语、东北话等）
- **批量处理**: 支持批量生成（最多100个）和批量下载 ZIP 打包
- **任务队列**: 优先级任务队列，支持后台异步执行，实时状态追踪
- **并发控制**: GPU 资源锁 + 令牌桶速率限制 + 模型级并发管理
- **Web 界面**: 美观的现代化 Web 交互界面，动态音波背景效果
- **统一 API**: RESTful API 设计，Swagger 自动文档
- **离线部署**: 支持无 HuggingFace 等外部网络环境的服务器部署
- **内存管理**: 推理完成后自动释放显存，防止内存泄漏
- **说话人管理**: 统一管理参考人声，支持录制、上传、删除，带自定义确认弹窗

### 技术特点

- 基于 FastAPI 的高性能后端服务
- 支持 CUDA 加速推理
- 模型文件本地化管理（统一存放于 `models/` 目录）                                                              
- 完善的日志系统（自动轮转、清理）
- 内置登录认证（默认: `admin`       ）
- 浏览器 Logo 图标支持
- 多用户并发访问支持
- 所有配置项集中管理，防止硬编码

---

## 🎙️ 支持的TTS算法

### 当前前端可用算法（6种）

| 算法 | 能力 | 音色克隆 | 预设音色 | 指令控制 | 方言支持 | 流式 | 服务类型 | 状态 |
|------|------|----------|----------|----------|----------|------|----------|------|
| **Qwen3-TTS** | 全功能 | ✅ (3秒) | 9种 | ✅ | 北京话、四川话 | ✅ (97ms) | 主服务 | 🟢 前端可用 |
| **VoxCPM** | 全功能 | ✅ | 多种 | ✅ | 30种语言+9种方言 | ❌ | 主服务 | 🟢 前端可用 |
| **OmniVoice** | 全功能 | ✅ | 12种方言音色 | ✅ | 12种中文方言 | ❌ | 独立服务 | 🟢 前端可用 |
| **CosyVoice** | 全功能 | ✅ | 18+方言 | ✅ | 18+种中文方言 | ❌ | 独立服务 | 🟢 前端可用 |
| **GPT-SoVITS** | 克隆专用 | ✅ 优秀 | ❌ | ❌ | 中文为主 | ❌ | 独立服务 | 🟢 前端可用 |
| **PilotTTS** | 全功能 | ✅ | 11种情感+14种方言 | ✅ | 14种中文方言 | ❌ | 独立服务 | 🟢 前端可用 |

### 后端API可用算法（前端已隐藏，5种）

| 算法 | 能力 | 音色克隆 | 预设音色 | 状态 | 隐藏原因 |
|------|------|----------|----------|------|----------|
| **ChatTTS** | 对话优化 | ✅ | ❌ | 🔴 前端已屏蔽 | 禁止商业使用 |
| **F5-TTS** | 克隆专用 | ✅ | ❌ | 🔴 前端已屏蔽 | 禁止商业使用 |
| **OpenVoice** | 克隆专用 | ✅ | ❌ | 🔴 前端已屏蔽 | 音色相似度一般 |
| **IndexTTS** | 全功能 | ✅ | ❌ | 🔴 前端已屏蔽 | 哔哩哔哩开源 |
| **FireRedTTS2** | 对话优化 | ✅ | ❌ | 🔴 前端已屏蔽 | 模型较大(19.7GB) |

> **说明**: 被屏蔽的算法仍可通过后端 API 调用，前端界面暂时隐藏以优化用户体验。

### 算法选择建议

- **需要流式低延迟**: 选择 Qwen3-TTS（97ms 首包延迟）
- **需要音色设计/方言**: 选择 OmniVoice（12种方言）或 PilotTTS（14种方言+11种情感）
- **最佳克隆效果**: 选择 GPT-SoVITS（中文克隆之王）或 VoxCPM（极致克隆模式）
- **多方言覆盖**: CosyVoice (18+种方言，含粤语、闽南语等)
- **情感合成**: 选择 PilotTTS（11种情感：开心、悲伤、愤怒、惊讶等）
- **副语言合成**: 选择 PilotTTS（笑声、哭声、呼吸、咳嗽）
- **跨语言克隆**: 选择 CosyVoice 或 VoxCPM（30种语言）

### 方言支持详情

| 算法 | 方言种类 | 具体方言 |
|------|----------|----------|
| **PilotTTS** | 14种 | 东北话、山东话、河南话、山西话、闽南语、甘肃话、宁夏话、上海话、重庆话、湖北话、湖南话、江西话、贵州话、云南话 |
| **CosyVoice** | 18+种 | 粤语、闽南语、四川话、东北话、河南话、陕西话、山东话、上海话、天津话、山西话、宁夏话、甘肃话、客家话、湖南话、湖北话、河北话、安徽话、江苏话 |
| **OmniVoice** | 12种 | 四川话、东北话、河南话、陕西话、云南话、贵州话、桂林话、甘肃话、宁夏话、济南话、青岛话、石家庄话 |
| **Qwen3-TTS** | 2种 | 北京话、四川话 |
| **VoxCPM** | 30种语言+9种方言 | 四川话、粤语、闽南话等 |

---

## 📁 项目结构

```
VersTTS/
├── algorithms/                  # TTS算法目录（11个项目）
│   ├── ChatTTS/                # ChatTTS 项目（扩散+GPT架构）
│   ├── CosyVoice/              # CosyVoice 项目（流匹配+LLM）
│   ├── F5-TTS/                 # F5-TTS 项目（流匹配+DiT）
│   ├── FireRedTTS2/            # FireRedTTS2 项目（双Transformer）
│   ├── GPT-SoVITS/             # GPT-SoVITS 项目（VQ+GPT+VITS）
│   ├── IndexTTS/               # IndexTTS 项目（GPT自回归）
│   ├── OmniVoice/              # OmniVoice 项目（扩散语言模型）
│   ├── OpenVoice/              # OpenVoice 项目（VAE+VITS）
│   ├── PilotTTS/               # PilotTTS 项目（指令驱动TTS）
│   ├── Qwen3-TTS/              # Qwen3-TTS 项目（多码本LM）
│   └── VoxCPM/                 # VoxCPM 项目（扩散自回归）
│
├── backend/                     # 后端服务
│   ├── main.py                 # FastAPI 主入口
│   ├── config.py               # 全局配置（路径、模型、服务地址）
│   ├── task_queue.py           # 任务队列核心（优先级调度、并发控制）
│   ├── task_handlers.py        # 任务处理器（处理各算法TTS任务）
│   ├── logger_config.py        # 日志配置
│   ├── core/                   # 核心模块
│   │   ├── concurrency.py      # 并发控制（GPU锁 + 速率限制 + 任务队列）
│   │   ├── memory_utils.py     # 内存管理（显存释放 + GC回收）
│   │   └── audio_utils.py      # 音频工具（统一命名、保存）
│   ├── engines/                # 算法引擎（各TTS算法推理封装）
│   │   ├── qwen3tts_engine.py
│   │   ├── voxcpm_engine.py
│   │   ├── chattts_engine.py
│   │   ├── f5tts_engine.py
│   │   ├── openvoice_engine.py
│   │   ├── indextts_engine.py
│   │   ├── fireredtts2_engine.py
│   │   ├── pilottts_engine.py.bak
│   │   ├── cosyvoice_engine.py.bak
│   │   ├── gptsovits_engine.py.bak
│   │   ├── omnivoice_engine.py.bak
│   │   └── transformers_compat.py  # Transformers 版本兼容工具
│   ├── routers/                # API 路由
│   │   ├── tts/                # TTS 算法路由（11个）
│   │   │   ├── qwen3tts.py
│   │   │   ├── voxcpm.py
│   │   │   ├── gptsovits.py
│   │   │   ├── cosyvoice.py
│   │   │   ├── omnivoice.py
│   │   │   ├── pilottts.py
│   │   │   ├── chattts.py
│   │   │   ├── f5tts.py
│   │   │   ├── openvoice.py
│   │   │   ├── indextts.py
│   │   │   └── fireredtts.py
│   │   ├── batch.py            # 批量生成 API
│   │   ├── task_queue.py       # 任务队列 API
│   │   ├── speakers.py         # 说话人管理 API
│   │   ├── concurrency.py      # 并发管理 API
│   │   ├── system.py           # 系统管理 API
│   │   └── services.py         # 服务状态 API
│   ├── models/                 # 数据模型（Pydantic schemas）
│   └── services/               # 业务逻辑层
│
├── frontend/                    # 前端界面
│   ├── login.html              # 登录页面
│   ├── index.html              # 主页面（模型选择卡片）
│   ├── assets/                 # 静态资源
│   │   ├── logo.svg            # 浏览器 Logo
│   │   └── css/                # 样式文件
│   └── pages/                  # 各算法页面
│       ├── qwen3tts.html       # Qwen3-TTS 页面
│       ├── voxcpm.html         # VoxCPM 页面
│       ├── omnivoice.html      # OmniVoice 页面
│       ├── cosyvoice.html      # CosyVoice 页面
│       ├── gptsovits.html      # GPT-SoVITS 页面
│       ├── pilottts.html       # PilotTTS 页面
│       ├── chattts.html        # ChatTTS 页面（隐藏入口）
│       ├── f5tts.html          # F5-TTS 页面（隐藏入口）
│       ├── openvoice.html      # OpenVoice 页面（隐藏入口）
│       ├── indextts.html       # IndexTTS 页面（隐藏入口）
│       ├── fireredtts.html     # FireRedTTS2 页面（隐藏入口）
│       └── tasks.html          # 任务队列管理页面
│
├── models/                      # 模型文件存放目录
│   ├── Qwen3-TTS/              # Qwen3-TTS 模型 (~18GB)
│   ├── CosyVoice/              # CosyVoice 模型 (~9.1GB)
│   ├── VoxCPM/                 # VoxCPM 模型 (~4.7GB)
│   ├── GPT-SoVITS/             # GPT-SoVITS 模型 (~4.5GB)
│   ├── OmniVoice/              # OmniVoice 模型 (~3.1GB)
│   ├── PilotTTS/               # PilotTTS 模型 (~6.6GB)
│   └── wenet/                  # WeNet 模型
│
├── lib/                         # 独立依赖库（解决版本冲突）
│   ├── transformers5/          # OmniVoice 专用 transformers 5.x
│   └── transformers4/          # CosyVoice/GPT-SoVITS 专用 transformers 4.51.3
│
├── scripts/                     # 工具脚本
│   ├── batch_tts_client.py     # 批量 TTS 客户端脚本
│   ├── README.md               # 脚本使用说明
│   └── examples/               # 示例文件
│
├── test_scripts/                # 测试脚本目录
│   └── test_all_tts.py         # 全量 TTS 功能测试
│
├── records/                     # 工作记录文档（163个）
│
├── outputs/                     # 音频输出目录
├── logs/                        # 日志文件目录
│   ├── server.log              # 主服务日志
│   ├── omnivoice_service.log   # OmniVoice 服务日志
│   ├── cosyvoice_service.log   # CosyVoice 服务日志
│   ├── pilottts_service.log    # PilotTTS 服务日志
│   └── gptsovits_service.log   # GPT-SoVITS 服务日志
│
├── speakers/                    # 说话人数据库
│   └── speakers_db.json        # 说话人元数据
│
├── start_server.sh              # 统一服务管理脚本
├── cosyvoice_service.py         # CosyVoice 独立服务
├── omnivoice_service.py         # OmniVoice 独立服务
├── pilottts_service.py          # PilotTTS 独立服务
├── gptsovits_service.py         # GPT-SoVITS 独立服务
├── requirements.txt             # Python 基础依赖
├── requirements_full.txt        # Python 完整依赖
└── README.md                    # 项目说明文档
```

---

## 🚀 快速开始

### 环境要求

- Python 3.10+
- CUDA 11.8+（推荐）
- 24GB+ GPU 显存（推荐 RTX 3090/4090，运行全部服务需要约 46GB 模型空间）
- 16GB+ 系统内存
- 100GB+ 磁盘空间（模型文件约 46GB）

### 1. 克隆项目

```bash
git clone <项目地址>
cd VersTTS
```

### 2. 创建虚拟环境

```bash
# 使用 conda（推荐）
conda create -n verstts python=3.10 -y
conda activate verstts

# 或使用 venv
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
```

### 3. 安装依赖

```bash
pip install -r requirements.txt

# 可选：安装 FlashAttention 2 以优化性能
pip install flash-attn --no-build-isolation
```

### 4. 下载模型文件

模型文件需手动下载并存放至 `models/` 目录：

```bash
# Qwen3-TTS 模型（~18GB）
modelscope download --model Qwen/Qwen3-TTS-1.7B --local_dir ./models/Qwen3-TTS/

# VoxCPM 模型（~4.7GB）
modelscope download --model OpenBMB/VoxCPM-2B --local_dir ./models/VoxCPM/

# CosyVoice 模型（~9.1GB）
# 下载 Fun-CosyVoice3-0.5B 到 ./models/CosyVoice/

# PilotTTS 模型（~6.6GB）
# 下载到 ./models/PilotTTS/

# GPT-SoVITS 模型（~4.5GB）
# 下载 BERT、HuBERT、V2/V4 模型到 ./models/GPT-SoVITS/

# OmniVoice 模型（~3.1GB）
# 下载到 ./models/OmniVoice/
```

> 完整模型下载说明请参考各算法目录下的 `readme.md`。

### 5. 启动服务

```bash
# 一键启动所有服务（推荐）
./start_server.sh start-all                # 启动主服务 + 所有独立服务
./start_server.sh stop-all                 # 停止所有服务
./start_server.sh restart-all              # 重启所有服务

# 单独控制主服务
./start_server.sh start                    # 启动主服务（端口 8006）
./start_server.sh stop                     # 停止主服务
./start_server.sh restart                  # 重启主服务
./start_server.sh status                   # 查看主服务状态

# 单独控制独立服务
./start_server.sh start-omnivoice          # 启动 OmniVoice 服务（端口 8007）
./start_server.sh stop-omnivoice           # 停止 OmniVoice 服务
./start_server.sh status-omnivoice         # 查看 OmniVoice 状态

./start_server.sh start-cosyvoice          # 启动 CosyVoice 服务（端口 8008）
./start_server.sh stop-cosyvoice           # 停止 CosyVoice 服务
./start_server.sh status-cosyvoice         # 查看 CosyVoice 状态

./start_server.sh start-pilottts           # 启动 PilotTTS 服务（端口 8009）
./start_server.sh stop-pilottts            # 停止 PilotTTS 服务
./start_server.sh status-pilottts          # 查看 PilotTTS 状态

./start_server.sh start-gptsovits          # 启动 GPT-SoVITS 服务（端口 8010）
./start_server.sh stop-gptsovits           # 停止 GPT-SoVITS 服务
./start_server.sh status-gptsovits         # 查看 GPT-SoVITS 状态
```

### 6. 访问系统

- 前端界面: http://localhost:8006/static/login.html
- API 文档: http://localhost:8006/docs

---

## 📖 安装部署

### 环境配置

所有配置项集中在 `start_server.sh` 脚本顶部，部署到服务器时直接修改脚本中的变量即可：

```bash
nano start_server.sh
```

**可配置项说明（位于 `start_server.sh` 顶部）：**

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `HOST` | `0.0.0.0` | 主服务监听地址 |
| `PORT` | `8006` | 主服务端口 |
| `OMNIVOICE_HOST` | `127.0.0.1` | OmniVoice 服务地址 |
| `OMNIVOICE_PORT` | `8007` | OmniVoice 服务端口 |
| `COSYVOICE_HOST` | `127.0.0.1` | CosyVoice 服务地址 |
| `COSYVOICE_PORT` | `8008` | CosyVoice 服务端口 |
| `PILOTTS_HOST` | `127.0.0.1` | PilotTTS 服务地址 |
| `PILOTTS_PORT` | `8009` | PilotTTS 服务端口 |
| `GPTSOVITS_HOST` | `127.0.0.1` | GPT-SoVITS 服务地址 |
| `GPTSOVITS_PORT` | `8010` | GPT-SoVITS 服务端口 |
| `MAIN_GPU` | `0` | 主服务使用的 GPU |
| `OMNIVOICE_GPU` | `0` | OmniVoice 服务使用的 GPU |
| `COSYVOICE_GPU` | `0` | CosyVoice 服务使用的 GPU |
| `PILOTTS_GPU` | `0` | PilotTTS 服务使用的 GPU |
| `GPTSOVITS_GPU` | `0` | GPT-SoVITS 服务使用的 GPU |
| `MAX_CONCURRENT_MODELS` | `2` | 最大并发模型数 |
| `MODELS_DIR` | `models` | 模型文件目录 |
| `OUTPUTS_DIR` | `outputs` | 音频输出目录 |
| `LOGS_DIR` | `logs` | 日志目录 |
| `SPEAKERS_DIR` | `speakers` | 说话人数据库目录 |
| `TRANSFORMERS_OFFLINE` | `1` | 离线模式（1=启用） |
| `HF_HUB_OFFLINE` | `1` | HuggingFace 离线模式 |
| `LOG_LEVEL` | `INFO` | 日志级别 |
| `IDLE_TIMEOUT` | `300` | 空闲超时秒数（默认5分钟） |
| `PRELOAD_OMNIVOICE` | `0` | 启动时预加载 OmniVoice（1=启用） |
| `PRELOAD_COSYVOICE` | `0` | 启动时预加载 CosyVoice（1=启用） |
| `PRELOAD_PILOTTS` | `0` | 启动时预加载 PilotTTS（1=启用） |
| `PRELOAD_GPTSOVITS` | `0` | 启动时预加载 GPT-SoVITS（1=启用） |

### 离线部署

项目支持完全离线部署，无需访问 HuggingFace 等外部资源：

```bash
# 使用离线模式启动
./start_server.sh start --offline
./start_server.sh start-all --offline

# 或设置环境变量
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
```

> 离线部署前需确保所有模型文件已预先下载并放置到 `models/` 目录对应位置。

### 服务器部署注意事项

1. **录音功能**: 需要通过 `https://` 或 `localhost` 访问才能使用浏览器录音功能（浏览器安全策略）
2. **端口配置**: 如需修改端口，统一在 `start_server.sh` 顶部修改
3. **GPU分配**: 多 GPU 服务器可为不同服务分配不同 GPU
4. **日志文件**: 日志自动轮转（50MB/文件，保留5个备份），避免磁盘占满

---

## 💡 使用指南

### Web 界面使用

1. **登录系统**: 访问 `http://localhost:8006/static/login.html`，使用默认账号登录
2. **选择算法**: 在主页面选择想要使用的 TTS 算法
3. **输入文本**: 在文本框中输入要合成的内容
4. **配置参数**:
   - 选择合成模式（基础生成/音色克隆/语音设计等）
   - 选择预设音色或说话人（如支持）
   - 调整语速、情感等参数
5. **生成语音**: 点击"生成语音"按钮，任务自动提交到任务队列
6. **查看结果**: 任务完成后自动显示音频播放器和下载按钮
7. **批量处理**: 支持批量生成（最多100个），批量下载 ZIP 打包
8. **任务队列**: 前往任务列表页面查看所有任务状态

### API 调用示例

#### 单次 TTS 生成

```python
import requests

url = "http://localhost:8006/tts/qwen3-tts"

data = {
    "text": "你好，这是语音合成测试。",
    "speaker_id": "Vivian",  # 预设音色
    "mode": "sft"
}

response = requests.post(url, json=data)

with open("output.wav", "wb") as f:
    f.write(response.content)
```

#### 音色克隆

```python
import requests

url = "http://localhost:8006/tts/voxcpm"

data = {
    "text": "使用参考音频的声音说这句话。",
    "mode": "clone",
    "clone_speaker_id": "speaker_001"
}

response = requests.post(url, json=data)
```

#### 任务队列模式

```python
import requests
import time

# 提交任务
url = "http://localhost:8006/tasks/submit"
data = {
    "algorithm": "qwen3-tts",
    "params": {
        "text": "这是通过任务队列生成的语音。",
        "mode": "sft",
        "speaker_id": "Vivian"
    }
}

response = requests.post(url, json=data)
task_id = response.json()["task_id"]

# 轮询任务状态
while True:
    status_resp = requests.get(f"http://localhost:8006/tasks/{task_id}/status")
    status = status_resp.json()
    if status["status"] == "completed":
        # 下载音频
        audio_url = f"http://localhost:8006/tasks/{task_id}/download"
        audio = requests.get(audio_url)
        with open("output.wav", "wb") as f:
            f.write(audio.content)
        break
    elif status["status"] == "failed":
        print(f"任务失败: {status['error']}")
        break
    time.sleep(3)
```

#### 批量 TTS 生成（客户端脚本）

```bash
# 使用批量 TTS 客户端
python scripts/batch_tts_client.py \
    --input scripts/examples/sample_texts.csv \
    --algorithm qwen3-tts \
    --output ./batch_output/

# JSON 格式输入
python scripts/batch_tts_client.py \
    --input scripts/examples/sample_texts.json \
    --algorithm voxcpm \
    --concurrency 2 \
    --retry 3
```

---

## 📚 API 文档

### 主要 API 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/` | GET | 服务状态检查 |
| `/tts/{algorithm}` | POST | 文本转语音（11种算法） |
| `/tts/batch/generate` | POST | 批量 TTS 生成（2-100个） |
| `/tts/batch/download-zip` | POST | 批量下载 ZIP 打包 |
| `/tts/batch/config` | GET | 批量生成并发配置 |
| `/tasks/submit` | POST | 提交任务到队列 |
| `/tasks/batch/submit` | POST | 批量提交任务 |
| `/tasks/list` | GET | 获取任务列表 |
| `/tasks/{task_id}/status` | GET | 查询任务状态 |
| `/tasks/{task_id}/download` | GET | 下载任务结果 |
| `/tasks/{task_id}/cancel` | POST | 取消任务 |
| `/tasks/{task_id}/retry` | POST | 重试失败任务 |
| `/tasks/{task_id}` | DELETE | 删除任务 |
| `/tasks/queue/status` | GET | 队列状态（模型繁忙度等） |
| `/tasks/queue/clear-completed` | POST | 清理已完成任务 |
| `/algorithms` | GET | 获取支持的算法列表 |
| `/speakers` | GET/POST/DELETE | 说话人管理 |
| `/system/gpu-memory` | GET | GPU 显存状态 |
| `/system/cleanup` | POST | 手动触发显存清理 |
| `/system/models` | GET | 已加载模型列表 |
| `/concurrency/status` | GET | 并发状态查询 |
| `/concurrency/session` | GET | 会话信息 |
| `/concurrency/queue/wait-time` | GET | 预估等待时间 |
| `/concurrency/config` | GET | 并发配置参数 |

### 完整 API 文档

启动服务后访问: http://localhost:8006/docs

### 支持的 TTS 算法端点

| 端点 | 算法 | 支持模式 |
|------|------|----------|
| `/tts/qwen3-tts` | Qwen3-TTS | sft / voice_clone / voice_design / custom_voice |
| `/tts/voxcpm` | VoxCPM | base / voice_design / clone / ultimate_clone |
| `/tts/omnivoice` | OmniVoice | auto_voice / voice_clone / voice_design |
| `/tts/cosyvoice` | CosyVoice | zero_shot / instruct |
| `/tts/gptsovits` | GPT-SoVITS | clone（多个版本） |
| `/tts/pilottts` | PilotTTS | voice_clone / emotion / dialect / paralanguage |
| `/tts/chattts` | ChatTTS | random |
| `/tts/f5tts` | F5-TTS | clone |
| `/tts/openvoice` | OpenVoice | clone |
| `/tts/indextts` | IndexTTS | free / controlled |
| `/tts/fireredtts` | FireRedTTS2 | clone / random |

---

## 🔬 算法分析

各算法的详细分析文档已整理在对应目录：

| 算法 | 分析文档 | 核心架构 | 关键特点 |
|------|----------|----------|----------|
| **Qwen3-TTS** | [algorithms/Qwen3-TTS/readme.md](algorithms/Qwen3-TTS/readme.md) | 离散多码本LM | 97ms流式延迟，9种预设音色，支持声音设计 |
| **VoxCPM** | [algorithms/VoxCPM/readme.md](algorithms/VoxCPM/readme.md) | 无Tokenizer扩散自回归 | 30语言支持，4种模式（含极致克隆），基于MiniCPM-4 |
| **GPT-SoVITS** | [algorithms/GPT-SoVITS/readme.md](algorithms/GPT-SoVITS/readme.md) | VQ+GPT+VITS | 中文克隆之王，少样本即可高质量克隆，最新版V4 |
| **CosyVoice** | [algorithms/CosyVoice/readme.md](algorithms/CosyVoice/readme.md) | 流匹配+LLM | 18+方言支持，CosyVoice 3.0，Zero-shot+Instruct |
| **OmniVoice** | [algorithms/OmniVoice/readme.md](algorithms/OmniVoice/readme.md) | 扩散语言模型 | 600+语言，12种中文方言，语音设计 |
| **PilotTTS** | [algorithms/PilotTTS/readme.md](algorithms/PilotTTS/readme.md) | 指令驱动TTS | 11种情感+14种方言+副语言合成 |
| **ChatTTS** | [algorithms/ChatTTS/readme.md](algorithms/ChatTTS/readme.md) | 扩散+GPT | 对话优化，韵律丰富（已屏蔽前端） |
| **F5-TTS** | [algorithms/F5-TTS/readme.md](algorithms/F5-TTS/readme.md) | 流匹配+DiT | 高效快速，非自回归（已屏蔽前端） |
| **OpenVoice** | [algorithms/OpenVoice/readme.md](algorithms/OpenVoice/readme.md) | VAE+VITS | 音色风格解耦，V2升级（已屏蔽前端） |
| **IndexTTS** | [algorithms/IndexTTS/readme.md](algorithms/IndexTTS/readme.md) | 自回归GPT | 哔哩哔哩开源，情感可控（已屏蔽前端） |
| **FireRedTTS2** | [algorithms/FireRedTTS2/readme.md](algorithms/FireRedTTS2/readme.md) | 双Transformer | 长对话优化，7语言支持（已屏蔽前端） |

### 算法对比总结

```
┌──────────────┬──────────────┬──────────┬──────────┬──────────┬──────────────┐
│    算法      │   架构类型    │ 克隆质量  │ 生成速度  │ 模型大小  │ 中文自然度   │
├──────────────┼──────────────┼──────────┼──────────┼──────────┼──────────────┤
│ GPT-SoVITS   │ VQ+GPT+VITS  │   ★★★★★  │   ★★★    │  4.5GB   │   极佳       │
│ Qwen3-TTS    │ 多码本LM     │   ★★★★   │   ★★★★★  │  18GB    │   优秀       │
│ VoxCPM       │ 扩散自回归   │   ★★★★   │   ★★★    │  4.7GB   │   优秀       │
│ CosyVoice    │ 流匹配+LLM   │   ★★★★   │   ★★★    │  9.1GB   │   优秀       │
│ OmniVoice    │ 扩散LM       │   ★★★★   │   ★★★    │  3.1GB   │   优秀       │
│ PilotTTS     │ 指令驱动TTS  │   ★★★    │   ★★★    │  6.6GB   │   良好       │
│ ChatTTS      │ 扩散+GPT     │   ★★     │   ★★     │   中     │   良好       │
│ F5-TTS       │ 流匹配+DiT   │   ★★★    │   ★★★★   │   低     │   良好       │
│ OpenVoice    │ VAE+TTS      │   ★★     │   ★★★    │   低     │   一般       │
│ IndexTTS     │ GPT自回归    │   ★★★    │   ★★     │  6.2GB   │   良好       │
│ FireRedTTS2  │ 双Transformer│  ★★★    │   ★★     │  19.7GB  │   良好       │
└──────────────┴──────────────┴──────────┴──────────┴──────────┴──────────────┘
```

### 独立服务说明

由于不同 TTS 算法依赖不同版本的 transformers 库，采用独立进程架构解决依赖冲突：

| 服务 | 端口 | Transformers 版本 | 包含算法 | 模型大小 |
|------|------|-------------------|----------|----------|
| 主服务 | 8006 | 4.57.3 | Qwen3-TTS、VoxCPM | ~22.7GB |
| OmniVoice 服务 | 8007 | 5.x | OmniVoice | ~3.1GB |
| CosyVoice 服务 | 8008 | 4.51.3 | CosyVoice | ~9.1GB |
| PilotTTS 服务 | 8009 | 4.57.3 | PilotTTS | ~6.6GB |
| GPT-SoVITS 服务 | 8010 | 4.51.3 | GPT-SoVITS | ~4.5GB |

> 四个独立服务相互独立，可单独启停。通过 `restart-all` 可一键重启所有服务。主服务通过 HTTP API 与各独立服务通信。

---

## 🖥️ 前端功能

### 当前前端可用算法功能详情

#### Qwen3-TTS

| 功能 | 状态 | 说明 |
|------|------|------|
| SFT 模式（预设音色） | ✅ | 9种预设音色（Vivian、Dylan、Eric等） |
| 声音克隆 | ✅ | 3秒参考音频即可克隆 |
| 音色设计 | ✅ | 自然语言描述生成音色（不含方言） |
| 自定义音色 | ✅ | 选择预设音色+文本合成 |
| 流式生成 | ✅ | 97ms 低延迟首包 |
| 模型选择 | ✅ | 仅支持 1.7B 模型 |
| 方言支持 | ✅ | 北京话（Dylan）、四川话（Eric） |
| 批量生成 | ✅ | 2-100个，任务队列模式 |

#### VoxCPM

| 功能 | 状态 | 说明 |
|------|------|------|
| 基础生成 | ✅ | 默认音色随机生成 |
| 声音设计 | ✅ | 自然语言描述音色（支持方言） |
| 声音克隆 | ✅ | Reference-only 模式 |
| 极致克隆 | ✅ | Combined 模式（需参考文本） |
| 控制指令 | ✅ | 支持语速/情绪/风格控制 |
| 方言支持 | ✅ | 30种语言 + 9种方言 |
| 批量生成 | ✅ | 2-100个，任务队列模式 |

#### OmniVoice

| 功能 | 状态 | 说明 |
|------|------|------|
| 自动音色 | ✅ | 随机音色生成 |
| 声音克隆 | ✅ | 参考音频克隆 |
| 声音设计 | ✅ | 12种方言 + 性别 + 年龄组合 |
| 语速控制 | ✅ | 0.5-2.0x |
| 方言支持 | ✅ | 12种中文方言 |
| 批量生成 | ✅ | 2-100个，并行生成 |

#### CosyVoice

| 功能 | 状态 | 说明 |
|------|------|------|
| Zero-shot 克隆 | ✅ | 3-10秒参考音频克隆 |
| Instruct 指令 | ✅ | 自然语言控制说话风格 |
| 方言支持 | ✅ | 18+种中文方言（含粤语、闽南语） |
| 跨语言克隆 | ✅ | 支持多语言音色迁移 |
| 批量生成 | ✅ | 2-100个，任务队列模式 |

#### GPT-SoVITS

| 功能 | 状态 | 说明 |
|------|------|------|
| 声音克隆 | ✅ | 少样本高质量克隆 |
| 多版本支持 | ✅ | V2/V4 模型可选 |
| 中文效果 | ✅ | 中文克隆之王 |
| 批量生成 | ✅ | 2-100个，任务队列模式 |

#### PilotTTS

| 功能 | 状态 | 说明 |
|------|------|------|
| 声音克隆 | ✅ | 零样本声音克隆 |
| 情感合成 | ✅ | 11种情感（开心/悲伤/愤怒/惊讶/恐惧/厌恶/严肃/关切/忧郁/轻蔑/中立） |
| 方言合成 | ✅ | 14种中文方言 |
| 副语言合成 | ✅ | 笑声/哭声/呼吸/咳嗽 |
| 批量生成 | ✅ | 2-100个，任务队列模式 |

### 通用功能

- ✅ 模型选择卡片界面（6种前端可用算法）
- ✅ 文本输入与预览
- ✅ 说话人选择与管理（统一说话人管理模块）
- ✅ 自定义删除确认弹窗（红色主题、动画效果）
- ✅ 生成结果播放与下载
- ✅ 批量文本生成（2-100个）+ ZIP 打包下载
- ✅ 任务队列管理页面（实时状态仪表盘、模型繁忙度）
- ✅ 首页任务队列红点提示（实时显示等待/执行中任务数）
- ✅ 浏览器 Logo 图标
- ✅ 动态音波背景效果
- ✅ 服务状态监控

---

## 🔧 系统架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                 前端界面                                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────────────────────────────┐  │
│  │ 登录页面  │  │ 主应用页面 │  │  6个算法页面 + 任务队列页面 + 说话人管理  │  │
│  └──────────┘  └──────────┘  └──────────────────────────────────────────┘  │
└──────────────────────────────┬──────────────────────────────────────────────┘
                               │ HTTP
┌──────────────────────────────▼──────────────────────────────────────────────┐
│                     FastAPI 主服务（端口 8006）                                │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                         统一 API 接口层                                  │ │
│  │     /tts/*  /tasks/*  /speakers  /system  /concurrency                 │ │
│  └──────────────────────────────┬─────────────────────────────────────────┘ │
│                                 │ 算法调度层 + 任务队列 + 并发控制            │
│  ┌──────────────┬──────────────┐                                            │
│  │ Qwen3-TTS    │   VoxCPM     │  ← 本地 GPU 模型（主服务进程内）            │
│  │ 引擎         │   引擎       │                                            │
│  └──────────────┴──────────────┘                                            │
│                                                                             │
│  ┌──────────────┬──────────────┬──────────────┬──────────────┐              │
│  │ ChatTTS      │ F5-TTS       │ OpenVoice    │ IndexTTS     │              │
│  │ 引擎         │ 引擎         │ 引擎         │ 引擎         │              │
│  └──────────────┴──────────────┴──────────────┴──────────────┘              │
│  ┌──────────────┐                                                            │
│  │ FireRedTTS2  │                                                            │
│  │ 引擎         │                                                            │
│  └──────────────┘                                                            │
└──────────────────────────────┬──────────────────────────────────────────────┘
                               │ HTTP 通信
┌──────────────────────────────▼──────────────────────────────────────────────┐
│                           独立服务进程                                       │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐              │
│  │ OmniVoice 服务   │  │ CosyVoice 服务   │  │ PilotTTS 服务    │              │
│  │ 端口: 8007       │  │ 端口: 8008       │  │ 端口: 8009       │              │
│  │ Transformers 5.x │  │ Transformers     │  │ Transformers     │              │
│  │                  │  │ 4.51.3           │  │ 4.57.3           │              │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘              │
│           │                    │                    │                        │
│  ┌────────┴────────┐                                                         │
│  │ GPT-SoVITS 服务  │                                                         │
│  │ 端口: 8010       │                                                         │
│  │ Transformers     │                                                         │
│  │ 4.51.3           │                                                         │
│  └────────┬────────┘                                                         │
│           │                                                                  │
└───────────┼──────────────────────────────────────────────────────────────────┘
            │
    ┌───────┴──────────────────────────────────────┐
    │              models/ 模型文件目录               │
    │  Qwen3-TTS(18G) │ VoxCPM(4.7G) │ OmniVoice   │
    │  CosyVoice(9.1G)│ GPT-SoVITS(4.5G)│PilotTTS   │
    │  (6.6G)         │                │             │
    └──────────────────────────────────────────────┘
```

---

## 📝 开发记录

所有工作记录已按时间戳整理在 `records/` 目录（共 163 个记录文件）：

```bash
# 查看最新记录
ls -lt records/ | head -20
```

主要记录类型：
- 各算法部署调试记录
- API 开发与适配记录
- 前端开发与美化记录
- 问题排查与修复记录
- 功能验证与测试记录
- 模型下载与迁移记录
- 性能优化记录
- 并发控制与任务队列记录

最近更新（2026年7月）：
- GPT-SoVITS 更新到最新版本并重新部署（独立服务，端口 8010）
- 任务列表时间显示修复与取消任务逻辑完善
- 任务队列优化（模型级并发控制、批量进度显示、文件名规范化）
- PilotTTS 集成（11种情感 + 14种方言 + 副语言合成）
- 模型文件统一迁移到 models/ 目录

---

## ❓ 常见问题

### Q1: 为什么需要多个独立服务？

由于不同 TTS 算法依赖不同版本的 transformers 库，而 Python 无法在同一进程中加载多个版本的同一库：

| 服务 | Transformers 版本 | 原因 |
|------|-------------------|------|
| 主服务 | 4.57.3 | Qwen3-TTS、VoxCPM 等主流算法兼容版本 |
| OmniVoice 服务 | 5.x | OmniVoice 需要最新版 transformers |
| CosyVoice 服务 | 4.51.3 | CosyVoice 3.0 特定版本要求 |
| GPT-SoVITS 服务 | 4.51.3 | 最新版 GPT-SoVITS 需要兼容版本 |
| PilotTTS 服务 | 4.57.3 | 与主服务兼容，独立进程避免显存冲突 |

因此采用独立进程架构，各服务通过 HTTP API 通信。

### Q2: 如何启动完整服务？

```bash
# 一键启动所有服务（推荐）
./start_server.sh start-all                # 启动主服务 + 4个独立服务
./start_server.sh stop-all                 # 停止所有服务
./start_server.sh restart-all              # 重启所有服务

# 或单独控制各服务
./start_server.sh start                    # 主服务（端口 8006）
./start_server.sh start-omnivoice          # OmniVoice（端口 8007）
./start_server.sh start-cosyvoice          # CosyVoice（端口 8008）
./start_server.sh start-pilottts           # PilotTTS（端口 8009）
./start_server.sh start-gptsovits          # GPT-SoVITS（端口 8010）

# 查看日志
tail -f logs/server.log                    # 主服务日志
tail -f logs/omnivoice_service.log         # OmniVoice 服务日志
tail -f logs/cosyvoice_service.log         # CosyVoice 服务日志
tail -f logs/pilottts_service.log          # PilotTTS 服务日志
tail -f logs/gptsovits_service.log         # GPT-SoVITS 服务日志
```

### Q3: 模型文件存放位置？

所有模型文件统一存放在 `models/` 目录下：

```
models/
├── Qwen3-TTS/           # Qwen3-TTS 模型 (~18GB)
├── CosyVoice/           # CosyVoice 模型 (~9.1GB)
│   └── Fun-CosyVoice3-0.5B-2512/
├── PilotTTS/            # PilotTTS 模型 (~6.6GB)
│   ├── pilot_tts.pt / pilot_tts_instruct.pt
│   ├── Qwen3-0.6B / w2v-bert-2.0
│   └── Fun-CosyVoice3-0.5B
├── VoxCPM/              # VoxCPM 模型 (~4.7GB)
├── GPT-SoVITS/          # GPT-SoVITS 模型 (~4.5GB)
│   ├── chinese-hubert-base / chinese-roberta-wwm-ext-large
│   ├── gsv-v2final-pretrained / gsv-v4-pretrained
│   └── s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt
├── OmniVoice/           # OmniVoice 模型 (~3.1GB)
└── wenet/               # WeNet 模型
```

> **注意**: 模型文件较大，总计约 46GB。首次启动前需手动下载并放置到正确位置，或提前配置离线模式。

### Q4: 各算法支持哪些方言？

- **PilotTTS**（14种）：东北话、山东话、河南话、山西话、闽南语、甘肃话、宁夏话、上海话、重庆话、湖北话、湖南话、江西话、贵州话、云南话
- **CosyVoice**（18+种）：粤语、闽南语、四川话、东北话、河南话、陕西话、山东话、上海话、天津话、山西话、宁夏话、甘肃话、客家话、湖南话、湖北话、河北话、安徽话、江苏话
- **OmniVoice**（12种）：四川话、东北话、河南话、陕西话、云南话、贵州话、桂林话、甘肃话、宁夏话、济南话、青岛话、石家庄话
- **Qwen3-TTS**（2种）：北京话（Dylan预设音色）、四川话（Eric预设音色）
- **VoxCPM**：30种语言 + 9种方言

### Q5: 录音功能无法使用？

浏览器录音需要安全上下文（HTTPS 或 localhost）：

- ✅ `http://localhost:8006` — 支持录音
- ✅ `https://your-domain.com` — 支持录音（需配置 SSL 证书）
- ❌ `http://192.168.x.x:8006` — 不支持录音（浏览器安全限制）

> 配置 SSL 证书：在 `start_server.sh` 中设置 `SSL_CERT` 和 `SSL_KEY` 变量。

### Q6: 如何实现批量 TTS 生成？

**方式一：前端页面批量生成**
- 在各算法页面选择"批量生成"模式
- 输入文本列表或条数
- 任务自动提交到队列
- 完成后可逐个下载或 ZIP 打包下载

**方式二：批量 TTS 客户端脚本**
```bash
python scripts/batch_tts_client.py --input texts.csv --algorithm qwen3-tts
```

**方式三：任务队列 API**
```python
POST /tasks/batch/submit  # 批量提交
GET  /tasks/{task_id}/status  # 查询状态
GET  /tasks/{task_id}/download  # 下载结果
POST /tts/batch/download-zip  # ZIP打包下载
```

### Q7: 任务队列如何工作？

1. 提交任务到队列（支持优先级）
2. 后台异步执行器按模型并发控制策略调度
3. 本地 GPU 模型（Qwen3-TTS、VoxCPM）串行执行，HTTP 服务（2并发）
4. 任务完成后自动释放显存
5. 7天前的任务记录和音频自动清理

### Q8: 如何查看详细的模型信息和服务日志？

```bash
# 查看各服务日志中的模型信息
tail -f logs/server.log | grep "模型信息"            # 主服务
tail -f logs/omnivoice_service.log | grep "模型信息"  # OmniVoice
tail -f logs/cosyvoice_service.log | grep "模型信息"  # CosyVoice
tail -f logs/pilottts_service.log | grep "模型信息"   # PilotTTS
tail -f logs/gptsovits_service.log | grep "模型信息"  # GPT-SoVITS
```

日志内容包括：模型名称、版本、路径、支持的语言/方言、采样率等。

---

## 📄 许可证

本项目采用 Apache 2.0 许可证。

各 TTS 算法遵循其原始开源许可证：

| 算法 | 许可证 |
|------|--------|
| ChatTTS | AGPL-3.0 |
| CosyVoice | Apache 2.0 |
| F5-TTS | MIT |
| FireRedTTS2 | Apache 2.0 |
| GPT-SoVITS | MIT |
| IndexTTS | 自定义开源许可 |
| OpenVoice | MIT |
| Qwen3-TTS | Apache 2.0 |
| VoxCPM | Apache 2.0 |
| OmniVoice | Apache 2.0 |
| PilotTTS | Apache 2.0 |

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

**更新时间**: 2026-07-19
