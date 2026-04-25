# VersTTS - 统一语音合成平台

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/CUDA-支持-green.svg" alt="CUDA Support">
  <img src="https://img.shields.io/badge/支持-6种TTS算法-orange.svg" alt="6 TTS Algorithms">
  <img src="https://img.shields.io/badge/许可-Apache%202.0-yellow.svg" alt="License">
</p>

VersTTS 是一个统一的语音合成（Text-to-Speech）平台，集成了业界领先的 6 种开源 TTS 算法，提供统一的后端 API 服务和美观的前端交互界面，支持音色克隆、预设音色、语音设计等多种语音合成能力。

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
- [开发记录](#-开发记录)
- [常见问题](#-常见问题)

---

## 🎯 项目简介

### 核心功能

- **多算法集成**: 一键切换 6 种主流 TTS 算法
- **音色克隆**: 支持使用参考音频进行声音克隆
- **预设音色**: 提供多种优质预设人声
- **语音设计**: 通过自然语言描述生成特定音色
- **批量处理**: 支持批量文本上传和音频下载
- **Web界面**: 美观的现代化 Web 交互界面
- **统一API**: RESTful API 设计，易于集成

### 技术特点

- 基于 FastAPI 的高性能后端服务
- 支持 CUDA 加速推理
- 模型文件本地化管理（非 .cache 目录）
- 完善的日志系统
- 内置登录认证（默认: admin / tp123456）

---

## 🎙️ 支持的TTS算法

| 算法 | 能力 | 音色克隆 | 预设音色 | 指令控制 | 流式 | 中文效果 |
|------|------|----------|----------|----------|------|----------|
| **Qwen3-TTS** | 全功能 | ✅ (3秒) | 9种 | ✅ | ✅ (97ms) | 优秀 |
| **GPT-SoVITS** | 克隆专用 | ✅ 优秀 | ❌ | ❌ | ❌ | 优秀 |
| **CosyVoice** | 全功能 | ✅ | ✅ | 有限 | ✅ | 优秀 |
| **ChatTTS** | 对话优化 | ✅ | ❌ | 有限 | ❌ | 优秀 |
| **F5-TTS** | 克隆专用 | ✅ | ❌ | 有限 | ❌ | 良好 |
| **OpenVoice** | 克隆专用 | ✅ | ❌ | ❌ | ❌ | 良好 |

### 算法选择建议

- **需要流式低延迟**: 选择 Qwen3-TTS、CosyVoice
- **需要音色设计**: 选择 Qwen3-TTS (VoiceDesign)
- **最佳克隆效果**: 选择 GPT-SoVITS
- **对话场景**: 选择 ChatTTS
- **轻量部署**: 选择 F5-TTS、OpenVoice

---

## 📁 项目结构

```
VersTTS/
├── algorithms/              # 六种TTS算法目录
│   ├── ChatTTS/            # ChatTTS 项目
│   ├── CosyVoice/          # CosyVoice 项目
│   ├── F5-TTS/             # F5-TTS 项目
│   ├── GPT-SoVITS/         # GPT-SoVITS 项目
│   ├── OpenVoice/          # OpenVoice 项目
│   └── Qwen3-TTS/          # Qwen3-TTS 项目
│
├── backend/                 # 后端服务
│   ├── api_server.py       # 统一API服务主文件
│   ├── logger_config.py    # 日志配置
│   └── ...
│
├── frontend/                # 前端界面
│   ├── app.html            # 主应用页面
│   ├── login.html          # 登录页面
│   └── ...
│
├── test_scripts/            # 测试脚本
│   ├── test_all_tts.py     # 统一测试入口
│   ├── test_chattts.py
│   ├── test_cosyvoice.py
│   ├── test_f5_tts.py
│   ├── test_gpt_sovits.py
│   ├── test_openvoice.py
│   └── test_qwen3_tts.py
│
├── logs/                    # 日志文件目录
├── records/                 # 工作记录文档
├── models/                  # 模型文件存放目录
├── requirements.txt         # Python依赖
└── readme.md               # 项目说明文档
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

# 其他模型请参考各算法目录下的 readme.md
```

### 5. 启动服务

```bash
# 启动后端API服务
python backend/api_server.py

# 或启动前端页面 (直接打开 frontend/login.html 或 frontend/app.html)
```

### 6. 访问系统

- 前端界面: 打开 `frontend/login.html`
- 默认账号: `admin` / `tp123456`
- API文档: http://localhost:8000/docs

---

## 📖 安装部署

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
| ChatTTS | [algorithms/ChatTTS/README.md](algorithms/ChatTTS/README.md) |
| CosyVoice | [algorithms/CosyVoice/README.md](algorithms/CosyVoice/README.md) |
| F5-TTS | [algorithms/F5-TTS/README.md](algorithms/F5-TTS/README.md) |
| GPT-SoVITS | [algorithms/GPT-SoVITS/README.md](algorithms/GPT-SoVITS/README.md) |
| OpenVoice | [algorithms/OpenVoice/README.md](algorithms/OpenVoice/README.md) |
| Qwen3-TTS | [algorithms/Qwen3-TTS/README.md](algorithms/Qwen3-TTS/README.md) |

#### 步骤4: 验证安装

```bash
# 运行测试脚本
python test_scripts/test_all_tts.py
```

---

## 💡 使用指南

### Web界面使用

1. **登录系统**: 访问 `frontend/login.html`，使用默认账号登录
2. **选择算法**: 在主页面选择想要使用的 TTS 算法
3. **输入文本**: 在文本框中输入要合成的内容
4. **配置参数**:
   - 选择预设音色（如支持）
   - 上传参考音频进行克隆（如支持）
   - 调整生成参数
5. **生成语音**: 点击生成按钮，等待结果
6. **批量处理**: 支持批量上传文本文件，批量下载生成的音频

### API调用示例

```python
import requests

# API端点
url = "http://localhost:8000/tts"

# 请求参数
data = {
    "algorithm": "qwen3-tts",
    "text": "你好，这是语音合成测试。",
    "speaker": "Vivian",  # 预设音色
    "language": "Chinese"
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

url = "http://localhost:8000/tts/clone"

# 上传参考音频进行克隆
files = {
    "ref_audio": open("reference.wav", "rb")
}
data = {
    "algorithm": "gpt-sovits",
    "text": "使用参考音频的声音说这句话。",
    "ref_text": "参考音频对应的文本内容"
}

response = requests.post(url, files=files, data=data)
```

---

## 📚 API文档

### 主要API端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/` | GET | 服务状态检查 |
| `/tts` | POST | 文本转语音（预设音色） |
| `/tts/clone` | POST | 音色克隆 |
| `/algorithms` | GET | 获取支持的算法列表 |
| `/speakers/{algorithm}` | GET | 获取指定算法的预设音色 |

### 请求参数说明

#### TTS请求 (POST /tts)

```json
{
  "algorithm": "qwen3-tts",      // 算法名称
  "text": "要合成的文本",         // 必填
  "speaker": "Vivian",           // 预设音色（可选）
  "language": "Chinese",         // 语言（可选，默认Auto）
  "instruct": "语气描述",         // 指令控制（可选）
  "speed": 1.0,                  // 语速（可选）
  "temperature": 0.9             // 随机性（可选）
}
```

#### 音色克隆请求 (POST /tts/clone)

```json
{
  "algorithm": "gpt-sovits",
  "text": "要合成的文本",
  "ref_text": "参考音频的文本内容"  // 部分算法需要
}
```

同时需要在 FormData 中上传 `ref_audio` 文件。

完整 API 文档可在启动服务后访问: http://localhost:8000/docs

---

## 🔬 算法分析

各算法的详细分析文档已整理在对应目录：

| 算法 | 分析文档 | 核心特点 |
|------|----------|----------|
| **ChatTTS** | [algorithms/ChatTTS/readme.md](algorithms/ChatTTS/readme.md) | 扩散+GPT架构，对话优化 |
| **Qwen3-TTS** | [algorithms/Qwen3-TTS/readme.md](algorithms/Qwen3-TTS/readme.md) | 离散多码本LM，97ms流式延迟 |

### 算法对比总结

```
┌─────────────┬────────────┬──────────┬──────────┬──────────┐
│   算法      │  架构类型   │ 克隆质量  │ 生成速度  │ 内存占用  │
├─────────────┼────────────┼──────────┼──────────┼──────────┤
│ Qwen3-TTS   │ 多码本LM   │    ★★★   │   ★★★   │   高     │
│ GPT-SoVITS  │ VQ+GPT     │    ★★★★  │   ★★    │   中     │
│ CosyVoice   │ 流匹配+LM  │    ★★★   │   ★★★   │   高     │
│ ChatTTS     │ 扩散+GPT   │    ★★    │   ★★    │   中     │
│ F5-TTS      │ 流匹配     │    ★★★   │   ★★★   │   低     │
│ OpenVoice   │ VAE+TTS    │    ★★    │   ★★★   │   低     │
└─────────────┴────────────┴──────────┴──────────┴──────────┘
```

---

## 📝 开发记录

所有工作记录已按时间戳整理在 `records/` 目录：

- 项目结构整理记录
- 各算法部署调试记录
- API开发记录
- 前端开发记录
- 问题排查记录

查看记录文件：
```bash
ls -la records/
```

---

## ❓ 常见问题

### Q1: 启动时提示 CUDA out of memory

**A**: 尝试以下方法：
1. 使用较小的模型（如 0.6B 版本）
2. 减少 batch size
3. 使用 `device_map="auto"` 让模型自动分配显存
4. 关闭其他占用显存的程序

### Q2: 模型文件应该放在哪里？

**A**: 模型文件统一放在 `models/` 目录下，按算法分子目录存放，不要放在 `.cache` 目录。

### Q3: 如何使用 CPU 运行？

**A**: 修改代码中的 `device_map` 参数为 `"cpu"`，但生成速度会较慢。

### Q4: 前端无法登录？

**A**: 检查：
1. 默认账号密码是否正确
2. 浏览器 localStorage 是否正常
3. 尝试清除浏览器缓存

### Q5: 音色克隆效果不佳？

**A**: 建议：
1. 使用 3-10 秒的高质量参考音频
2. 参考音频应只包含单人声音
3. 避免背景噪音
4. 部分算法（如 ChatTTS）的克隆效果本身较弱

---

## 🔧 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                        前端界面                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │  登录页面    │  │  主应用页面  │  │    Web录音功能      │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└────────────────────────┬────────────────────────────────────┘
                         │ HTTP/WebSocket
┌────────────────────────▼────────────────────────────────────┐
│                    FastAPI 后端服务                          │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              统一API接口层                           │   │
│  │   /tts  /tts/clone  /algorithms  /speakers/...     │   │
│  └────────────────────┬────────────────────────────────┘   │
│                       │ 算法调度层                          │
│  ┌─────────┬─────────┬─────────┬─────────┬─────────┐      │
│  │ChatTTS  │CosyVoice│ F5-TTS  │GPT-SoVITS│OpenVoice│      │
│  │ 适配器  │ 适配器  │ 适配器  │ 适配器  │ 适配器  │      │
│  └────┬────┴────┬────┴────┬────┴────┬────┴────┬────┘      │
│       │         │         │         │         │            │
└───────┼─────────┼─────────┼─────────┼─────────┼────────────┘
        │         │         │         │         │
   ┌────┴────┐ ┌─┴────┐ ┌──┴───┐ ┌───┴───┐ ┌───┴────┐
   │ ChatTTS │ │CosyVo│ │F5-TTS│ │GPT-SoV│ │OpenVoic│
   │  模型   │ │模型  │ │ 模型 │ │模型   │ │模型    │
   └─────────┘ └──────┘ └──────┘ └───────┘ └────────┘
```

---

## 📄 许可证

本项目采用 Apache 2.0 许可证。

各 TTS 算法遵循其原始开源许可证：
- ChatTTS: AGPL-3.0
- CosyVoice: Apache 2.0
- F5-TTS: MIT
- GPT-SoVITS: MIT
- OpenVoice: MIT
- Qwen3-TTS: Apache 2.0

---

## 🤝 贡献指南

欢迎提交 Issue 和 PR！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request
