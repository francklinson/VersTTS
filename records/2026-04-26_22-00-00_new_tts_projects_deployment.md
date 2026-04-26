# 新增TTS算法部署工作记录

**时间:** 2026-04-26 22:00:00

## 任务概述
根据需求.txt，部署三个新的TTS算法项目：
1. VoxCPM - https://github.com/OpenBMB/VoxCPM/
2. IndexTTS - https://github.com/index-tts/index-tts
3. FireRedTTS2 - https://github.com/FireRedTeam/FireRedTTS2

## 已完成工作

### [1] 项目克隆
- ✅ VoxCPM 克隆成功
- ✅ IndexTTS 克隆成功 (第二次尝试成功)
- ✅ FireRedTTS2 克隆成功

### [2] 虚拟环境创建
- ✅ 创建 .venv_voxcpm
- ✅ 创建 .venv_fireredtts
- ✅ 创建 .venv_indextts

### [3] PyTorch安装 (进行中)
- ⏳ VoxCPM: PyTorch 2.5.1 + CUDA 12.1 安装中
- ⏳ FireRedTTS2: PyTorch 2.7.1 + CUDA 12.6 安装中

## 项目分析

### VoxCPM
- **类型**: 无tokenizer TTS系统
- **核心特点**: 
  - 直接生成连续语音表示
  - 端到端扩散自回归架构
  - 支持30种语言
  - 声音设计功能
  - 可控声音克隆
  - 48kHz高质量音频输出
- **模型大小**: 2B参数
- **训练数据**: 超过200万小时多语言语音数据
- **依赖**: Python ≥ 3.10, PyTorch ≥ 2.5.0, CUDA ≥ 12.0
- **安装**: `pip install voxcpm`

### IndexTTS
- **类型**: 自回归零样本TTS
- **核心特点**:
  - 精确的语音时长控制
  - 情感表达控制
  - 说话人身份解耦
  - 支持可控和不可控两种生成模式
- **模型版本**: IndexTTS-2 (最新)
- **依赖管理**: 使用 uv 包管理器
- **安装**: `uv sync --all-extras`

### FireRedTTS2
- **类型**: 长对话语音生成系统
- **核心特点**:
  - 支持多说话人对话生成
  - 3分钟长对话支持
  - 多语言支持 (英/中/日/韩/法/德/俄)
  - 超低延迟 (首包延迟140ms)
  - 流式生成
  - 随机音色生成
- **模型架构**: 双Transformer架构
- **分词器**: 12.5Hz流式语音分词器
- **依赖**: PyTorch 2.7.1, transformers, einops等

## 下一步工作
1. 完成PyTorch安装
2. 安装各项目依赖
3. 下载模型文件
4. 编写API接口
5. 测试各项目功能
6. 编写readme文档

## 遇到的问题
1. IndexTTS克隆时遇到TLS连接问题，第二次尝试成功
2. PyTorch下载时出现SSL错误，正在重试

## 备注
- 所有项目都放在 algorithms/ 文件夹下
- 每个项目使用独立的虚拟环境
- 需要确保CUDA版本兼容性
