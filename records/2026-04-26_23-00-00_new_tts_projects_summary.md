# 新增TTS算法部署工作记录 - 总结

**时间:** 2026-04-26 23:00:00

## 任务完成情况

### 已完成的任务

#### [1] 项目克隆 ✅
- ✅ VoxCPM - https://github.com/OpenBMB/VoxCPM/
- ✅ IndexTTS - https://github.com/index-tts/index-tts
- ✅ FireRedTTS2 - https://github.com/FireRedTeam/FireRedTTS2

#### [2] 虚拟环境创建 ✅
- ✅ .venv_voxcpm
- ✅ .venv_indextts
- ✅ .venv_fireredtts

#### [3] 项目分析与文档整理 ✅
- ✅ VoxCPM 算法分析与验证文档 - algorithms/VoxCPM/readme.md
- ✅ IndexTTS 算法分析与验证文档 - algorithms/IndexTTS/readme.md
- ✅ FireRedTTS2 算法分析与验证文档 - algorithms/FireRedTTS2/readme.md

### 进行中的任务

#### [4] 环境配置与依赖安装 ⏳
- ⏳ PyTorch安装遇到版本兼容性问题
- ⏳ 需要解决依赖版本匹配

## 遇到的问题

### 1. PyTorch版本兼容性问题
VoxCPM环境安装时出现以下错误：
```
ERROR: Could not find a version that satisfies the requirement torchvision==0.20.1
ERROR: No matching distribution found for torchvision==0.20.1
```

解决方案：需要检查正确的torchvision版本号，可能是0.20.1+cu121

### 2. 网络连接问题
FireRedTTS2环境安装时出现以下错误：
```
ERROR: Could not find a version that satisfies the requirement fsspec (from torch)
ERROR: No matching distribution found for fsspec
```

解决方案：可能需要使用清华镜像或阿里镜像

### 3. IndexTTS依赖管理
IndexTTS要求使用uv包管理器，但系统环境限制需要特殊处理

## 三个新TTS项目概览

### 1. VoxCPM
- **核心特点**: 无tokenizer设计，直接生成连续语音表示
- **模型大小**: 2B参数
- **支持语言**: 30种语言 + 9种中文方言
- **特色功能**: 声音设计、可控克隆、极致克隆、48kHz输出
- **适用场景**: 高质量多语言TTS、声音克隆、播客制作

### 2. IndexTTS
- **核心特点**: 首个支持精确时长控制的自回归TTS
- **模型版本**: IndexTTS-2 (最新)
- **支持语言**: 中英文为主
- **特色功能**: 情感控制、时长控制、情感-音色解耦
- **适用场景**: 视频配音、需要精确时长的场景、情感丰富的语音

### 3. FireRedTTS2
- **核心特点**: 专注于长对话语音生成
- **模型架构**: 双Transformer + 12.5Hz流式分词器
- **支持语言**: 英/中/日/韩/法/德/俄
- **特色功能**: 3分钟长对话、多说话人、流式生成、140ms首包延迟
- **适用场景**: 播客生成、有声书、客服系统、对话数据生成

## 项目能力对比

| 能力 | VoxCPM | IndexTTS | FireRedTTS2 |
|------|--------|----------|-------------|
| 基础TTS | ✅ | ✅ | ✅ |
| 零样本克隆 | ✅ | ✅ | ✅ |
| 跨语言克隆 | ✅ | ✅ | ✅ |
| 声音设计 | ✅ | ❌ | ❌ |
| 情感控制 | 文本描述 | 精确控制 | 有限支持 |
| 时长控制 | ❌ | ✅ | ❌ |
| 流式生成 | ✅ | ❌ | ✅ |
| 多说话人对话 | ❌ | ❌ | ✅ |
| 输出采样率 | 48kHz | 通常24kHz | 24kHz |
| 支持语言数 | 30种 | 多种 | 7种 |

## 下一步工作计划

### 待完成任务

#### [1] 环境配置修复
```bash
# VoxCPM - 使用正确版本
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# FireRedTTS2 - 使用清华镜像
pip install torch==2.7.1+cu126 torchvision==0.22.1+cu126 torchaudio==2.7.1+cu126 \
    --index-url https://download.pytorch.org/whl/cu126 -i https://pypi.tuna.tsinghua.edu.cn/simple

# IndexTTS - 使用uv安装
cd algorithms/IndexTTS
uv sync --all-extras
```

#### [2] 模型下载
- 下载VoxCPM2模型 (约2B参数)
- 下载IndexTTS-2模型
- 下载FireRedTTS2模型

#### [3] API接口开发
- 为每个项目开发FastAPI接口
- 集成到后端api_server.py
- 添加健康检查和状态监控

#### [4] 前端集成
- 在app.html添加三个新模型的选择卡片
- 为每个模型添加参数配置界面
- 测试前端与后端API的交互

#### [5] 功能测试
- 测试每个模型的基础TTS功能
- 测试声音克隆功能
- 测试特殊功能(声音设计、情感控制、对话生成)

#### [6] 前端优化
- 根据TTS方案拆分前端文件
- 添加动态效果(粒子、声波等)
- 优化页面加载速度

## 文件变更清单

### 新增文件
1. `algorithms/VoxCPM/` - VoxCPM项目代码
2. `algorithms/IndexTTS/` - IndexTTS项目代码
3. `algorithms/FireRedTTS2/` - FireRedTTS2项目代码
4. `algorithms/VoxCPM/readme.md` - VoxCPM分析文档
5. `algorithms/IndexTTS/readme.md` - IndexTTS分析文档
6. `algorithms/FireRedTTS2/readme.md` - FireRedTTS2分析文档
7. `records/2026-04-26_22-00-00_new_tts_projects_deployment.md` - 部署记录
8. `records/2026-04-26_23-00-00_new_tts_projects_summary.md` - 本总结文档

### 虚拟环境
1. `.venv_voxcpm/` - VoxCPM虚拟环境
2. `.venv_indextts/` - IndexTTS虚拟环境
3. `.venv_fireredtts/` - FireRedTTS虚拟环境

## 备注

- 所有项目都符合需求.txt的要求：放在algorithms文件夹下
- 每个项目都有独立的虚拟环境
- 分析文档已按照需求整理核心思路/用法/局限性
- 前端拆分和美化工作待后续完成
- 环境配置问题需要进一步解决
