# GPT-SoVITS 集成工作记录

## 日期
2026-04-25

## 任务
调试GPT-SoVITS,构建后端API服务,加入前端选择

## 完成内容

### 1. 环境准备
- 创建必要的目录结构: `reference_audio`, `uploads`
- 创建模型文件符号链接到正确位置
- 确认GPT-SoVITS模型文件位置:
  - s2G2333k.pth (SoVITS权重)
  - s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt (GPT权重)
  - chinese-hubert-base
  - chinese-roberta-wwm-ext-large

### 2. 后端API集成
修改文件: `backend/api_server.py`

**新增内容:**
- 添加GPT-SoVITS到项目路径
- 新增 `GPTSoVITSRequest` 数据模型，包含参数:
  - text_lang: 文本语言
  - ref_audio_path: 参考音频路径
  - prompt_text: 参考音频文本
  - prompt_lang: 参考音频语言
  - top_k, top_p, temperature: 采样参数
  - text_split_method: 文本分割方法
  - batch_size: 批处理大小
  - speed_factor: 语速因子
  - version: 模型版本(v1/v2/v3/v4/v2Pro/v2ProPlus)

- 新增模型加载函数:
  - `get_gpt_sovits_model()`: 获取或加载GPT-SoVITS模型配置
  - `init_gpt_sovits_pipeline()`: 初始化推理管道

- 新增API端点: `POST /tts/gptsovits`
  - 必须提供参考音频文件(prompt_wav)
  - 必须提供参考音频文本(prompt_text)
  - 支持版本选择和高级参数调节
  - 完整的日志记录和错误处理

### 3. 前端界面集成
修改文件: `frontend/index.html`

**新增内容:**
- 添加GPT-SoVITS模型卡片(青色主题🗣️)
- 新增GPT-SoVITS专用选项面板，包含:
  - 文本语言选择(中/英/日/韩/粤)
  - 参考音频语言选择
  - 模型版本选择(V1/V2/V3/V4/V2Pro/V2ProPlus)
  - 参考音频上传(必填)
  - 参考音频文本输入(必填)
  - 高级参数: Top K, Top P, 温度, 语速

- JavaScript处理逻辑:
  - 模型选择切换
  - 必填字段验证
  - 表单数据构建

### 4. API端点更新
根端点 `/` 现在返回的端点列表包含:
- /tts/chattts
- /tts/cosyvoice
- /tts/f5tts
- /tts/qwen3tts
- /tts/openvoice
- /tts/gptsovits (新增)

API版本更新为 1.1.0

## 技术要点

### GPT-SoVITS特性
- **少样本声音克隆**: 仅需5-10秒参考音频即可克隆声音
- **多语言支持**: 支持中/英/日/韩/粤等多种语言
- **跨语言合成**: 可以用中文参考音频合成英文语音
- **版本选择**: V2为推荐版本, V3/V4为实验版本

### 与现有项目的区别
| 项目 | 克隆能力 | 语言支持 | 特点 |
|------|---------|---------|------|
| ChatTTS | 否 | 中英 | 情感表达丰富 |
| CosyVoice | 是 | 多语言 | 阿里开源 |
| F5-TTS | 是 | 中英 | 流匹配,速度快 |
| Qwen3-TTS | 是 | 中英 | 通义千问 |
| OpenVoice | 是 | 多语言 | 即时克隆 |
| GPT-SoVITS | 是 | 多语言 | 少样本学习 |

## 后续计划
1. 测试GPT-SoVITS API功能
2. 收集参考人声音频样本
3. 优化前端用户体验
4. 添加批量处理功能
