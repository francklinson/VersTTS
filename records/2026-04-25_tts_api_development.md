# TTS API 开发记录

## 日期
2026-04-25

## 工作内容

### 1. 统一后端API服务开发

创建了 `/home/zhouchenghao/PycharmProjects/VersTTS/backend/api_server.py`,实现以下功能:

#### 支持的TTS模型
- **ChatTTS**: 开源对话式TTS,支持情感表达
- **CosyVoice**: 阿里开源,支持多语言和声音克隆
- **F5-TTS**: 高效流匹配TTS,快速推理
- **Qwen3-TTS**: 通义千问TTS,强大的语音克隆能力
- **OpenVoice**: 即时声音克隆,多语言支持

#### API端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/` | GET | 服务信息 |
| `/health` | GET | 健康检查 |
| `/tts/chattts` | POST | ChatTTS语音合成 |
| `/tts/cosyvoice` | POST | CosyVoice语音合成 |
| `/tts/f5tts` | POST | F5-TTS语音合成 |
| `/tts/qwen3tts` | POST | Qwen3-TTS语音合成 |
| `/tts/openvoice` | POST | OpenVoice语音合成 |
| `/audio/{filename}` | GET | 获取音频文件 |

#### 特性
- 模型懒加载: 首次请求时加载对应模型,节省内存
- 多模式支持: 支持基础合成、声音克隆、跨语言等模式
- 参数可调: 温度、Top P、语速等参数可配置
- 文件上传: 支持参考音频上传进行声音克隆
- 输出格式: 支持URL和Base64两种输出格式

### 2. 前端界面开发

创建了 `/home/zhouchenghao/PycharmProjects/VersTTS/frontend/index.html`,包含:

#### 功能特性
- 模型选择卡片界面
- 各模型专属参数配置面板
- 文本输入区域
- 音频播放和下载
- 服务健康状态指示器
- 响应式设计,支持移动端

#### 支持的参数配置

**ChatTTS:**
- 温度 (0-1)
- Top P (0-1)
- Top K (1-100)

**CosyVoice:**
- 模式: SFT / Zero-shot / 跨语言 / 指令控制
- 说话人选择
- 参考音频上传
- 指令文本

**F5-TTS:**
- 参考音频上传 (必填)
- 参考文本 (必填)
- NFE步数
- CFG强度
- 语速

**Qwen3-TTS:**
- 模型大小: 0.6B / 1.7B
- 模式: 基础合成 / 声音克隆
- 参考音频上传

**OpenVoice:**
- 语言: 中文 / 英文
- 风格: 默认 / 低语
- 语速
- 参考音频 (可选)

### 3. 服务启动脚本

创建了 `/home/zhouchenghao/PycharmProjects/VersTTS/start_server.py`:
- 环境检查 (Python版本、CUDA、依赖包)
- 目录结构初始化
- 服务启动管理

## 使用方法

### 1. 启动服务

```bash
# 使用虚拟环境
source .venv/bin/activate

# 启动服务
python start_server.py

# 指定端口
python start_server.py --port 8080

# 开发模式(自动重载)
python start_server.py --reload
```

### 2. API调用示例

```bash
# ChatTTS
curl -X POST "http://localhost:8000/tts/chattts" \
  -F "text=你好,这是测试" \
  -F "temperature=0.3"

# CosyVoice
curl -X POST "http://localhost:8000/tts/cosyvoice" \
  -F "text=你好,这是测试" \
  -F "mode=sft" \
  -F "speaker_id=中文女"

# F5-TTS (需要参考音频)
curl -X POST "http://localhost:8000/tts/f5tts" \
  -F "text=你好,这是测试" \
  -F "ref_text=参考文本" \
  -F "ref_wav=@reference.wav"
```

### 3. 访问前端界面

打开浏览器访问: `http://localhost:8000`

## 项目结构

```
VersTTS/
├── backend/
│   └── api_server.py       # 统一API服务
├── frontend/
│   └── index.html          # Web界面
├── records/
│   └── *.md                # 工作记录
├── start_server.py         # 启动脚本
├── ChatTTS/                # ChatTTS项目
├── CosyVoice/              # CosyVoice项目
├── F5-TTS/                 # F5-TTS项目
├── Qwen3-TTS/              # Qwen3-TTS项目
├── OpenVoice/              # OpenVoice项目
└── GPT-SoVITS/             # GPT-SoVITS项目 (待调试)
```

## 待完成事项

1. **GPT-SoVITS**: 模型调试尚未完成,需要进一步处理
2. **性能优化**: 考虑添加模型预热和缓存机制
3. **错误处理**: 完善各模型的错误处理逻辑
4. **测试**: 编写单元测试和集成测试

## 技术栈

- **后端**: FastAPI, Uvicorn, PyTorch
- **前端**: HTML5, CSS3, JavaScript (原生)
- **模型**: ChatTTS, CosyVoice, F5-TTS, Qwen3-TTS, OpenVoice

## 备注

- 所有模型文件已按项目需求下载到对应文件夹,未使用.cache
- CUDA支持已配置,优先使用GPU进行推理
- 中文语音生成为主要目标,所有模型均支持中文
