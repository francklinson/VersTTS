# 项目代码整理与Git管理记录

## 日期
2026-04-25

## 任务
将项目中实际需要用到的代码文件加入git，有一些是原项目做示例的，可以不用加入git

## 完成内容

### 1. 创建 .gitignore 文件

**文件位置:** `.gitignore`

**主要规则:**
- Python缓存文件(__pycache__, *.pyc)
- 虚拟环境(.venv/)
- IDE配置(.idea/, .vscode/)
- 模型文件(*.pth, *.ckpt, *.safetensors, *.onnx)
- 输出目录(output/)
- 上传目录(uploads/)
- 日志目录(logs/)
- 临时文件

**例外规则:**
- `!reference_audio/**/*.wav` - 保留参考音频

### 2. 创建项目结构文档

**文件位置:** `PROJECT_STRUCTURE.md`

**包含内容:**
- 完整目录结构说明
- 核心文件说明
- Git管理规范
- 提交信息规范
- 开发流程指南
- API端点列表

### 3. 应该加入Git的文件清单

#### ✅ 后端代码
```
backend/
├── __init__.py              (待创建)
├── api_server.py            ✅ 已修改 - 添加GPT-SoVITS和批量处理
├── batch_processor.py       ✅ 新增 - 批量处理模块
└── logger_config.py         ✅ 已有 - 日志配置
```

#### ✅ 前端代码
```
frontend/
└── index.html               ✅ 已修改 - 添加GPT-SoVITS支持
```

#### ✅ 工具脚本
```
tools/
└── validate_references.py   ✅ 新增 - 参考音频验证工具
```

#### ✅ 参考音频管理
```
reference_audio/
├── README.md                ✅ 新增 - 管理指南
├── metadata.json            ✅ 新增 - 元数据配置
├── children/                ✅ 目录(保留结构)
├── teenagers/               ✅ 目录(保留结构)
└── adults/                  ✅ 目录(保留结构)
```

#### ✅ 文档和配置
```
├── requirements.txt         ✅ 新增 - 项目依赖
├── .gitignore              ✅ 新增 - Git忽略配置
├── PROJECT_STRUCTURE.md    ✅ 新增 - 项目结构文档
└── start_server.py         ✅ 已有 - 启动脚本
```

#### ✅ 工作记录
```
records/
├── 2026-04-25_gptsovits_integration.md      ✅ 新增
├── 2026-04-25_tts_capabilities_analysis.md  ✅ 新增
├── 2026-04-25_batch_processing.md           ✅ 新增
├── 2026-04-25_reference_audio_setup.md      ✅ 新增
└── 2026-04-25_project_git_setup.md          ✅ 新增
```

### 4. 不应该加入Git的文件

#### ❌ 模型文件(大文件)
```
*.pth
*.ckpt
*.safetensors
*.bin
*.pt
*.onnx
ChatTTS/models/asset/
CosyVoice/models/iic/
F5-TTS/models/
Qwen3-TTS/models/
OpenVoice/checkpoints_*/
GPT-SoVITS/GPT_SoVITS/pretrained_models/
GPT-SoVITS/GPT_SoVITS/text/XXXXRT/
```

#### ❌ 生成文件
```
output/
uploads/*
logs/
*.wav
*.mp3
*.ogg
```

#### ❌ 环境文件
```
.venv/
__pycache__/
.idea/
.vscode/
```

#### ❌ 子项目中的示例和文档(可选)
```
ChatTTS/docs/
ChatTTS/examples/
ChatTTS/tests/
CosyVoice/examples/
F5-TTS/src/f5_tts/train/
F5-TTS/src/f5_tts/eval/
OpenVoice/demo*.ipynb
```

### 5. Git提交建议

#### 建议的提交分组

**提交1: 后端API更新**
```bash
git add backend/api_server.py
git commit -m "[Feature] 添加GPT-SoVITS和批量处理API支持

- 集成GPT-SoVITS模型到统一API
- 添加批量TTS处理端点
- 更新API版本到1.2.0"
```

**提交2: 批量处理模块**
```bash
git add backend/batch_processor.py
git commit -m "[Feature] 实现批量TTS处理模块

- 支持CSV/JSON/TXT格式文本上传
- 实现任务状态管理和进度追踪
- 支持ZIP包下载结果"
```

**提交3: 前端更新**
```bash
git add frontend/index.html
git commit -m "[Feature] 前端添加GPT-SoVITS支持

- 添加GPT-SoVITS模型卡片和选项
- 完善必填字段验证
- 更新模型名称映射"
```

**提交4: 参考音频管理**
```bash
git add reference_audio/ tools/validate_references.py
git commit -m "[Feature] 创建参考音频管理系统

- 添加参考音频目录结构
- 创建元数据配置
- 实现音频质量验证工具"
```

**提交5: 项目配置和文档**
```bash
git add requirements.txt .gitignore PROJECT_STRUCTURE.md records/
git commit -m "[Config] 添加项目配置和文档

- 创建requirements.txt
- 配置.gitignore规则
- 添加项目结构文档
- 记录工作进展"
```

### 6. 项目依赖说明

**核心依赖 (requirements.txt):**
- FastAPI + Uvicorn - Web服务
- PyTorch + Transformers - 深度学习
- Librosa + SoundFile - 音频处理
- 各TTS项目特定依赖

**完整依赖 (requirements_full.txt):**
- 通过 `pip freeze` 生成
- 包含所有安装的包
- 用于环境复现

### 7. 文件变更统计

**新增文件:**
- backend/batch_processor.py
- reference_audio/README.md
- reference_audio/metadata.json
- tools/validate_references.py
- requirements.txt
- .gitignore
- PROJECT_STRUCTURE.md
- records/*.md (5个)

**修改文件:**
- backend/api_server.py (添加GPT-SoVITS和批量处理)
- frontend/index.html (添加GPT-SoVITS选项)

**总计:**
- 新增: 13个文件
- 修改: 2个文件

## 后续建议

1. **首次提交**
   - 执行上述建议的提交分组
   - 确保所有文件正确提交

2. **模型文件管理**
   - 使用Git LFS管理大文件(可选)
   - 或保持模型文件在.gitignore中
   - 提供模型下载脚本

3. **持续维护**
   - 定期更新requirements.txt
   - 保持records/目录的工作记录
   - 及时更新文档

4. **版本标签**
   ```bash
   git tag -a v1.2.0 -m "集成GPT-SoVITS和批量处理功能"
   git push origin v1.2.0
   ```
