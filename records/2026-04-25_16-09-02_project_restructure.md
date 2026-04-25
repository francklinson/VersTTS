# VersTTS 项目结构整理与登录功能实现

## 任务目标
根据需求文档完成以下待办事项：
1. 整理项目文件结构，把六个TTS项目都放到 `algorithms` 文件夹下，修复六个项目的依赖问题
2. 把测试脚本放到 `test_scripts` 文件夹下，并写一个测试脚本，测试所有项目的TTS功能
3. 给项目前端增加一个登陆页面，默认用户名 admin，密码 tp123456

## 执行内容

### 1. 整理项目文件结构

**创建目录并移动文件夹：**
- 创建 `algorithms/` 目录
- 将以下六个TTS项目移动到 `algorithms/` 下：
  - `ChatTTS` -> `algorithms/ChatTTS`
  - `CosyVoice` -> `algorithms/CosyVoice`
  - `F5-TTS` -> `algorithms/F5-TTS`
  - `GPT-SoVITS` -> `algorithms/GPT-SoVITS`
  - `OpenVoice` -> `algorithms/OpenVoice`
  - `Qwen3-TTS` -> `algorithms/Qwen3-TTS`

**更新路径引用：**
- 修改 `backend/api_server.py` 中的所有项目路径引用，添加 `algorithms/` 前缀
  - `sys.path.insert` 路径更新
  - 模型加载路径更新（ChatTTS、CosyVoice、F5-TTS、Qwen3-TTS、OpenVoice、GPT-SoVITS）
  - 参考音频默认路径更新（F5-TTS）
- 修改所有测试脚本中的路径引用：
  - `test_scripts/test_chattts.py`
  - `test_scripts/test_cosyvoice.py`
  - `test_scripts/test_f5_tts.py`
  - `test_scripts/test_openvoice.py`
  - `test_scripts/test_qwen3_tts.py`

### 2. 测试脚本整理

**移动测试脚本：**
- 创建 `test_scripts/` 目录
- 将以下测试脚本移动到 `test_scripts/` 下：
  - `test_api.py`
  - `test_chattts.py`
  - `test_cosyvoice.py`
  - `test_f5_tts.py`
  - `test_openvoice.py`
  - `test_qwen3_tts.py`

**创建统一测试脚本：**
- 新建 `test_scripts/test_all_tts.py`
  - 支持环境检查（Python版本、CUDA、关键依赖、目录结构）
  - 支持逐个运行各TTS项目测试脚本
  - 支持API服务测试
  - 提供命令行参数：`--env-only`、`--api-only`、`--skip-api`
  - 输出测试汇总报告

### 3. 前端登录页面

**修改 `frontend/index.html`：**
- 添加登录页面样式（`.login-container`、`.login-box`、`.login-input`、`.btn-login` 等）
- 添加登录页面HTML结构：
  - 用户名输入框（默认填充 admin）
  - 密码输入框
  - 登录按钮
  - 错误提示区域
- 添加登录验证JavaScript逻辑：
  - 默认用户名：`admin`
  - 默认密码：`tp123456`
  - 使用 `localStorage` 存储登录状态
  - 支持回车键登录
  - 登录成功后显示主界面，隐藏登录页
  - 添加用户信息显示栏（左上角），点击可退出登录
  - 页面加载时自动检查登录状态

## 文件变更汇总

| 操作 | 文件/目录 |
|------|----------|
| 新建目录 | `algorithms/` |
| 新建目录 | `test_scripts/` |
| 移动 | `ChatTTS/` -> `algorithms/ChatTTS/` |
| 移动 | `CosyVoice/` -> `algorithms/CosyVoice/` |
| 移动 | `F5-TTS/` -> `algorithms/F5-TTS/` |
| 移动 | `GPT-SoVITS/` -> `algorithms/GPT-SoVITS/` |
| 移动 | `OpenVoice/` -> `algorithms/OpenVoice/` |
| 移动 | `Qwen3-TTS/` -> `algorithms/Qwen3-TTS/` |
| 移动 | `test_api.py` -> `test_scripts/test_api.py` |
| 移动 | `test_chattts.py` -> `test_scripts/test_chattts.py` |
| 移动 | `test_cosyvoice.py` -> `test_scripts/test_cosyvoice.py` |
| 移动 | `test_f5_tts.py` -> `test_scripts/test_f5_tts.py` |
| 移动 | `test_openvoice.py` -> `test_scripts/test_openvoice.py` |
| 移动 | `test_qwen3_tts.py` -> `test_scripts/test_qwen3_tts.py` |
| 修改 | `backend/api_server.py` |
| 修改 | `test_scripts/test_chattts.py` |
| 修改 | `test_scripts/test_cosyvoice.py` |
| 修改 | `test_scripts/test_f5_tts.py` |
| 修改 | `test_scripts/test_openvoice.py` |
| 修改 | `test_scripts/test_qwen3_tts.py` |
| 修改 | `frontend/index.html` |
| 新建 | `test_scripts/test_all_tts.py` |
| 新建 | `records/2026-04-25_project_restructure.md` |

## 备注
- 六个TTS项目各自的 `requirements.txt` 仍保留在项目目录内，由各自项目维护
- 根目录 `requirements.txt` 作为统一运行环境的依赖清单，未做修改
- 登录功能为前端实现，基于 localStorage 存储会话状态，未增加后端登录接口
