# lib/ — Transformers 多版本隔离管理

由于不同的 TTS 算法依赖不同版本的 transformers，项目采用 **独立服务 + sys.path 隔离** 的方案解决版本冲突。

## 版本总览

| 目录 | transformers 版本 | 使用方 | 运行模式 |
|---|---|---|---|
| `.venv` (全局) | **4.57.3** | 主服务（Qwen3TTS、VoxCPM、ChatTTS、F5-TTS、FireRedTTS2、IndexTTS、OpenVoice） | 主服务进程内 |
| `lib/transformers4` | **4.51.3** | CosyVoice、PilotTTS、GPT-SoVITS | 独立服务 (端口 8008/8009/8010) |
| `lib/transformers5` | **5.14.1** | OmniVoice | 独立服务 (端口 8007) |
| `lib/wenet` | 跟随全局 | ASR 语音识别 | 被主服务/GPT-SoVITS 引用 |

## 架构说明

```
┌────────────────────────────────────────────────┐
│  start_server.sh                               │
│  ├─ do_start          → 主服务 (端口 8006)     │
│  │   └─ transformers 4.57.3 (全局 .venv)       │
│  ├─ do_start_omnivoice → OmniVoice 服务         │
│  │   └─ lib/transformers5 (5.14.1)             │
│  ├─ do_start_cosyvoice → CosyVoice 服务         │
│  │   └─ lib/transformers4 (4.51.3)             │
│  ├─ do_start_pilottts  → PilotTTS 服务          │
│  │   └─ lib/transformers4 (4.51.3)             │
│  └─ do_start_gptsovits → GPT-SoVITS 服务        │
│      └─ lib/transformers4 (4.51.3)             │
└────────────────────────────────────────────────┘
```

每个独立服务脚本在 `import` 任何模块之前，通过 `sys.path.insert(0, ...)` 优先加载对应版本的 transformers，实现版本隔离。

## 版本约束策略

不同环境采用不同的检查严格度，由各算法的实际兼容性决定：

| 环境 | 版本约束 | 严格度 | 原因 |
|---|---|---|---|
| `.venv` (全局) | `>= 4.57.0, < 5.0.0` | 中等 | Qwen3TTS 运行时检查 `>= 4.57.0`；5.x 破坏所有现有 API，必须阻止 |
| `lib/transformers4` | `== 4.51.3` (锁定) | **高** | PilotTTS 声明上限 `<= 4.52.4`；CosyVoice 的 Fun-CosyVoice3-0.5B tokenizer 在 `>= 4.52` 时出错；GPT-SoVITS 的 HiggsAudioV2 在 4.57.x 有兼容问题 |
| `lib/transformers5` | `>= 5.3.0` | 低 | OmniVoice 依赖 5.x 新 API，5.x 系列内子版本兼容性好 |

start_server.sh 的 `check_transformers_versions()` 函数会在主服务启动时自动执行上述检查，超出约束范围时给出明确错误/警告。

## 各版本详细说明

### lib/transformers4 (4.51.3)

- **版本**: 4.51.3
- **安装命令**: `pip install --target lib/transformers4 transformers==4.51.3`
- **使用服务**: CosyVoice、PilotTTS、GPT-SoVITS
- **独立服务脚本**:
  - `cosyvoice_service.py` → 端口 8008
  - `pilottts_service.py` → 端口 8009
  - `gptsovits_service.py` → 端口 8010
- **冲突原因**: CosyVoice 依赖 Fun-CosyVoice3-0.5B 模型，该模型的 tokenizer 与 transformers >= 4.52 不兼容；PilotTTS 声明依赖 `transformers>=4.40.0,<=4.52.4`；GPT-SoVITS 的 HiggsAudioV2 模型在 4.57.x 下有兼容性问题

### lib/transformers5 (5.14.1)

- **版本**: 5.14.1
- **安装命令**: `pip install --target lib/transformers5 transformers>=5.3.0`
- **使用服务**: OmniVoice
- **独立服务脚本**: `omnivoice_service.py` → 端口 8007
- **冲突原因**: OmniVoice 依赖 transformers 5.x 的新 API（如 `HiggsAudioV2TokenizerModel`），与全局 4.57.3 不兼容

### .venv transformers (4.57.3)

- **版本**: 4.57.3
- **安装方式**: 通过 `requirements.txt` / `requirements_full.txt` 全局安装
- **使用方**:
  - Qwen3-TTS（需要 >= 4.57.0）
  - VoxCPM（兼容 4.57.x）
  - ChatTTS、F5-TTS、FireRedTTS2、IndexTTS、OpenVoice（兼容 4.57.x）

## 兼容性机制

### 启动脚本检查 (start_server.sh)

各独立服务的 `do_start_*` 函数在启动前检查对应 `lib/` 子目录是否存在：
- `do_start_omnivoice` → 检查 `lib/transformers5/`
- `do_start_cosyvoice` → 检查 `lib/transformers4/`
- `do_start_pilottts` → 检查 `lib/transformers4/`
- `do_start_gptsovits` → 检查 `lib/transformers4/`

### Python 运行时兼容

| 文件 | 用途 |
|---|---|
| `backend/engines/transformers_compat.py` | 全局 transformers 4.57.3 兼容层，注册缺失的类/配置 |
| `backend/engines/transformers_isolation.py` | 运行时版本切换（activate/restore），用于同进程内切换 |
| `backend/main.py` (第 48-60 行) | 启动时补丁，为 CosyVoice 等添加缺失函数 |
| `check_env.py` | 环境信息报告，列出当前 transformers 版本 |

## 新增算法时注意事项

1. 确认新算法需要的 transformers 版本范围
2. 如果与全局 4.57.3 兼容 → 直接在主服务进程中使用
3. 如果需要 4.x 特定版本 → 复用 `lib/transformers4` 或新建独立服务
4. 如果需要 5.x → 复用 `lib/transformers5` 或新建独立服务
5. 更新此文档和 `start_server.sh` 中的版本检查

## 相关记录

- `records/2026-04-26_12-58-00_transformers_version_analysis.md`
- `records/2026-04-26_13-10-00_transformers_4.55.2_compatibility_fix.md`
- `records/2026-04-26_14-10-00_transformers_version_conflict_summary.md`
- `records/2026-05-08_07-30-00_OmniVoice_部署与版本隔离.md`
