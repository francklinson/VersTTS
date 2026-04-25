# Qwen3-TTS 特色高级功能前端开放实现记录

**时间**: 2026-04-25 22:10:00  
**任务**: 将 Qwen3-TTS 的特色高级功能（预设音色、音色设计、指令控制）在前端开放给用户  
**状态**: ✅ 已完成

---

## 1. 实现的功能

### 1.1 新增模式选项

在 Qwen3-TTS 的模式选择中添加了两种新模式：
- **预设音色** (`custom_voice`) - 使用 9 种预设音色进行合成
- **音色设计** (`voice_design`) - 通过自然语言描述设计自定义音色

### 1.2 预设音色选择 (Custom Voice)

添加了 9 种预设音色的选择界面：

| 音色名称 | 描述 | 语言 |
|---------|------|------|
| Vivian | 明亮女声 | 中文 |
| Serena | 温暖女声 | 中文 |
| Uncle_Fu | 低沉男声 | 中文 |
| Dylan | 北京男声 | 中文方言 |
| Eric | 成都男声 | 四川方言 |
| Ryan | 动态男声 | 英文 |
| Aiden | 阳光男声 | 英文 |
| Ono_Anna | 俏皮女声 | 日文 |
| Sohee | 温暖女声 | 韩文 |

### 1.3 音色设计 (Voice Design)

添加了音色描述输入框，用户可以通过自然语言描述来设计独特的音色。

**示例描述**:
- "体现撒娇稚嫩的萝莉女声，音调偏高且起伏明显"
- "成熟稳重的男声，低沉醇厚，适合新闻播报"

### 1.4 指令控制 (Instruct)

在 `custom_voice` 和 `voice_design` 模式下支持指令控制：

- 添加指令输入框用于控制语音的情感和风格
- 支持自然语言指令，如：
  - "用特别愤怒的语气说"
  - "非常开心地"
  - "悲伤地"

### 1.5 克隆模式选择

在 `voice_clone` 模式下添加了克隆模式选择：
- **ICL 模式** (默认) - 质量更高
- **纯说话人嵌入模式** - 速度更快

---

## 2. 修改的文件

### 2.1 后端 API (`backend/api_server.py`)

#### 修改 `Qwen3TTSRequest` 类
```python
class Qwen3TTSRequest(BaseTTSRequest):
    model_size: str = Field(default="1.7B", description="模型大小: 0.6B, 1.7B")
    mode: str = Field(default="base", description="模式: base, voice_clone, custom_voice, voice_design")
    speaker: Optional[str] = Field(default=None, description="预设音色: Vivian, Serena...")
    ref_audio: Optional[str] = Field(default=None, description="参考音频URL/base64")
    ref_text: Optional[str] = Field(default=None, description="参考文本")
    voice_design_prompt: Optional[str] = Field(default=None, description="音色设计描述")
    instruct_text: Optional[str] = Field(default=None, description="指令控制文本")
    streaming: bool = Field(default=False, description="是否使用流式生成")
    x_vector_only_mode: bool = Field(default=False, description="是否仅使用说话人嵌入模式")
```

#### 修改 `get_qwen3tts_model` 函数
- 添加 `model_type` 参数支持加载不同类型的模型（Base, CustomVoice, VoiceDesign）
- 添加模型路径检查和回退逻辑

#### 修改 `tts_qwen3tts` API 端点
- 添加新参数：`speaker`, `voice_design_prompt`, `instruct_text`, `x_vector_only_mode`
- 实现四种模式的处理逻辑：
  - `base`: 基础合成（使用默认参考音频）
  - `voice_clone`: 音色克隆（支持上传参考音频）
  - `custom_voice`: 预设音色生成
  - `voice_design`: 音色设计

### 2.2 前端页面 (`frontend/app.html`)

#### HTML 修改
1. **模式选择** - 添加 `custom_voice` 和 `voice_design` 选项
2. **预设音色选择** - 添加 9 种预设音色的下拉菜单
3. **音色设计输入** - 添加音色描述 textarea
4. **指令控制输入** - 添加指令控制 input
5. **克隆模式选择** - 添加 ICL/纯说话人嵌入模式选择

#### JavaScript 修改
1. **更新 `updateQwen3Options` 函数**
   - 根据选择的模式显示/隐藏相应的选项组
   - 支持四种模式的选项切换

2. **更新 `generateTTS` 函数中的 qwen3tts 处理逻辑**
   - `voice_clone` 模式：添加 `x_vector_only_mode` 参数
   - `custom_voice` 模式：添加 `speaker` 和 `instruct_text` 参数
   - `voice_design` 模式：添加 `voice_design_prompt` 和 `instruct_text` 参数

---

## 3. 功能使用说明

### 3.1 预设音色模式

1. 选择 Qwen3-TTS 模型
2. 选择"预设音色"模式
3. 从下拉菜单中选择喜欢的音色（如 Vivian, Serena 等）
4. （可选）输入指令控制情感，如"开心地"
5. 点击生成

### 3.2 音色设计模式

1. 选择 Qwen3-TTS 模型
2. 选择"音色设计"模式
3. 在"音色描述"框中输入对音色的描述
4. （可选）输入指令控制情感
5. 点击生成

### 3.3 声音克隆模式

1. 选择 Qwen3-TTS 模型
2. 选择"声音克隆"模式
3. 上传参考音频或选择内置人声
4. 输入参考音频对应的文本
5. （可选）选择克隆模式（ICL/纯说话人嵌入）
6. 点击生成

---

## 4. 注意事项

### 4.1 模型文件要求

要使用新功能，需要下载相应的模型文件：
- **CustomVoice 模型**: `Qwen3-TTS-12Hz-1.7B-CustomVoice` 或 `Qwen3-TTS-12Hz-0.6B-CustomVoice`
- **VoiceDesign 模型**: `Qwen3-TTS-12Hz-1.7B-VoiceDesign`

模型文件应放置在 `algorithms/Qwen3-TTS/models/Qwen/` 目录下。

### 4.2 回退机制

如果指定的模型文件不存在，系统会自动回退到 Base 模型：
- `custom_voice` 模式会回退到使用 Base 模型的默认参考音频
- `voice_design` 模式会回退到使用 Base 模型的默认参考音频

### 4.3 流式生成

流式生成功能已在后端 API 中添加参数支持，但前端暂未实现实时音频流传输界面。

---

## 5. 测试建议

### 5.1 功能测试

1. **预设音色模式**
   - 测试所有 9 种预设音色
   - 测试带指令和不带指令的情况

2. **音色设计模式**
   - 测试不同的音色描述
   - 测试带指令的情况

3. **声音克隆模式**
   - 测试上传参考音频
   - 测试内置人声选择
   - 测试两种克隆模式

4. **基础模式**
   - 确保原有功能正常工作

### 5.2 边界测试

- 不输入音色描述直接生成
- 不选择预设音色直接生成
- 上传非音频文件
- 输入超长文本

---

## 6. 后续优化建议

1. **流式生成** - 实现 WebSocket 或 SSE 实时音频流传输
2. **音色预览** - 添加预设音色的试听功能
3. **音色设计模板** - 提供常用的音色描述模板
4. **批量生成优化** - 实现可复用克隆 Prompt 功能
5. **前端代码拆分** - 将 JavaScript 代码拆分到单独的文件中

---

## 7. 总结

本次修改成功将 Qwen3-TTS 的三大特色功能开放给用户：
- ✅ 预设音色（9 种优质音色）
- ✅ 音色设计（自然语言描述）
- ✅ 指令控制（情感和风格控制）

用户现在可以充分体验 Qwen3-TTS 的强大功能，根据不同场景选择合适的合成方式。
