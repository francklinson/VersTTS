# Qwen3-TTS 根据模型能力动态显示前端UI

**时间**: 2026-04-25 23:15:00  
**任务**: 根据模型的实际能力（模型文件是否存在）动态显示前端UI选项  
**状态**: ✅ 已完成

---

## 1. 功能概述

实现根据 Qwen3-TTS 模型文件的实际存在情况，动态显示或隐藏前端UI选项，避免用户选择不可用的功能。

---

## 2. 实现方案

### 2.1 后端 API

新增端点 `/tts/qwen3tts/status` 用于检查模型状态：

```python
GET /tts/qwen3tts/status

返回示例:
{
    "base_available": true,
    "custom_voice_available": false,
    "voice_design_available": false,
    "model_sizes": ["0.6B", "1.7B"],
    "message": "仅 Base 模型可用，CustomVoice 和 VoiceDesign 功能将使用默认音色"
}
```

### 2.2 前端逻辑

1. **模型状态变量**
   ```javascript
   let qwen3TTSStatus = {
       base_available: false,
       custom_voice_available: false,
       voice_design_available: false,
       model_sizes: [],
       message: ''
   };
   ```

2. **状态检查函数** (`checkQwen3TTSStatus`)
   - 页面加载时调用
   - 查询后端API获取模型状态
   - 更新全局状态变量
   - 触发UI更新

3. **动态更新模式选项** (`updateQwen3TTSModeOptions`)
   - 根据 `custom_voice_available` 显示/隐藏 "预设音色" 选项
   - 根据 `voice_design_available` 显示/隐藏 "音色设计" 选项
   - "基础合成" 和 "声音克隆" 始终显示

4. **状态提示消息** (`showQwen3TTSStatusMessage`)
   - 当模型不完整时显示警告提示
   - 提示用户哪些功能可用/不可用
   - 说明缺失功能将使用默认音色

---

## 3. 修改的文件

### 3.1 `backend/api_server.py`

新增内容：
- `Qwen3TTSModelStatus` 数据模型类
- `/tts/qwen3tts/status` GET 端点
- 检查三种模型（Base, CustomVoice, VoiceDesign）的文件是否存在
- 返回模型可用状态和用户友好的消息

### 3.2 `frontend/app.html`

新增内容：
- `qwen3TTSStatus` 全局变量
- `checkQwen3TTSStatus()` 函数
- `updateQwen3TTSModeOptions()` 函数
- `showQwen3TTSStatusMessage()` 函数
- 在 DOMContentLoaded 初始化时调用 `checkQwen3TTSStatus()`

---

## 4. UI 行为

### 4.1 所有模型可用
- 显示所有4种模式选项：
  - ✅ 基础合成
  - ✅ 声音克隆
  - ✅ 预设音色
  - ✅ 音色设计
- 不显示警告消息

### 4.2 仅 Base 模型可用
- 显示2种模式选项：
  - ✅ 基础合成
  - ✅ 声音克隆
  - ❌ 预设音色（隐藏）
  - ❌ 音色设计（隐藏）
- 显示警告消息：
  > ⚠️ 模型状态提示  
  > 仅 Base 模型可用，CustomVoice 和 VoiceDesign 功能将使用默认音色

### 4.3 Base + CustomVoice 可用
- 显示3种模式选项：
  - ✅ 基础合成
  - ✅ 声音克隆
  - ✅ 预设音色
  - ❌ 音色设计（隐藏）
- 显示警告消息

### 4.4 Base + VoiceDesign 可用
- 显示3种模式选项：
  - ✅ 基础合成
  - ✅ 声音克隆
  - ❌ 预设音色（隐藏）
  - ✅ 音色设计
- 显示警告消息

---

## 5. 用户体验改进

### 5.1 避免错误
用户无法选择不可用的功能，减少操作错误。

### 5.2 清晰提示
当功能受限时，用户会收到明确的提示，了解：
- 哪些功能可用
- 哪些功能需要额外模型
- 缺失功能的行为（使用默认音色）

### 5.3 渐进增强
系统可以正常工作，即使只有 Base 模型：
- 基础功能始终可用
- 高级功能在有模型时自动启用
- 无需手动配置

---

## 6. 测试建议

### 6.1 场景测试

| 场景 | 操作 | 预期结果 |
|------|------|----------|
| 完整模型 | 放置所有模型文件 | 显示所有4个选项，无警告 |
| 仅 Base | 只放置 Base 模型 | 显示2个选项，显示警告 |
| 无模型 | 不放置任何模型 | 显示2个选项，显示错误警告 |

### 6.2 功能测试
- 切换模型大小（0.6B/1.7B）后检查状态
- 删除/添加模型文件后刷新页面
- 验证选项是否正确显示/隐藏

---

## 7. 后续优化建议

1. **实时刷新**: 添加刷新按钮，无需刷新页面即可检查模型状态
2. **模型下载指引**: 在警告消息中添加模型下载链接或指引
3. **模型管理界面**: 提供模型上传/下载管理功能
4. **其他模型**: 为 CosyVoice、F5-TTS 等其他模型实现类似功能

---

## 8. 总结

通过动态UI方案，用户界面现在能够：
- ✅ 自动检测模型文件是否存在
- ✅ 根据检测结果动态显示可用选项
- ✅ 对受限功能提供清晰的提示说明
- ✅ 确保基础功能始终可用

这大大提升了用户体验，避免了因模型缺失导致的错误操作和困惑。
