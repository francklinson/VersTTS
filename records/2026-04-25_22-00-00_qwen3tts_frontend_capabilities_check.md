# Qwen3-TTS 前端功能开放情况检查报告

**时间**: 2026-04-25 22:00:00  
**检查人**: AI Assistant  
**任务**: 根据 algorithms/Qwen3-TTS/readme.md 检查 Qwen3-TTS 支持的功能是否都在前端页面上开放给用户

---

## 1. Qwen3-TTS 支持的功能（根据 readme.md）

### 1.1 模型系列

| 模型 | 功能特点 | 语言支持 | 流式支持 | 指令控制 |
|------|----------|----------|----------|----------|
| Qwen3-TTS-12Hz-1.7B-VoiceDesign | 基于用户描述进行音色设计 | 中英日韩德法俄葡西意 | ✅ | ✅ |
| Qwen3-TTS-12Hz-1.7B-CustomVoice | 通过指令对目标音色进行风格控制，支持 9 种优质音色 | 中英日韩德法俄葡西意 | ✅ | ✅ |
| Qwen3-TTS-12Hz-1.7B-Base | 基础模型，支持 3 秒快速音色克隆，可用于微调 | 中英日韩德法俄葡西意 | ✅ | |
| Qwen3-TTS-12Hz-0.6B-CustomVoice | 轻量版，支持 9 种预设音色 | 中英日韩德法俄葡西意 | ✅ | |
| Qwen3-TTS-12Hz-0.6B-Base | 轻量版基础模型，支持 3 秒快速音色克隆 | 中英日韩德法俄葡西意 | ✅ | |

### 1.2 支持的 9 种预设音色

| 音色名称 | 音色描述 | 母语 |
|----------|----------|------|
| Vivian | 明亮、略带锐利的年轻女声 | 中文 |
| Serena | 温暖、柔和的年轻女声 | 中文 |
| Uncle_Fu | 经验丰富的男声，低沉醇厚 | 中文 |
| Dylan | 年轻的北京男声，清晰自然 | 中文（北京方言） |
| Eric | 活泼的成都男声，略带沙哑的明亮感 | 中文（四川方言） |
| Ryan | 富有节奏感的动态男声 | 英文 |
| Aiden | 阳光的美国男声，中音清晰 | 英文 |
| Ono_Anna | 俏皮的日本女声，轻快灵活 | 日文 |
| Sohee | 温暖、富有情感的韩语女声 | 韩文 |

### 1.3 核心功能

1. **音色克隆 (Voice Clone)** - Base 模型，支持 3 秒快速音色克隆
2. **预设音色生成 (Custom Voice)** - CustomVoice 模型，支持 9 种预设音色
3. **音色设计 (Voice Design)** - VoiceDesign 模型，通过自然语言描述设计音色
4. **指令控制 (Instruct)** - 通过自然语言指令控制语音风格（情感、语气等）
5. **流式生成** - 极低延迟流式生成（97ms）

---

## 2. 前端页面实际开放的功能

### 2.1 前端配置选项（app.html）

```html
<!-- Qwen3-TTS 选项 -->
<div id="qwen3tts-options" class="model-options" style="display:none">
    <div class="options-grid">
        <div class="option-group">
            <label>模型大小</label>
            <select id="qwen3tts-size">
                <option value="1.7B">1.7B (推荐)</option>
                <option value="0.6B">0.6B (轻量)</option>
            </select>
        </div>
        <div class="option-group">
            <label>模式</label>
            <select id="qwen3tts-mode" onchange="updateQwen3Options()">
                <option value="base">基础合成</option>
                <option value="voice_clone">声音克隆</option>
            </select>
        </div>
        <!-- 参考音频和参考文本选项（仅在 voice_clone 模式下显示） -->
    </div>
</div>
```

### 2.2 后端 API 支持（api_server.py）

```python
class Qwen3TTSRequest(BaseTTSRequest):
    model_size: str = Field(default="1.7B", description="模型大小: 0.6B, 1.7B")
    mode: str = Field(default="base", description="模式: base, voice_clone")
    ref_audio: Optional[str] = Field(default=None, description="参考音频URL/base64")
    ref_text: Optional[str] = Field(default=None, description="参考文本")
```

---

## 3. 功能对比分析

| 功能 | readme.md 说明 | 前端支持 | 后端支持 | 状态 |
|------|----------------|----------|----------|------|
| **模型大小选择** | 1.7B / 0.6B | ✅ 支持 | ✅ 支持 | ✅ 已开放 |
| **音色克隆** | Base 模型，3秒快速克隆 | ✅ 支持 (voice_clone) | ✅ 支持 | ✅ 已开放 |
| **基础合成** | 使用默认参考音频 | ✅ 支持 (base) | ✅ 支持 | ✅ 已开放 |
| **预设音色生成** | CustomVoice 模型，9种预设音色 | ❌ 不支持 | ❌ 不支持 | ❌ 未开放 |
| **音色设计** | VoiceDesign 模型，自然语言描述音色 | ❌ 不支持 | ❌ 不支持 | ❌ 未开放 |
| **指令控制** | 通过自然语言指令控制风格 | ❌ 不支持 | ❌ 不支持 | ❌ 未开放 |
| **流式生成** | 97ms 极低延迟流式生成 | ❌ 不支持 | ❌ 不支持 | ❌ 未开放 |
| **两种克隆模式** | ICL 模式 / 纯说话人嵌入模式 | ❌ 不支持 | ❌ 不支持 (固定为False) | ❌ 未开放 |
| **可复用克隆 Prompt** | 批量生成时复用 prompt | ❌ 不支持 | ❌ 不支持 | ❌ 未开放 |

---

## 4. 结论

### 4.1 已开放功能

Qwen3-TTS 在前端页面上**已开放**的功能包括：
1. ✅ 模型大小选择（1.7B / 0.6B）
2. ✅ 音色克隆（voice_clone 模式，支持上传参考音频和参考文本）
3. ✅ 基础合成（base 模式，使用默认参考音频）

### 4.2 未开放功能

Qwen3-TTS 在前端页面上**未开放**的功能包括：

| 未开放功能 | 说明 | 影响 |
|------------|------|------|
| ❌ 预设音色生成 (Custom Voice) | 9 种优质预设音色（Vivian, Serena 等） | 用户无法使用预设音色，只能使用基础音色或上传参考音频 |
| ❌ 音色设计 (Voice Design) | 通过自然语言描述设计音色 | 用户无法通过描述创建自定义音色 |
| ❌ 指令控制 (Instruct) | 通过自然语言指令控制语音风格 | 用户无法精细控制语音的情感、语气等 |
| ❌ 流式生成 | 97ms 极低延迟实时合成 | 无法实现实时交互场景 |
| ❌ 两种克隆模式选择 | ICL 模式 vs 纯说话人嵌入模式 | 用户无法选择克隆质量与速度的权衡 |
| ❌ 可复用克隆 Prompt | 批量生成时复用 prompt 提高效率 | 批量生成时效率较低 |

---

## 5. 建议

### 5.1 高优先级建议

1. **增加 Custom Voice 模式支持**
   - 添加 `custom_voice` 模式选项
   - 实现 9 种预设音色的选择界面
   - 后端需要加载 CustomVoice 模型

2. **增加 Voice Design 模式支持**
   - 添加 `voice_design` 模式选项
   - 添加音色描述文本输入框
   - 后端需要加载 VoiceDesign 模型

### 5.2 中优先级建议

3. **增加指令控制支持**
   - 在 Custom Voice 和 Voice Design 模式下添加指令输入框
   - 允许用户通过自然语言控制语音风格

4. **增加流式生成开关**
   - 添加流式生成选项
   - 需要前后端配合实现 WebSocket 或 SSE 流式传输

### 5.3 低优先级建议

5. **增加克隆模式选择**
   - 在 voice_clone 模式下添加 ICL/纯说话人嵌入模式切换

6. **优化批量生成**
   - 实现可复用克隆 Prompt 功能，提高批量生成效率

---

## 6. 总结

根据本次检查，**Qwen3-TTS 的核心基础功能（模型选择、音色克隆、基础合成）已在前端页面开放给用户**。但是，其**特色高级功能（预设音色、音色设计、指令控制、流式生成）目前尚未在前端开放**。

如果希望用户能够充分体验 Qwen3-TTS 的全部能力，建议按照上述优先级逐步增加这些功能的开放。
