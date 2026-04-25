# 说话人管理与 ChatTTS 解耦实现记录

**时间**: 2026-04-26 00:10:00  
**任务**: 将说话人管理模块与 ChatTTS 解耦，不再提取 ChatTTS 特定的特征  
**状态**: ✅ 已完成

---

## 1. 解耦前的问题

### 原设计的问题
- 说话人管理**深度绑定 ChatTTS**
- 需要提取 ChatTTS 特定的 `spk_smp` embedding
- 只能用于 ChatTTS 模型，无法用于其他 TTS 模型
- 流程复杂：上传 → 提取特征 → 保存 embedding

### 解耦后的优势
- ✅ **模型无关**：说话人音频可用于任何 TTS 模型
- ✅ **流程简化**：上传音频 → 直接保存
- ✅ **通用性**：一套参考人声，全模型共享
- ✅ **可维护性**：不再依赖特定模型的特征提取

---

## 2. 后端修改

### 2.1 API 变更

| 原 API | 新 API | 说明 |
|--------|--------|------|
| `POST /speakers/extract` | `POST /speakers/upload` | 不再提取 embedding，只上传音频 |
| `POST /speakers/save` | `POST /speakers/save` | 移除 `embedding` 参数，改为 `audio_path` 必填 |

### 2.2 `/speakers/upload` 新接口

```python
@app.post("/speakers/upload")
async def upload_speaker_audio(
    audio: UploadFile = File(...),
    speaker_name: str = Form(...),
    reference_text: Optional[str] = Form(None)
):
    """上传说话人音频文件（与模型解耦，只保存音频和文本）"""
    # 1. 验证文件格式
    # 2. 统一转换为 WAV 格式（24kHz，单声道）
    # 3. 保存音频文件
    # 4. 返回音频路径（不再提取 embedding）
```

### 2.3 数据模型变更

```python
# 原数据结构
speaker = {
    "id": "...",
    "name": "...",
    "embedding": "<base64-encoded-spk_smp>",  # ❌ 已移除
    "audio_path": "...",
    "reference_text": "...",
    "model_type": "chattts"  # ❌ 已改为 "universal"
}

# 新数据结构（解耦后）
speaker = {
    "id": "...",
    "name": "...",
    "embedding": None,  # 不再保存模型特定特征
    "audio_path": "...",
    "reference_text": "...",
    "model_type": "universal"  # 通用类型
}
```

### 2.4 修改的文件

**`backend/api_server.py`**:
- ✅ 新增 `/speakers/upload` 接口
- ✅ 修改 `/speakers/save` 接口（移除 embedding 参数）
- ✅ 修改 `add_speaker()` 函数（embedding 可选）
- ✅ 修改数据模型（model_type 改为 "universal"）

---

## 3. 前端修改

### 3.1 流程简化

```
解耦前：
录音/上传 → 提取 ChatTTS 特征 → 保存 embedding + 音频 → 仅 ChatTTS 可用

解耦后：
录音/上传 → 保存音频 + 文本 → 所有 TTS 模型可用
```

### 3.2 JavaScript 变更

| 变更项 | 原代码 | 新代码 |
|--------|--------|--------|
| API 调用 | `/speakers/extract` | `/speakers/upload` |
| 变量 | `extractedEmbedding` | ❌ 已移除 |
| 保存参数 | `embedding` + `audio_path` | 仅 `audio_path` |
| 刷新列表 | `updateChatTTSSpeakerSelect()` | `loadReferenceVoices()` |

### 3.3 修改的文件

**`frontend/app.html`**:
- ✅ 移除 `extractedEmbedding` 变量
- ✅ 修改 API 调用（extract → upload）
- ✅ 简化保存流程（不再处理 embedding）
- ✅ 刷新参考人声列表（用于所有 TTS 模型）

---

## 4. 使用流程对比

### 4.1 创建参考人声（解耦后）

1. 点击"说话人管理"
2. 选择朗读文本片段
3. 朗读并录音（或上传音频）
4. 预览确认
5. **一步保存**：音频 + 文本直接保存
6. 自动同步到所有 TTS 模型的"参考人声"列表

### 4.2 使用参考人声

在任意 TTS 模型的**声音克隆**模式中：
1. 选择"参考人声"下拉框
2. 选择标注 `[我的]` 的自定义说话人
3. 参考文本自动填充
4. 输入要合成的内容
5. 生成语音

---

## 5. 技术细节

### 5.1 音频格式处理

上传的音频统一转换为标准格式：
- **格式**: WAV
- **采样率**: 24000Hz
- **声道**: 单声道
- **编码**: 16-bit PCM

### 5.2 兼容的 TTS 模型

解耦后的参考人声可用于：
- ✅ ChatTTS
- ✅ CosyVoice
- ✅ F5-TTS
- ✅ Qwen3-TTS
- ✅ OpenVoice
- ✅ GPT-SoVITS

### 5.3 参考人声数据结构

```json
{
  "id": "spk_...",
  "name": "我的声音",
  "audio_path": "/path/to/audio.wav",
  "reference_text": "欢迎使用智能语音合成系统...",
  "created_at": "2026-04-26T00:10:00",
  "model_type": "universal"
}
```

---

## 6. 注意事项

### 6.1 数据兼容性
- 旧的说话人数据（带 embedding）仍然可用
- 但新的说话人不再生成 embedding
- 建议逐步迁移到新的通用格式

### 6.2 性能考虑
- 由于不再预提取特征，声音克隆时可能需要额外处理
- 但现代 TTS 模型通常都能快速处理参考音频
- 权衡：存储空间 vs 计算时间

### 6.3 存储优化
- 音频文件统一转换为 WAV，可能占用更多空间
- 可考虑添加压缩选项（如 OGG/MP3）

---

## 7. 后续优化建议

1. **音频压缩**：支持上传时选择压缩格式
2. **批量导入**：支持批量上传参考音频
3. **音频编辑**：提供简单的裁剪、降噪功能
4. **质量检测**：自动检测音频质量和时长
5. **分类管理**：支持按用途分类管理参考人声

---

## 8. 总结

通过将说话人管理与 ChatTTS 解耦，实现了：

- ✅ **模型无关**：一套参考人声，全模型共享
- ✅ **流程简化**：上传 → 保存，一步到位
- ✅ **可扩展性**：易于支持未来的新 TTS 模型
- ✅ **维护性**：不再依赖特定模型的特征提取

这是向通用语音合成平台迈出的重要一步。
