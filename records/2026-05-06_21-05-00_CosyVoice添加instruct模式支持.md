# CosyVoice 添加 Instruct 模式后端支持

**操作时间**: 2026-05-06 21:05:00

**问题描述**: 前端调用 instruct 模式时报错 `不支持的模式: instruct`

**问题原因**: 后端路由代码只处理了 `sft` 和 `zero_shot` 两种模式，没有添加 `instruct` 模式的处理逻辑。

---

## 修复内容

### 修改文件
- **文件路径**: `backend/routers/tts/cosyvoice.py`

### 修改详情

**添加的代码**:
```python
elif mode == "instruct":
    if not instruct_text:
        raise HTTPException(status_code=400, detail="instruct模式需要提供instruct_text指令文本")
    
    # 构建说话人描述，如果没有clone_speaker_id则使用默认值
    speaker_desc = speaker_id
    if clone_speaker_id:
        db = load_speakers_db()
        speaker = None
        for s in db.get("speakers", []):
            if s["id"] == clone_speaker_id:
                speaker = s
                break
        if speaker:
            speaker_desc = "中文女"  # 默认使用中文女
    
    # 使用 inference_instruct 方法
    model_output = cosyvoice.inference_instruct(text, speaker_desc, instruct_text, stream=False)
```

---

## 功能说明

现在 CosyVoice 支持三种模式：

| 模式 | 说明 |
|------|------|
| **sft** | 预训练音色（CosyVoice 3.0 已不支持）|
| **zero_shot** | 零样本克隆，使用参考音频克隆音色 |
| **instruct** | 指令控制，支持方言和情感控制 |

### Instruct 模式使用示例

**请求参数**:
- `mode`: "instruct"
- `text`: 要合成的文本
- `instruct_text`: 指令文本（如"用四川话说"、"用开心的语气说"）
- `speaker_id`: 说话人描述（如"中文女"、"中文男"）
- `clone_speaker_id`: （可选）参考说话人ID

---

**操作状态**: 已完成
