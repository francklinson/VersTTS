# CosyVoice 修复 generator 错误

**操作时间**: 2026-05-06 20:40:00

**问题描述**: CosyVoice 生成语音时报错：
```
CosyVoice错误: 'generator' object is not subscriptable
```

**问题原因**: CosyVoice 的 `inference_zero_shot` 和 `inference_cross_lingual` 方法返回的是 generator 对象，但代码尝试像访问字典一样用 `['tts_speech']` 去索引它。

---

## 修复内容

### 修改文件
- **文件路径**: `backend/routers/tts/cosyvoice.py`

### 修改详情

**修复前**:
```python
# 保存音频
sr = 22050
audio_data = model_output['tts_speech'].numpy().squeeze()
audio_path = save_temp_audio(audio_data, sr)

# 清理显存
if torch.cuda.is_available():
    del model_output
```

**修复后**:
```python
# 处理 generator 输出
# CosyVoice 返回的是 generator，需要遍历获取结果
output_list = list(model_output)
if not output_list:
    raise HTTPException(status_code=500, detail="模型未返回音频数据")

# 获取第一个输出结果
first_output = output_list[0]
sr = 22050
audio_data = first_output['tts_speech'].numpy().squeeze()
audio_path = save_temp_audio(audio_data, sr)

# 清理显存
if torch.cuda.is_available():
    del first_output
    del output_list
```

---

## 修复要点

1. 将 generator 转换为 list: `output_list = list(model_output)`
2. 检查结果是否为空
3. 获取第一个输出结果: `first_output = output_list[0]`
4. 从第一个结果中提取音频数据
5. 清理显存时删除正确的对象

---

**操作状态**: 已完成
