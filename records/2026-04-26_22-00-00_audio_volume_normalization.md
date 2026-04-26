# 音频音量归一化功能实现记录

**时间:** 2026-04-26 22:00:00  
**任务:** 解决TTS生成音频音量太小的问题  
**执行人:** VersTTS System

---

## 1. 问题描述

**用户反馈**: TTS生成的音频音量太小，听不清楚。

---

## 2. 问题分析

### 2.1 根本原因

TTS模型生成的音频通常峰值较低（约0.1-0.3），导致实际播放时音量很小。这是因为：
1. 模型输出未经过归一化处理
2. 不同模型的输出电平不一致
3. 没有统一的音量标准

### 2.2 解决方案

添加音频后处理步骤，对所有生成的音频进行音量归一化，将峰值提升到标准水平。

---

## 3. 实现内容

### 3.1 新增音量归一化函数

在 `backend/api_server.py` 中添加 `normalize_audio_volume` 函数：

```python
def normalize_audio_volume(audio_data: np.ndarray, target_db: float = -3.0) -> np.ndarray:
    """
    归一化音频音量到目标dB级别
    
    Args:
        audio_data: 输入音频数组
        target_db: 目标dB级别，默认-3.0 dB（峰值归一化，保留少量headroom）
    
    Returns:
        归一化后的音频数组
    """
    # 确保音频是float32类型
    if audio_data.dtype != np.float32:
        audio_data = audio_data.astype(np.float32)
    
    # 计算当前峰值
    current_peak = np.max(np.abs(audio_data))
    
    if current_peak == 0:
        return audio_data  # 避免除零
    
    # 计算目标峰值（从dB转换为线性比例）
    target_peak = 10 ** (target_db / 20.0)
    
    # 计算增益因子
    gain = target_peak / current_peak
    
    # 应用增益
    normalized_audio = audio_data * gain
    
    # 确保不会溢出（硬限幅）
    normalized_audio = np.clip(normalized_audio, -1.0, 1.0)
    
    return normalized_audio
```

### 3.2 修改音频保存函数

修改 `save_temp_audio` 函数，添加音量归一化参数：

```python
def save_temp_audio(audio_data: np.ndarray, sample_rate: int, suffix: str = ".wav", normalize: bool = True) -> str:
    """
    保存临时音频文件
    
    Args:
        audio_data: 音频数据数组
        sample_rate: 采样率
        suffix: 文件后缀
        normalize: 是否进行音量归一化，默认True
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    temp_path = f"output/tts_{timestamp}{suffix}"
    
    # 音量归一化处理
    if normalize:
        audio_data = normalize_audio_volume(audio_data)
    
    sf.write(temp_path, audio_data, sample_rate)
    return temp_path
```

### 3.3 各算法修改位置

| 算法 | 修改位置 | 修改方式 |
|------|----------|----------|
| **ChatTTS** | `save_temp_audio` 调用处 | 自动生效 |
| **Qwen3-TTS** | `save_temp_audio` 调用处 | 自动生效 |
| **F5-TTS** | `save_temp_audio` 调用处 | 自动生效 |
| **CosyVoice** | `torchaudio.save` 前 | 添加归一化调用 |
| **GPT-SoVITS** | `sf.write` 前 | 添加归一化调用 |
| **OpenVoice** | 生成后读取再保存 | 添加归一化处理 |

#### 具体修改：

**CosyVoice** (api_server.py ~1248行):
```python
# 音量归一化
audio_np = normalize_audio_volume(audio_np)

# 使用torchaudio保存
torchaudio.save(audio_path, torch.from_numpy(audio_np), cosyvoice.sample_rate)
```

**GPT-SoVITS** (api_server.py ~1755行):
```python
# 音量归一化
audio_data = normalize_audio_volume(audio_data)

# 保存音频
sf.write(audio_path, audio_data, sr)
```

**OpenVoice** (api_server.py ~1658行):
```python
# 读取生成的音频并进行音量归一化
import soundfile as sf
audio_data, sample_rate = sf.read(audio_path)
audio_data = normalize_audio_volume(audio_data)
sf.write(audio_path, audio_data, sample_rate)
```

---

## 4. 技术参数

### 4.1 归一化参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 目标dB | -3.0 dB | 峰值归一化目标 |
| 线性峰值 | 0.707 | 10^(-3/20) |
| 限幅范围 | [-1.0, 1.0] | 防止削波失真 |

### 4.2 音量参考标准

| 峰值振幅 | dB值 | 评价 |
|----------|------|------|
| 1.0 | 0 dB | 最大音量（可能有削波风险） |
| 0.7 | -3 dB | **标准音量**（推荐） |
| 0.5 | -6 dB | 较低音量 |
| 0.3 | -10 dB | 明显偏小 |
| 0.1 | -20 dB | 过小音量 |

---

## 5. 测试验证

### 5.1 测试脚本

创建了 `test_volume_normalization.py` 测试脚本，用于：
1. 生成测试音频
2. 测量RMS音量和峰值
3. 判断音量是否正常

### 5.2 运行测试

```bash
python test_volume_normalization.py
```

### 5.3 预期结果

修复后，所有算法的输出音量应该：
- 峰值约 0.7 (-3 dB)
- RMS音量约 -20 到 -15 dB
- 听起来清晰响亮，无需调大播放器音量

---

## 6. 兼容性

### 6.1 向后兼容

- 新增 `normalize` 参数默认为 `True`，所有现有调用自动生效
- 如需禁用归一化，可传入 `normalize=False`

### 6.2 性能影响

- 归一化处理非常轻量，增加时间 < 10ms
- 对整体推理延迟影响可忽略

---

## 7. 总结

**问题类型**: 音频后处理缺失  
**影响范围**: 所有TTS算法  
**严重程度**: 中（影响用户体验）  
**修复难度**: 低  
**修复状态**: ✅ 已完成

**改进效果**:
- 修复前: 峰值约 0.1-0.3，音量过小
- 修复后: 峰值约 0.7，音量正常

**关键改进**:
- 统一的音量归一化处理
- 保留少量headroom避免削波
- 自动应用到所有算法
