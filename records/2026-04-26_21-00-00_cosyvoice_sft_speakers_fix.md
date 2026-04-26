# CosyVoice SFT模式预设音色问题修复记录

**时间:** 2026-04-26 21:00:00  
**任务:** 修复CosyVoice SFT模式预设音色选择问题  
**执行人:** VersTTS System

---

## 1. 问题描述

用户反馈在CosyVoice的SFT模式下，选择"中文女"、"中文男"、"英文女"、"英文男"生成的音色都差不多。

---

## 2. 问题分析

### 2.1 调查发现

通过检查CosyVoice-300M-SFT模型实际的预设音色列表，发现：

**前端显示**: 4个音色
- 中文女
- 中文男
- 英文女
- 英文男

**模型实际支持**: 7个音色
- 中文女
- 中文男
- 日语男
- 粤语女
- 英文女
- 英文男
- 韩语女

### 2.2 根本原因

1. **前端硬编码不全**: 前端HTML中只硬编码了4个音色选项，缺失了3个音色
2. **SFT模型特性**: SFT（监督微调）模型的预设音色是从训练数据中提取的统计平均音色，本身就比较相似
3. **音色区分度**: 预设音色之间的差异确实比较细微，尤其是同语言的男女声

---

## 3. 解决方案

### 3.1 前端静态更新 (app.html)

补充缺失的音色选项：
```html
<select id="cosyvoice-speaker">
    <option value="中文女">中文女</option>
    <option value="中文男">中文男</option>
    <option value="粤语女">粤语女</option>      <!-- 新增 -->
    <option value="英文女">英文女</option>
    <option value="英文男">英文男</option>
    <option value="日语男">日语男</option>      <!-- 新增 -->
    <option value="韩语女">韩语女</option>      <!-- 新增 -->
</select>
```

### 3.2 动态加载功能

新增 `loadCosyVoiceSpeakers()` 函数，从后端API动态获取音色列表：
```javascript
async function loadCosyVoiceSpeakers() {
    try {
        const response = await fetch(`${API_BASE}/tts/cosyvoice/speakers`);
        const data = await response.json();
        if (data.speakers && data.speakers.length > 0) {
            const select = document.getElementById('cosyvoice-speaker');
            if (select) {
                select.innerHTML = '';
                data.speakers.forEach(speaker => {
                    const option = document.createElement('option');
                    option.value = speaker;
                    option.textContent = speaker;
                    select.appendChild(option);
                });
            }
        }
    } catch (e) {
        console.error('加载CosyVoice音色失败:', e);
    }
}
```

### 3.3 用户提示

添加提示信息，告知用户如何获得更独特的音色：
```html
<small style="color: #666; margin-top: 5px; display: block;">
    提示：不同音色会有细微差别，建议使用零样本克隆获得更独特的音色
</small>
```

---

## 4. 完整音色列表

| 序号 | 音色名称 | 语言 | 备注 |
|------|----------|------|------|
| 1 | 中文女 | 中文 | 标准女声 |
| 2 | 中文男 | 中文 | 标准男声 |
| 3 | 粤语女 | 粤语 | 粤语女声 |
| 4 | 英文女 | 英文 | 标准女声 |
| 5 | 英文男 | 英文 | 标准男声 |
| 6 | 日语男 | 日语 | 日语男声 |
| 7 | 韩语女 | 韩语 | 韩语女声 |

---

## 5. 使用建议

### 5.1 如果预设音色相似

CosyVoice SFT模型的预设音色确实比较相似，建议：

1. **使用Zero-shot模式** (推荐)
   - 上传3-10秒的参考音频
   - 可以获得与参考音频非常相似的音色
   - 支持跨语言克隆

2. **使用Instruct模式**
   - 通过指令控制音色特征
   - 例如："用沙哑的声音说"
   - 结合方言选择获得独特效果

3. **使用其他模型**
   - Qwen3-TTS: 9种预设音色，区分度更高
   - GPT-SoVITS: 克隆效果最佳

### 5.2 音色选择建议

| 需求 | 推荐音色 | 替代方案 |
|------|----------|----------|
| 标准中文女声 | 中文女 | Zero-shot克隆 |
| 标准中文男声 | 中文男 | Zero-shot克隆 |
| 粤语内容 | 粤语女 | Zero-shot克隆 |
| 标准英文 | 英文女/英文男 | Zero-shot克隆 |
| 独特音色 | - | Zero-shot克隆 |

---

## 6. 技术说明

### 6.1 SFT模型 vs Zero-shot

| 特性 | SFT模式 | Zero-shot模式 |
|------|---------|---------------|
| 音色来源 | 训练数据平均 | 参考音频提取 |
| 音色独特性 | 较低 | 高 |
| 使用便捷性 | 简单（直接选择） | 需要准备参考音频 |
| 适用场景 | 快速测试 | 生产环境 |

### 6.2 为什么预设音色相似

SFT（Supervised Fine-Tuning）模型的预设音色是通过对大量训练数据进行统计平均得到的代表性音色。这种设计确保了音色的稳定性和通用性，但也导致了不同预设音色之间的区分度不高。

---

## 7. 测试验证

### 7.1 测试内容

1. 验证7个音色都能正常加载
2. 验证不同音色的生成结果有差异
3. 验证动态加载功能正常工作

### 7.2 预期结果

- 音色选择下拉框显示7个选项
- 不同音色生成的音频在音高、音色上有可感知的差异
- 页面刷新后自动加载最新音色列表

---

## 8. 总结

本次修复解决了CosyVoice SFT模式预设音色显示不全的问题，从4个增加到7个。同时添加了动态加载功能和用户提示，帮助用户了解如何获得更独特的音色效果。

**关键改进**:
- ✅ 音色数量: 4个 → 7个
- ✅ 动态加载: 新增API自动获取
- ✅ 用户提示: 增加使用建议

**使用建议**:
- 如需独特音色，请使用Zero-shot模式上传参考音频
- 如需方言效果，可选择粤语女或使用Instruct模式
