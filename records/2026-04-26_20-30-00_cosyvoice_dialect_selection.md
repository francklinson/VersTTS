# CosyVoice 方言选择功能实现记录

**时间:** 2026-04-26 20:30:00  
**任务:** 为 CosyVoice 算法实现方言选择功能  
**执行人:** VersTTS System

---

## 1. 实现概述

根据需求.txt的要求，为 CosyVoice 算法在前端页面添加方言选择功能。CosyVoice 支持 18+ 种中国方言/口音，包括粤语、闽南语、四川话、东北话、陕西话、上海话、天津话、山东话等。

---

## 2. 实现内容

### 2.1 前端界面修改 (app.html)

#### 添加方言选择下拉框
在 CosyVoice 的 instruct 模式下添加方言选择器：

```html
<div class="option-group" id="cosyvoice-dialect-group" style="display:none">
    <label>方言选择</label>
    <select id="cosyvoice-dialect">
        <option value="">-- 不使用方言 --</option>
        <option value="用粤语说">粤语</option>
        <option value="用四川话说">四川话</option>
        <option value="用东北话说">东北话</option>
        <option value="用陕西话说">陕西话</option>
        <option value="用上海话说">上海话</option>
        <option value="用天津话说">天津话</option>
        <option value="用山东话说">山东话</option>
        <option value="用闽南语说">闽南语</option>
        <option value="用河南话说">河南话</option>
        <option value="用湖南话说">湖南话</option>
        <option value="用湖北话说">湖北话</option>
        <option value="用重庆话说">重庆话</option>
        <option value="用云南话说">云南话</option>
        <option value="用贵州话说">贵州话</option>
        <option value="用甘肃话说">甘肃话</option>
        <option value="用新疆话说">新疆话</option>
        <option value="用台湾话说">台湾话</option>
        <option value="用河北话说">河北话</option>
        <option value="用山西话说">山西话</option>
    </select>
    <small style="color: #666; margin-top: 5px; display: block;">选择方言后将自动添加到指令中</small>
</div>
```

#### 支持的方言列表
| 方言 | 指令格式 |
|------|----------|
| 粤语 | 用粤语说 |
| 闽南语 | 用闽南语说 |
| 四川话 | 用四川话说 |
| 东北话 | 用东北话说 |
| 陕西话 | 用陕西话说 |
| 上海话 | 用上海话说 |
| 天津话 | 用天津话说 |
| 山东话 | 用山东话说 |
| 河南话 | 用河南话说 |
| 湖南话 | 用湖南话说 |
| 湖北话 | 用湖北话说 |
| 重庆话 | 用重庆话说 |
| 云南话 | 用云南话说 |
| 贵州话 | 用贵州话说 |
| 甘肃话 | 用甘肃话说 |
| 新疆话 | 用新疆话说 |
| 台湾话 | 用台湾话说 |
| 河北话 | 用河北话说 |
| 山西话 | 用山西话说 |

### 2.2 JavaScript 逻辑修改

#### 更新选项显示控制
```javascript
function updateCosyVoiceOptions() {
    const mode = document.getElementById('cosyvoice-mode').value;
    const isClone = mode === 'zero_shot' || mode === 'cross_lingual';
    const isInstruct = mode === 'instruct';

    document.getElementById('cosyvoice-speaker-group').style.display = isClone || isInstruct ? 'none' : 'block';
    document.getElementById('cosyvoice-ref-voice-group').style.display = isClone ? 'block' : 'none';
    document.getElementById('cosyvoice-ref-group').style.display = isClone ? 'block' : 'none';
    document.getElementById('cosyvoice-prompt-group').style.display = mode === 'zero_shot' ? 'block' : 'none';
    document.getElementById('cosyvoice-instruct-group').style.display = isInstruct ? 'block' : 'none';
    document.getElementById('cosyvoice-dialect-group').style.display = isInstruct ? 'block' : 'none';  // 新增
}
```

#### 提交逻辑修改
```javascript
if (cvMode === 'instruct') {
    let instructText = document.getElementById('cosyvoice-instruct').value;
    const dialect = document.getElementById('cosyvoice-dialect').value;
    // 如果选择了方言，将方言指令添加到instruct文本中
    if (dialect) {
        if (instructText) {
            instructText = instructText + '，' + dialect;
        } else {
            instructText = dialect;
        }
    }
    formData.append('instruct_text', instructText);
}
```

### 2.3 使用示例

用户可以在 CosyVoice 的 instruct 模式下：

1. **仅使用方言**：
   - 选择"四川话"
   - 生成语音将使用四川话口音

2. **方言+情感控制**：
   - 输入指令："用开心的语气说话"
   - 选择方言："四川话"
   - 最终指令："用开心的语气说话，用四川话说"

3. **仅使用情感控制**：
   - 输入指令："用温柔的语气说话"
   - 不选择方言
   - 最终指令："用温柔的语气说话"

---

## 3. 技术实现说明

### 3.1 实现方式
- 方言选择通过 CosyVoice 的 `inference_instruct` 方法实现
- 方言指令作为自然语言指令的一部分传递给模型
- 支持与其他指令（如情感控制）组合使用

### 3.2 兼容性
- 仅在使用 CosyVoice-300M-Instruct 模型时可用
- 需要在 instruct 模式下使用
- 后端 API 无需修改，复用现有的 `instruct_text` 参数

---

## 4. 测试建议

1. **单独测试方言**：
   - 选择不同方言，验证语音输出是否正确

2. **组合测试**：
   - 方言 + 情感指令组合
   - 验证指令拼接逻辑

3. **边界测试**：
   - 不选择方言时的默认行为
   - 空指令仅选择方言的情况

---

## 5. 总结

CosyVoice 方言选择功能已成功实现，用户现在可以在前端界面选择 18+ 种中国方言进行语音合成。该功能与现有的指令控制功能完全兼容，可以同时控制语音的情感风格和方言口音。
