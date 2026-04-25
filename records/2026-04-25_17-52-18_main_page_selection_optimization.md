# 主页面模型选择效果优化记录

**时间**: 2026-04-25 17:52:18
**任务**: 优化主页面模型选择后的视觉效果

---

## 1. 优化目标

改进用户选择TTS模型后的视觉反馈，让用户清楚知道当前选中了哪个模型。

---

## 2. 实现内容

### 2.1 选中卡片视觉效果增强

**CSS 更新**:
- 添加 `.model-card.active` 选中状态样式
- 边框从透明变为 3px 彩色实线
- 多层阴影效果：外发光 + 内发光
- 卡片轻微上浮 (transform: translateY(-3px))

**新增元素**:
- 右上角勾选标记 (✓) - 带弹入动画
- 微光扫过效果 (shimmer animation)

### 2.2 按钮状态变化

**默认状态**:
- 白底紫字，带边框
- 文字: "选择"

**悬停状态**:
- 填充变色
- 上浮阴影
- 波纹扩散效果

**选中状态**:
- 文字变为: "✓ 已选择"
- 渐变背景色
- 阴影发光效果

### 2.3 模型专属主题色

为每个模型配置独特的主题色：

| 模型 | 主题色 | 色值 |
|------|--------|------|
| ChatTTS | 蓝色 | #2196f3 |
| CosyVoice | 紫色 | #9c27b0 |
| F5-TTS | 绿色 | #4caf50 |
| Qwen3-TTS | 橙色 | #ff9800 |
| OpenVoice | 粉色 | #e91e63 |
| GPT-SoVITS | 青色 | #00bcd4 |

### 2.4 标题标签

- 在模型名称旁添加 "当前选择" 标签
- 标签带脉动呼吸动画 (tagPulse)
- 颜色与模型主题一致

### 2.5 交互增强

**JavaScript 更新** (`selectModel` 函数):
- 切换选中状态时更新按钮文字
- 添加选中动画触发
- 自动滚动到配置区域

---

## 3. 代码变更

### 3.1 CSS 样式新增

```css
/* 选中卡片样式 */
.model-card.active {
    border: 3px solid #667eea;
    box-shadow: 
        0 15px 50px rgba(102, 126, 234, 0.3),
        0 0 0 4px rgba(102, 126, 234, 0.1),
        inset 0 0 30px rgba(102, 126, 234, 0.05);
    transform: translateY(-3px);
}

/* 勾选标记 */
.model-card.active::before {
    content: '✓';
    /* 位置和样式 */
    animation: checkPop 0.3s ease-out;
}

/* 选中标签 */
.selected-tag {
    display: none;
    /* 基础样式 */
    animation: tagPulse 1.5s ease-in-out infinite;
}

.model-card.active .selected-tag {
    display: inline-block;
}

/* 各模型主题色 */
.model-card.active[data-model="chattts"] { border-color: #2196f3; ... }
.model-card.active[data-model="cosyvoice"] { border-color: #9c27b0; ... }
/* ... */
```

### 3.2 JavaScript 更新

```javascript
function selectModel(model) {
    currentModel = model;

    // 更新所有卡片样式
    document.querySelectorAll('.model-card').forEach(card => {
        card.classList.remove('active');
        const btn = card.querySelector('.btn-select');
        if (btn) btn.innerHTML = '<span>选择</span>';
    });

    // 设置选中状态
    const selectedCard = document.querySelector(`[data-model="${model}"]`);
    selectedCard.classList.add('active');
    
    // 更新按钮文字
    const selectedBtn = selectedCard.querySelector('.btn-select');
    if (selectedBtn) selectedBtn.innerHTML = '<span>✓ 已选择</span>';

    // 触发选中动画
    selectedCard.style.animation = 'none';
    setTimeout(() => selectedCard.style.animation = '', 10);

    // 显示对应选项
    document.querySelectorAll('.model-options').forEach(opt => {
        opt.style.display = 'none';
    });
    document.getElementById(`${model}-options`).style.display = 'block';

    // 滚动到配置区域
    setTimeout(() => {
        const optionsSection = document.querySelector('.main-interface');
        if (optionsSection) {
            optionsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }
    }, 100);
}
```

### 3.3 HTML 更新

在每个模型卡片的标题中添加选中标签：

```html
<h3>ChatTTS <span class="selected-tag">当前选择</span></h3>
```

---

## 4. 效果预览

选中模型后，用户可以看到：

1. **卡片边框** 变为彩色
2. **勾选标记** 出现在右上角
3. **按钮文字** 变为 "✓ 已选择"
4. **标题旁** 显示 "当前选择" 标签
5. **自动滚动** 到配置选项区域

---

## 5. 文件变更

- **修改**: `frontend/index.html`
  - CSS 样式添加
  - JavaScript 函数更新
  - HTML 结构微调

---

## 6. 测试验证

- ✅ 选择不同模型时正确显示主题色
- ✅ 勾选标记动画正常
- ✅ 按钮状态切换正确
- ✅ 自动滚动功能正常
- ✅ 各模型颜色区分明显

---

**状态**: ✅ 已完成
