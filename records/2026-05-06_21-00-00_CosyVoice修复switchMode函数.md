# CosyVoice 修复 switchMode 函数

**操作时间**: 2026-05-06 21:00:00

**问题描述**: 用户报告 CosyVoice 的指令控制模式没有生效，查看日志发现所有请求都是 `zero_shot` 模式。

**问题原因**: 前端 HTML 中调用了 `switchMode` 函数来切换模式，但该函数未定义，导致点击"指令控制"标签时无法切换模式。

---

## 修复内容

### 修改文件
- **文件路径**: `frontend/pages/cosyvoice.html`

### 修改详情

**添加的代码**:
```javascript
function switchMode(mode) {
    currentMode = mode;
    
    // 更新标签样式
    document.querySelectorAll('.mode-tab').forEach(tab => {
        tab.classList.remove('active');
    });
    event.target.classList.add('active');
    
    // 更新内容显示
    document.querySelectorAll('.mode-content').forEach(content => {
        content.classList.remove('active');
    });
    document.getElementById('mode-' + mode).classList.add('active');
}
```

---

## 功能说明

现在 CosyVoice 支持两种模式：

### 1. 零样本克隆 (zero_shot)
- 选择说话人
- 使用参考音频克隆音色

### 2. 指令控制 (instruct)
- 选择说话人
- 输入指令文本（如"用四川话说"、"用开心的语气说"）
- 支持 18+ 种方言和情感控制

---

**操作状态**: 已完成
