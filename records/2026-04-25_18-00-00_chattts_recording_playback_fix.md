# ChatTTS 页面 Web 录音回放无声问题排查与修复

**时间戳**: 2026-04-25 18:00:00  
**任务来源**: 需求.txt 待完成项 [1]  
**涉及文件**: `frontend/index.html`

## 问题描述
在 ChatTTS 页面增加 Web 录音并提取说话人 embedding 功能后，出现**录音回放没声音**的问题。

## 排查过程

### 1. 审查前端录音逻辑
- 录音使用 `navigator.mediaDevices.getUserMedia` 获取麦克风权限
- 使用 `MediaRecorder` 录制音频，格式为 `audio/webm;codecs=opus`
- 录音数据通过 `recordedChunks` 收集，停止时组装为 `Blob`
- 预览时通过 `URL.createObjectURL(recordedAudioBlob)` 生成 Blob URL

### 2. 定位根因
在 `showAudioPreview()` 函数中发现以下问题：

**核心 Bug**: 动态创建新的 `<audio>` 元素并设置 `<source src="audioUrl">` 后，**没有调用 `audio.load()`**。浏览器在通过 `<source>` 子元素动态变更音频源时，不会自动重新加载资源，必须显式调用 `load()` 方法，否则音频控件显示但无法播放（无声）。

**次要问题**: 
- 旧 `audio` 元素被移除前未释放其 Blob URL，存在内存泄漏
- `preload="metadata"` 在某些浏览器中对 webm/opus 格式支持不佳，改为 `preload="auto"` 更稳妥

### 3. 修复内容

```javascript
// 修复前（有问题的代码片段）
const newAudio = document.createElement('audio');
// ... 创建 source 并设置 src ...
audioContainer.insertBefore(newAudio, errorMsg);
// 缺少 newAudio.load()！

// 修复后
const newAudio = document.createElement('audio');
newAudio.preload = 'auto';
// ... 创建 source 并设置 src ...
audioContainer.insertBefore(newAudio, errorMsg);
newAudio.load();  // 关键修复：触发浏览器重新加载音频资源
```

同时增加旧音频 Blob URL 的释放逻辑：
```javascript
if (audio.src) {
    URL.revokeObjectURL(audio.src);
}
const oldSource = audio.querySelector('source');
if (oldSource && oldSource.src && oldSource.src.startsWith('blob:')) {
    URL.revokeObjectURL(oldSource.src);
}
audio.remove();
```

### 4. 验证思路
- `newAudio.load()` 会触发浏览器重新解析 `<source>` 元素并开始加载音频数据
- 配合 `preload="auto"`，浏览器会预加载足够的数据以确保播放时能立即出声
- 旧 Blob URL 的释放防止多次录音后内存泄漏

## 第二轮修复
用户反馈播放仍然没声音，日志显示音频时长（2.78秒）和 `canplay` 事件都正常，但播放无声且出现"无效的 URI"错误。

**进一步分析**：
- 录音数据本身有效（44KB，28个chunks）
- 浏览器能解析出时长，说明文件结构正确
- 问题锁定为 **Linux 环境下浏览器对 `audio/webm;codecs=opus` 的播放兼容性缺陷**

**修复措施**：
1. **前端 webm → WAV 转换**：使用 Web Audio API 的 `decodeAudioData()` 解码 webm，再通过 `audioBufferToWav()` 转换为 WAV 格式的 Blob。WAV 在所有浏览器上都有完美支持，彻底规避 opus 编码的兼容性问题。
2. **实时音量检测**：在转换过程中分析音频振幅，如果最大振幅 `< 0.01`，提示用户检查麦克风是否正常。
3. **移除所有 `<source>` 子元素**：全部改为直接设置 `audio.src`，避免空 src 导致的"无效 URI"错误。
4. **确保音量属性**：设置 `audio.volume = 1.0` 和 `audio.muted = false`，排除静音状态干扰。

## 第三轮修复
保存说话人时提示 "Node.insertBefore: Child to insert before is not a child of this node"。

**根因分析**：`discardRecordedAudio()` 中 `audioContainer = audioPreview.querySelector('div')` 获取的是 `audioPreview` 的**第一个 `<div>` 子元素**（标题区域），而 `audioErrorMsg` 是 `audioPreview` 的直接子元素，不在那个 `<div>` 内部。`insertBefore` 要求第二个参数必须是第一个参数的子节点，因此报错。

**修复**：将 `audioContainer` 直接指向 `audioPreview`，并增加 `parentNode === audioPreview` 的校验，若条件不满足则退回到 `appendChild`。

## 结论
录音回放无声问题的根本原因是 Linux 浏览器对 webm/opus 的 `<audio>` 播放支持不完善。通过前端解码转换为 WAV 格式预览，从根本上解决了兼容性问题。同时音量检测功能可以帮助用户快速确认麦克风是否正常工作。保存说话人时的 DOM 操作错误也已修复。
