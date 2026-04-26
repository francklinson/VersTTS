# Transformers 版本冲突问题总结

**时间**: 2026-04-26 14:10:00

## 问题背景

- **CosyVoice**: 需要 `transformers==4.51.3` 才能生成正常音频
- **Qwen3-TTS**: 需要 `transformers>=4.57.0` 才能正常运行

## 兼容性分析

### 尝试 transformers 4.55.2
- CosyVoice: 可以加载，但生成的音频有杂音 ❌
- Qwen3-TTS: 无法加载（缺少大量 API）❌

### 尝试 transformers 4.51.3
- CosyVoice: 缺少 `rope_config_validation` 函数 ❌
- Qwen3-TTS: 缺少大量 API (masking_utils, layer_type_validation, 等) ❌

### 尝试 transformers 4.57.3
- CosyVoice: 可以加载，但生成的音频有杂音 ❌
- Qwen3-TTS: 正常工作 ✅

## 结论

**两个模型存在根本性的 transformers 版本冲突，无法在同一环境中同时正常工作。**

### 可能的解决方案

1. **子进程隔离**（方案2）
   - 为 Qwen3-TTS 创建独立的 Python 进程/服务
   - 主服务使用 transformers 4.51.3
   - Qwen3-TTS 服务使用 transformers 4.57.3
   - 通过进程间通信或 HTTP API 调用

2. **Docker 隔离**
   - 为两个模型分别创建 Docker 容器
   - 每个容器安装不同版本的 transformers

3. **模型选择**
   - 根据需求选择使用 CosyVoice 或 Qwen3-TTS
   - 每次切换时重新安装对应版本的 transformers

## 当前状态

- transformers 版本: 4.51.3
- CosyVoice: 无法使用（缺少 `rope_config_validation`）
- Qwen3-TTS: 明确返回版本不兼容错误（HTTP 503）

## 建议

推荐实现**子进程隔离方案**，为 Qwen3-TTS 创建独立的子进程服务，这样可以同时满足两个模型的版本要求。
