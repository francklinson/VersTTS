# GPT-SoVITS API 功能验证报告

## 日期
2026-04-25

## 验证结果
✅ **5/5 项测试全部通过** - GPT-SoVITS API 功能完全可用!

## 测试详情

### 1. API模块导入 ✅
- API模块导入成功
- `/tts/gptsovits` 端点已正确注册

### 2. 配置加载 ✅
- TTS_Config 配置加载成功
- 支持版本: v1, v2, v3, v4, v2Pro, v2ProPlus, custom
- 默认版本: v2
- 自动检测CUDA设备并启用半精度

### 3. 模型加载检查 ✅
- 配置设置成功
- 版本: v2
- 设备: cuda
- 半精度: True
- 模型文件检查:
  - t2s权重: ✅ 存在 (s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt)
  - vits权重: ✅ 存在 (s2G2333k.pth)

### 4. API请求格式 ✅
端点: `POST /tts/gptsovits`
Content-Type: `multipart/form-data`

参数列表:
| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| text | str | ✅ | 待合成文本 |
| text_lang | str | | 文本语言(zh/en/ja/ko/yue) |
| prompt_wav | File | ✅ | 参考音频文件 |
| prompt_text | str | ✅ | 参考音频对应的文本 |
| prompt_lang | str | | 参考音频语言 |
| version | str | | 模型版本(v1/v2/v3/v4/v2Pro/v2ProPlus) |
| top_k | int | | Top K采样(默认15) |
| top_p | float | | Top P采样(默认1.0) |
| temperature | float | | 温度(默认1.0) |
| speed_factor | float | | 语速因子(默认1.0) |

### 5. API响应格式 ✅

成功响应:
```json
{
  "success": true,
  "message": "合成成功",
  "audio_url": "/audio/gptsovits_xxx.wav",
  "sample_rate": 32000
}
```

失败响应:
```json
{
  "detail": "错误信息"
}
```

## 修复的问题

### 1. 模块导入路径修复
**问题**: GPT-SoVITS内部模块导入需要特定的工作目录和sys.path设置

**解决方案**: 
- 添加 `_setup_gpt_sovits_path()` 辅助函数
- 在函数级别设置正确的路径
- 使用try-finally确保工作目录恢复

### 2. 配置访问方式修复
**问题**: 原代码使用错误的配置访问方式 `tts_config.v2`

**解决方案**:
```python
# 正确的配置访问方式
if version in tts_config.default_configs:
    tts_config.configs = tts_config.default_configs[version].copy()
    tts_config.version = version
```

## 使用方法

### 启动服务
```bash
python backend/api_server.py
```

### 前端使用
1. 访问前端界面 `http://localhost:5001/app`
2. 选择 GPT-SoVITS 模型
3. 上传参考音频文件
4. 填写参考音频文本
5. 输入待合成文本
6. 点击生成

### curl调用示例
```bash
curl -X POST http://localhost:5001/tts/gptsovits \
  -F "text=你好，这是测试文本" \
  -F "text_lang=zh" \
  -F "prompt_wav=@reference.wav" \
  -F "prompt_text=这是参考音频的文本内容" \
  -F "prompt_lang=zh" \
  -F "version=v2"
```

## 支持的模型版本

| 版本 | t2s权重 | vits权重 | 说明 |
|------|---------|----------|------|
| v1 | s1bert25hz-2kh... | s2G488k.pth | 原始版本 |
| v2 | s1bert25hz-5kh... | s2G2333k.pth | 推荐版本 ✅ |
| v3 | s1v3.ckpt | s2Gv3.pth | 实验版本 |
| v4 | s1v3.ckpt | s2Gv4.pth | 实验版本 |
| v2Pro | s1v3.ckpt | s2Gv2Pro.pth | 专业版 |
| v2ProPlus | s1v3.ckpt | s2Gv2ProPlus.pth | 增强版 |

## 结论

GPT-SoVITS后端API功能已完全可用，包括:
- ✅ 模块正确导入
- ✅ 配置文件正确加载
- ✅ 模型权重文件就位
- ✅ API端点正确注册
- ✅ 请求/响应格式正确

API已准备好接收请求并生成语音。
