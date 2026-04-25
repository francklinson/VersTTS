# GPT-SoVITS 音频生成成功

## 日期
2026-04-25

## 执行结果
✅ **成功使用GPT-SoVITS生成音频**

## 生成的音频文件
- **文件路径**: `output/gptsovits_generated_20260425_152318.wav`
- **文件大小**: 208KB (212,524 bytes)
- **采样率**: 32000Hz
- **时长**: 约6.6秒

## 生成参数
- **模型版本**: v2
- **文本语言**: 中文 (zh)
- **参考音频**: `output/case1_promptSingle_synSingle_direct_icl_0.wav` (5.04秒)
- **参考文本**: "你好，欢迎使用语音合成服务。"
- **待合成文本**: "你好，这是使用GPT-SoVITS生成的语音。声音克隆技术真的很神奇！"
- **Top K**: 15
- **Top P**: 1.0
- **温度**: 1.0
- **语速因子**: 1.0

## 修复的问题

### 问题1: G2PW模型路径
**问题**: 代码尝试下载G2PW模型到 `GPT_SoVITS/text/G2PWModel_1.1.zip`，但路径不存在

**解决方案**:
- 创建符号链接 `GPT_SoVITS -> GPT-SoVITS`
- 复制zip文件到正确位置

### 问题2: BERT模型路径
**问题**: `chinese2.py` 使用环境变量 `bert_path`，默认是相对路径 `GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large`

**解决方案**:
- 在后端API的 `_setup_gpt_sovits_path()` 函数中设置 `bert_path` 环境变量为绝对路径

```python
# 设置BERT模型路径环境变量（避免使用相对路径）
bert_path = os.path.join(gpt_sovits_module, "pretrained_models", "chinese-roberta-wwm-ext-large")
os.environ["bert_path"] = bert_path
```

### 问题3: 默认配置路径
**问题**: `TTS.py` 中的 `default_configs` 使用相对路径

**解决方案**:
- 将 `TTS.py` 中的 `default_configs` 所有路径改为绝对路径
- 将 `tts_infer.yaml` 中的所有路径改为绝对路径

## 使用方法

### 通过API调用
```python
import requests

url = 'http://localhost:5001/tts/gptsovits'
data = {
    'text': '你好，这是使用GPT-SoVITS生成的语音。',
    'text_lang': 'zh',
    'prompt_text': '你好，欢迎使用语音合成服务。',
    'prompt_lang': 'zh',
    'version': 'v2',
}

with open('reference.wav', 'rb') as f:
    files = {'prompt_wav': f}
    response = requests.post(url, data=data, files=files)

result = response.json()
if result.get('success'):
    print(f"音频URL: {result.get('audio_url')}")
```

### 通过前端界面
1. 访问 `http://localhost:5001/app`
2. 选择 **GPT-SoVITS** 模型
3. 上传参考音频文件 (3-10秒)
4. 填写参考音频对应的文本
5. 输入待合成文本
6. 点击生成按钮

## 注意事项
1. 参考音频时长必须在 **3-10秒** 范围内
2. 必须提供参考音频对应的文本 (`prompt_text`)
3. 首次加载模型需要 **2-3分钟**
4. 支持跨语言合成（如用中文参考音频合成英文）

## 支持的模型版本
- ✅ v1 - 原始版本
- ✅ v2 - 推荐版本
- ✅ v3 - 实验版本
- ✅ v4 - 实验版本
- ✅ v2Pro - 专业版
- ✅ v2ProPlus - 增强版
