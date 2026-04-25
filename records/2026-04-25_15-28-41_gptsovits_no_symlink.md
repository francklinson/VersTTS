# GPT-SoVITS 无符号链接实现

## 日期
2026-04-25

## 目标
移除所有符号链接，通过修改代码正确处理路径

## 已删除的符号链接
1. ✅ `GPT_SoVITS -> GPT-SoVITS` (项目根目录)
2. ✅ `GPT-SoVITS/text -> GPT_SoVITS/text` (GPT-SoVITS目录)

## 修改的文件

### 1. `GPT-SoVITS/GPT_SoVITS/text/g2pw/onnx_api.py`
修改 `download_and_decompress` 函数，使用绝对路径：

```python
def download_and_decompress(model_dir: str = "G2PWModel/"):
    # 将相对路径转换为绝对路径（基于当前文件位置）
    if not os.path.isabs(model_dir):
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(current_file_dir, model_dir)
    
    model_dir = os.path.normpath(model_dir)
    ...
```

### 2. `GPT-SoVITS/GPT_SoVITS/TTS_infer_pack/TTS.py`
修改 `default_configs`，使用绝对路径：
- 所有模型路径改为绝对路径
- 例如：`"bert_base_path": "/home/zhouchenghao/PycharmProjects/VersTTS/..."`

### 3. `GPT-SoVITS/GPT_SoVITS/configs/tts_infer.yaml`
所有配置项使用绝对路径

### 4. `backend/api_server.py`
修改 `_setup_gpt_sovits_path` 函数：
```python
def _setup_gpt_sovits_path():
    ...
    # 设置BERT模型路径环境变量（使用绝对路径）
    bert_path = os.path.join(gpt_sovits_module, "pretrained_models", "chinese-roberta-wwm-ext-large")
    os.environ["bert_path"] = bert_path
    
    # 设置G2PW模型路径环境变量（使用绝对路径）
    g2pw_model_path = os.path.join(gpt_sovits_module, "text", "G2PWModel")
    os.environ["g2pw_model"] = g2pw_model_path
    
    # 确保G2PW模型目录存在
    os.makedirs(g2pw_model_path, exist_ok=True)
    ...
```

## 验证结果

### 音频生成成功
- **文件**: `output/gptsovits_no_symlink_20260425_152802.wav`
- **大小**: 291,884 bytes
- **采样率**: 32000Hz
- **生成时间**: 2026-04-25 15:28

### 确认无符号链接
```
✅ GPT_SoVITS 符号链接已删除
✅ text 符号链接已删除
```

## 正确做法总结

### ❌ 不推荐的做法
- 使用符号链接解决路径问题
- 依赖相对路径
- 假设工作目录

### ✅ 推荐的做法
1. **代码中使用绝对路径**
   - 基于 `__file__` 计算绝对路径
   - 使用 `os.path.abspath()` 和 `os.path.dirname()`

2. **环境变量传递路径**
   - `bert_path` - BERT模型路径
   - `g2pw_model` - G2PW模型路径

3. **创建必要的目录**
   - 使用 `os.makedirs(path, exist_ok=True)`

4. **配置文件中统一使用绝对路径**
   - YAML配置文件
   - Python默认配置字典

## 好处
- 不依赖文件系统链接
- 代码可移植性更好
- 路径问题在代码层面解决
- 更清晰的路径管理
