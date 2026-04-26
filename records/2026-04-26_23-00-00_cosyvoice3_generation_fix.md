# 工作记录：修复 CosyVoice 3.0 生成语音混乱问题

**时间**: 2026-04-26 23:00:00
**任务编号**: [23]
**任务描述**: 使用了 CosyVoice 3.0 的模型，但是生成的语音完全是混乱的，无论克隆模式还是 instruct 模式

---

## 1. 问题分析

### 1.1 现象
- 使用 `Fun-CosyVoice3-0.5B` 模型进行语音合成时，无论 `zero_shot` 模式还是 `instruct` 模式，生成的音频都是混乱的（完全听不清内容）

### 1.2 根因定位
通过对 `backend/api_server.py` 中 CosyVoice 调用逻辑的深入分析，发现以下关键问题：

**问题1：zero_shot 模式错误调用了 `inference_cross_lingual`**
- 后端代码在 `zero_shot` 模式下调用了 `cosyvoice.inference_cross_lingual(formatted_text, audio_path)`
- `inference_cross_lingual` 内部调用 `frontend_cross_lingual`，其逻辑为：
  ```python
  def frontend_cross_lingual(self, tts_text, prompt_wav, resample_rate, zero_shot_spk_id):
      model_input = self.frontend_zero_shot(tts_text, '', prompt_wav, resample_rate, zero_shot_spk_id)
      del model_input['prompt_text']
      del model_input['prompt_text_len']
      del model_input['llm_prompt_speech_token']
      del model_input['llm_prompt_speech_token_len']
      return model_input
  ```
- 这意味着 **LLM 完全看不到参考音频的文本内容和 speech token**，只能依赖 speaker embedding 和 flow 模型进行合成
- 对于 CosyVoice 3.0 这种基于大语言模型的架构，缺少 prompt speech token 会导致生成结果严重退化，甚至完全混乱

**问题2：未使用 `reference_text` 进行 zero_shot 克隆**
- 说话人数据库 (`speakers_db.json`) 中已存储了每个说话人的 `reference_text`
- 但后端 `zero_shot` 模式完全没有读取和使用该字段
- CosyVoice 3.0 的 `inference_zero_shot` 要求 `prompt_text` 格式为：`You are a helpful assistant.<|endofprompt|>{参考音频文本}`

**问题3：cross_lingual 模式文本前缀缺失**
- `cross_lingual` 模式下，CosyVoice 3.0 要求 `tts_text` 必须包含前缀 `You are a helpful assistant.<|endofprompt|>`
- 原代码中直接传递了原始 `text`，未添加前缀

---

## 2. 修复内容

### 2.1 修改 `backend/api_server.py`

#### zero_shot 模式修复
- **原逻辑**: 统一使用 `inference_cross_lingual`，不依赖参考文本
- **新逻辑**:
  - 优先读取说话人的 `reference_text`
  - 如果存在 `reference_text`：
    - 格式化为 `You are a helpful assistant.<|endofprompt|>{reference_text}`
    - 调用 `cosyvoice.inference_zero_shot(text, prompt_text, audio_path, stream=False)`
  - 如果不存在 `reference_text`：
    - 回退到 `inference_cross_lingual`，并打印警告日志
    - 确保 `tts_text` 包含 `You are a helpful assistant.<|endofprompt|>` 前缀

#### cross_lingual 模式修复
- 为 `tts_text` 添加 `You are a helpful assistant.<|endofprompt|>` 前缀
- 明确使用 `stream=False` 参数

### 2.2 修改 `test_scripts/test_cosyvoice.py`
- 新增 `Fun-CosyVoice3-0.5B` 模型测试（重点测试）
- 测试用例覆盖：
  - `inference_zero_shot`（带参考文本）
  - `inference_cross_lingual`（带文本前缀）
  - `inference_instruct2`（指令控制）
- 使用项目内置参考音频 `../algorithms/CosyVoice/asset/zero_shot_prompt.wav`
- 添加 `traceback.print_exc()` 以便调试时查看完整堆栈

---

## 3. 关键代码变更

### `backend/api_server.py` - zero_shot 模式
```python
ref_text = speaker.get("reference_text", "")
logger.info(f"zero_shot模式: 使用说话人 {speaker['name']} 的音频: {audio_path}, 参考文本: {ref_text[:50] if ref_text else '无'}")

if ref_text:
    prompt_text = f"You are a helpful assistant.<|endofprompt|>{ref_text}"
    logger.info(f"使用 inference_zero_shot 进行声音克隆，prompt_text 前缀已添加")
    model_output = cosyvoice.inference_zero_shot(text, prompt_text, audio_path, stream=False)
else:
    logger.warning(f"说话人 {speaker['name']} 没有参考文本，回退到 cross_lingual 模式")
    formatted_text = f"You are a helpful assistant.<|endofprompt|>{text}"
    model_output = cosyvoice.inference_cross_lingual(formatted_text, audio_path, stream=False)
```

### `backend/api_server.py` - cross_lingual 模式
```python
formatted_text = f"You are a helpful assistant.<|endofprompt|>{text}"
logger.info(f"cross_lingual模式: 使用格式化文本前缀")
model_output = cosyvoice.inference_cross_lingual(formatted_text, tmp_path, stream=False)
```

---

## 4. 验证计划

1. 启动后端服务：`./start_server.sh`
2. 进入前端页面，选择 CosyVoice 模型
3. 选择 `zero_shot` 模式，选择一个带有 `reference_text` 的参考人声
4. 输入合成文本，点击生成
5. 检查生成的音频是否清晰、自然
6. 运行测试脚本：`python test_scripts/test_cosyvoice.py`
7. 检查 `output_cosyvoice/` 目录下生成的 `cosyvoice3_*.wav` 文件

---

## 5. 参考文档

- `algorithms/CosyVoice/example.py` 中 `cosyvoice3_example()` 函数展示了 CosyVoice 3.0 的正确用法
- `algorithms/CosyVoice/cosyvoice/cli/frontend.py` 中 `frontend_cross_lingual` 和 `frontend_zero_shot` 的实现逻辑
- `algorithms/CosyVoice/cosyvoice/cli/cosyvoice.py` 中 `CosyVoice3` 类的定义

---

## 6. 待后续验证

- [ ] 运行测试脚本验证修复效果
- [ ] 前端实际测试 zero_shot 模式
- [ ] 前端实际测试 instruct 模式
- [ ] 若仍有异常，需排查模型文件完整性（llm.pt / flow.pt / hift.pt / speech_tokenizer_v3.onnx）
