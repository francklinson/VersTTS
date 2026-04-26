# CosyVoice 与 Qwen3-TTS transformers 版本依赖分析

**时间**: 2026-04-26 12:58:00

## 问题背景

- **CosyVoice** requirements.txt 指定: `transformers==4.51.3`
- **Qwen3-TTS** 要求: `transformers==4.57.3`

## 分析结果

### 1. CosyVoice 实际依赖分析

通过检查 CosyVoice 代码，发现其对 transformers 的依赖主要是：

```python
from transformers import Qwen2ForCausalLM
from transformers import AutoTokenizer
```

这些都是 transformers 的基础功能，不涉及版本特定的内部 API。

### 2. 兼容性测试

**测试环境**: transformers==4.57.3, tokenizers==0.22.2

| 模型 | 测试模式 | 结果 | 音频文件 |
|-----|---------|------|---------|
| CosyVoice 3.0 | zero_shot | ✅ 成功 | 384KB, 24kHz |
| Qwen3-TTS | voice_clone | ✅ 成功 | 146KB, 24kHz |

### 3. 版本指定原因分析

CosyVoice 指定 `transformers==4.51.3` 的原因可能是：

1. **开发/测试时的稳定版本**: 4.51.3 是 CosyVoice 开发时测试通过的版本
2. **预防未来兼容性问题**: 锁定版本避免 transformers 更新带来的潜在问题
3. **vLLM 兼容性**: CosyVoice 使用 vLLM 进行推理，需要与特定 transformers 版本配合

### 4. 实际兼容性结论

**CosyVoice 可以在 transformers 4.57.3 下正常工作**，原因：

- 只使用了 transformers 的公开 API (`Qwen2ForCausalLM`, `AutoTokenizer`)
- 没有使用版本特定的内部功能
- 4.51.3 → 4.57.3 是向后兼容的升级

## 建议

### 方案 1: 统一使用 transformers 4.57.3 (推荐)

两个模型都可以在此版本下正常工作，无需维护两套环境。

```bash
pip install transformers==4.57.3
```

### 方案 2: 如果需要回退到 4.51.3

```bash
pip install transformers==4.51.3 tokenizers==0.20.3
```

**注意**: 回退后 Qwen3-TTS 会再次失败。

## 当前状态

当前环境已统一使用：
- `transformers==4.57.3`
- `tokenizers==0.22.2`

两个模型均测试通过 ✅
