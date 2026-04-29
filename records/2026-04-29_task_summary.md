# VersTTS 项目工作记录 - 2026-04-29

**日期**: 2026-04-29  
**工作时长**: 全天  
**任务状态**: 主要任务已完成

---

## 一、任务清单完成情况

### ✅ 已完成任务

#### 1. GPT-SoVITS 调试与集成 (核心任务)

| 子任务 | 状态 | 详情 |
|--------|------|------|
| 检查模型文件 | ✅ | 发现模型缺失，已下载所有必需模型 |
| 下载模型文件 | ✅ | 从HuggingFace下载V1/V2/V3/V4所有版本模型 |
| 修复配置文件 | ✅ | 修复 `tts_infer.yaml` 和 `TTS.py` 中的路径问题 |
| 修改后端API | ✅ | 支持 `clone_speaker_id` 参数，从说话人管理模块获取音频 |
| 修改前端页面 | ✅ | 移除参考音频上传，改为说话人选择下拉框 |
| 修复V3/V4版本 | ✅ | 修复版本切换时管道未重新初始化的问题 |
| 测试验证 | ✅ | 所有版本(V1/V2/V3/V4)均可正常生成语音 |

**关键修改文件**:
- `backend/api_server.py` - 添加说话人选择支持，修复版本切换逻辑
- `frontend/pages/gptsovits.html` - 移除文件上传，添加说话人选择
- `algorithms/GPT-SoVITS/GPT_SoVITS/TTS_infer_pack/TTS.py` - 修复模型路径
- `algorithms/GPT-SoVITS/GPT_SoVITS/configs/tts_infer.yaml` - 修复配置路径

#### 2. 输出目录统一修改

| 修改项 | 状态 | 详情 |
|--------|------|------|
| 目录迁移 | ✅ | 将 `output/` 目录内容迁移到 `outputs/` |
| 代码修改 | ✅ | 修改 `api_server.py` 中所有输出路径 |
| 删除旧目录 | ✅ | 删除空的 `output` 目录 |

**影响**: 所有TTS方案的语音输出现在都统一放到 `outputs` 目录

#### 3. Git代码管理

| 子任务 | 状态 | 详情 |
|--------|------|------|
| 分析项目结构 | ✅ | 分析8个TTS项目的文件结构 |
| 创建.gitignore | ✅ | 为每个项目创建Git忽略配置文件 |
| 添加核心代码 | ✅ | 添加所有项目的Python源码和配置文件到Git |
| 提交更改 | ✅ | 完成Git提交 |

**提交统计**:
- GPT-SoVITS: 161个文件 (之前已提交)
- 其他7个TTS: 449个文件
- 总计: 610个文件，约20万行代码

---

## 二、Git仓库代码统计

### 总览

| 指标 | 数值 |
|------|------|
| **总文件数** | 835 |
| **总代码行数** | **478,851 行** |

### 按类型统计

| 类型 | 文件数 | 代码行数 |
|------|--------|----------|
| Python (.py) | 592 | 156,415 |
| HTML (.html) | 16 | 8,666 |
| JavaScript (.js) | 2 | 1,498 |
| CSS (.css) | 2 | 2,413 |

### TTS算法代码排行

| 排名 | 算法 | Python文件数 | 代码行数 |
|------|------|-------------|----------|
| 1 | **GPT-SoVITS** | 151 | 41,236 |
| 2 | **IndexTTS** | 134 | 40,439 |
| 3 | **CosyVoice** | 124 | 25,465 |
| 4 | **F5-TTS** | 40 | 12,886 |
| 5 | **ChatTTS** | 68 | 12,353 |
| 6 | **Qwen3-TTS** | 25 | 10,425 |
| 7 | **FireRedTTS2** | 21 | 4,896 |
| 8 | **OpenVoice** | 17 | 3,677 |

---

## 三、技术难点与解决方案

### 1. GPT-SoVITS V3版本生成无效音频

**问题**: 切换V3版本时，生成的音频是静音或无效

**原因**: 版本切换时，TTS推理管道未重新初始化，继续使用V2的模型

**解决方案**: 修改 `init_gpt_sovits_pipeline` 函数，添加版本检查逻辑
```python
# 检查是否需要重新初始化管道
cached_version = model_info.get("pipeline_version")
current_version = model_info.get("version")

if pipeline is None or cached_version != current_version:
    # 清理旧管道并重新初始化
    ...
```

### 2. 模型路径配置错误

**问题**: TTS.py中的默认模型路径缺少 `algorithms/` 目录

**解决方案**: 批量替换路径
```bash
sed -i 's|/home/.../GPT-SoVITS/|/home/.../algorithms/GPT-SoVITS/|g' TTS.py
```

### 3. CUDA环境问题

**问题**: 出现 `CUDA error: device busy` 错误

**解决方案**: 
- 杀死占用GPU的进程
- 重启后端服务
- 确保GPU内存充足

---

## 四、模型文件清单

### GPT-SoVITS 已下载模型 (~4.6GB)

| 模型 | 文件 | 大小 | 用途 |
|------|------|------|------|
| BERT | chinese-roberta-wwm-ext-large | 622MB | 文本编码 |
| HuBERT | chinese-hubert-base | 181MB | 特征提取 |
| V1 GPT | s1bert25hz-2kh | 148MB | GPT模型 |
| V1 SoVITS | s2G488k | 101MB | V1生成器 |
| V2 GPT | s1bert25hz-5kh | 148MB | V2 GPT模型 |
| V2 SoVITS | s2G2333k | 101MB | V2生成器 |
| V3 GPT | s1v3 | 149MB | V3/V4 GPT模型 |
| V3 SoVITS | s2Gv3 | 734MB | V3生成器 |
| V4 SoVITS | s2Gv4 | 734MB | V4生成器 |

---

## 五、文件变更清单

### 修改的文件

1. `backend/api_server.py`
   - 修改输出目录: output -> outputs
   - 修改GPT-SoVITS API支持说话人选择
   - 修复版本切换逻辑

2. `frontend/pages/gptsovits.html`
   - 移除参考音频上传
   - 添加说话人选择下拉框

3. `algorithms/GPT-SoVITS/GPT_SoVITS/TTS_infer_pack/TTS.py`
   - 修复模型路径配置

4. `algorithms/GPT-SoVITS/GPT_SoVITS/configs/tts_infer.yaml`
   - 修复路径为绝对路径

### 新增的文件

1. `algorithms/GPT-SoVITS/download_models.py` - 模型下载脚本
2. `algorithms/GPT-SoVITS/.gitignore` - Git忽略配置
3. `algorithms/GPT-SoVITS/GIT_MANAGEMENT_GUIDE.md` - Git管理指南
4. `test_scripts/test_gptsovits.py` - GPT-SoVITS测试脚本
5. 其他7个TTS项目的 `.gitignore` 文件

---

## 六、测试验证结果

### GPT-SoVITS 测试结果

| 版本 | 状态 | 耗时 | 音频时长 |
|------|------|------|----------|
| V1 | ✅ 通过 | ~2s | 5.5s |
| V2 | ✅ 通过 | ~7s | 5.5s |
| V3 | ✅ 通过 | ~1s | 5.5s |
| V4 | ✅ 通过 | ~2s | 5.5s |

### 后端API测试

- ✅ 健康检查接口正常
- ✅ 说话人列表API正常
- ✅ 所有TTS API正常响应

---

## 七、待优化项

### 低优先级

1. **VoxCPM**: 尚未集成到Git，代码行数统计显示为0
2. **代码优化**: 部分TTS方案的代码有重复，可进一步抽象
3. **文档完善**: 各TTS方案的使用文档需要补充

---

## 八、Git提交记录

```
d70560c - Add TTS algorithms core source code and configurations
919b73d - Add GPT-SoVITS core source code and configuration files
```

**提交详情**:
- 总提交数: 2次
- 新增文件: 610个
- 新增代码行: ~200,000行

---

## 九、总结

今日主要完成了GPT-SoVITS的调试集成工作，包括模型下载、前后端修改、版本切换修复等。同时完成了所有TTS项目的Git代码管理，将核心代码提交到仓库，排除了大体积的模型文件。

**核心成果**:
1. ✅ GPT-SoVITS 完全可用，支持V1/V2/V3/V4四个版本
2. ✅ 输出目录统一为 `outputs`
3. ✅ Git仓库包含所有TTS核心代码，共478,851行

**记录时间**: 2026-04-29 19:30:00  
**记录人**: AI Assistant
