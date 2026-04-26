# CosyVoice 音色参数Bug修复记录

**时间:** 2026-04-26 21:30:00  
**任务:** 修复CosyVoice选择"英文男"却输出女声的Bug  
**执行人:** VersTTS System

---

## 1. 问题描述

**用户反馈**: 在CosyVoice的SFT模式下，选择说话人为"英文男"，但实际输出的是女声。

---

## 2. 问题分析

### 2.1 代码审查

**前端代码** (app.html):
```javascript
formData.append('speaker', document.getElementById('cosyvoice-speaker').value);
```
参数名: `speaker`

**后端API** (api_server.py):
```python
speaker_id: str = Form("中文女"),
```
参数名: `speaker_id`

### 2.2 根本原因

**参数名不匹配**！
- 前端使用 `speaker` 作为参数名
- 后端期望 `speaker_id` 作为参数名

由于参数名不匹配，后端无法获取前端传递的音色选择，只能使用默认值 `"中文女"`。因此无论用户选择什么音色，最终都会使用"中文女"的音色。

---

## 3. 解决方案

### 3.1 修复前端代码

将参数名从 `speaker` 改为 `speaker_id`:

```javascript
// 修复前
formData.append('speaker', document.getElementById('cosyvoice-speaker').value);

// 修复后
formData.append('speaker_id', document.getElementById('cosyvoice-speaker').value);
```

### 3.2 修复位置

文件: `frontend/app.html`  
行号: 约第2753行

---

## 4. 验证测试

### 4.1 测试脚本

创建了测试脚本 `test_cosyvoice_speaker.py`，测试所有7个音色：
- 中文女
- 中文男
- 粤语女
- 英文女
- 英文男
- 日语男
- 韩语女

### 4.2 预期结果

修复后，选择不同音色应该产生明显不同的声音特征：

| 音色 | 预期性别 | 预期语言特征 |
|------|----------|--------------|
| 中文女 | 女声 | 标准普通话 |
| 中文男 | 男声 | 标准普通话 |
| 粤语女 | 女声 | 粤语口音 |
| 英文女 | 女声 | 标准英语 |
| 英文男 | 男声 | 标准英语 |
| 日语男 | 男声 | 日语发音 |
| 韩语女 | 女声 | 韩语发音 |

---

## 5. 技术说明

### 5.1 FastAPI Form参数

后端使用 FastAPI 的 `Form` 参数来接收表单数据：
```python
speaker_id: str = Form("中文女"),
```
- 参数名是 `speaker_id`
- 默认值是 `"中文女"`

### 5.2 参数不匹配的后果

当前端传递的参数名与后端期望的不匹配时：
1. FastAPI 无法找到名为 `speaker_id` 的参数
2. 使用默认值 `"中文女"`
3. 导致无论用户选择什么音色，都使用"中文女"

---

## 6. 预防措施

为避免类似问题，建议：

1. **统一参数命名规范**
   - 前后端使用相同的参数名
   - 使用驼峰命名或下划线命名保持一致

2. **添加参数验证日志**
   - 在API入口添加参数日志
   - 记录实际接收到的参数值

3. **创建API文档**
   - 明确每个API的参数名称和类型
   - 前后端开发人员共同维护

---

## 7. 测试方法

### 7.1 手动测试

1. 启动服务: `./start_server.sh start`
2. 打开前端页面
3. 选择CosyVoice算法
4. 选择SFT模式
5. 选择不同音色（如英文男、英文女）
6. 生成音频并试听，确认性别差异

### 7.2 自动测试

```bash
# 运行测试脚本
python test_cosyvoice_speaker.py
```

脚本会自动测试所有7个音色，生成对应的音频文件。

---

## 8. 总结

**问题类型**: 参数名不匹配  
**影响范围**: CosyVoice SFT模式的所有音色选择  
**严重程度**: 高（功能完全错误）  
**修复难度**: 低（单字符修改）  
**修复状态**: ✅ 已修复

**关键教训**: 
- 前后端参数名必须严格一致
- 表单数据传递时要注意参数名匹配
- 测试时要验证实际效果，而不仅仅是HTTP请求成功
