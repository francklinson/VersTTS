# CosyVoice 3.0 前端调用音频异常问题修复记录

**时间**: 2026-04-26 12:45:00  
**问题描述**: 通过前端调用生成的 CosyVoice 3.0 音频还是异常的

## 问题排查过程

### 1. 后端 API 测试正常
直接测试后端 API (`/tts/cosyvoice`) 发现：
- API 响应正常（200 OK）
- 音频文件成功生成
- 文件格式正确（RIFF WAVE, 16bit, mono 24000Hz）

### 2. 发现问题根源
检查当前 Python 环境发现：
```python
import matcha
# ModuleNotFoundError: No module named 'matcha'
```

**根本原因是**：API 服务进程启动时，`Matcha-TTS` 路径没有被正确添加到 `sys.path` 中。虽然 `api_server.py` 代码中有添加路径的语句，但正在运行的服务进程是在之前启动的（PID 355698，启动时间 10:21），没有包含该路径。

### 3. 修复措施
**重启 API 服务**，让新的服务进程加载正确的路径配置：

```bash
# 停止旧服务
pkill -f "uvicorn backend.api_server"

# 启动新服务
nohup python -m uvicorn backend.api_server:app --host 0.0.0.0 --port 8000 > logs/server.log 2>&1 &
```

## 验证结果

重启后测试 CosyVoice 3.0 API：

| 测试项目 | 结果 |
|---------|------|
| API 响应 | ✅ 200 OK |
| 音频生成 | ✅ 成功 |
| 文件大小 | ✅ 172878 bytes |
| 文件格式 | ✅ RIFF WAVE, 16bit, mono 24000Hz |

## 结论

CosyVoice 3.0 前端调用音频异常的问题通过**重启 API 服务**解决。问题原因是运行中的服务进程缺少 `Matcha-TTS` 模块路径，导致 CosyVoice 模型加载不完整。

**建议**: 以后修改了路径配置相关的代码后，需要重启服务才能生效。
