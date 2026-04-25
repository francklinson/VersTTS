# 日志系统优化

**时间戳**: 2026-04-25 20:20:00  
**任务来源**: 用户反馈 - 日志文件太多，audit日志为空  
**修改文件**: backend/logger_config.py, backend/api_server.py

## 优化前问题

1. **日志文件过多**: 每天产生5个日志文件 (system, operation, audit, performance, server)
2. **audit日志为空**: 没有记录配置变更、说话人操作等审计信息
3. **缺少自动清理**: 日志文件会无限增长
4. **缺少日志轮转**: 单个日志文件可能过大

## 优化内容

### 1. 减少日志文件数量

从5个日志减少到3个核心日志：
- `system.log` - 系统日志、错误日志、初始化信息
- `operation.log` - 业务操作、API请求、TTS合成
- `audit.log` - 审计日志（配置变更、说话人操作、错误）

### 2. 添加日志轮转

使用 `RotatingFileHandler` 实现自动轮转：
- 单个文件最大：10MB
- 备份文件数：3个
- 文件格式：`xxx.log`, `xxx.log.1`, `xxx.log.2`, `xxx.log.3`

### 3. 添加自动清理

在日志模块初始化时自动清理：
- 保留天数：7天
- 自动删除超过7天的 `.log*` 文件

```python
def cleanup_old_logs():
    """清理超过保留天数的旧日志文件"""
    cutoff_time = datetime.now() - timedelta(days=LOG_RETENTION_DAYS)
    # 清理逻辑...
```

### 4. 修复audit日志

添加以下审计记录：
- 配置变更 (`log_config_change`)
- 说话人操作 (`log_speaker_operation`)
- 错误信息 (`log_error`)

在 api_server.py 中添加：
```python
# 保存说话人时记录审计
OperationLogger.log_speaker_operation("创建", speaker["name"], speaker["id"])

# 删除说话人时记录审计
OperationLogger.log_speaker_operation("删除", speaker.get("name", "unknown"), speaker_id)
```

### 5. 优化日志内容

- **参数简化**: 限制字符串长度，避免日志过大
- **减少重复**: API请求只记录到operation日志
- **性能日志优化**: 只记录耗时超过1秒的操作
- **系统状态优化**: 只在资源使用超过80%时记录

## 日志文件说明

| 日志文件 | 内容 | 审计级别 |
|---------|------|---------|
| system.log | 系统启动/关闭、模型加载、错误信息 | 高 |
| operation.log | API请求、TTS合成、文件操作 | 中 |
| audit.log | 配置变更、说话人操作、所有错误 | 最高 |

## 配置参数

```python
LOG_RETENTION_DAYS = 7      # 保留7天
MAX_LOG_SIZE = 10MB         # 单个文件最大10MB
BACKUP_COUNT = 3            # 保留3个备份
```

## 使用建议

1. **日常查看**: 主要查看 `system.log` 了解系统状态
2. **问题排查**: 结合 `system.log` 和 `operation.log`
3. **安全审计**: 查看 `audit.log` 了解重要操作

## 手动清理

如需手动清理日志：
```bash
# 清理所有日志
rm -f logs/*.log*

# 或保留最近7天（自动）
python -c "from backend.logger_config import cleanup_old_logs; cleanup_old_logs()"
```
