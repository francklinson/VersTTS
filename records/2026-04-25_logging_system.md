# VersTTS 日志系统文档

## 日期
2026-04-25

## 概述
VersTTS 日志系统提供全面的操作审计、性能监控和用户行为记录功能。

## 日志文件结构

```
logs/
├── system_YYYYMMDD.log       # 系统日志
├── operation_YYYYMMDD.log    # 操作日志
├── audit_YYYYMMDD.log        # 审计日志（配置变更）
└── performance_YYYYMMDD.log  # 性能日志
```

## 日志类型详解

### 1. 系统日志 (system.log)
记录系统级事件和状态信息。

**包含内容：**
- 项目初始化信息
- 环境信息（Python版本、项目路径）
- 硬件信息（CUDA、GPU内存）
- 目录初始化
- 配置加载状态
- 模型加载详情
- 系统状态监控

**示例：**
```
2026-04-25 13:08:16 | INFO     | versTTS.system       | 【项目初始化】VersTTS 服务启动
2026-04-25 13:08:16 | INFO     | versTTS.system       | 【硬件信息】CUDA可用: NVIDIA GeForce RTX 3090 (CUDA 12.8)
2026-04-25 13:08:16 | INFO     | versTTS.system       | 【模型加载】ChatTTS | 状态: 成功 | 耗时: 2.434s | GPU内存: 1.04GB
2026-04-25 13:08:58 | INFO     | versTTS.system       | 【系统状态】CPU: 21.8% | 内存: 21.4% | GPU: 1.05GB / 23.54GB
```

### 2. 操作日志 (operation.log)
记录用户操作和API请求详情。

**包含内容：**
- API请求（端点、方法、客户端IP）
- 文件操作（保存路径、文件大小）
- TTS合成请求（模型、文本、参数、耗时）
- 用户操作详情

**示例：**
```
2026-04-25 13:08:48 | INFO     | versTTS.operation    | 【API请求】POST /tts/chattts | 客户端: 127.0.0.1 | 参数: {"text_preview": "测试日志系统功能", "temperature": 0.3}
2026-04-25 13:08:52 | INFO     | versTTS.operation    | 【文件操作】保存音频 | 路径: output/tts_20260425_130852_181946.wav | 大小: 72.45KB | 状态: 成功
2026-04-25 13:08:52 | INFO     | versTTS.operation    | 【TTS合成】模型: ChatTTS | 文本: 测试日志系统功能 | 耗时: 3.389s | 状态: 成功
```

### 3. 审计日志 (audit.log)
记录配置变更和敏感操作。

**包含内容：**
- 配置项变更（旧值、新值、操作用户）
- 重要系统设置修改

**示例：**
```
2026-04-25 13:10:00 | INFO     | versTTS.audit        | 【配置变更】temperature | 用户: admin | 旧值: 0.3 | 新值: 0.5
```

### 4. 性能日志 (performance.log)
记录性能指标和资源使用情况。

**包含内容：**
- 模型加载时间
- 推理耗时
- GPU内存使用
- CPU/内存占用

**示例：**
```
2026-04-25 13:08:51 | INFO     | versTTS.performance  | 【性能】ChatTTS加载 | 耗时: 2.434s | 内存: 0.00MB | GPU: 1.04%
```

## 日志格式

### 标准格式
```
时间戳 | 日志级别 | 记录器名称 | 消息内容
```

**示例：**
```
2026-04-25 13:08:16 | INFO     | versTTS.system       | 【服务启动】初始化应用生命周期
```

### 日志级别
- **INFO**: 一般信息（系统启动、请求处理、操作完成）
- **WARNING**: 警告信息（资源不足、配置缺失）
- **ERROR**: 错误信息（操作失败、异常捕获）
- **DEBUG**: 调试信息（详细参数、调用堆栈）

## 使用方式

### 查看日志

```bash
# 实时查看系统日志
tail -f logs/system_$(date +%Y%m%d).log

# 查看操作日志
cat logs/operation_$(date +%Y%m%d).log

# 查看所有日志
ls -lh logs/
```

### 日志分析

```bash
# 统计API请求数量
grep "【API请求】" logs/operation_$(date +%Y%m%d).log | wc -l

# 查找错误日志
grep "ERROR" logs/system_$(date +%Y%m%d).log

# 统计TTS合成次数
grep "【TTS合成】" logs/operation_$(date +%Y%m%d).log | wc -l
```

## 记录的操作类型

### 项目初始化
- ✅ 服务启动/关闭
- ✅ 环境信息检测
- ✅ 硬件资源检测
- ✅ 目录结构初始化
- ✅ 配置加载

### 模型操作
- ✅ 模型加载（开始/完成/失败）
- ✅ 模型加载耗时
- ✅ GPU内存使用
- ✅ 模型缓存状态

### API请求
- ✅ 请求端点和方法
- ✅ 客户端IP地址
- ✅ 请求参数
- ✅ 响应耗时
- ✅ 请求结果

### TTS合成
- ✅ 模型类型
- ✅ 输入文本
- ✅ 合成参数
- ✅ 合成耗时
- ✅ 输出文件路径
- ✅ 文件大小

### 文件操作
- ✅ 音频保存
- ✅ 参考音频上传
- ✅ 文件大小统计

### 系统状态
- ✅ CPU使用率
- ✅ 内存使用率
- ✅ GPU内存使用
- ✅ 健康检查状态

## 配置说明

### 日志保留策略
- 日志文件按日期分割
- 每天生成新的日志文件
- 建议定期清理旧日志（如保留30天）

### 日志文件大小
- 单个日志文件无大小限制
- 建议监控日志目录大小
- 可配置日志轮转策略

## 最佳实践

### 1. 日常监控
```bash
# 监控实时日志
tail -f logs/system_$(date +%Y%m%d).log

# 检查系统状态
watch -n 5 'grep "【系统状态】" logs/system_$(date +%Y%m%d).log | tail -1'
```

### 2. 故障排查
```bash
# 查找错误
grep -i "error\|失败\|异常" logs/*.log

# 查看特定时间段
sed -n '/2026-04-25 13:08:00/,/2026-04-25 13:09:00/p' logs/system_20260425.log
```

### 3. 性能分析
```bash
# 统计平均响应时间
grep "【TTS合成】" logs/operation_$(date +%Y%m%d).log | awk -F'耗时: ' '{print $2}' | awk '{sum+=$1; count++} END {print "平均耗时:", sum/count, "s"}'

# 查看模型加载时间
grep "【性能】" logs/performance_$(date +%Y%m%d).log
```

## 扩展功能

### 未来可添加的功能
1. **日志聚合** - 使用ELK Stack进行日志分析
2. **实时监控** - Web界面展示实时日志
3. **告警机制** - 错误率达到阈值时发送告警
4. **日志搜索** - 提供全文搜索功能
5. **可视化报表** - 生成操作统计图表

## 总结

VersTTS 日志系统提供全面的操作记录能力：
- ✅ 项目初始化详细记录
- ✅ 用户点击/操作记录
- ✅ 配置变更审计
- ✅ 性能监控指标
- ✅ 错误追踪和排查

所有日志按类型分离，便于查询和分析。
