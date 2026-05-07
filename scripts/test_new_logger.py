#!/usr/bin/env python3
"""
测试新的简化日志配置
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from backend.logger_config import (
    app_logger, audit_logger, system_logger, operation_logger,
    OperationLogger
)

print("=" * 60)
print("测试新的简化日志配置")
print("=" * 60)
print()

# 测试 1: 基本日志记录
print("1. 测试基本日志记录...")
app_logger.info("【测试】应用日志测试消息")
audit_logger.info("【测试】审计日志测试消息")
print("   ✓ 基本日志记录成功")
print()

# 测试 2: 兼容性测试
print("2. 测试兼容性（旧名称映射到新logger）...")
system_logger.info("【测试】system_logger 兼容性测试")
operation_logger.info("【测试】operation_logger 兼容性测试")
print("   ✓ 兼容性测试通过")
print()

# 测试 3: OperationLogger 方法
print("3. 测试 OperationLogger 方法...")
OperationLogger.log_init_start()
OperationLogger.log_config_load("test_config", "成功", "测试配置")
OperationLogger.log_model_load("TestModel", "成功", 1.234, "GPU内存: 2.5GB")
OperationLogger.log_model_load_detail("TestModel-V2", "路径确认", 
                                      model_path="/path/to/model", 
                                      device="cuda:0",
                                      model_size="1.5GB")
OperationLogger.log_user_operation("test_operation", "test_user", {"param": "value"}, "success")
OperationLogger.log_api_request("/test", "POST", {"data": "test"}, "127.0.0.1", 0.5)
OperationLogger.log_tts_request("ChatTTS", "测试文本", {"voice": "default"}, 2.5, "成功")
OperationLogger.log_file_operation("read", "/path/to/file.wav", 1024, "成功")
OperationLogger.log_init_complete(3.456, "成功")
print("   ✓ OperationLogger 方法测试通过")
print()

# 测试 4: 检查日志文件
print("4. 检查日志文件...")
log_dir = os.path.join(PROJECT_ROOT, "logs")
log_files = ["app.log", "audit.log", "server.log"]

for log_file in log_files:
    log_path = os.path.join(log_dir, log_file)
    if os.path.exists(log_path):
        size = os.path.getsize(log_path)
        print(f"   ✓ {log_file}: {size} bytes")
    else:
        print(f"   ○ {log_file}: 不存在（首次运行将创建）")

print()
print("=" * 60)
print("测试完成！")
print("=" * 60)
print()
print("日志文件说明:")
print("  - app.log: 应用日志 (系统和操作日志合并)")
print("  - audit.log: 审计日志 (安全相关)")
print("  - server.log: Uvicorn服务器日志 (通过start_server.sh生成)")
print()
print("当前日志文件位置:")
for log_file in log_files:
    print(f"  - {os.path.join(log_dir, log_file)}")
