#!/usr/bin/env python3
"""
测试模型加载日志记录功能
"""

import os
import sys

# 添加项目路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from backend.logger_config import OperationLogger, system_logger

print("=" * 60)
print("测试模型加载日志记录")
print("=" * 60)
print()

# 测试 1: 基本模型加载日志
print("1. 测试基本模型加载日志...")
OperationLogger.log_model_load("TestModel", "开始加载")
OperationLogger.log_model_load("TestModel", "成功", 1.234, "GPU内存: 2.5GB")
print("   ✓ 基本日志测试通过")
print()

# 测试 2: 详细模型加载日志
print("2. 测试详细模型加载日志...")
OperationLogger.log_model_load_detail(
    "TestModel-V2",
    "路径确认",
    model_path="/path/to/model",
    device="cuda:0"
)

OperationLogger.log_model_load_detail(
    "TestModel-V2",
    "文件检查",
    model_path="/path/to/model",
    model_size="1.5GB",
    extra_info={"文件数": 15, "格式": "safetensors"}
)

OperationLogger.log_model_load_detail(
    "TestModel-V2",
    "加载中",
    device="cuda:0",
    extra_info={"精度": "fp16", "框架": "transformers"}
)

OperationLogger.log_model_load_detail(
    "TestModel-V2",
    "完成",
    device="cuda:0",
    memory_usage=2.5,
    extra_info={"耗时": "3.456s", "状态": "成功"}
)
print("   ✓ 详细日志测试通过")
print()

# 测试 3: 查看日志文件
print("3. 查看日志文件...")
log_file = os.path.join(PROJECT_ROOT, "logs", "system.log")
if os.path.exists(log_file):
    print(f"   日志文件: {log_file}")
    print(f"   文件大小: {os.path.getsize(log_file)} bytes")
    print()
    print("   最近的日志条目:")
    
    # 读取最后几行
    with open(log_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        # 找到最后10条模型加载相关的日志
        model_logs = [l for l in lines if '模型加载' in l]
        for line in model_logs[-5:]:
            print(f"   {line.strip()}")
else:
    print(f"   日志文件不存在: {log_file}")

print()
print("=" * 60)
print("测试完成!")
print("=" * 60)
print()
print("日志文件位置:")
print(f"  - system.log: {os.path.join(PROJECT_ROOT, 'logs', 'system.log')}")
print(f"  - operation.log: {os.path.join(PROJECT_ROOT, 'logs', 'operation.log')}")
print(f"  - audit.log: {os.path.join(PROJECT_ROOT, 'logs', 'audit.log')}")
