#!/usr/bin/env python3
"""
VersTTS 日志配置模块
提供详细的操作审计、性能监控和用户行为日志
"""

import os
import sys
import logging
import json
import time
from datetime import datetime
from functools import wraps
from typing import Optional, Dict, Any

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(PROJECT_ROOT, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

# 日志文件路径
SYSTEM_LOG = os.path.join(LOG_DIR, f'system_{datetime.now().strftime("%Y%m%d")}.log')
OPERATION_LOG = os.path.join(LOG_DIR, f'operation_{datetime.now().strftime("%Y%m%d")}.log')
AUDIT_LOG = os.path.join(LOG_DIR, f'audit_{datetime.now().strftime("%Y%m%d")}.log')
PERFORMANCE_LOG = os.path.join(LOG_DIR, f'performance_{datetime.now().strftime("%Y%m%d")}.log')

# 日志格式
DETAILED_FORMATTER = logging.Formatter(
    '%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

JSON_FORMATTER = logging.Formatter(
    '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": %(message)s}'
)


def setup_logger(name: str, log_file: str, level=logging.INFO, formatter=None) -> logging.Logger:
    """设置日志记录器"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 避免重复添加处理器
    if logger.handlers:
        return logger
    
    # 文件处理器
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter or DETAILED_FORMATTER)
    logger.addHandler(file_handler)
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter or DETAILED_FORMATTER)
    logger.addHandler(console_handler)
    
    return logger


# 各类日志记录器
system_logger = setup_logger('versTTS.system', SYSTEM_LOG)
operation_logger = setup_logger('versTTS.operation', OPERATION_LOG)
audit_logger = setup_logger('versTTS.audit', AUDIT_LOG)
performance_logger = setup_logger('versTTS.performance', PERFORMANCE_LOG)


class OperationLogger:
    """操作日志记录器"""
    
    @staticmethod
    def log_init_start():
        """记录项目初始化开始"""
        system_logger.info("=" * 80)
        system_logger.info("【项目初始化】VersTTS 服务启动")
        system_logger.info("=" * 80)
    
    @staticmethod
    def log_init_complete(duration: float, status: str = "成功"):
        """记录项目初始化完成"""
        system_logger.info(f"【项目初始化】完成 | 状态: {status} | 耗时: {duration:.3f}s")
        system_logger.info("=" * 80)
    
    @staticmethod
    def log_config_load(config_name: str, status: str = "成功", details: str = ""):
        """记录配置加载"""
        system_logger.info(f"【配置加载】{config_name} | 状态: {status} {details}")
    
    @staticmethod
    def log_model_load(model_name: str, status: str = "成功", duration: float = 0, details: str = ""):
        """记录模型加载"""
        msg = f"【模型加载】{model_name} | 状态: {status}"
        if duration > 0:
            msg += f" | 耗时: {duration:.3f}s"
        if details:
            msg += f" | {details}"
        system_logger.info(msg)
    
    @staticmethod
    def log_user_operation(operation: str, user_info: str = "anonymous", params: Dict = None, result: str = ""):
        """记录用户操作"""
        params_str = json.dumps(params, ensure_ascii=False) if params else "{}"
        operation_logger.info(f"【用户操作】{operation} | 用户: {user_info} | 参数: {params_str} | 结果: {result}")
    
    @staticmethod
    def log_config_change(config_name: str, old_value: Any, new_value: Any, user_info: str = "anonymous"):
        """记录配置变更"""
        audit_logger.info(
            f"【配置变更】{config_name} | 用户: {user_info} | "
            f"旧值: {old_value} | 新值: {new_value}"
        )
    
    @staticmethod
    def log_api_request(endpoint: str, method: str, params: Dict = None, client_ip: str = "", duration: float = 0):
        """记录API请求"""
        params_str = json.dumps(params, ensure_ascii=False) if params else "{}"
        operation_logger.info(
            f"【API请求】{method} {endpoint} | 客户端: {client_ip} | "
            f"耗时: {duration:.3f}s | 参数: {params_str}"
        )
    
    @staticmethod
    def log_tts_request(model: str, text: str, params: Dict = None, duration: float = 0, status: str = "成功"):
        """记录TTS请求"""
        text_preview = text[:50] + "..." if len(text) > 50 else text
        params_str = json.dumps(params, ensure_ascii=False) if params else "{}"
        operation_logger.info(
            f"【TTS合成】模型: {model} | 文本: {text_preview} | "
            f"耗时: {duration:.3f}s | 状态: {status} | 参数: {params_str}"
        )
    
    @staticmethod
    def log_error(error_type: str, message: str, stack_trace: str = ""):
        """记录错误"""
        system_logger.error(f"【错误】{error_type} | {message}")
        if stack_trace:
            system_logger.error(f"堆栈: {stack_trace}")
    
    @staticmethod
    def log_performance(operation: str, duration: float, memory_usage: float = 0, gpu_usage: float = 0):
        """记录性能指标"""
        performance_logger.info(
            f"【性能】{operation} | 耗时: {duration:.3f}s | "
            f"内存: {memory_usage:.2f}MB | GPU: {gpu_usage:.2f}%"
        )
    
    @staticmethod
    def log_file_operation(operation: str, file_path: str, size: int = 0, status: str = "成功"):
        """记录文件操作"""
        size_str = f"{size / 1024:.2f}KB" if size > 0 else ""
        operation_logger.info(
            f"【文件操作】{operation} | 路径: {file_path} | "
            f"大小: {size_str} | 状态: {status}"
        )
    
    @staticmethod
    def log_system_status(cpu_percent: float, memory_percent: float, gpu_info: str = ""):
        """记录系统状态"""
        system_logger.info(
            f"【系统状态】CPU: {cpu_percent:.1f}% | 内存: {memory_percent:.1f}% | GPU: {gpu_info}"
        )


def log_operation(operation_name: str):
    """装饰器：记录函数操作"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            operation_logger.info(f"【操作开始】{operation_name}")
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                operation_logger.info(f"【操作完成】{operation_name} | 耗时: {duration:.3f}s")
                return result
            except Exception as e:
                duration = time.time() - start_time
                operation_logger.error(f"【操作失败】{operation_name} | 耗时: {duration:.3f}s | 错误: {str(e)}")
                raise
        return wrapper
    return decorator


def log_api_call(endpoint: str):
    """装饰器：记录API调用"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            client_ip = kwargs.get('request', {}).get('client', [{}])[0].get('host', 'unknown') if 'request' in kwargs else 'unknown'
            
            # 提取参数
            params = {}
            for key, value in kwargs.items():
                if key not in ['request', 'background_tasks']:
                    params[key] = str(value)[:100]  # 限制长度
            
            operation_logger.info(f"【API调用开始】{endpoint} | 客户端: {client_ip}")
            
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                operation_logger.info(f"【API调用完成】{endpoint} | 耗时: {duration:.3f}s | 状态: 成功")
                return result
            except Exception as e:
                duration = time.time() - start_time
                operation_logger.error(f"【API调用失败】{endpoint} | 耗时: {duration:.3f}s | 错误: {str(e)}")
                raise
        return wrapper
    return decorator


# 兼容性：保持原有logger可用
logger = system_logger

__all__ = [
    'OperationLogger',
    'log_operation',
    'log_api_call',
    'system_logger',
    'operation_logger',
    'audit_logger',
    'performance_logger',
    'logger'
]
