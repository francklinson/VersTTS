#!/usr/bin/env python3
"""
VersTTS 日志配置模块
提供详细的操作审计、性能监控和用户行为日志
优化版：减少日志文件数量，添加自动清理
"""

import os
import sys
import logging
import json
import time
import glob
from datetime import datetime, timedelta
from functools import wraps
from typing import Optional, Dict, Any
from logging.handlers import RotatingFileHandler

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(PROJECT_ROOT, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

# 日志保留天数
LOG_RETENTION_DAYS = 7
# 单个日志文件最大大小 (10MB)
MAX_LOG_SIZE = 10 * 1024 * 1024
# 备份文件数量
BACKUP_COUNT = 3

# 日志文件路径 - 简化为3个核心日志
SYSTEM_LOG = os.path.join(LOG_DIR, 'system.log')
OPERATION_LOG = os.path.join(LOG_DIR, 'operation.log')
AUDIT_LOG = os.path.join(LOG_DIR, 'audit.log')

# 日志格式
DETAILED_FORMATTER = logging.Formatter(
    '%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

SIMPLE_FORMATTER = logging.Formatter(
    '%(asctime)s | %(message)s',
    datefmt='%H:%M:%S'
)


def cleanup_old_logs():
    """清理超过保留天数的旧日志文件"""
    try:
        cutoff_time = datetime.now() - timedelta(days=LOG_RETENTION_DAYS)
        log_files = glob.glob(os.path.join(LOG_DIR, '*.log*'))
        
        removed_count = 0
        for file_path in log_files:
            try:
                # 获取文件修改时间
                file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                if file_mtime < cutoff_time:
                    os.remove(file_path)
                    removed_count += 1
            except Exception as e:
                print(f"清理日志文件失败 {file_path}: {e}")
        
        if removed_count > 0:
            print(f"已清理 {removed_count} 个过期日志文件")
    except Exception as e:
        print(f"日志清理过程出错: {e}")


def setup_logger(name: str, log_file: str, level=logging.INFO, formatter=None, max_bytes=None, backup_count=None) -> logging.Logger:
    """设置日志记录器 - 使用RotatingFileHandler实现自动轮转"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 避免重复添加处理器
    if logger.handlers:
        return logger
    
    # 使用RotatingFileHandler实现日志轮转
    max_bytes = max_bytes or MAX_LOG_SIZE
    backup_count = backup_count or BACKUP_COUNT
    
    file_handler = RotatingFileHandler(
        log_file, 
        maxBytes=max_bytes, 
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter or DETAILED_FORMATTER)
    logger.addHandler(file_handler)
    
    # 控制台处理器 - 只输出INFO及以上级别
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter or DETAILED_FORMATTER)
    logger.addHandler(console_handler)
    
    return logger


# 执行日志清理
cleanup_old_logs()

# 各类日志记录器 - 简化为3个核心日志
system_logger = setup_logger('versTTS.system', SYSTEM_LOG)
operation_logger = setup_logger('versTTS.operation', OPERATION_LOG)
audit_logger = setup_logger('versTTS.audit', AUDIT_LOG)


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
        """记录配置变更 - 同时记录到audit日志"""
        msg = f"【配置变更】{config_name} | 用户: {user_info} | 旧值: {old_value} | 新值: {new_value}"
        audit_logger.info(msg)
        system_logger.info(msg)
    
    @staticmethod
    def log_speaker_operation(operation: str, speaker_name: str, speaker_id: str = "", user_info: str = "anonymous"):
        """记录说话人相关操作 - 重要操作记录到audit"""
        msg = f"【说话人操作】{operation} | 名称: {speaker_name} | ID: {speaker_id} | 用户: {user_info}"
        audit_logger.info(msg)
        operation_logger.info(msg)
    
    @staticmethod
    def log_api_request(endpoint: str, method: str, params: Dict = None, client_ip: str = "", duration: float = 0):
        """记录API请求 - 只记录到operation日志，减少重复"""
        # 简化参数，避免日志过大
        simplified_params = {}
        if params:
            for key, value in params.items():
                if isinstance(value, str) and len(value) > 100:
                    simplified_params[key] = value[:100] + "..."
                else:
                    simplified_params[key] = value
        
        params_str = json.dumps(simplified_params, ensure_ascii=False) if simplified_params else "{}"
        operation_logger.info(
            f"【API请求】{method} {endpoint} | 客户端: {client_ip} | "
            f"耗时: {duration:.3f}s | 参数: {params_str}"
        )
    
    @staticmethod
    def log_tts_request(model: str, text: str, params: Dict = None, duration: float = 0, status: str = "成功"):
        """记录TTS请求"""
        text_preview = text[:50] + "..." if len(text) > 50 else text
        # 简化参数
        simplified_params = {}
        if params:
            for key, value in params.items():
                if isinstance(value, str) and len(value) > 50:
                    simplified_params[key] = value[:50] + "..."
                else:
                    simplified_params[key] = value
        
        params_str = json.dumps(simplified_params, ensure_ascii=False) if simplified_params else "{}"
        operation_logger.info(
            f"【TTS合成】模型: {model} | 文本: {text_preview} | "
            f"耗时: {duration:.3f}s | 状态: {status} | 参数: {params_str}"
        )
    
    @staticmethod
    def log_error(error_type: str, message: str, stack_trace: str = ""):
        """记录错误 - 错误同时记录到system和audit"""
        system_logger.error(f"【错误】{error_type} | {message}")
        audit_logger.error(f"【错误】{error_type} | {message}")
        if stack_trace:
            system_logger.error(f"堆栈: {stack_trace}")
    
    @staticmethod
    def log_performance(operation: str, duration: float, memory_usage: float = 0, gpu_usage: float = 0):
        """记录性能指标 - 只记录超过阈值的操作"""
        # 只记录耗时超过1秒或有资源使用的情况
        if duration > 1 or memory_usage > 0 or gpu_usage > 0:
            operation_logger.info(
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
        """记录系统状态 - 只在异常时记录"""
        # 只在资源使用较高时记录
        if cpu_percent > 80 or memory_percent > 80:
            system_logger.warning(
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
            
            # 简化参数
            params = {}
            for key, value in kwargs.items():
                if key not in ['request', 'background_tasks']:
                    val_str = str(value)
                    params[key] = val_str[:100] + "..." if len(val_str) > 100 else val_str
            
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                operation_logger.info(f"【API调用】{endpoint} | 客户端: {client_ip} | 耗时: {duration:.3f}s | 状态: 成功")
                return result
            except Exception as e:
                duration = time.time() - start_time
                operation_logger.error(f"【API调用】{endpoint} | 客户端: {client_ip} | 耗时: {duration:.3f}s | 错误: {str(e)}")
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
    'logger',
    'cleanup_old_logs',
    'LOG_RETENTION_DAYS'
]
