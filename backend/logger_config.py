#!/usr/bin/env python3
"""
VersTTS 日志配置模块
简化版：2个核心日志文件
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

# 优先从 config.py 读取配置，否则使用默认值
try:
    from backend.config import LOGS_DIR, LOG_MAX_SIZE, LOG_BACKUP_COUNT
    LOG_DIR = LOGS_DIR
    MAX_LOG_SIZE = LOG_MAX_SIZE * 1024 * 1024  # config 中单位为 MB
    BACKUP_COUNT = LOG_BACKUP_COUNT
    LOG_RETENTION_DAYS = int(os.environ.get("LOG_RETENTION_DAYS", "7"))
except (ImportError, AttributeError):
    LOG_DIR = os.path.join(PROJECT_ROOT, 'logs')
    LOG_RETENTION_DAYS = 7
    MAX_LOG_SIZE = 50 * 1024 * 1024  # 默认 50MB
    BACKUP_COUNT = 5

os.makedirs(LOG_DIR, exist_ok=True)

# 日志文件路径 - 简化为2个核心日志
# 1. app.log: 应用日志 (系统 + 操作)
# 2. audit.log: 审计日志 (安全相关)
APP_LOG = os.path.join(LOG_DIR, 'app.log')
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

# 各类日志记录器 - 简化为2个核心日志
# app_logger: 应用日志 (系统和操作日志合并)
app_logger = setup_logger('versTTS.app', APP_LOG)
# audit_logger: 审计日志 (安全相关)
audit_logger = setup_logger('versTTS.audit', AUDIT_LOG)


class OperationLogger:
    """操作日志记录器"""
    
    @staticmethod
    def log_init_start():
        """记录项目初始化开始"""
        app_logger.info("=" * 80)
        app_logger.info("【项目初始化】VersTTS 服务启动")
        app_logger.info("=" * 80)
    
    @staticmethod
    def log_init_complete(duration: float, status: str = "成功"):
        """记录项目初始化完成"""
        app_logger.info(f"【项目初始化】完成 | 状态: {status} | 耗时: {duration:.3f}s")
        app_logger.info("=" * 80)
    
    @staticmethod
    def log_config_load(config_name: str, status: str = "成功", details: str = ""):
        """记录配置加载"""
        app_logger.info(f"【配置加载】{config_name} | 状态: {status} {details}")
    
    @staticmethod
    def log_model_load(model_name: str, status: str = "成功", duration: float = 0, details: str = ""):
        """记录模型加载"""
        msg = f"【模型加载】{model_name} | 状态: {status}"
        if duration > 0:
            msg += f" | 耗时: {duration:.3f}s"
        if details:
            msg += f" | {details}"
        app_logger.info(msg)
    
    @staticmethod
    def log_model_load_detail(model_name: str, stage: str, model_path: str = "", model_size: str = "", 
                               device: str = "", memory_usage: float = 0, extra_info: dict = None):
        """记录模型加载详细信息"""
        msg = f"【模型加载详情】{model_name} | 阶段: {stage}"
        
        if model_path:
            msg += f" | 路径: {model_path}"
        if model_size:
            msg += f" | 大小: {model_size}"
        if device:
            msg += f" | 设备: {device}"
        if memory_usage > 0:
            msg += f" | GPU内存: {memory_usage:.2f}GB"
        if extra_info:
            extra_str = " | ".join([f"{k}: {v}" for k, v in extra_info.items()])
            msg += f" | {extra_str}"
        
        app_logger.info(msg)
    
    @staticmethod
    def log_user_operation(operation: str, user_info: str = "anonymous", params: Dict = None, result: str = ""):
        """记录用户操作"""
        params_str = json.dumps(params, ensure_ascii=False) if params else "{}"
        app_logger.info(f"【用户操作】{operation} | 用户: {user_info} | 参数: {params_str} | 结果: {result}")
    
    @staticmethod
    def log_config_change(config_name: str, old_value: Any, new_value: Any, user_info: str = "anonymous"):
        """记录配置变更 - 同时记录到audit日志"""
        msg = f"【配置变更】{config_name} | 用户: {user_info} | 旧值: {old_value} | 新值: {new_value}"
        audit_logger.info(msg)
        app_logger.info(msg)
    
    @staticmethod
    def log_speaker_operation(operation: str, speaker_name: str, speaker_id: str = "", user_info: str = "anonymous"):
        """记录说话人相关操作 - 重要操作记录到audit"""
        msg = f"【说话人操作】{operation} | 名称: {speaker_name} | ID: {speaker_id} | 用户: {user_info}"
        audit_logger.info(msg)
        app_logger.info(msg)
    
    @staticmethod
    def log_api_request(endpoint: str, method: str, params: Dict = None, client_ip: str = "", duration: float = 0):
        """记录API请求"""
        # 简化参数，避免日志过大
        simplified_params = {}
        if params:
            for key, value in params.items():
                if isinstance(value, str) and len(value) > 100:
                    simplified_params[key] = value[:100] + "..."
                else:
                    simplified_params[key] = value
        
        params_str = json.dumps(simplified_params, ensure_ascii=False) if simplified_params else "{}"
        app_logger.info(
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
        app_logger.info(
            f"【TTS合成】模型: {model} | 文本: {text_preview} | "
            f"耗时: {duration:.3f}s | 状态: {status} | 参数: {params_str}"
        )
    
    @staticmethod
    def log_error(error_type: str, message: str, stack_trace: str = ""):
        """记录错误 - 错误同时记录到app和audit"""
        app_logger.error(f"【错误】{error_type} | {message}")
        audit_logger.error(f"【错误】{error_type} | {message}")
        if stack_trace:
            app_logger.error(f"堆栈: {stack_trace}")
    
    @staticmethod
    def log_performance(operation: str, duration: float, memory_usage: float = 0, gpu_usage: float = 0):
        """记录性能指标 - 只记录超过阈值的操作"""
        # 只记录耗时超过1秒或有资源使用的情况
        if duration > 1 or memory_usage > 0 or gpu_usage > 0:
            app_logger.info(
                f"【性能】{operation} | 耗时: {duration:.3f}s | "
                f"内存: {memory_usage:.2f}MB | GPU: {gpu_usage:.2f}%"
            )
    
    @staticmethod
    def log_file_operation(operation: str, file_path: str, size: int = 0, status: str = "成功"):
        """记录文件操作"""
        size_str = f"{size / 1024:.2f}KB" if size > 0 else ""
        app_logger.info(
            f"【文件操作】{operation} | 路径: {file_path} | "
            f"大小: {size_str} | 状态: {status}"
        )
    
    @staticmethod
    def log_system_status(cpu_percent: float, memory_percent: float, gpu_info: str = ""):
        """记录系统状态 - 只在异常时记录"""
        # 只在资源使用较高时记录
        if cpu_percent > 80 or memory_percent > 80:
            app_logger.warning(
                f"【系统状态】CPU: {cpu_percent:.1f}% | 内存: {memory_percent:.1f}% | GPU: {gpu_info}"
            )


def log_operation(operation_name: str):
    """装饰器：记录函数操作"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            app_logger.info(f"【操作开始】{operation_name}")
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                app_logger.info(f"【操作完成】{operation_name} | 耗时: {duration:.3f}s")
                return result
            except Exception as e:
                duration = time.time() - start_time
                app_logger.error(f"【操作失败】{operation_name} | 耗时: {duration:.3f}s | 错误: {str(e)}")
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
                app_logger.info(f"【API调用】{endpoint} | 客户端: {client_ip} | 耗时: {duration:.3f}s | 状态: 成功")
                return result
            except Exception as e:
                duration = time.time() - start_time
                app_logger.error(f"【API调用】{endpoint} | 客户端: {client_ip} | 耗时: {duration:.3f}s | 错误: {str(e)}")
                raise
        return wrapper
    return decorator


# 兼容性：保持原有logger可用
# 将旧名称映射到新logger
system_logger = app_logger
operation_logger = app_logger
logger = app_logger

__all__ = [
    'OperationLogger',
    'log_operation',
    'log_api_call',
    'app_logger',
    'audit_logger',
    'system_logger',  # 兼容性
    'operation_logger',  # 兼容性
    'logger',
    'cleanup_old_logs',
    'LOG_RETENTION_DAYS'
]
