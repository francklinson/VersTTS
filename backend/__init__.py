#!/usr/bin/env python3
"""
VersTTS 模块化后端包

该包是 backend/api_server.py 的模块化重构版本，将庞大的单体文件拆分为：
- config: 配置管理
- models: 数据模型 (Pydantic)
- core: 核心工具函数
- services: 业务逻辑
- engines: TTS 引擎加载器
- routers: API 路由
- main: 应用入口

使用方法:
    from backend.main import app
    import uvicorn
    uvicorn.run(app)
"""

__version__ = "2.0.0"
__author__ = "VersTTS Team"

# 便捷导入
from .main import app

__all__ = ['app', '__version__']
