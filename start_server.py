#!/usr/bin/env python3
"""
VersTTS 服务启动脚本
用于启动前后端联调部署的服务
"""

import sys
import os

# 添加项目根目录到Python路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from backend.api_server import app
import uvicorn

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║                    VersTTS 服务启动                          ║
╠══════════════════════════════════════════════════════════════╣
║  前端访问: http://localhost:8000                             ║
║  登录页面: http://localhost:8000/login.html                  ║
║  API文档: http://localhost:8000/docs                         ║
║  健康检查: http://localhost:8000/health                      ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,
        access_log=True
    )
