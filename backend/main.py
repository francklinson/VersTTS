#!/usr/bin/env python3
"""
VersTTS 模块化后端服务主入口
支持: ChatTTS, CosyVoice, F5-TTS, Qwen3-TTS, OpenVoice, GPT-SoVITS, VoxCPM, IndexTTS, FireRedTTS2

使用方法:
    python -m backend.main
    或
    uvicorn backend.main:app --host 0.0.0.0 --port 8000
"""

import os
import sys

# ========== 离线部署环境变量配置 ==========
# 必须在导入任何依赖 HuggingFace 的库之前设置
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 设置项目内 HuggingFace 缓存目录（优先使用项目内缓存）
HF_CACHE_PATH = os.path.join(PROJECT_ROOT, "models", "hf_cache")
if os.path.exists(HF_CACHE_PATH):
    # 如果项目内有缓存目录，强制使用它
    os.environ['HF_HOME'] = HF_CACHE_PATH
    os.environ['HUGGINGFACE_HUB_CACHE'] = HF_CACHE_PATH
    os.environ['TRANSFORMERS_CACHE'] = os.path.join(PROJECT_ROOT, "models", "transformers_cache")
    print(f"[配置] 使用项目内 HF 缓存: {HF_CACHE_PATH}")

# 检查是否启用离线模式
if os.environ.get('TRANSFORMERS_OFFLINE') == '1' or os.environ.get('HF_HUB_OFFLINE') == '1':
    print("[离线部署] 检测到离线模式环境变量，禁用HuggingFace在线访问")
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['HF_DATASETS_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_HUB_DISABLE_DOWNLOADS'] = '1'

# 打印当前缓存配置
if 'HF_HOME' in os.environ:
    print(f"[配置] HF_HOME: {os.environ['HF_HOME']}")

# ========== Transformers 兼容性补丁 ==========
# 必须在导入 transformers 之前加载
try:
    import transformers

    # 添加 rope_config_validation (CosyVoice 需要)
    if not hasattr(transformers.modeling_rope_utils, 'rope_config_validation'):
        def rope_config_validation(config):
            pass

        transformers.modeling_rope_utils.rope_config_validation = rope_config_validation
        print("[API Server] CosyVoice rope_config_validation patch loaded")
except Exception as e:
    print(f"[API Server] Warning: Failed to load CosyVoice compatibility patch: {e}")

# ========== 路径配置 ==========
sys.path.insert(0, PROJECT_ROOT)

# ========== 导入应用组件 ==========
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, RedirectResponse

from backend.logger_config import OperationLogger, system_logger
from backend.config import (
    PROJECT_ROOT, 
    setup_algorithm_paths, 
    ensure_directories,
    OUTPUTS_DIR
)
from backend.core import lifespan
from backend.routers import router as api_router

# 记录项目启动
OperationLogger.log_init_start()

# 设置算法路径
setup_algorithm_paths()

# ========== FastAPI 应用 ==========
app = FastAPI(
    title="VersTTS API (Modular)",
    description="统一文本转语音API服务 - 模块化重构版",
    version="2.0.0",
    lifespan=lifespan
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册 API 路由
app.include_router(api_router)

# 静态文件服务 - 音频输出
if os.path.exists(OUTPUTS_DIR):
    app.mount("/audio", StaticFiles(directory=OUTPUTS_DIR), name="audio")

# 挂载前端静态文件 - 子目录
frontend_dir = os.path.join(PROJECT_ROOT, "frontend")
assets_dir = os.path.join(frontend_dir, "assets")
pages_dir = os.path.join(frontend_dir, "pages")
if os.path.exists(assets_dir):
    app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")
if os.path.exists(pages_dir):
    app.mount("/pages", StaticFiles(directory=pages_dir), name="pages")


# ========== 根路由 - 重定向到登录页面 ==========
@app.get("/")
async def root():
    """根路径重定向到登录页面"""
    return RedirectResponse(url="/login.html")


# ========== 前端页面路由 ==========
@app.get("/login.html")
async def login_page():
    """登录页面"""
    frontend_path = os.path.join(PROJECT_ROOT, "frontend", "login.html")
    if os.path.exists(frontend_path):
        return FileResponse(frontend_path)
    raise HTTPException(status_code=404, detail="登录页面不存在")


@app.get("/index.html")
async def index_page():
    """首页"""
    frontend_path = os.path.join(PROJECT_ROOT, "frontend", "index.html")
    if os.path.exists(frontend_path):
        return FileResponse(frontend_path)
    raise HTTPException(status_code=404, detail="首页不存在")


@app.get("/app.html")
async def app_page():
    """应用页面 - 重定向到首页"""
    return RedirectResponse(url="/index.html")


@app.get("/favicon.ico")
async def favicon():
    """网站图标 - 返回空响应避免 404 错误"""
    return FileResponse(os.path.join(frontend_dir, "assets", "favicon.ico")) if os.path.exists(os.path.join(frontend_dir, "assets", "favicon.ico")) else RedirectResponse(url="/assets/logo.png") if os.path.exists(os.path.join(frontend_dir, "assets", "logo.png")) else ""


# ========== 主函数 ==========
if __name__ == "__main__":
    import uvicorn
    
    # 确保必要的目录存在
    ensure_directories()
    
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 8000))
    
    system_logger.info(f"【服务启动】启动VersTTS模块化服务 at {host}:{port}")
    
    uvicorn.run(
        "backend.main:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )
