#!/usr/bin/env python3
"""
服务注册与 GPU 资源管理路由

功能：
- 服务注册/注销：独立服务启动时注册，关闭时注销
- 心跳上报：各服务定期上报模型状态和显存占用
- OOM 驱逐：某服务加载模型 OOM 时，按 LRU 顺序驱逐同 GPU 上其他模型
"""

import time
import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, List

logger = logging.getLogger(__name__)

router = APIRouter()

# ========== 服务注册表 ==========
# { service_id: { host, port, gpu_id, model_loaded, vram_used_mb, last_used_time, last_heartbeat } }
_service_registry: Dict[str, dict] = {}


class RegisterRequest(BaseModel):
    service_id: str
    host: str
    port: int
    gpu_id: str = "0"


class UnregisterRequest(BaseModel):
    service_id: str


class HeartbeatRequest(BaseModel):
    service_id: str
    model_loaded: bool = False
    current_version: Optional[str] = None
    vram_used_mb: int = 0
    last_used_time: Optional[float] = None
    gpu_id: str = "0"


class EvictRequest(BaseModel):
    gpu_id: str
    exclude_service: str
    needed_mb: int = 3000


# ========== 注册/注销 ==========

@router.post("/register")
async def register_service(req: RegisterRequest):
    """服务注册"""
    _service_registry[req.service_id] = {
        "host": req.host,
        "port": req.port,
        "gpu_id": req.gpu_id,
        "model_loaded": False,
        "vram_used_mb": 0,
        "last_used_time": None,
        "last_heartbeat": time.time(),
    }
    logger.info(f"【服务注册】{req.service_id} | GPU: {req.gpu_id} | 地址: {req.host}:{req.port}")
    return {"success": True}


@router.post("/unregister")
async def unregister_service(req: UnregisterRequest):
    """服务注销"""
    if req.service_id in _service_registry:
        info = _service_registry.pop(req.service_id)
        logger.info(f"【服务注销】{req.service_id} | GPU: {info.get('gpu_id')}")
    return {"success": True}


# ========== 心跳 ==========

@router.post("/heartbeat")
async def heartbeat(req: HeartbeatRequest):
    """心跳上报"""
    if req.service_id not in _service_registry:
        # 自动注册
        _service_registry[req.service_id] = {
            "host": "unknown",
            "port": 0,
            "gpu_id": req.gpu_id,
        }

    _service_registry[req.service_id].update({
        "model_loaded": req.model_loaded,
        "current_version": req.current_version,
        "vram_used_mb": req.vram_used_mb,
        "last_used_time": req.last_used_time,
        "last_heartbeat": time.time(),
        "gpu_id": req.gpu_id,
    })
    return {"success": True}


# ========== OOM 驱逐 ==========

@router.post("/evict")
async def evict_service(req: EvictRequest):
    """
    驱逐同 GPU 上最久未使用的模型，为新模型腾出显存

    按 LRU (Least Recently Used) 顺序驱逐，直到释放足够显存或无可驱逐服务
    支持驱逐独立服务（HTTP 调用 /model/unload）和主进程内模型（model_manager.unload）
    """
    import requests as http_requests

    # 找到同 GPU 上的其他已加载模型的服务，按 last_used_time 排序（最久未用的在前）
    same_gpu_services = []
    for sid, info in _service_registry.items():
        if sid == req.exclude_service:
            continue
        if info.get("gpu_id") != req.gpu_id:
            continue
        if not info.get("model_loaded", False):
            continue
        same_gpu_services.append((sid, info))

    # 按 last_used_time 升序排列（最久未用的优先驱逐）
    same_gpu_services.sort(key=lambda x: x[1].get("last_used_time") or 0)

    evicted_services = []
    freed_mb = 0

    for sid, info in same_gpu_services:
        if freed_mb >= req.needed_mb:
            break

        try:
            logger.info(f"【OOM驱逐】正在驱逐 {sid} (GPU: {req.gpu_id}, 已释放: {freed_mb}MB/{req.needed_mb}MB)")

            if sid.startswith("main_"):
                # 主进程内模型：直接通过 model_manager 卸载
                model_key = sid[len("main_"):]
                from backend.core.model_manager import model_manager
                if model_manager.unload(model_key):
                    freed_mb += info.get("vram_used_mb", 0)
                    evicted_services.append(sid)
                    _service_registry[sid]["model_loaded"] = False
                    _service_registry[sid]["vram_used_mb"] = 0
                    logger.info(f"【OOM驱逐】主进程模型 {sid} 已卸载，释放约 {info.get('vram_used_mb', 0)}MB")
                else:
                    logger.warning(f"【OOM驱逐】主进程模型 {sid} 卸载失败")
            else:
                # 独立服务：通过 HTTP 调用 /model/unload
                host = info.get("host", "127.0.0.1")
                port = info.get("port", 0)
                resp = http_requests.post(
                    f"http://{host}:{port}/model/unload",
                    timeout=15,
                )
                if resp.status_code == 200:
                    freed_mb += info.get("vram_used_mb", 0)
                    evicted_services.append(sid)
                    _service_registry[sid]["model_loaded"] = False
                    _service_registry[sid]["vram_used_mb"] = 0
                    logger.info(f"【OOM驱逐】{sid} 已卸载，释放约 {info.get('vram_used_mb', 0)}MB")
                else:
                    logger.warning(f"【OOM驱逐】{sid} 卸载失败: {resp.status_code}")
        except Exception as e:
            logger.warning(f"【OOM驱逐】{sid} 卸载请求失败: {e}")

    logger.info(f"【OOM驱逐】完成 | 驱逐: {evicted_services} | 释放: {freed_mb}MB")
    return {
        "success": True,
        "evicted_services": evicted_services,
        "freed_mb": freed_mb,
    }


# ========== 查询 ==========

@router.get("/list")
async def list_services():
    """列出所有已注册服务"""
    now = time.time()
    result = {}
    for sid, info in _service_registry.items():
        result[sid] = {
            **info,
            "heartbeat_age": int(now - info.get("last_heartbeat", now)),
        }
    return result


@router.get("/status")
async def services_status():
    """服务总览：按 GPU 分组显示模型加载状态和显存占用"""
    gpu_groups: Dict[str, list] = {}
    for sid, info in _service_registry.items():
        gpu_id = info.get("gpu_id", "unknown")
        if gpu_id not in gpu_groups:
            gpu_groups[gpu_id] = []
        gpu_groups[gpu_id].append({
            "service_id": sid,
            "model_loaded": info.get("model_loaded", False),
            "current_version": info.get("current_version"),
            "vram_used_mb": info.get("vram_used_mb", 0),
            "last_used_time": info.get("last_used_time"),
        })

    # 计算每个 GPU 的总显存占用
    result = {}
    for gpu_id, services in gpu_groups.items():
        total_vram = sum(s["vram_used_mb"] for s in services)
        result[gpu_id] = {
            "services": services,
            "total_vram_used_mb": total_vram,
        }
    return result


# ========== 主进程内模型管理 ==========

class MainModelUnloadRequest(BaseModel):
    model_key: str  # 模型 key，如 "chattts", "qwen3tts_1.7B_Base" 等


@router.post("/main/unload")
async def main_model_unload(req: MainModelUnloadRequest):
    """卸载主进程内指定的模型"""
    from backend.core.model_manager import model_manager

    if model_manager.unload(req.model_key):
        # 同步更新注册表
        sid = f"main_{req.model_key}"
        if sid in _service_registry:
            _service_registry[sid]["model_loaded"] = False
            _service_registry[sid]["vram_used_mb"] = 0
        return {"success": True, "message": f"模型 {req.model_key} 已卸载"}
    else:
        return {"success": False, "message": f"模型 {req.model_key} 未加载或卸载失败"}


@router.get("/main/status")
async def main_model_status():
    """获取主进程内所有模型的状态"""
    from backend.core.model_manager import model_manager
    return model_manager.get_status()


@router.post("/main/unload_all")
async def main_model_unload_all():
    """卸载主进程内所有模型"""
    from backend.core.model_manager import model_manager
    model_manager.unload_all()
    # 同步更新注册表
    for sid in list(_service_registry.keys()):
        if sid.startswith("main_"):
            _service_registry[sid]["model_loaded"] = False
            _service_registry[sid]["vram_used_mb"] = 0
    return {"success": True, "message": "所有主进程模型已卸载"}
