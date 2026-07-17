#!/usr/bin/env python3
"""
主进程内模型生命周期管理器

功能：
- 懒加载：模型首次调用时加载
- 空闲超时卸载：超过 IDLE_TIMEOUT 未使用的模型自动卸载
- OOM 驱逐：加载 OOM 时请求主服务按 LRU 驱逐同 GPU 上其他模型
- 心跳上报：定期向主服务上报模型状态和显存占用
"""

import os
import gc
import time
import asyncio
import logging
from typing import Optional, Dict, Callable

import torch

from backend.logger_config import system_logger

logger = logging.getLogger(__name__)

# ========== 配置 ==========
IDLE_TIMEOUT = int(os.environ.get("IDLE_TIMEOUT", "300"))  # 默认 5 分钟
HEARTBEAT_INTERVAL = int(os.environ.get("HEARTBEAT_INTERVAL", "60"))  # 心跳间隔秒
GPU_ID = os.environ.get("GPU_ID", "0")  # 主服务使用的 GPU

# 主服务地址（自身）
MAIN_HOST = os.environ.get("HOST", "0.0.0.0")
MAIN_PORT = os.environ.get("PORT", "8000")


class ModelMeta:
    """单个模型的元数据"""
    __slots__ = ("key", "display_name", "last_used_time", "estimated_vram_mb", "unload_fn")

    def __init__(self, key: str, display_name: str, estimated_vram_mb: int, unload_fn: Optional[Callable] = None):
        self.key = key
        self.display_name = display_name
        self.last_used_time: Optional[float] = None
        self.estimated_vram_mb = estimated_vram_mb
        self.unload_fn = unload_fn  # 卸载回调


class ModelManager:
    """
    主进程内模型生命周期管理器

    与 backend/config.py 中的 models dict 配合使用：
    - models dict 存储实际模型对象
    - ModelManager 存储元数据（last_used_time、显存估算、卸载回调）
    """

    def __init__(self):
        # { model_key: ModelMeta }
        self._registry: Dict[str, ModelMeta] = {}
        self._idle_check_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None

    def register(self, key: str, display_name: str, estimated_vram_mb: int, unload_fn: Optional[Callable] = None):
        """注册一个模型到管理器"""
        self._registry[key] = ModelMeta(key, display_name, estimated_vram_mb, unload_fn)
        logger.info(f"【ModelManager】注册模型: {key} ({display_name}, ~{estimated_vram_mb}MB)")

    def touch(self, key: str):
        """更新模型的最后使用时间"""
        if key in self._registry:
            self._registry[key].last_used_time = time.time()

    def get_last_used_time(self, key: str) -> Optional[float]:
        """获取模型最后使用时间"""
        if key in self._registry:
            return self._registry[key].last_used_time
        return None

    def is_loaded(self, key: str) -> bool:
        """检查模型是否已加载"""
        from backend.config import models
        return key in models

    def unload(self, key: str) -> bool:
        """
        卸载指定模型，释放显存

        Returns:
            True 表示成功卸载，False 表示模型未加载或卸载失败
        """
        from backend.config import models

        if key not in models:
            return False

        meta = self._registry.get(key)
        display_name = meta.display_name if meta else key

        logger.info(f"【ModelManager】正在卸载模型: {display_name} ({key})")
        system_logger.info(f"【模型卸载】正在卸载 {display_name}...")

        try:
            # 优先使用自定义卸载回调
            if meta and meta.unload_fn:
                meta.unload_fn(key)
            else:
                # 默认卸载：从 models dict 中删除并清理
                model = models.pop(key, None)
                if model is not None:
                    del model

            # 更新元数据
            if meta:
                meta.last_used_time = None

            # 强制 GC 和显存清理
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info(f"【ModelManager】模型已卸载: {display_name}")
            system_logger.info(f"【模型卸载】{display_name} 已卸载，显存已清理")
            return True

        except Exception as e:
            logger.error(f"【ModelManager】卸载模型 {key} 失败: {e}")
            system_logger.error(f"【模型卸载】卸载 {display_name} 失败: {e}")
            return False

    def unload_all(self):
        """卸载所有已加载的模型"""
        from backend.config import models

        for key in list(models.keys()):
            self.unload(key)

    def get_loaded_models(self) -> list:
        """获取所有已加载模型的 key 列表"""
        from backend.config import models
        return list(models.keys())

    def get_idle_models(self) -> list:
        """获取所有已超时的模型 key 列表（按空闲时间降序，最久的在前）"""
        from backend.config import models
        now = time.time()
        idle = []
        for key in models.keys():
            meta = self._registry.get(key)
            if meta and meta.last_used_time is not None:
                idle_seconds = now - meta.last_used_time
                if idle_seconds > IDLE_TIMEOUT:
                    idle.append((key, idle_seconds))
        # 按空闲时间降序排列
        idle.sort(key=lambda x: x[1], reverse=True)
        return [key for key, _ in idle]

    def get_lru_loaded_models(self, exclude_key: str = None) -> list:
        """获取已加载模型按 last_used_time 升序排列（最久未用的在前），用于 OOM 驱逐"""
        from backend.config import models
        result = []
        for key in models.keys():
            if key == exclude_key:
                continue
            meta = self._registry.get(key)
            if meta:
                result.append((key, meta.last_used_time or 0, meta.estimated_vram_mb))
        # 按 last_used_time 升序排列
        result.sort(key=lambda x: x[1])
        return result

    def get_total_vram_mb(self) -> int:
        """获取所有已加载模型的估算显存总和"""
        from backend.config import models
        total = 0
        for key in models.keys():
            meta = self._registry.get(key)
            if meta:
                total += meta.estimated_vram_mb
        return total

    def get_actual_vram_mb(self) -> int:
        """获取实际 GPU 显存占用（MB）"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() // (1024 * 1024)
        return 0

    # ========== 后台任务 ==========

    async def start_idle_check(self):
        """启动空闲检查后台任务"""
        self._idle_check_task = asyncio.create_task(self._idle_check_loop())
        logger.info(f"【ModelManager】空闲检查已启动 (超时: {IDLE_TIMEOUT}s)")

    async def start_heartbeat(self):
        """启动心跳上报后台任务（向自身 /api/services/heartbeat 注册主进程内的模型）"""
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        logger.info(f"【ModelManager】心跳上报已启动 (间隔: {HEARTBEAT_INTERVAL}s)")

    async def stop(self):
        """停止所有后台任务"""
        if self._idle_check_task:
            self._idle_check_task.cancel()
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
        logger.info("【ModelManager】后台任务已停止")

    async def _idle_check_loop(self):
        """定时检查空闲超时，自动卸载"""
        while True:
            await asyncio.sleep(HEARTBEAT_INTERVAL)
            idle_keys = self.get_idle_models()
            for key in idle_keys:
                meta = self._registry.get(key)
                name = meta.display_name if meta else key
                idle_seconds = time.time() - (meta.last_used_time if meta and meta.last_used_time else 0)
                logger.info(f"【空闲超时】{name} 已空闲 {idle_seconds:.0f}s > {IDLE_TIMEOUT}s，自动卸载")
                system_logger.info(f"【模型管理】{name} 空闲超时，自动卸载")
                self.unload(key)

    async def _heartbeat_loop(self):
        """定时向主服务上报主进程内各模型的状态"""
        while True:
            await asyncio.sleep(HEARTBEAT_INTERVAL)
            self._report_heartbeat()

    def _report_heartbeat(self):
        """向主服务 /api/services/heartbeat 上报主进程内各模型的状态"""
        try:
            import requests as http_requests
            from backend.config import models

            # 主进程内的模型按 service_id = "main_{key}" 注册
            for key, meta in self._registry.items():
                loaded = key in models
                vram_mb = meta.estimated_vram_mb if loaded else 0
                http_requests.post(
                    f"http://{MAIN_HOST}:{MAIN_PORT}/api/services/heartbeat",
                    json={
                        "service_id": f"main_{key}",
                        "model_loaded": loaded,
                        "current_version": None,
                        "vram_used_mb": vram_mb,
                        "last_used_time": meta.last_used_time,
                        "gpu_id": GPU_ID,
                    },
                    timeout=5,
                )
        except Exception:
            pass  # 心跳失败不影响服务

    def register_to_main_service(self):
        """向主服务注册主进程内的所有模型"""
        try:
            import requests as http_requests
            for key, meta in self._registry.items():
                http_requests.post(
                    f"http://{MAIN_HOST}:{MAIN_PORT}/api/services/register",
                    json={
                        "service_id": f"main_{key}",
                        "host": MAIN_HOST,
                        "port": int(MAIN_PORT),
                        "gpu_id": GPU_ID,
                    },
                    timeout=5,
                )
            logger.info(f"【ModelManager】已向主服务注册 {len(self._registry)} 个模型")
        except Exception as e:
            logger.warning(f"【ModelManager】注册主服务失败: {e}")

    def unregister_from_main_service(self):
        """从主服务注销"""
        try:
            import requests as http_requests
            for key in self._registry.keys():
                http_requests.post(
                    f"http://{MAIN_HOST}:{MAIN_PORT}/api/services/unregister",
                    json={"service_id": f"main_{key}"},
                    timeout=5,
                )
            logger.info("【ModelManager】已从主服务注销")
        except Exception as e:
            logger.warning(f"【ModelManager】注销主服务失败: {e}")

    def request_eviction(self, needed_mb: int, exclude_key: str = None):
        """
        OOM 时请求主服务驱逐同 GPU 上其他模型

        先尝试卸载主进程内的 LRU 模型，再请求主服务驱逐其他独立服务
        """
        # 1. 先尝试在主进程内按 LRU 卸载
        lru_models = self.get_lru_loaded_models(exclude_key=exclude_key)
        freed_mb = 0
        for key, _, est_mb in lru_models:
            if freed_mb >= needed_mb:
                break
            meta = self._registry.get(key)
            name = meta.display_name if meta else key
            logger.info(f"【OOM驱逐】正在卸载主进程模型: {name} (已释放: {freed_mb}MB/{needed_mb}MB)")
            if self.unload(key):
                freed_mb += est_mb

        if freed_mb >= needed_mb:
            logger.info(f"【OOM驱逐】主进程内已释放 {freed_mb}MB，满足需求")
            return True

        # 2. 还不够，请求主服务驱逐其他独立服务
        remaining_needed = needed_mb - freed_mb
        try:
            import requests as http_requests
            resp = http_requests.post(
                f"http://{MAIN_HOST}:{MAIN_PORT}/api/services/evict",
                json={
                    "gpu_id": GPU_ID,
                    "exclude_service": "main",  # 不驱逐主进程内的模型（已自行处理）
                    "needed_mb": remaining_needed,
                },
                timeout=30,
            )
            if resp.status_code == 200:
                result = resp.json()
                logger.info(f"【OOM驱逐】主服务已驱逐: {result.get('evicted_services', [])}")
                return True
            else:
                logger.warning(f"【OOM驱逐】主服务返回: {resp.status_code}")
                return False
        except Exception as e:
            logger.warning(f"【OOM驱逐】请求主服务失败: {e}")
            return False

    def get_status(self) -> dict:
        """获取所有模型的当前状态"""
        from backend.config import models
        now = time.time()
        result = {}
        for key, meta in self._registry.items():
            loaded = key in models
            idle_seconds = None
            if loaded and meta.last_used_time:
                idle_seconds = int(now - meta.last_used_time)
            result[key] = {
                "display_name": meta.display_name,
                "loaded": loaded,
                "estimated_vram_mb": meta.estimated_vram_mb if loaded else 0,
                "last_used_time": meta.last_used_time,
                "idle_seconds": idle_seconds,
            }
        return result


# ========== 全局单例 ==========
model_manager = ModelManager()
