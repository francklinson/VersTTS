#!/usr/bin/env python3
"""
VersTTS 批量TTS生成客户端脚本

功能：
1. 支持从CSV/JSON/TXT文件读取文本
2. 支持所有9种TTS算法
3. 支持并发请求控制
4. 支持下载结果ZIP包
5. 错误重试机制
6. 进度条显示

使用方法:
    python batch_tts_client.py --input texts.csv --model chattts --output ./results
    python batch_tts_client.py --input texts.json --model gptsovits --speaker  speaker_001 --concurrent 3

作者: VersTTS
日期: 2025-04-30
"""

import argparse
import asyncio
import csv
import json
import os
import sys
import time
import zipfile
from pathlib import Path
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
from urllib.parse import urljoin

import aiohttp
from aiohttp import ClientTimeout, ClientSession
from tqdm import tqdm


# ========== 配置 ==========
DEFAULT_API_BASE = "http://localhost:8000"
DEFAULT_TIMEOUT = 300  # 5分钟
DEFAULT_RETRY_COUNT = 3
DEFAULT_RETRY_DELAY = 2.0

# 支持的TTS模型
SUPPORTED_MODELS = [
    "chattts",
    "cosyvoice",
    "f5tts",
    "qwen3tts",
    "openvoice",
    "gptsovits",
    "voxcpm",
    "indextts",
    "fireredtts"
]


@dataclass
class TTSConfig:
    """TTS配置"""
    model: str
    speaker_id: Optional[str] = None
    temperature: float = 0.3
    top_p: float = 0.7
    top_k: int = 20
    speed: float = 1.0
    # 模型特有参数
    version: Optional[str] = None  # gptsovits版本
    mode: Optional[str] = None  # 生成模式
    emotion: Optional[str] = None  # 情感
    style: Optional[str] = None  # 风格


@dataclass
class TTSTask:
    """TTS任务"""
    id: int
    text: str
    speaker_id: Optional[str] = None
    status: str = "pending"  # pending, processing, completed, failed
    error: Optional[str] = None
    audio_url: Optional[str] = None
    audio_path: Optional[str] = None
    retry_count: int = 0


class BatchTTSClient:
    """批量TTS客户端"""
    
    def __init__(
        self,
        api_base: str = DEFAULT_API_BASE,
        timeout: int = DEFAULT_TIMEOUT,
        max_concurrent: int = 1,
        retry_count: int = DEFAULT_RETRY_COUNT,
        retry_delay: float = DEFAULT_RETRY_DELAY
    ):
        self.api_base = api_base.rstrip("/")
        self.timeout = ClientTimeout(total=timeout)
        self.max_concurrent = max_concurrent
        self.retry_count = retry_count
        self.retry_delay = retry_delay
        self.session: Optional[ClientSession] = None
        self.session_id: Optional[str] = None
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        connector = aiohttp.TCPConnector(limit=self.max_concurrent * 2)
        self.session = ClientSession(
            connector=connector,
            timeout=self.timeout,
            headers={"Accept": "application/json"}
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        if self.session:
            await self.session.close()
    
    async def check_health(self) -> bool:
        """检查服务健康状态"""
        try:
            url = urljoin(self.api_base, "/health")
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("status") == "healthy"
                return False
        except Exception as e:
            print(f"⚠️  健康检查失败: {e}")
            return False
    
    async def get_queue_status(self) -> Dict:
        """获取队列状态"""
        try:
            url = urljoin(self.api_base, "/concurrency/queue/wait-time")
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return data
                return {}
        except Exception as e:
            print(f"⚠️  获取队列状态失败: {e}")
            return {}
    
    async def generate_tts(
        self,
        text: str,
        config: TTSConfig,
        task_id: int = 0
    ) -> Dict:
        """
        生成单个TTS
        
        Args:
            text: 要合成的文本
            config: TTS配置
            task_id: 任务ID（用于日志）
        
        Returns:
            包含audio_url或error的字典
        """
        # 构建请求数据
        data = {"text": text}
        
        # 添加配置参数
        if config.temperature is not None:
            data["temperature"] = config.temperature
        if config.top_p is not None:
            data["top_P"] = config.top_p
        if config.top_k is not None:
            data["top_K"] = config.top_k
        if config.speed is not None:
            data["speed"] = config.speed
        
        # 添加说话人ID
        speaker_id = config.speaker_id
        if speaker_id:
            data["clone_speaker_id"] = speaker_id
        
        # 添加版本参数（GPT-SoVITS）
        if config.version and config.model == "gptsovits":
            data["version"] = config.version
        
        # 添加模式参数
        if config.mode:
            data["mode"] = config.mode
        
        # 添加情感参数
        if config.emotion:
            data["emotion"] = config.emotion
        
        # 构建URL
        endpoint = f"/tts/{config.model}"
        url = urljoin(self.api_base, endpoint)
        
        # 准备请求头
        headers = {}
        if self.session_id:
            headers["X-Session-ID"] = self.session_id
        
        # 重试机制
        for attempt in range(self.retry_count + 1):
            try:
                async with self.session.post(
                    url,
                    data=data,
                    headers=headers
                ) as response:
                    # 保存session_id
                    if "X-Session-ID" in response.headers:
                        self.session_id = response.headers["X-Session-ID"]
                    
                    if response.status == 200:
                        result = await response.json()
                        if result.get("success"):
                            return {
                                "success": True,
                                "audio_url": result.get("audio_url"),
                                "audio_base64": result.get("audio_base64"),
                                "sample_rate": result.get("sample_rate", 24000)
                            }
                        else:
                            return {
                                "success": False,
                                "error": result.get("message", "未知错误")
                            }
                    elif response.status == 429:
                        # 限流，等待后重试
                        wait_time = int(response.headers.get("Retry-After", 60))
                        if attempt < self.retry_count:
                            print(f"  任务 {task_id}: 触发限流，等待 {wait_time} 秒后重试...")
                            await asyncio.sleep(wait_time)
                            continue
                        else:
                            return {
                                "success": False,
                                "error": f"请求过于频繁 (HTTP 429)"
                            }
                    elif response.status == 503:
                        # 服务繁忙
                        if attempt < self.retry_count:
                            print(f"  任务 {task_id}: GPU繁忙，等待 {self.retry_delay} 秒后重试...")
                            await asyncio.sleep(self.retry_delay)
                            continue
                        else:
                            return {
                                "success": False,
                                "error": "GPU资源繁忙，请稍后重试"
                            }
                    else:
                        error_text = await response.text()
                        return {
                            "success": False,
                            "error": f"HTTP {response.status}: {error_text[:200]}"
                        }
                        
            except asyncio.TimeoutError:
                if attempt < self.retry_count:
                    print(f"  任务 {task_id}: 请求超时，正在重试 ({attempt + 1}/{self.retry_count})...")
                    await asyncio.sleep(self.retry_delay)
                    continue
                return {
                    "success": False,
                    "error": "请求超时"
                }
            except Exception as e:
                if attempt < self.retry_count:
                    print(f"  任务 {task_id}: 错误 {e}，正在重试 ({attempt + 1}/{self.retry_count})...")
                    await asyncio.sleep(self.retry_delay)
                    continue
                return {
                    "success": False,
                    "error": str(e)
                }
        
        return {
            "success": False,
            "error": "达到最大重试次数"
        }
    
    async def download_audio(
        self,
        audio_url: str,
        output_path: str,
        task_id: int = 0
    ) -> bool:
        """下载音频文件"""
        try:
            # 构建完整URL
            if audio_url.startswith("/"):
                url = urljoin(self.api_base, audio_url)
            else:
                url = audio_url
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    # 确保输出目录存在
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    
                    # 保存文件
                    with open(output_path, "wb") as f:
                        async for chunk in response.content.iter_chunked(8192):
                            f.write(chunk)
                    return True
                else:
                    print(f"  任务 {task_id}: 下载失败 HTTP {response.status}")
                    return False
        except Exception as e:
            print(f"  任务 {task_id}: 下载错误 {e}")
            return False
    
    async def process_batch(
        self,
        tasks: List[TTSTask],
        config: TTSConfig,
        output_dir: str,
        progress_callback: Optional[Callable] = None
    ) -> List[TTSTask]:
        """
        批量处理TTS任务
        
        Args:
            tasks: 任务列表
            config: TTS配置
            output_dir: 输出目录
            progress_callback: 进度回调函数
        
        Returns:
            处理后的任务列表
        """
        semaphore = asyncio.Semaphore(self.max_concurrent)
        completed_count = [0]  # 使用列表以便在闭包中修改
        
        async def process_single(task: TTSTask) -> TTSTask:
            async with semaphore:
                task.status = "processing"
                
                # 生成TTS
                result = await self.generate_tts(
                    task.text,
                    config,
                    task.id
                )
                
                if result["success"]:
                    task.audio_url = result.get("audio_url")
                    
                    # 下载音频
                    if task.audio_url:
                        output_path = os.path.join(
                            output_dir,
                            f"audio_{task.id:04d}.wav"
                        )
                        
                        if await self.download_audio(
                            task.audio_url,
                            output_path,
                            task.id
                        ):
                            task.audio_path = output_path
                            task.status = "completed"
                        else:
                            task.status = "failed"
                            task.error = "下载音频失败"
                    else:
                        task.status = "completed"
                else:
                    task.status = "failed"
                    task.error = result.get("error", "未知错误")
                
                completed_count[0] += 1
                if progress_callback:
                    progress_callback(completed_count[0], len(tasks))
                
                return task
        
        # 创建所有任务
        coroutines = [process_single(task) for task in tasks]
        
        # 等待所有任务完成
        results = await asyncio.gather(*coroutines, return_exceptions=True)
        
        # 处理异常结果
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                tasks[i].status = "failed"
                tasks[i].error = str(result)
        
        return tasks


def parse_input_file(file_path: str) -> List[Dict]:
    """
    解析输入文件
    
    支持格式:
    - CSV: text,speaker_id(可选)
    - JSON: [{"text": "...", "speaker_id": "..."}, ...]
    - TXT: 每行一个文本
    """
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    suffix = path.suffix.lower()
    
    if suffix == ".csv":
        tasks = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                task = {"text": row.get("text", "").strip()}
                if "speaker_id" in row and row["speaker_id"]:
                    task["speaker_id"] = row["speaker_id"].strip()
                tasks.append(task)
        return tasks
    
    elif suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return [
                    {
                        "text": item.get("text", ""),
                        "speaker_id": item.get("speaker_id")
                    }
                    for item in data
                ]
            else:
                raise ValueError("JSON文件必须是对象数组")
    
    elif suffix == ".txt":
        tasks = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                text = line.strip()
                if text:
                    tasks.append({"text": text})
        return tasks
    
    else:
        raise ValueError(f"不支持的文件格式: {suffix}")


def save_report(
    tasks: List[TTSTask],
    output_dir: str,
    config: TTSConfig,
    duration: float
):
    """保存处理报告"""
    report = {
        "generated_at": datetime.now().isoformat(),
        "config": asdict(config),
        "duration_seconds": round(duration, 2),
        "total": len(tasks),
        "completed": sum(1 for t in tasks if t.status == "completed"),
        "failed": sum(1 for t in tasks if t.status == "failed"),
        "tasks": [
            {
                "id": t.id,
                "text": t.text[:100] + "..." if len(t.text) > 100 else t.text,
                "status": t.status,
                "error": t.error,
                "audio_path": t.audio_path
            }
            for t in tasks
        ]
    }
    
    report_path = os.path.join(output_dir, "report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    # 同时生成CSV
    csv_path = os.path.join(output_dir, "mapping.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "text", "status", "audio_file"])
        for t in tasks:
            audio_file = os.path.basename(t.audio_path) if t.audio_path else ""
            text = t.text.replace('"', '""')
            writer.writerow([t.id, text, t.status, audio_file])
    
    return report_path, csv_path


def create_zip_package(output_dir: str, zip_name: str = "tts_results.zip") -> str:
    """创建ZIP包"""
    zip_path = os.path.join(output_dir, zip_name)
    
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                if file == zip_name:
                    continue
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, output_dir)
                zf.write(file_path, arcname)
    
    return zip_path


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="VersTTS 批量TTS生成客户端",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # CSV格式输入
  python batch_tts_client.py -i texts.csv -m chattts -o ./results
  
  # JSON格式输入，指定说话人
  python batch_tts_client.py -i texts.json -m gptsovits -s speaker_001 -o ./results
  
  # TXT格式输入，并发处理
  python batch_tts_client.py -i texts.txt -m f5tts -c 3 -o ./results
  
  # GPT-SoVITS指定版本
  python batch_tts_client.py -i texts.csv -m gptsovits --version v2 -s speaker_001
        """
    )
    
    # 输入输出
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="输入文件路径 (支持 .csv, .json, .txt)"
    )
    parser.add_argument(
        "-o", "--output",
        default="./tts_results",
        help="输出目录 (默认: ./tts_results)"
    )
    
    # TTS配置
    parser.add_argument(
        "-m", "--model",
        required=True,
        choices=SUPPORTED_MODELS,
        help=f"TTS模型 ({', '.join(SUPPORTED_MODELS)})"
    )
    parser.add_argument(
        "-s", "--speaker",
        help="说话人ID (从说话人管理模块选择)"
    )
    parser.add_argument(
        "--version",
        help="模型版本 (仅GPT-SoVITS支持: v1, v2, v3, v4)"
    )
    parser.add_argument(
        "--mode",
        help="生成模式 (取决于具体模型)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="采样温度 (默认: 0.3)"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.7,
        help="Top-P采样 (默认: 0.7)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Top-K采样 (默认: 20)"
    )
    
    # 连接配置
    parser.add_argument(
        "-u", "--url",
        default=DEFAULT_API_BASE,
        help=f"API基础URL (默认: {DEFAULT_API_BASE})"
    )
    parser.add_argument(
        "-c", "--concurrent",
        type=int,
        default=1,
        help="并发请求数 (默认: 1)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"请求超时时间秒 (默认: {DEFAULT_TIMEOUT})"
    )
    parser.add_argument(
        "--retry",
        type=int,
        default=DEFAULT_RETRY_COUNT,
        help=f"错误重试次数 (默认: {DEFAULT_RETRY_COUNT})"
    )
    
    # 输出选项
    parser.add_argument(
        "--zip",
        action="store_true",
        help="生成ZIP包"
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="不下载音频，只生成报告"
    )
    
    args = parser.parse_args()
    
    # 打印欢迎信息
    print("=" * 60)
    print("VersTTS 批量TTS生成客户端")
    print("=" * 60)
    print(f"API地址: {args.url}")
    print(f"输入文件: {args.input}")
    print(f"输出目录: {args.output}")
    print(f"TTS模型: {args.model}")
    print(f"并发数: {args.concurrent}")
    print("=" * 60)
    
    # 解析输入文件
    try:
        input_tasks = parse_input_file(args.input)
        print(f"✅ 成功加载 {len(input_tasks)} 个任务")
    except Exception as e:
        print(f"❌ 加载输入文件失败: {e}")
        sys.exit(1)
    
    # 创建TTSTask列表
    tasks = [
        TTSTask(
            id=i,
            text=t["text"],
            speaker_id=t.get("speaker_id") or args.speaker
        )
        for i, t in enumerate(input_tasks)
    ]
    
    # 创建TTS配置
    config = TTSConfig(
        model=args.model,
        speaker_id=args.speaker,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        version=args.version,
        mode=args.mode
    )
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    # 初始化客户端并处理
    start_time = time.time()
    
    async with BatchTTSClient(
        api_base=args.url,
        timeout=args.timeout,
        max_concurrent=args.concurrent,
        retry_count=args.retry
    ) as client:
        # 检查服务健康
        print("\n🔍 检查服务状态...")
        if not await client.check_health():
            print("❌ 服务不可用，请检查API地址是否正确")
            sys.exit(1)
        print("✅ 服务正常")
        
        # 获取队列状态
        queue_status = await client.get_queue_status()
        if queue_status:
            print(f"📊 当前队列长度: {queue_status.get('queue_size', '未知')}")
            print(f"⏱️  预计等待时间: {queue_status.get('estimated_wait_formatted', '未知')}")
        
        # 处理任务
        print(f"\n🚀 开始处理 {len(tasks)} 个TTS任务...")
        print("-" * 60)
        
        # 创建进度条
        pbar = tqdm(total=len(tasks), desc="处理进度", unit="task")
        
        def progress_callback(completed, total):
            pbar.update(1)
        
        results = await client.process_batch(
            tasks,
            config,
            args.output,
            progress_callback
        )
        
        pbar.close()
    
    duration = time.time() - start_time
    
    # 统计结果
    completed = sum(1 for t in results if t.status == "completed")
    failed = sum(1 for t in results if t.status == "failed")
    
    print("-" * 60)
    print(f"\n✅ 处理完成!")
    print(f"   总任务: {len(results)}")
    print(f"   成功: {completed}")
    print(f"   失败: {failed}")
    print(f"   耗时: {duration:.2f} 秒")
    print(f"   平均: {duration/len(results):.2f} 秒/任务" if results else "")
    
    # 保存报告
    report_path, csv_path = save_report(results, args.output, config, duration)
    print(f"\n📝 报告已保存:")
    print(f"   JSON: {report_path}")
    print(f"   CSV: {csv_path}")
    
    # 创建ZIP包
    if args.zip:
        zip_path = create_zip_package(args.output)
        print(f"   ZIP: {zip_path}")
    
    # 显示失败任务
    if failed > 0:
        print("\n⚠️  失败任务:")
        for t in results:
            if t.status == "failed":
                text_preview = t.text[:50] + "..." if len(t.text) > 50 else t.text
                print(f"   [{t.id}] {text_preview}")
                print(f"       错误: {t.error}")
    
    print("\n" + "=" * 60)
    print("处理完成!")
    print("=" * 60)
    
    # 返回退出码
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
