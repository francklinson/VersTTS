#!/usr/bin/env python3
"""
批量TTS处理模块
支持批量文本上传、批量生成、打包下载
"""

import os
import io
import csv
import json
import zipfile
import tempfile
import asyncio
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

import soundfile as sf
import numpy as np
from fastapi import UploadFile


@dataclass
class BatchTask:
    """批量任务项"""
    id: int
    text: str
    speaker_id: Optional[str] = None
    status: str = "pending"  # pending, processing, completed, failed
    result: Optional[Dict] = None
    error: Optional[str] = None
    audio_path: Optional[str] = None


@dataclass
class BatchJob:
    """批量任务作业"""
    job_id: str
    model: str
    created_at: str
    total: int
    completed: int = 0
    failed: int = 0
    status: str = "pending"  # pending, processing, completed
    tasks: List[BatchTask] = None
    
    def __post_init__(self):
        if self.tasks is None:
            self.tasks = []


class BatchProcessor:
    """批量处理器"""
    
    def __init__(self, output_dir: str = "output"):
        self.output_dir = output_dir
        self.jobs: Dict[str, BatchJob] = {}
        os.makedirs(output_dir, exist_ok=True)
    
    def create_job(self, model: str, tasks_data: List[Dict]) -> BatchJob:
        """创建批量任务作业"""
        job_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        
        tasks = [
            BatchTask(id=i, **task_data)
            for i, task_data in enumerate(tasks_data)
        ]
        
        job = BatchJob(
            job_id=job_id,
            model=model,
            created_at=datetime.now().isoformat(),
            total=len(tasks),
            tasks=tasks
        )
        
        self.jobs[job_id] = job
        return job
    
    def get_job(self, job_id: str) -> Optional[BatchJob]:
        """获取任务作业"""
        return self.jobs.get(job_id)
    
    @staticmethod
    def parse_text_file(file_content: bytes, filename: str) -> List[Dict]:
        """解析上传的文本文件"""
        tasks = []
        content = file_content.decode('utf-8')
        
        if filename.endswith('.csv'):
            # CSV格式: text,speaker_id(可选)
            reader = csv.DictReader(io.StringIO(content))
            for row in reader:
                task = {"text": row.get('text', '').strip()}
                if 'speaker_id' in row and row['speaker_id']:
                    task['speaker_id'] = row['speaker_id'].strip()
                tasks.append(task)
                
        elif filename.endswith('.json'):
            # JSON格式: [{"text": "...", "speaker_id": "..."}, ...]
            data = json.loads(content)
            if isinstance(data, list):
                for item in data:
                    task = {"text": item.get('text', '').strip()}
                    if 'speaker_id' in item:
                        task['speaker_id'] = item['speaker_id']
                    tasks.append(task)
                    
        elif filename.endswith('.txt'):
            # TXT格式: 每行一个文本
            lines = content.strip().split('\n')
            for line in lines:
                line = line.strip()
                if line:
                    tasks.append({"text": line})
        else:
            raise ValueError(f"不支持的文件格式: {filename}")
        
        return tasks
    
    def create_zip_package(self, job_id: str) -> str:
        """创建音频ZIP包"""
        job = self.jobs.get(job_id)
        if not job:
            raise ValueError(f"任务不存在: {job_id}")
        
        zip_path = os.path.join(self.output_dir, f"batch_{job_id}.zip")
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            # 添加音频文件
            for task in job.tasks:
                if task.audio_path and os.path.exists(task.audio_path):
                    arcname = f"audio_{task.id:04d}.wav"
                    zf.write(task.audio_path, arcname)
            
            # 添加生成报告
            report = self._generate_report(job)
            zf.writestr("report.json", json.dumps(report, ensure_ascii=False, indent=2))
            
            # 添加文本映射
            mapping = self._generate_mapping(job)
            zf.writestr("mapping.csv", mapping)
        
        return zip_path
    
    def _generate_report(self, job: BatchJob) -> Dict:
        """生成任务报告"""
        return {
            "job_id": job.job_id,
            "model": job.model,
            "created_at": job.created_at,
            "status": job.status,
            "total": job.total,
            "completed": job.completed,
            "failed": job.failed,
            "success_rate": f"{(job.completed / job.total * 100):.1f}%" if job.total > 0 else "0%",
            "tasks": [
                {
                    "id": task.id,
                    "text": task.text[:100] + "..." if len(task.text) > 100 else task.text,
                    "status": task.status,
                    "error": task.error
                }
                for task in job.tasks
            ]
        }
    
    def _generate_mapping(self, job: BatchJob) -> str:
        """生成文件映射CSV"""
        lines = ["id,audio_file,text"]
        for task in job.tasks:
            audio_file = f"audio_{task.id:04d}.wav" if task.audio_path else ""
            text = task.text.replace('"', '""')  # CSV转义
            lines.append(f'{task.id},"{audio_file}","{text}"')
        return '\n'.join(lines)
    
    def update_task_result(self, job_id: str, task_id: int, 
                          status: str, audio_path: Optional[str] = None,
                          error: Optional[str] = None):
        """更新任务结果"""
        job = self.jobs.get(job_id)
        if not job:
            return
        
        for task in job.tasks:
            if task.id == task_id:
                task.status = status
                task.audio_path = audio_path
                task.error = error
                break
        
        # 更新统计
        job.completed = sum(1 for t in job.tasks if t.status == "completed")
        job.failed = sum(1 for t in job.tasks if t.status == "failed")
        
        if job.completed + job.failed >= job.total:
            job.status = "completed"


# 全局批量处理器实例
batch_processor = BatchProcessor()
