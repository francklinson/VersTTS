# 批量TTS处理功能实现记录

## 日期
2026-04-25

## 任务
支持批量上传待生成的文本内容,批量下载生成的音频文件

## 完成内容

### 1. 创建批量处理模块
文件: `backend/batch_processor.py`

**核心类:**
- `BatchTask`: 单个任务项，包含id, text, speaker_id, status, result, error, audio_path
- `BatchJob`: 批量作业，包含job_id, model, created_at, total, completed, failed, status, tasks
- `BatchProcessor`: 批量处理器，提供以下功能：
  - `create_job()`: 创建批量任务作业
  - `get_job()`: 获取任务作业
  - `parse_text_file()`: 解析上传的文本文件(CSV/JSON/TXT)
  - `create_zip_package()`: 创建音频ZIP包
  - `update_task_result()`: 更新任务结果

**支持文件格式:**
| 格式 | 说明 | 示例 |
|------|------|------|
| .txt | 每行一个文本 | 文本1\n文本2\n文本3 |
| .csv | text,speaker_id列 | text,speaker_id\n你好,zh_female |
| .json | JSON数组 | [{"text":"你好"}, {"text":"世界"}] |

### 2. 后端API端点
修改文件: `backend/api_server.py`

**新增端点:**

| 端点 | 方法 | 说明 |
|------|------|------|
| `/tts/batch/create` | POST | 创建批量任务，上传文本文件 |
| `/tts/batch/{job_id}/status` | GET | 查询任务状态 |
| `/tts/batch/{job_id}/process` | POST | 开始处理任务 |
| `/tts/batch/{job_id}/download` | GET | 下载ZIP结果包 |
| `/tts/batch/{job_id}/results` | GET | 获取详细结果 |

**ZIP包内容:**
- `audio_0000.wav` - 生成的音频文件
- `audio_0001.wav` - ...
- `report.json` - 生成报告(包含成功率、失败原因等)
- `mapping.csv` - 文件与文本映射关系

### 3. API版本更新
API版本从 1.1.0 更新到 1.2.0

根端点 `/` 现在返回的端点列表包含批量处理相关端点。

## 技术要点

### 批量处理流程
```
1. 用户上传文本文件(CSV/JSON/TXT)
2. 系统创建批量任务，返回job_id
3. 用户调用process端点开始处理
4. 后台异步处理所有任务
5. 用户查询status获取进度
6. 完成后调用download下载ZIP包
```

### 数据结构设计
```python
BatchTask:
  - id: int              # 任务序号
  - text: str            # 待合成文本
  - speaker_id: str      # 说话人ID(可选)
  - status: str          # 状态: pending/processing/completed/failed
  - result: dict         # 结果数据
  - error: str           # 错误信息
  - audio_path: str      # 音频文件路径

BatchJob:
  - job_id: str          # 作业ID
  - model: str           # TTS模型
  - created_at: str      # 创建时间
  - total: int           # 总任务数
  - completed: int       # 完成数
  - failed: int          # 失败数
  - status: str          # 状态
  - tasks: List[BatchTask]  # 任务列表
```

## 后续计划
1. 实现完整的批量处理逻辑(调用各TTS模型)
2. 添加前端批量处理界面
3. 支持实时进度推送(WebSocket)
4. 添加批量任务队列管理
