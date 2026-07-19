#!/usr/bin/env python3
"""
OmniVoice 路由
通过独立服务调用 OmniVoice（需要 transformers 5.x）
"""

import time
import os
import re
import requests
import soundfile as sf
from datetime import datetime
from fastapi import APIRouter, Form, Request, HTTPException
from typing import Optional

from backend.logger_config import OperationLogger, system_logger
from backend.models import TTSResponse
from backend.services import get_speaker_by_id
from backend.core import audio_to_base64
from backend.config import OUTPUTS_DIR, OMNIVOICE_HOST, OMNIVOICE_PORT

router = APIRouter()

OMNIVOICE_SERVICE_URL = f"http://{OMNIVOICE_HOST}:{OMNIVOICE_PORT}/tts"
OMNIVOICE_HEALTH_URL = f"http://{OMNIVOICE_HOST}:{OMNIVOICE_PORT}/health"


def _check_omnivoice_service():
    """检查 OmniVoice 独立服务是否运行"""
    try:
        response = requests.get(OMNIVOICE_HEALTH_URL, timeout=3)
        return response.status_code == 200
    except Exception:
        return False


def _fix_voice_design_format(prompt: str) -> str:
    """
    修正声音设计属性的格式
    
    问题：用户可能输入 "宁夏话女，中年，高音调" 或 "女，老年甘肃话"（缺少逗号）
    修正："宁夏话，女，中年，高音调" 或 "女，老年，甘肃话"
    
    规则：
    1. 在相邻的中文属性之间添加逗号分隔
    2. 处理所有属性组合之间缺少分隔符的情况
    """
    if not prompt:
        return prompt
    
    # 定义所有可能的属性值
    dialects = ['四川话', '东北话', '河南话', '陕西话', '云南话', '贵州话', '桂林话',
                '甘肃话', '宁夏话', '济南话', '青岛话', '石家庄话']
    genders = ['男', '女']
    ages = ['儿童', '少年', '青年', '中年', '老年']
    pitches = ['极低音调', '低音调', '中音调', '高音调', '极高音调', '耳语']
    
    all_attributes = dialects + genders + ages + pitches
    
    result = prompt
    
    # 迭代处理直到没有变化（处理多重粘连）
    max_iterations = 10
    for _ in range(max_iterations):
        prev_result = result
        
        # 1. 处理方言和性别之间缺少分隔符
        # 例如："宁夏话女" -> "宁夏话，女"
        for dialect in dialects:
            for gender in genders:
                result = re.sub(f'{dialect}{gender}(?![\u4e00-\u9fff])', f'{dialect}，{gender}', result)
        
        # 2. 处理性别和年龄之间缺少分隔符
        # 例如："男中年" -> "男，中年"
        for gender in genders:
            for age in ages:
                result = re.sub(f'{gender}{age}(?![\u4e00-\u9fff])', f'{gender}，{age}', result)
        
        # 3. 处理年龄和方言之间缺少分隔符
        # 例如："老年甘肃话" -> "老年，甘肃话"
        for age in ages:
            for dialect in dialects:
                result = re.sub(f'{age}{dialect}(?![\u4e00-\u9fff])', f'{age}，{dialect}', result)
        
        # 4. 处理年龄和音调之间缺少分隔符
        # 例如："老年高音调" -> "老年，高音调"
        for age in ages:
            for pitch in pitches:
                result = re.sub(f'{age}{pitch}(?![\u4e00-\u9fff])', f'{age}，{pitch}', result)
        
        # 5. 处理性别和音调之间缺少分隔符
        # 例如："男高音调" -> "男，高音调"
        for gender in genders:
            for pitch in pitches:
                result = re.sub(f'{gender}{pitch}(?![\u4e00-\u9fff])', f'{gender}，{pitch}', result)
        
        # 6. 处理方言和音调之间缺少分隔符
        # 例如："四川话高音调" -> "四川话，高音调"
        for dialect in dialects:
            for pitch in pitches:
                result = re.sub(f'{dialect}{pitch}(?![\u4e00-\u9fff])', f'{dialect}，{pitch}', result)
        
        if result == prev_result:
            break  # 没有更多变化，退出迭代
    
    # 规范化分隔符：统一使用中文逗号，去除多余空格
    result = result.replace(',', '，')  # 英文逗号转中文
    result = re.sub(r'\s*，\s*', '，', result)  # 去除逗号前后的空格
    result = re.sub(r'，+', '，', result)  # 多个逗号合并为一个
    result = re.sub(r'^，|，$', '', result)  # 移除开头和结尾的逗号
    
    return result.strip()


@router.post("/")
async def tts_omnivoice(
        request: Request,
        text: str = Form(...),
        mode: str = Form("auto_voice"),
        clone_speaker_id: Optional[str] = Form(None),
        voice_design_prompt: Optional[str] = Form(None),
        num_steps: int = Form(32),
        speed: float = Form(1.0),
        output_format: str = Form("url")
):
    """OmniVoice语音合成 - 通过独立服务"""
    ref_path = None
    speaker_ref_text = None
    start_time = time.time()
    client_ip = request.client.host if request.client else "unknown"
    
    # 详细日志：请求开始
    system_logger.info(f"【OmniVoice】{'='*60}")
    system_logger.info(f"【OmniVoice】请求开始 | 模式: {mode} | 客户端: {client_ip}")
    system_logger.info(f"【OmniVoice】输入文本: {text[:100]}...")

    try:
        # 记录API请求参数
        request_params = {
            "text_preview": text[:50],
            "mode": mode,
            "clone_speaker_id": clone_speaker_id,
            "voice_design_prompt": voice_design_prompt[:50] if voice_design_prompt else None,
            "num_steps": num_steps,
            "speed": speed
        }
        OperationLogger.log_api_request("/tts/omnivoice", "POST", request_params, client_ip)
        system_logger.info(f"【OmniVoice】请求参数: {request_params}")

        # 检查独立服务状态
        system_logger.info(f"【OmniVoice】检查独立服务状态...")
        if not _check_omnivoice_service():
            system_logger.error(f"【OmniVoice】独立服务未运行")
            raise HTTPException(
                status_code=503,
                detail="OmniVoice 独立服务未运行。请执行: nohup python omnivoice_service.py > logs/omnivoice_service.log 2>&1 &"
            )
        system_logger.info(f"【OmniVoice】独立服务运行正常")
        
        # 记录模型信息
        system_logger.info(f"【OmniVoice】模型信息:")
        system_logger.info(f"【OmniVoice】   - 模型名称: OmniVoice")
        system_logger.info(f"【OmniVoice】   - 模型来源: HuggingFace")
        system_logger.info(f"【OmniVoice】   - 模型路径: /home/zhouchenghao/PycharmProjects/VersTTS/models/speechpi/OmniVoice")
        system_logger.info(f"【OmniVoice】   - Transformers版本: 5.x (独立服务)")
        system_logger.info(f"【OmniVoice】   - 支持模式: voice_clone, voice_design, auto_voice")
        system_logger.info(f"【OmniVoice】   - 推理步数: {num_steps}")
        system_logger.info(f"【OmniVoice】   - 语速范围: 0.5 - 2.0")
        system_logger.info(f"【OmniVoice】   - 支持方言: 四川话, 东北话, 河南话, 陕西话, 云南话, 贵州话, 桂林话, 甘肃话, 宁夏话, 济南话, 青岛话, 石家庄话")
        system_logger.info(f"【OmniVoice】   - 支持音调: 极低音调, 低音调, 中音调, 高音调, 极高音调, 耳语")

        # 获取说话人信息
        speaker = None
        if mode == "voice_clone" and clone_speaker_id:
            system_logger.info(f"【OmniVoice】查找说话人: {clone_speaker_id}")
            speaker = get_speaker_by_id(clone_speaker_id)
            if not speaker:
                system_logger.error(f"【OmniVoice】说话人不存在: {clone_speaker_id}")
                raise HTTPException(status_code=404, detail=f"说话人不存在: {clone_speaker_id}")
            ref_path = speaker.get("audio_path")
            speaker_ref_text = speaker.get("reference_text")
            system_logger.info(f"【OmniVoice】找到说话人 | 名称: {speaker.get('name')} | 音频: {ref_path}")
            
            if ref_path and os.path.exists(ref_path):
                file_size = os.path.getsize(ref_path)
                system_logger.info(f"【OmniVoice】参考音频存在 | 大小: {file_size / 1024:.2f} KB")
            else:
                system_logger.error(f"【OmniVoice】参考音频不存在: {ref_path}")
                raise HTTPException(status_code=404, detail=f"参考音频文件不存在: {ref_path}")

        # 语速参数自动矫正
        original_speed = speed
        actual_speed = speed
        if speed < 0.5 or speed > 2.0:
            actual_speed = max(0.5, min(2.0, speed))
            actual_speed = round(actual_speed, 1)  # 保留一位小数
            system_logger.warning(f"【OmniVoice】语速参数 {original_speed} 超出范围，已自动矫正为 {actual_speed}")
        elif speed != round(speed, 1):
            # 确保只有一位小数
            actual_speed = round(speed, 1)

        # 声音设计模式校验和格式修正
        actual_voice_design = voice_design_prompt
        if mode == "voice_design" and voice_design_prompt:
            system_logger.info(f"【OmniVoice】进入声音设计模式")
            has_chinese = bool(re.search(r'[\u4e00-\u9fff]', voice_design_prompt))
            has_english = bool(re.search(r'[a-zA-Z]', voice_design_prompt))
            if has_chinese and has_english:
                system_logger.error(f"【OmniVoice】中英文混用错误")
                raise HTTPException(
                    status_code=400,
                    detail="声音设计属性不能中英文混用！请使用纯中文（如'男，四川话'）或纯英文（如'male, high pitch'）"
                )
            
            # 修正格式：确保属性之间用逗号分隔
            # 处理 "宁夏话女" -> "宁夏话，女" 这样的情况
            actual_voice_design = _fix_voice_design_format(voice_design_prompt)
            system_logger.info(f"【OmniVoice】原始描述: {voice_design_prompt}")
            system_logger.info(f"【OmniVoice】修正后描述: {actual_voice_design}")

        # 调用独立服务
        system_logger.info(f"【OmniVoice】调用独立服务: {OMNIVOICE_SERVICE_URL}")
        data = {
            "text": text,
            "mode": mode,
            "num_steps": num_steps,
            "speed": actual_speed
        }
        
        actual_params = {
            "mode": mode,
            "num_steps": num_steps,
            "speed": actual_speed,
            "speed_corrected": actual_speed != original_speed
        }
        
        if mode == "voice_clone" and ref_path:
            data["ref_audio"] = ref_path
            actual_params["clone_speaker_id"] = clone_speaker_id
            actual_params["speaker_name"] = speaker.get('name') if speaker else None
            actual_params["ref_audio"] = ref_path
            actual_params["ref_text"] = speaker_ref_text[:50] if speaker_ref_text else None
            if speaker_ref_text:
                data["ref_text"] = speaker_ref_text
            system_logger.info(f"【OmniVoice】Voice Clone参数: ref_audio={ref_path}, ref_text={speaker_ref_text[:50] if speaker_ref_text else '无'}...")
        elif mode == "voice_design" and actual_voice_design:
            data["voice_design_prompt"] = actual_voice_design
            actual_params["voice_design_prompt"] = actual_voice_design[:100]
            actual_params["original_prompt"] = voice_design_prompt[:100] if voice_design_prompt != actual_voice_design else None
            system_logger.info(f"【OmniVoice】Voice Design参数: {actual_voice_design}")

        service_start = time.time()
        response = requests.post(OMNIVOICE_SERVICE_URL, data=data, timeout=120)
        service_duration = time.time() - service_start
        system_logger.info(f"【OmniVoice】独立服务响应 | 状态码: {response.status_code} | 耗时: {service_duration:.3f}s")

        if response.status_code != 200:
            error_detail = response.json().get("detail", "未知错误")
            system_logger.error(f"【OmniVoice】独立服务错误: {error_detail}")
            raise HTTPException(status_code=response.status_code, detail=error_detail)

        result = response.json()
        if not result.get("success"):
            system_logger.error(f"【OmniVoice】合成失败: {result}")
            raise HTTPException(status_code=500, detail="OmniVoice 合成失败")

        audio_path = result.get("audio_path")
        system_logger.info(f"【OmniVoice】独立服务返回音频: {audio_path}")
        
        if not audio_path or not os.path.exists(audio_path):
            system_logger.error(f"【OmniVoice】音频文件未生成: {audio_path}")
            raise HTTPException(status_code=500, detail="OmniVoice 音频文件未生成")

        # 保存音频到 outputs 目录
        system_logger.info(f"【OmniVoice】保存音频文件...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(OUTPUTS_DIR, f"omnivoice_{timestamp}.wav")
        save_start = time.time()
        audio_data, sample_rate = sf.read(audio_path)
        sf.write(output_path, audio_data, samplerate=sample_rate)
        save_duration = time.time() - save_start
        file_size = os.path.getsize(output_path)
        system_logger.info(f"【OmniVoice】音频保存完成: {output_path} | 大小: {file_size / 1024:.2f} KB | 耗时: {save_duration:.3f}s")
        
        actual_params["output_path"] = output_path
        actual_params["sample_rate"] = sample_rate
        actual_params["file_size_kb"] = round(file_size / 1024, 2)
        actual_params["service_duration"] = round(service_duration, 3)

        # 清理临时文件
        try:
            os.remove(audio_path)
            system_logger.info(f"【OmniVoice】清理服务临时文件: {audio_path}")
        except Exception as e:
            system_logger.warning(f"【OmniVoice】清理临时文件失败: {e}")

        total_duration = time.time() - start_time
        actual_params["total_duration"] = round(total_duration, 3)
        
        OperationLogger.log_tts_request("OmniVoice", text, actual_params, total_duration, "成功")
        system_logger.info(f"【OmniVoice】请求完成 | 总耗时: {total_duration:.3f}s | 输出: {output_path}")
        system_logger.info(f"【OmniVoice】{'='*60}")

        if output_format == "base64":
            audio_b64 = audio_to_base64(output_path)
            return TTSResponse(
                success=True,
                message="合成成功",
                audio_base64=audio_b64,
                sample_rate=sample_rate
            )
        else:
            return TTSResponse(
                success=True,
                message="合成成功",
                audio_url=f"/audio/{os.path.basename(output_path)}",
                sample_rate=sample_rate
            )

    except HTTPException:
        raise
    except Exception as e:
        total_duration = time.time() - start_time
        OperationLogger.log_error("OmniVoice合成错误", str(e))
        OperationLogger.log_tts_request("OmniVoice", text, {"mode": mode}, total_duration, f"失败: {str(e)}")
        system_logger.error(f"【OmniVoice】合成错误: {e}")
        system_logger.error(f"【OmniVoice】{'='*60}")
        raise HTTPException(status_code=500, detail=str(e))
