#!/usr/bin/env python3
"""
任务处理器模块
为任务队列注册各种TTS模型的处理函数
"""

import os
import sys
import asyncio
import time
from typing import Dict, Any, List, Optional
from datetime import datetime

from backend.task_queue import task_queue, TaskRecord, TaskStatus
from backend.logger_config import system_logger
from backend.config import OUTPUTS_DIR
from backend.core.audio_utils import save_temp_audio, verify_and_cleanup
from backend.services import get_speaker_by_id


def _pack_batch_results(audio_files: List[str], audio_urls: List[str], prefix: str) -> Dict[str, Any]:
    """将批量生成的音频打包成 ZIP 并返回结果"""
    import zipfile
    timestamp = datetime.now().strftime("%H%M%S")
    zip_name = f"{prefix}_batch_{timestamp}.zip"
    zip_path = os.path.join(OUTPUTS_DIR, zip_name)
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for filename in audio_files:
            file_path = os.path.join(OUTPUTS_DIR, filename)
            if os.path.exists(file_path):
                zf.write(file_path, filename)
    batch_results = [{"url": url, "filename": f} for url, f in zip(audio_urls, audio_files)]
    return {
        'audio_file': zip_path,
        'audio_url': f"/audio/{zip_name}",
        'batch_results': batch_results
    }


def _extract_instruct(params: Dict[str, Any]) -> Optional[str]:
    """从任务参数中提取指令文本，供文件命名使用。

    字段名因模型/模式而异，按优先级取首个非空值：
    instruct_text → control_prompt → voice_design_prompt
    （与 backend.routers.task_queue._extract_instruct_prompt 保持一致）
    """
    if not params:
        return None
    for key in ('instruct_text', 'control_prompt', 'voice_design_prompt'):
        val = params.get(key)
        if val and str(val).strip():
            return str(val).strip()
    return None


def _generate_meaningful_filename(model: str, mode: str, text: str, index: int = 0, batch_total: int = 1,
                                   speaker_name: str = None, prefix: str = None,
                                   instruct_prompt: str = None) -> str:
    """
    生成有意义的音频文件名（任务队列路径）。

    薄封装：转发到 backend.core.audio_utils.build_meaningful_filename，
    与即时生成路径（save_temp_audio）共用同一命名规则，避免两套逻辑漂移。

    格式: {prefix}_{model}_{mode}_{指令摘要}_{speaker}_{text摘要}_{HHMMSS}{_NNofMM}.wav
    """
    from backend.core.audio_utils import build_meaningful_filename
    return build_meaningful_filename(
        model=model, mode=mode, text=text, index=index, batch_total=batch_total,
        speaker_name=speaker_name, prefix=prefix, instruct_prompt=instruct_prompt
    )


async def handle_voxcpm_task(task: TaskRecord) -> Dict[str, Any]:
    """处理VoxCPM任务"""
    try:
        from backend.engines import get_voxcpm_model
        from backend.task_queue import task_queue, progress_updater

        params = task.params or {}
        batch_count = params.get('batch_count', 1)

        if batch_count > 1:
            import asyncio
            from backend.routers.batch import _batch_generate_voxcpm
            loop = asyncio.get_running_loop()

            # 设置批量任务总数
            task.batch_total = batch_count
            task.batch_completed = 0
            task_queue._save_task(task)

            # 定义进度回调函数
            def _progress_callback(completed: int, total: int):
                progress_updater.update_progress(
                    task_queue.update_batch_progress,
                    task.task_id, completed, total
                )

            # 取消检查：超时/取消后 _execute_task 会把 task.status 改为
            # FAILED/CANCELLED，批量同步线程据此主动跳出以释放 GPU
            _terminal = {TaskStatus.FAILED.value, TaskStatus.CANCELLED.value}

            def _cancel_check():
                return task.status in _terminal

            result = await loop.run_in_executor(
                None, _batch_generate_voxcpm,
                task.text, task.mode, batch_count,
                params.get('speaker_id'),
                params.get('voice_design_prompt'),
                params.get('control_prompt'),
                _progress_callback,
                _cancel_check
            )
            audio_files = result.get('audio_files', [])
            audio_urls = result.get('audio_urls', [])
            if not audio_files:
                raise Exception("批量生成未产生任何音频")
            return _pack_batch_results(audio_files, audio_urls, "voxcpm")

        def _sync_generate():
            _speaker_id = params.get('speaker_id')
            _voice_design_prompt = params.get('voice_design_prompt')
            _control_prompt = params.get('control_prompt')

            _ref_path = None
            _speaker_ref_text = None
            _speaker_name = None
            if _speaker_id:
                _speaker = get_speaker_by_id(_speaker_id)
                if _speaker:
                    _ref_path = _speaker.get("audio_path")
                    _speaker_ref_text = _speaker.get("reference_text")
                    _speaker_name = _speaker.get("name")

            _model = get_voxcpm_model()

            _generate_kwargs = {
                "cfg_value": 2.0,
                "inference_timesteps": 10
            }

            if task.mode == "voice_design" and _voice_design_prompt:
                _generate_kwargs["text"] = f"({_voice_design_prompt}){task.text}"
            elif task.mode in ["clone", "ultimate_clone"] and _ref_path:
                _generate_kwargs["text"] = task.text
                _generate_kwargs["reference_wav_path"] = _ref_path
                if _speaker_ref_text:
                    _generate_kwargs["reference_text"] = _speaker_ref_text
                if _control_prompt:
                    _generate_kwargs["text"] = f"({_control_prompt}){task.text}"
            else:
                _generate_kwargs["text"] = task.text

            _audio = _model.generate(**_generate_kwargs)
            _sample_rate = _model.tts_model.sample_rate

            # 使用有意义的文件名，传入说话人名称与指令文本
            _filename = _generate_meaningful_filename("voxcpm", task.mode, task.text,
                                                       speaker_name=_speaker_name,
                                                       instruct_prompt=_extract_instruct(params))
            _audio_path = os.path.join(OUTPUTS_DIR, _filename)

            import soundfile as sf
            sf.write(_audio_path, _audio, _sample_rate)

            # 内容校验（需求2）：不达标则删文件并抛 AudioVerifyError，
            # 由任务队列 _execute_task 捕获标记为可重试失败。
            verify_and_cleanup(_audio_path, task.text, model_tag="VoxCPM")

            return {
                'audio_file': _audio_path,
                'audio_url': f"/audio/{_filename}"
            }

        import asyncio
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _sync_generate)

    except Exception as e:
        system_logger.error(f"【任务处理器】VoxCPM任务失败: {e}")
        raise


async def handle_qwen3tts_task(task: TaskRecord) -> Dict[str, Any]:
    """处理Qwen3-TTS任务"""
    try:
        from backend.engines import get_qwen3tts_model
        from backend.task_queue import task_queue, progress_updater

        params = task.params or {}
        batch_count = params.get('batch_count', 1)

        if batch_count > 1:
            import asyncio
            from backend.routers.batch import _batch_generate_qwen3tts
            loop = asyncio.get_running_loop()

            # 设置批量任务总数
            task.batch_total = batch_count
            task.batch_completed = 0
            task_queue._save_task(task)

            # 定义进度回调函数
            def _progress_callback(completed: int, total: int):
                progress_updater.update_progress(
                    task_queue.update_batch_progress,
                    task.task_id, completed, total
                )

            # 取消检查：超时/取消后 _execute_task 会把 task.status 改为
            # FAILED/CANCELLED，批量同步线程据此主动跳出以释放 GPU
            _terminal = {TaskStatus.FAILED.value, TaskStatus.CANCELLED.value}

            def _cancel_check():
                return task.status in _terminal

            result = await loop.run_in_executor(
                None, _batch_generate_qwen3tts,
                task.text, task.mode, batch_count,
                params.get('speaker_id'),
                params.get('voice_design_prompt'),
                _progress_callback,
                _cancel_check
            )
            audio_files = result.get('audio_files', [])
            audio_urls = result.get('audio_urls', [])
            if not audio_files:
                raise Exception("批量生成未产生任何音频")
            return _pack_batch_results(audio_files, audio_urls, "qwen3tts")

        def _sync_generate():
            _text = task.text
            _mode = task.mode

            _model_type_map = {
                "voice_clone": "VoiceClone",
                "custom_voice": "CustomVoice",
                "voice_design": "VoiceDesign",
                "base": "Base"
            }
            _model_type = _model_type_map.get(_mode, "Base")
            _tts = get_qwen3tts_model("1.7B", _model_type)

            _wav = None
            _sr = 24000

            # 获取说话人名称（用于文件名）
            _speaker_name = None

            if _mode == "voice_clone":
                _speaker_id = params.get('speaker_id')
                if not _speaker_id:
                    raise ValueError("voice_clone 模式需要 speaker_id")

                _speaker_info = get_speaker_by_id(_speaker_id)
                if not _speaker_info:
                    raise ValueError(f"说话人不存在: {_speaker_id}")

                _ref_path = _speaker_info.get("audio_path")
                _ref_text = _speaker_info.get("reference_text", "")
                _speaker_name = _speaker_info.get("name")
                if not _ref_path or not os.path.exists(_ref_path):
                    raise ValueError(f"参考音频不存在: {_ref_path}")

                _x_vector_only_mode = False
                if not _ref_text:
                    _x_vector_only_mode = True

                _wavs, _sr = _tts.generate_voice_clone(
                    text=_text,
                    language="Auto",
                    ref_audio=_ref_path,
                    ref_text=_ref_text,
                    x_vector_only_mode=_x_vector_only_mode
                )
                _wav = _wavs[0] if isinstance(_wavs, list) else _wavs

            elif _mode == "custom_voice":
                _speaker = params.get('speaker') or params.get('speaker_id') or 'vivian'
                _speaker_name = _speaker
                _instruct_text = params.get('instruct_text', '')

                _custom_voice_success = False
                try:
                    if hasattr(_tts, 'generate_custom_voice'):
                        _wavs, _sr = _tts.generate_custom_voice(
                            text=_text,
                            language="Chinese",
                            speaker=_speaker,
                            instruct=_instruct_text or "",
                            do_sample=True,
                            temperature=0.9,
                            top_k=50,
                            top_p=1.0
                        )
                        _wav = _wavs[0] if isinstance(_wavs, list) else _wavs
                        _custom_voice_success = True
                except (ValueError, NotImplementedError) as e:
                    if "does not support generate_custom_voice" in str(e) or "not implemented" in str(e).lower():
                        system_logger.warning(f"【任务处理器】CustomVoice不支持，回退到Base: {e}")
                    else:
                        raise

                if not _custom_voice_success:
                    _tts_base = get_qwen3tts_model("1.7B", "Base")
                    _default_ref_audio = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone_1.wav"
                    _default_ref_text = "甚至出现交易几乎停滞的情况。"
                    _wavs, _sr = _tts_base.generate_voice_clone(
                        text=_text,
                        language="Auto",
                        ref_audio=_default_ref_audio,
                        ref_text=_default_ref_text,
                        x_vector_only_mode=True
                    )
                    _wav = _wavs[0] if isinstance(_wavs, list) else _wavs

            elif _mode == "voice_design":
                _voice_design_prompt = params.get('voice_design_prompt')
                if not _voice_design_prompt:
                    raise ValueError("voice_design 模式需要 voice_design_prompt")

                _voice_design_success = False
                try:
                    if hasattr(_tts, 'generate_voice_design'):
                        _wavs, _sr = _tts.generate_voice_design(
                            text=_text,
                            language="Auto",
                            instruct=_voice_design_prompt
                        )
                        _wav = _wavs[0] if isinstance(_wavs, list) else _wavs
                        _voice_design_success = True
                except ValueError as e:
                    if "does not support generate_voice_design" in str(e):
                        system_logger.warning(f"【任务处理器】VoiceDesign不支持，回退到Base: {e}")
                    else:
                        raise

                if not _voice_design_success:
                    _tts_base = get_qwen3tts_model("1.7B", "Base")
                    _default_ref_audio = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone_1.wav"
                    _default_ref_text = "甚至出现交易几乎停滞的情况。"
                    _wavs, _sr = _tts_base.generate_voice_clone(
                        text=_text,
                        language="Auto",
                        ref_audio=_default_ref_audio,
                        ref_text=_default_ref_text,
                        x_vector_only_mode=True
                    )
                    _wav = _wavs[0] if isinstance(_wavs, list) else _wavs
            else:
                _tts_base = get_qwen3tts_model("1.7B", "Base")
                _default_ref_audio = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone_1.wav"
                _default_ref_text = "甚至出现交易几乎停滞的情况。"
                _wavs, _sr = _tts_base.generate_voice_clone(
                    text=_text,
                    language="Auto",
                    ref_audio=_default_ref_audio,
                    ref_text=_default_ref_text,
                    x_vector_only_mode=True
                )
                _wav = _wavs[0] if isinstance(_wavs, list) else _wavs

            # 使用有意义的文件名，传入说话人名称与指令文本
            _filename = _generate_meaningful_filename("qwen3tts", task.mode, task.text,
                                                       speaker_name=_speaker_name,
                                                       instruct_prompt=_extract_instruct(params))
            _audio_path = os.path.join(OUTPUTS_DIR, _filename)

            import soundfile as sf
            sf.write(_audio_path, _wav, _sr)

            # 内容校验（需求2）：不达标则删文件并抛 AudioVerifyError，
            # 由任务队列 _execute_task 捕获标记为可重试失败。
            verify_and_cleanup(_audio_path, task.text, model_tag="Qwen3-TTS")

            return {
                'audio_file': _audio_path,
                'audio_url': f"/audio/{_filename}"
            }

        import asyncio
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _sync_generate)

    except Exception as e:
        system_logger.error(f"【任务处理器】Qwen3-TTS任务失败: {e}")
        raise


async def _check_subservice_health(host: str, port: int, name: str, start_cmd: str) -> None:
    """检查子服务是否在线，未在线时抛出友好提示"""
    import aiohttp
    health_url = f"http://{host}:{port}/health"
    try:
        timeout = aiohttp.ClientTimeout(total=3)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(health_url) as resp:
                if resp.status != 200:
                    raise ConnectionError(
                        f"{name} 服务未就绪 ({host}:{port})。\n"
                        f"请手动启动：{start_cmd}"
                    )
    except aiohttp.ClientConnectorError:
        raise ConnectionError(
            f"{name} 服务未启动或无法连接 ({host}:{port})。\n"
            f"请手动启动：{start_cmd}"
        )


async def handle_omnivoice_task(task: TaskRecord) -> Dict[str, Any]:
    """处理OmniVoice任务"""
    try:
        import aiohttp

        from backend.config import OMNIVOICE_HOST, OMNIVOICE_PORT
        await _check_subservice_health(
            OMNIVOICE_HOST, OMNIVOICE_PORT,
            "OmniVoice", "./start_server.sh start-omnivoice"
        )

        params = task.params or {}
        batch_count = params.get('batch_count', 1)

        if batch_count > 1:
            # 设置批量任务总数
            task.batch_total = batch_count
            task.batch_completed = 0
            task_queue._save_task(task)

            from backend.routers.batch import _batch_generate_omnivoice

            # 定义进度回调函数
            def _progress_callback(completed: int, total: int):
                task_queue.update_batch_progress(task.task_id, completed, total)

            result = await _batch_generate_omnivoice(
                text=task.text,
                mode=task.mode,
                count=batch_count,
                speaker_id=params.get('speaker_id'),
                voice_design_prompt=params.get('voice_design_prompt') or params.get('control_prompt'),
                speed=params.get('speed', 1.0),
                progress_callback=_progress_callback
            )
            audio_files = result.get('audio_files', [])
            audio_urls = result.get('audio_urls', [])
            if not audio_files:
                raise Exception("批量生成未产生任何音频")
            return _pack_batch_results(audio_files, audio_urls, "omnivoice")

        # 获取参数
        speaker_id = params.get('speaker_id')
        voice_design = params.get('voice_design_prompt')
        speed = params.get('speed', 1.0)

        # 构建表单数据
        form_data = aiohttp.FormData()
        form_data.add_field('text', task.text)
        form_data.add_field('mode', task.mode)
        form_data.add_field('speed', str(speed))

        # 获取说话人参考音频
        if speaker_id:
            speaker = get_speaker_by_id(speaker_id)
            if speaker:
                audio_path = speaker.get("audio_path")
                ref_text = speaker.get("reference_text")
                if audio_path:
                    form_data.add_field('ref_audio', audio_path)
                if ref_text:
                    form_data.add_field('ref_text', ref_text)

        if voice_design:
            form_data.add_field('voice_design_prompt', voice_design)

        # 调用OmniVoice服务
        url = f"http://{OMNIVOICE_HOST}:{OMNIVOICE_PORT}/tts"

        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=form_data) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"OmniVoice服务错误: {error_text}")

                result = await response.json()

                if not result.get("success"):
                    raise Exception(result.get("message", "OmniVoice生成失败"))

                audio_path = result.get('audio_path')
                if not audio_path or not os.path.exists(audio_path):
                    raise Exception(f"OmniVoice返回的音频文件不存在: {audio_path}")

                # 使用有意义的文件名并复制到 outputs 目录
                filename = _generate_meaningful_filename("omnivoice", task.mode, task.text,
                                                         instruct_prompt=_extract_instruct(params))
                dest_path = os.path.join(OUTPUTS_DIR, filename)
                import shutil
                shutil.copy2(audio_path, dest_path)

                # 内容校验（需求2）：不达标则删 outputs 文件并抛 AudioVerifyError。
                verify_and_cleanup(dest_path, task.text, model_tag="OmniVoice")

                return {
                    'audio_file': dest_path,
                    'audio_url': f"/audio/{filename}"
                }

    except Exception as e:
        system_logger.error(f"【任务处理器】OmniVoice任务失败: {e}")
        raise


async def handle_cosyvoice_task(task: TaskRecord) -> Dict[str, Any]:
    """处理CosyVoice任务"""
    try:
        import aiohttp
        from backend.services import get_speaker_by_id

        from backend.config import COSYVOICE_HOST, COSYVOICE_PORT
        await _check_subservice_health(
            COSYVOICE_HOST, COSYVOICE_PORT,
            "CosyVoice", "./start_server.sh start-cosyvoice"
        )

        # 检查文本长度，CosyVoice 需要至少3个字符
        if len(task.text.strip()) < 3:
            raise ValueError(f"CosyVoice 需要至少3个字符的文本，当前只有 {len(task.text.strip())} 个字符")

        params = task.params or {}
        batch_count = params.get('batch_count', 1)

        if batch_count > 1:
            # 设置批量任务总数
            task.batch_total = batch_count
            task.batch_completed = 0
            task_queue._save_task(task)

            from backend.routers.batch import _batch_generate_cosyvoice

            # 定义进度回调函数
            def _progress_callback(completed: int, total: int):
                task_queue.update_batch_progress(task.task_id, completed, total)

            result = await _batch_generate_cosyvoice(
                text=task.text,
                mode=task.mode,
                count=batch_count,
                speaker_id=params.get('speaker_id'),
                control_prompt=params.get('control_prompt'),
                progress_callback=_progress_callback
            )
            audio_files = result.get('audio_files', [])
            audio_urls = result.get('audio_urls', [])
            if not audio_files:
                raise Exception("批量生成未产生任何音频")
            return _pack_batch_results(audio_files, audio_urls, "cosyvoice")

        speaker_id = params.get('speaker_id')
        instruct_text = params.get('instruct_text') or params.get('control_prompt')

        # 构建表单数据
        form_data = aiohttp.FormData()
        form_data.add_field('text', task.text)
        form_data.add_field('mode', task.mode)

        # 获取说话人参考音频
        if speaker_id:
            speaker = get_speaker_by_id(speaker_id)
            if speaker:
                audio_path = speaker.get("audio_path")
                ref_text = speaker.get("reference_text")
                if audio_path:
                    form_data.add_field('prompt_wav_path', audio_path)
                if ref_text:
                    form_data.add_field('prompt_text', ref_text)

        # instruct 模式需要指令文本
        if task.mode == 'instruct' and instruct_text is not None:
            form_data.add_field('instruct_text', instruct_text)

        # 调用CosyVoice服务
        url = f"http://{COSYVOICE_HOST}:{COSYVOICE_PORT}/tts"

        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=form_data) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"CosyVoice服务错误: {error_text}")

                result = await response.json()

                if not result.get("success"):
                    raise Exception(result.get("message", "CosyVoice生成失败"))

                audio_path = result.get('audio_path')
                if not audio_path or not os.path.exists(audio_path):
                    raise Exception(f"CosyVoice返回的音频文件不存在: {audio_path}")

                # 使用有意义的文件名并复制到 outputs 目录
                filename = _generate_meaningful_filename("cosyvoice", task.mode, task.text,
                                                         instruct_prompt=_extract_instruct(params))
                dest_path = os.path.join(OUTPUTS_DIR, filename)
                import shutil
                shutil.copy2(audio_path, dest_path)

                # 内容校验（需求2）：不达标则删 outputs 文件并抛 AudioVerifyError。
                verify_and_cleanup(dest_path, task.text, model_tag="CosyVoice")

                return {
                    'audio_file': dest_path,
                    'audio_url': f"/audio/{filename}"
                }

    except Exception as e:
        system_logger.error(f"【任务处理器】CosyVoice任务失败: {e}")
        raise


async def handle_pilottts_task(task: TaskRecord) -> Dict[str, Any]:
    """处理PilotTTS任务 — 通过独立服务调用"""
    try:
        import aiohttp
        from backend.services import get_speaker_by_id
        from backend.config import PILOTTS_HOST, PILOTTS_PORT

        await _check_subservice_health(
            PILOTTS_HOST, PILOTTS_PORT,
            "PilotTTS", "./start_server.sh start-pilottts"
        )

        params = task.params or {}
        batch_count = params.get('batch_count', 1)

        if batch_count > 1:
            task.batch_total = batch_count
            task.batch_completed = 0
            task_queue._save_task(task)

            from backend.routers.batch import _batch_generate_pilottts

            def _progress_callback(completed: int, total: int):
                task_queue.update_batch_progress(task.task_id, completed, total)

            result = await _batch_generate_pilottts(
                text=task.text,
                mode=task.mode,
                count=batch_count,
                speaker_id=params.get('speaker_id'),
                emotion=params.get('emotion'),
                language=params.get('language', 'zh'),
                progress_callback=_progress_callback
            )
            audio_files = result.get('audio_files', [])
            audio_urls = result.get('audio_urls', [])
            if not audio_files:
                raise Exception("批量生成未产生任何音频")
            return _pack_batch_results(audio_files, audio_urls, "pilottts")

        # 获取说话人
        speaker_id = params.get('speaker_id')
        speaker_name = None
        ref_path = None
        if speaker_id:
            speaker = get_speaker_by_id(speaker_id)
            if speaker:
                ref_path = speaker.get("audio_path")
                speaker_name = speaker.get("name")
            if not ref_path or not os.path.exists(ref_path):
                raise ValueError(f"参考音频不存在: {ref_path}")

        # 构建请求数据
        form_data = aiohttp.FormData()
        form_data.add_field('text', task.text)
        form_data.add_field('mode', task.mode)
        form_data.add_field('ref_path', ref_path)
        form_data.add_field('language', params.get('language', 'zh'))

        emotion = params.get('emotion')
        if task.mode == "emotion" and emotion:
            form_data.add_field('emotion', emotion)

        # 调用 PilotTTS 独立服务
        url = f"http://{PILOTTS_HOST}:{PILOTTS_PORT}/tts"

        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=form_data, timeout=aiohttp.ClientTimeout(total=300)) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"PilotTTS服务错误: {error_text}")

                result = await response.json()

                if not result.get("success"):
                    raise Exception(result.get("message", "PilotTTS生成失败"))

                audio_path = result.get('audio_path')
                if not audio_path or not os.path.exists(audio_path):
                    raise Exception(f"PilotTTS返回的音频文件不存在: {audio_path}")

                # 使用有意义的文件名并复制到 outputs 目录（emotion 视为情感指令）
                filename = _generate_meaningful_filename("pilottts", task.mode, task.text,
                                                         speaker_name=speaker_name,
                                                         instruct_prompt=emotion)
                dest_path = os.path.join(OUTPUTS_DIR, filename)
                import shutil
                shutil.copy2(audio_path, dest_path)

                # 内容校验（需求2）：不达标则删 outputs 文件并抛 AudioVerifyError。
                verify_and_cleanup(dest_path, task.text, model_tag="PilotTTS")

                return {
                    'audio_file': dest_path,
                    'audio_url': f"/audio/{filename}"
                }

    except Exception as e:
        system_logger.error(f"【任务处理器】PilotTTS任务失败: {e}")
        raise


async def handle_gptsovits_task(task: TaskRecord) -> Dict[str, Any]:
    """处理GPT-SoVITS任务 — 通过独立服务调用"""
    try:
        import aiohttp
        from backend.services import get_speaker_by_id
        from backend.config import GPTSOVITS_HOST, GPTSOVITS_PORT

        await _check_subservice_health(
            GPTSOVITS_HOST, GPTSOVITS_PORT,
            "GPT-SoVITS", "./start_server.sh start-gptsovits"
        )

        params = task.params or {}
        batch_count = params.get('batch_count', 1)
        version = params.get('version', 'v2')

        if batch_count > 1:
            task.batch_total = batch_count
            task.batch_completed = 0
            task_queue._save_task(task)

            from backend.routers.batch import _batch_generate_gptsovits

            def _progress_callback(completed: int, total: int):
                task_queue.update_batch_progress(task.task_id, completed, total)

            result = await _batch_generate_gptsovits(
                text=task.text,
                mode=task.mode,
                count=batch_count,
                speaker_id=params.get('speaker_id'),
                version=version,
                progress_callback=_progress_callback
            )
            audio_files = result.get('audio_files', [])
            audio_urls = result.get('audio_urls', [])
            if not audio_files:
                raise Exception("批量生成未产生任何音频")
            return _pack_batch_results(audio_files, audio_urls, "gptsovits")

        # 单次生成
        speaker_id = params.get('speaker_id')
        speaker_name = None
        ref_path = None
        prompt_text = None

        if speaker_id:
            speaker = get_speaker_by_id(speaker_id)
            if speaker:
                ref_path = speaker.get("audio_path")
                prompt_text = speaker.get("reference_text")
                speaker_name = speaker.get("name")
            if not ref_path or not os.path.exists(ref_path):
                raise ValueError(f"参考音频不存在: {ref_path}")
            if not prompt_text:
                raise ValueError("该说话人缺少参考文本，请在说话人管理中补充参考文本")

        # 构建请求数据
        form_data = aiohttp.FormData()
        form_data.add_field('text', task.text)
        form_data.add_field('text_lang', 'zh')
        form_data.add_field('prompt_text', prompt_text)
        form_data.add_field('prompt_lang', 'zh')
        form_data.add_field('ref_audio_path', ref_path)
        form_data.add_field('version', version)
        form_data.add_field('output_format', 'url')

        # 调用 GPT-SoVITS 独立服务
        url = f"http://{GPTSOVITS_HOST}:{GPTSOVITS_PORT}/tts"

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120)) as session:
            async with session.post(url, data=form_data) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"GPT-SoVITS服务错误: {error_text}")

                result = await response.json()

                if not result.get("success"):
                    raise Exception(result.get("message", "GPT-SoVITS生成失败"))

                audio_path = result.get('audio_path')
                if not audio_path or not os.path.exists(audio_path):
                    raise Exception(f"GPT-SoVITS返回的音频文件不存在: {audio_path}")

                # 读取音频保存到 outputs 目录
                import soundfile as sf
                from backend.core import save_temp_audio
                audio_data, sample_rate = sf.read(audio_path)
                filename = _generate_meaningful_filename("gptsovits", task.mode, task.text,
                                                         speaker_name=speaker_name)
                dest_path = os.path.join(OUTPUTS_DIR, filename)
                sf.write(dest_path, audio_data, sample_rate)

                # 清理子服务临时文件
                try:
                    os.remove(audio_path)
                except Exception:
                    pass

                # 内容校验（需求2）：不达标则删 outputs 文件并抛 AudioVerifyError，
                # 由任务队列 _execute_task 捕获标记为可重试失败。
                verify_and_cleanup(dest_path, task.text, model_tag="GPT-SoVITS")

                return {
                    'audio_file': dest_path,
                    'audio_url': f"/audio/{filename}"
                }

    except Exception as e:
        system_logger.error(f"【任务处理器】GPT-SoVITS任务失败: {e}")
        raise


async def handle_dotstts_task(task: TaskRecord) -> Dict[str, Any]:
    """处理dots.tts任务"""
    try:
        from backend.engines import get_dotstts_model
        from backend.task_queue import task_queue, progress_updater

        params = task.params or {}
        batch_count = params.get('batch_count', 1)

        if batch_count > 1:
            import asyncio
            from backend.routers.batch import _batch_generate_dotstts
            loop = asyncio.get_running_loop()

            # 设置批量任务总数
            task.batch_total = batch_count
            task.batch_completed = 0
            task_queue._save_task(task)

            # 定义进度回调函数
            def _progress_callback(completed: int, total: int):
                progress_updater.update_progress(
                    task_queue.update_batch_progress,
                    task.task_id, completed, total
                )

            # 取消检查
            _terminal = {TaskStatus.FAILED.value, TaskStatus.CANCELLED.value}

            def _cancel_check():
                return task.status in _terminal

            result = await loop.run_in_executor(
                None, _batch_generate_dotstts,
                task.text, task.mode, batch_count,
                params.get('speaker_id'),
                params.get('num_steps', 16),
                params.get('guidance_scale', 1.2),
                params.get('speaker_scale', 1.5),
                params.get('language'),
                _progress_callback,
                _cancel_check
            )
            audio_files = result.get('audio_files', [])
            audio_urls = result.get('audio_urls', [])
            if not audio_files:
                raise Exception("批量生成未产生任何音频")
            return _pack_batch_results(audio_files, audio_urls, "dotstts")

        def _sync_generate():
            import numpy as np
            import torch

            _speaker_id = params.get('speaker_id')
            _num_steps = params.get('num_steps', 16)
            _guidance_scale = params.get('guidance_scale', 1.2)
            _speaker_scale = params.get('speaker_scale', 1.5)
            _language = params.get('language')

            _ref_path = None
            _prompt_text = None
            _speaker_name = None
            if _speaker_id:
                _speaker = get_speaker_by_id(_speaker_id)
                if _speaker:
                    _ref_path = _speaker.get("audio_path")
                    _prompt_text = _speaker.get("reference_text")
                    _speaker_name = _speaker.get("name")

            _model = get_dotstts_model()

            # 模板映射
            _template_map = {
                "voice_clone": "tts",
                "instruct": "instruction_tts",
            }
            _template_name = _template_map.get(task.mode, "tts")

            _generate_kwargs = {
                "text": task.text,
                "num_steps": _num_steps,
                "guidance_scale": _guidance_scale,
                "speaker_scale": _speaker_scale,
                "template_name": _template_name,
            }

            if _language:
                _generate_kwargs["language"] = _language

            if task.mode == "voice_clone" and _ref_path:
                _generate_kwargs["prompt_audio_path"] = _ref_path
                if _prompt_text:
                    _generate_kwargs["prompt_text"] = _prompt_text
            elif task.mode == "voice_clone" and not _ref_path:
                _generate_kwargs["template_name"] = "tts"

            _result = _model.generate(**_generate_kwargs)

            # 提取音频数据
            _audio = _result["audio"]
            _sample_rate = _result["sample_rate"]

            if torch.is_tensor(_audio):
                _audio_np = _audio.cpu().numpy().squeeze()
            elif isinstance(_audio, np.ndarray):
                _audio_np = _audio.squeeze()
            else:
                _audio_np = np.array(_audio).squeeze()

            # 保存音频
            _audio_path = save_temp_audio(
                _audio_np, _sample_rate, prefix="dotstts", mode=task.mode,
                text=task.text, speaker_name=_speaker_name
            )

            verify_and_cleanup(_audio_path, task.text, model_tag="dots.tts")

            return {
                'audio_file': _audio_path,
                'audio_url': f"/audio/{os.path.basename(_audio_path)}"
            }

        import asyncio
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _sync_generate)

    except Exception as e:
        system_logger.error(f"【任务处理器】dots.tts任务失败: {e}")
        raise


async def handle_fishspeech_task(task: TaskRecord) -> Dict[str, Any]:
    """处理Fish-Speech任务 — 通过独立服务调用"""
    try:
        import aiohttp
        from backend.config import FISHSPEECH_HOST, FISHSPEECH_PORT

        await _check_subservice_health(
            FISHSPEECH_HOST, FISHSPEECH_PORT,
            "Fish-Speech", "./start_server.sh start-fishspeech"
        )

        params = task.params or {}
        batch_count = params.get('batch_count', 1)

        if batch_count > 1:
            # 设置批量任务总数
            task.batch_total = batch_count
            task.batch_completed = 0
            task_queue._save_task(task)

            from backend.routers.batch import _batch_generate_fishspeech

            def _progress_callback(completed: int, total: int):
                task_queue.update_batch_progress(task.task_id, completed, total)

            result = await _batch_generate_fishspeech(
                text=task.text,
                mode=task.mode,
                count=batch_count,
                clone_speaker_id=params.get('speaker_id'),
                reference_text=params.get('reference_text'),
                temperature=params.get('temperature', 0.8),
                top_p=params.get('top_p', 0.8),
                repetition_penalty=params.get('repetition_penalty', 1.1),
                progress_callback=_progress_callback
            )
            audio_files = result.get('audio_files', [])
            audio_urls = result.get('audio_urls', [])
            if not audio_files:
                raise Exception("批量生成未产生任何音频")
            return _pack_batch_results(audio_files, audio_urls, "fishspeech")

        # 单次生成
        clone_speaker_id = params.get('speaker_id')
        speaker_name = None
        speaker_ref_text = None
        if clone_speaker_id:
            speaker = get_speaker_by_id(clone_speaker_id)
            if speaker:
                speaker_name = speaker.get("name")
                speaker_ref_text = speaker.get("reference_text")

        # 构建请求数据
        form_data = aiohttp.FormData()
        form_data.add_field('text', task.text)
        form_data.add_field('mode', task.mode)
        form_data.add_field('temperature', str(params.get('temperature', 0.8)))
        form_data.add_field('top_p', str(params.get('top_p', 0.8)))
        form_data.add_field('repetition_penalty', str(params.get('repetition_penalty', 1.1)))

        if clone_speaker_id:
            form_data.add_field('clone_speaker_id', clone_speaker_id)
        if speaker_ref_text:
            form_data.add_field('reference_text', speaker_ref_text)

        # 调用 Fish-Speech 独立服务
        url = f"http://{FISHSPEECH_HOST}:{FISHSPEECH_PORT}/tts"

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=180)) as session:
            async with session.post(url, data=form_data) as response:
                if response.status not in (200, 201):
                    error_text = await response.text()
                    raise Exception(f"Fish-Speech服务错误: {error_text}")

                result = await response.json()

                audio_path = result.get('audio_path')
                if not audio_path or not os.path.exists(audio_path):
                    raise Exception(f"Fish-Speech返回的音频文件不存在: {audio_path}")

                sample_rate = result.get('sample_rate', 44100)

                # 使用有意义的文件名并复制到 outputs 目录
                filename = _generate_meaningful_filename("fishspeech", task.mode, task.text,
                                                         speaker_name=speaker_name)
                dest_path = os.path.join(OUTPUTS_DIR, filename)
                import shutil
                shutil.copy2(audio_path, dest_path)

                # 内容校验
                verify_and_cleanup(dest_path, task.text, model_tag="Fish-Speech")

                return {
                    'audio_file': dest_path,
                    'audio_url': f"/audio/{filename}"
                }

    except Exception as e:
        system_logger.error(f"【任务处理器】Fish-Speech任务失败: {e}")
        raise


async def handle_indextts_task(task: TaskRecord) -> Dict[str, Any]:
    """处理IndexTTS任务"""
    try:
        from backend.engines import get_indextts_model
        from backend.task_queue import task_queue, progress_updater

        params = task.params or {}
        batch_count = params.get('batch_count', 1)

        if batch_count > 1:
            import asyncio
            from backend.routers.batch import _batch_generate_indextts
            loop = asyncio.get_running_loop()

            # 设置批量任务总数
            task.batch_total = batch_count
            task.batch_completed = 0
            task_queue._save_task(task)

            # 定义进度回调函数
            def _progress_callback(completed: int, total: int):
                progress_updater.update_progress(
                    task_queue.update_batch_progress,
                    task.task_id, completed, total
                )

            # 取消检查
            _terminal = {TaskStatus.FAILED.value, TaskStatus.CANCELLED.value}

            def _cancel_check():
                return task.status in _terminal

            result = await loop.run_in_executor(
                None, _batch_generate_indextts,
                task.text, task.mode, batch_count,
                params.get('speaker_id'),
                params.get('emotion_text'),
                _progress_callback,
                _cancel_check
            )
            audio_files = result.get('audio_files', [])
            audio_urls = result.get('audio_urls', [])
            if not audio_files:
                raise Exception("批量生成未产生任何音频")
            return _pack_batch_results(audio_files, audio_urls, "indextts")

        def _sync_generate():
            _speaker_id = params.get('speaker_id')
            _emotion_text = params.get('emotion_text')

            _ref_path = None
            _speaker_name = None
            if _speaker_id:
                _speaker = get_speaker_by_id(_speaker_id)
                if _speaker:
                    _ref_path = _speaker.get("audio_path")
                    _speaker_name = _speaker.get("name")
                if not _ref_path or not os.path.exists(_ref_path):
                    raise ValueError(f"参考音频不存在: {_ref_path}")

            if not _ref_path:
                raise ValueError("IndexTTS需要提供参考音频（speaker_id）")

            _model = get_indextts_model()

            # 生成音频路径
            _filename = _generate_meaningful_filename("indextts", task.mode, task.text,
                                                       speaker_name=_speaker_name,
                                                       instruct_prompt=_emotion_text)
            _audio_path = os.path.join(OUTPUTS_DIR, _filename)

            # 准备 infer 参数
            _infer_kwargs = {
                "spk_audio_prompt": _ref_path,
                "text": task.text,
                "output_path": _audio_path,
                "verbose": True
            }

            # 情感描述支持
            if _emotion_text and _emotion_text.strip():
                _infer_kwargs["use_emo_text"] = True
                _infer_kwargs["emo_text"] = _emotion_text.strip()
                _infer_kwargs["emo_alpha"] = 0.6

            _model.infer(**_infer_kwargs)

            verify_and_cleanup(_audio_path, task.text, model_tag="IndexTTS")

            return {
                'audio_file': _audio_path,
                'audio_url': f"/audio/{_filename}"
            }

        import asyncio
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _sync_generate)

    except Exception as e:
        system_logger.error(f"【任务处理器】IndexTTS任务失败: {e}")
        raise


def register_all_handlers():
    """注册所有任务处理器"""
    task_queue.register_handler("voxcpm", handle_voxcpm_task)
    task_queue.register_handler("qwen3tts", handle_qwen3tts_task)
    task_queue.register_handler("omnivoice", handle_omnivoice_task)
    task_queue.register_handler("cosyvoice", handle_cosyvoice_task)
    task_queue.register_handler("pilottts", handle_pilottts_task)
    task_queue.register_handler("gptsovits", handle_gptsovits_task)
    task_queue.register_handler("dotstts", handle_dotstts_task)
    task_queue.register_handler("fishspeech", handle_fishspeech_task)
    task_queue.register_handler("indextts", handle_indextts_task)

    system_logger.info("【任务处理器】所有处理器已注册")


# 启动时自动注册
register_all_handlers()
