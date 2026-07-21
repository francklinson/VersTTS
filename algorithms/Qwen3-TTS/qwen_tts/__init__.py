# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
qwen_tts: Qwen-TTS package.
"""

# 首先加载兼容性补丁（必须在导入 transformers 之前）
from .core.transformers_compat import *

# 延迟导入，避免在加载兼容性补丁时触发模型导入
# from .inference.qwen3_tts_model import Qwen3TTSModel, VoiceClonePromptItem
# from .inference.qwen3_tts_tokenizer import Qwen3TTSTokenizer

__all__ = ["__version__"]

# 提供延迟导入的辅助函数
def __getattr__(name):
    if name == "Qwen3TTSModel" or name == "VoiceClonePromptItem":
        from .inference.qwen3_tts_model import Qwen3TTSModel, VoiceClonePromptItem
        return locals()[name]
    if name == "Qwen3TTSTokenizer":
        from .inference.qwen3_tts_tokenizer import Qwen3TTSTokenizer
        return Qwen3TTSTokenizer
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")