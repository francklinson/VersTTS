"""
Transformers 版本兼容性补丁
为 Qwen3-TTS 提供在 transformers 4.51.3 下的兼容性支持
"""
import sys
import types
import warnings

# 检查 transformers 版本
try:
    import transformers
    TRANSFORMERS_VERSION = transformers.__version__
    VERSION_PARTS = [int(x) for x in TRANSFORMERS_VERSION.split('.')[:3]]
except:
    TRANSFORMERS_VERSION = "unknown"
    VERSION_PARTS = [0, 0, 0]

def is_version_at_least(major, minor, patch=0):
    """检查 transformers 版本是否 >= 指定版本"""
    current = VERSION_PARTS + [0] * (3 - len(VERSION_PARTS))
    target = [major, minor, patch]
    for c, t in zip(current, target):
        if c > t:
            return True
        if c < t:
            return False
    return True

# 为 4.51.3-4.57.x 创建所有缺失的模块或修复问题
if not is_version_at_least(4, 57, 0):
    
    # 1. layer_type_validation
    def layer_type_validation(layer_types, num_hidden_layers=None):
        """占位函数，4.51.3 中不需要此验证"""
        pass
    transformers.configuration_utils.layer_type_validation = layer_type_validation
    
    # 2. auto_docstring - 需要处理带参数和不带参数的情况
    def auto_docstring(*args, **kwargs):
        """占位装饰器，支持 @auto_docstring 和 @auto_docstring(custom_intro=...)"""
        if args and callable(args[0]):
            return args[0]
        # If used as decorator factory with kwargs, return a decorator that ignores the kwargs
        def decorator(func):
            return func
        return decorator
    transformers.utils.auto_docstring = auto_docstring
    
    # 3. check_model_inputs - 兼容 @check_model_inputs() 语法
    def check_model_inputs_wrapper(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda func: func
    
    try:
        from transformers.utils.generic import check_model_inputs as _orig_check
        import inspect
        if len(inspect.signature(_orig_check).parameters) == 1:
            check_model_inputs = check_model_inputs_wrapper
        else:
            check_model_inputs = _orig_check
    except ImportError:
        check_model_inputs = check_model_inputs_wrapper
    
    transformers.utils.generic.check_model_inputs = check_model_inputs

# 修复 4.57.x 版本中 check_model_inputs 的问题
def check_model_inputs_fixed(*args, **kwargs):
    """
    修复 check_model_inputs 装饰器的问题。
    原装饰器会返回 wrapped_fn，它会错误地处理 inputs_embeds 参数。
    这里我们返回一个直接传递所有参数的包装器。
    """
    def decorator(func):
        import functools
        @functools.wraps(func)
        def wrapper(*func_args, **func_kwargs):
            # 直接调用原函数，不做任何参数修改
            return func(*func_args, **func_kwargs)
        return wrapper
    
    # 支持 @check_model_inputs 和 @check_model_inputs() 两种用法
    if args and callable(args[0]):
        return decorator(args[0])
    return decorator

# 替换 transformers 中的 check_model_inputs
try:
    transformers.utils.generic.check_model_inputs = check_model_inputs_fixed
    print("[Transformers Compat] Fixed check_model_inputs for version 4.57.x")
except Exception as e:
    print(f"[Transformers Compat] Warning: Failed to fix check_model_inputs: {e}")

# 4. masking_utils - 独立的模块，不依赖于前面的 try-except
try:
    import transformers.masking_utils
except ImportError:
    def create_causal_mask(seq_len, device):
        import torch
        return torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    
    def create_sliding_window_causal_mask(seq_len, window_size, device):
        import torch
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        if window_size > 0:
            mask = mask | torch.tril(torch.ones(seq_len, seq_len, device=device), diagonal=-window_size)
        return mask.bool()
    
    masking_utils = types.ModuleType('transformers.masking_utils')
    masking_utils.create_causal_mask = create_causal_mask
    masking_utils.create_sliding_window_causal_mask = create_sliding_window_causal_mask
    sys.modules['transformers.masking_utils'] = masking_utils
    transformers.masking_utils = masking_utils
    print("[Transformers Compat] Created masking_utils module")

# 5. FlashAttentionKwargs 和相关函数
try:
    from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
except ImportError:
    class FlashAttentionKwargs:
        pass
    
    def flash_attn_supports_top_left_mask():
        return False
    
    def is_flash_attn_available():
        return False
    
    modeling_flash_attention_utils = types.ModuleType('transformers.modeling_flash_attention_utils')
    modeling_flash_attention_utils.FlashAttentionKwargs = FlashAttentionKwargs
    modeling_flash_attention_utils.flash_attn_supports_top_left_mask = flash_attn_supports_top_left_mask
    modeling_flash_attention_utils.is_flash_attn_available = is_flash_attn_available
    sys.modules['transformers.modeling_flash_attention_utils'] = modeling_flash_attention_utils
    transformers.modeling_flash_attention_utils = modeling_flash_attention_utils

# 6. GradientCheckpointingLayer
try:
    from transformers.modeling_layers import GradientCheckpointingLayer
except ImportError:
    from torch.nn import Module as GradientCheckpointingLayer
    modeling_layers = types.ModuleType('transformers.modeling_layers')
    modeling_layers.GradientCheckpointingLayer = GradientCheckpointingLayer
    sys.modules['transformers.modeling_layers'] = modeling_layers
    transformers.modeling_layers = modeling_layers

# 7. modeling_rope_utils
try:
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
except ImportError:
    ROPE_INIT_FUNCTIONS = {}
    def dynamic_rope_update(*args, **kwargs):
        pass
    modeling_rope_utils = types.ModuleType('transformers.modeling_rope_utils')
    modeling_rope_utils.ROPE_INIT_FUNCTIONS = ROPE_INIT_FUNCTIONS
    modeling_rope_utils.dynamic_rope_update = dynamic_rope_update
    sys.modules['transformers.modeling_rope_utils'] = modeling_rope_utils
    transformers.modeling_rope_utils = modeling_rope_utils

# 8. deprecate_kwarg
try:
    from transformers.utils.deprecation import deprecate_kwarg
except ImportError:
    def deprecate_kwarg(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    deprecation = types.ModuleType('transformers.utils.deprecation')
    deprecation.deprecate_kwarg = deprecate_kwarg
    sys.modules['transformers.utils.deprecation'] = deprecation
    transformers.utils.deprecation = deprecation

# 9. Unpack - 使用 typing_extensions 或创建兼容版本
try:
    from transformers.processing_utils import Unpack
except ImportError:
    try:
        from typing_extensions import Unpack
    except ImportError:
        # 创建一个支持泛型的 Unpack 类
        class _UnpackMeta(type):
            def __getitem__(self, item):
                return item
        
        class Unpack(metaclass=_UnpackMeta):
            pass
    
    processing_utils = types.ModuleType('transformers.processing_utils')
    processing_utils.Unpack = Unpack
    sys.modules['transformers.processing_utils'] = processing_utils
    transformers.processing_utils = processing_utils

# 10. use_kernel_forward_from_hub
try:
    from transformers.integrations import use_kernel_forward_from_hub
except ImportError:
    def use_kernel_forward_from_hub(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    integrations = types.ModuleType('transformers.integrations')
    integrations.use_kernel_forward_from_hub = use_kernel_forward_from_hub
    sys.modules['transformers.integrations'] = integrations
    transformers.integrations = integrations

print(f"[Transformers Compat] Loaded for transformers {TRANSFORMERS_VERSION}")
