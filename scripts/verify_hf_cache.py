#!/usr/bin/env python3
"""
验证 HuggingFace 缓存配置脚本
检查项目是否正确使用 models/hf_cache 目录
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

print("=" * 60)
print("VersTTS HuggingFace 缓存配置验证")
print("=" * 60)
print()

# 检查项目内缓存目录
hf_cache_path = PROJECT_ROOT / "models" / "hf_cache"
print(f"1. 项目内缓存目录:")
print(f"   路径: {hf_cache_path}")
print(f"   存在: {'✓ 是' if hf_cache_path.exists() else '✗ 否'}")
if hf_cache_path.exists():
    import subprocess
    result = subprocess.run(['du', '-sh', str(hf_cache_path)], capture_output=True, text=True)
    size = result.stdout.split()[0] if result.stdout else "未知"
    print(f"   大小: {size}")
print()

# 导入 backend.config 测试 HF_HOME 设置
print("2. 导入 backend.config 测试环境变量设置...")
try:
    # 保存原始环境变量
    original_hf_home = os.environ.get('HF_HOME')
    
    # 清除 HF_HOME 以测试自动设置
    if 'HF_HOME' in os.environ:
        del os.environ['HF_HOME']
    
    # 导入 config 模块
    from backend import config
    
    # 检查 HF_HOME 是否被设置
    current_hf_home = os.environ.get('HF_HOME')
    print(f"   HF_HOME 设置: {'✓ 成功' if current_hf_home else '✗ 失败'}")
    if current_hf_home:
        print(f"   HF_HOME 值: {current_hf_home}")
        expected_path = str(hf_cache_path)
        if current_hf_home == expected_path:
            print(f"   路径匹配: ✓ 是 (使用项目内缓存)")
        else:
            print(f"   路径匹配: ✗ 否")
            print(f"   期望路径: {expected_path}")
    
    # 恢复原始环境变量
    if original_hf_home:
        os.environ['HF_HOME'] = original_hf_home
    elif 'HF_HOME' in os.environ:
        del os.environ['HF_HOME']
        
except Exception as e:
    print(f"   错误: {e}")

print()

# 检查其他缓存位置
print("3. 其他缓存位置检查:")
cache_locations = [
    ("项目内缓存", PROJECT_ROOT / "models" / "hf_cache"),
    ("用户缓存", Path.home() / ".cache" / "huggingface"),
    ("系统缓存", Path("/tmp")),
]

for name, path in cache_locations:
    exists = path.exists()
    if exists:
        try:
            import subprocess
            result = subprocess.run(['du', '-sh', str(path)], capture_output=True, text=True)
            size = result.stdout.split()[0] if result.stdout else "0B"
        except:
            size = "无法计算"
    else:
        size = "不存在"
    print(f"   {name}: {path} ({size})")

print()

# 检查已缓存的模型
print("4. 项目内缓存的模型:")
if hf_cache_path.exists():
    hub_dir = hf_cache_path / "hub"
    if hub_dir.exists():
        models = [d.name for d in hub_dir.iterdir() if d.is_dir() and d.name.startswith("models--")]
        if models:
            for model in sorted(models):
                print(f"   ✓ {model.replace('models--', '').replace('--', '/')}")
        else:
            print("   (无)")
    else:
        print("   hub 目录不存在")
else:
    print("   缓存目录不存在")

print()
print("=" * 60)
print("验证完成!")
print("=" * 60)
