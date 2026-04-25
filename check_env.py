#!/usr/bin/env python3
"""
检查Python环境是否正常
主要检查：
1. torch GPU是否可用
2. flash attention是否可用
"""

import sys
import subprocess


def check_torch():
    """检查torch和CUDA状态"""
    print("=" * 50)
    print("检查 PyTorch 环境")
    print("=" * 50)
    
    try:
        import torch
        print(f"✓ PyTorch 版本: {torch.__version__}")
        
        # 检查CUDA是否可用
        if torch.cuda.is_available():
            print(f"✓ CUDA 可用")
            print(f"  - CUDA 版本: {torch.version.cuda}")
            print(f"  - cuDNN 版本: {torch.backends.cudnn.version()}")
            print(f"  - GPU 数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
            
            # 测试GPU tensor
            try:
                test_tensor = torch.tensor([1.0, 2.0]).cuda()
                print(f"✓ GPU tensor 测试成功: {test_tensor}")
            except Exception as e:
                print(f"✗ GPU tensor 测试失败: {e}")
        else:
            print("✗ CUDA 不可用")
            
        return True
    except ImportError:
        print("✗ PyTorch 未安装")
        return False


def check_flash_attention():
    """检查Flash Attention是否可用"""
    print("\n" + "=" * 50)
    print("检查 Flash Attention")
    print("=" * 50)
    
    # 尝试导入flash_attn
    try:
        import flash_attn
        print(f"✓ flash_attn 已安装，版本: {flash_attn.__version__}")
        
        # 检查flash_attn_2_cuda
        try:
            import flash_attn_2_cuda
            print("✓ flash_attn_2_cuda 模块可用")
        except ImportError:
            print("✗ flash_attn_2_cuda 模块不可用")
            
        # 检查flash_attn_cuda (旧版本)
        try:
            import flash_attn_cuda
            print("✓ flash_attn_cuda 模块可用")
        except ImportError:
            print("✗ flash_attn_cuda 模块不可用")
            
        # 尝试运行简单的flash attention操作
        try:
            import torch
            if torch.cuda.is_available():
                from flash_attn import flash_attn_func
                
                # 创建测试数据
                batch_size, seqlen, nheads, headdim = 2, 64, 4, 64
                q = torch.randn(batch_size, seqlen, nheads, headdim, device='cuda', dtype=torch.float16)
                k = torch.randn(batch_size, seqlen, nheads, headdim, device='cuda', dtype=torch.float16)
                v = torch.randn(batch_size, seqlen, nheads, headdim, device='cuda', dtype=torch.float16)
                
                # 运行flash attention
                out = flash_attn_func(q, k, v)
                print(f"✓ Flash Attention 运行测试成功，输出形状: {out.shape}")
            else:
                print("⚠ 跳过 Flash Attention 功能测试 (CUDA 不可用)")
        except Exception as e:
            print(f"✗ Flash Attention 运行测试失败: {e}")
            
        return True
    except ImportError:
        print("✗ flash_attn 未安装")
        print("  安装命令: pip install flash-attn --no-build-isolation")
        return False


def check_xformers():
    """检查xformers是否可用（作为flash attention的替代）"""
    print("\n" + "=" * 50)
    print("检查 xFormers (Flash Attention 替代)")
    print("=" * 50)
    
    try:
        import xformers
        print(f"✓ xformers 已安装")
        
        try:
            from xformers.ops import memory_efficient_attention
            print("✓ memory_efficient_attention 可用")
            
            # 测试运行
            import torch
            if torch.cuda.is_available():
                B, M, H, K = 2, 64, 4, 64
                q = torch.randn(B, M, H, K, device='cuda', dtype=torch.float16)
                k = torch.randn(B, M, H, K, device='cuda', dtype=torch.float16)
                v = torch.randn(B, M, H, K, device='cuda', dtype=torch.float16)
                out = memory_efficient_attention(q, k, v)
                print(f"✓ xFormers attention 测试成功，输出形状: {out.shape}")
        except Exception as e:
            print(f"✗ xformers attention 测试失败: {e}")
            
        return True
    except ImportError:
        print("✗ xformers 未安装")
        print("  安装命令: pip install xformers")
        return False


def check_other_dependencies():
    """检查其他常见依赖"""
    print("\n" + "=" * 50)
    print("检查其他常用依赖")
    print("=" * 50)
    
    packages = [
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("transformers", "transformers"),
        ("accelerate", "accelerate"),
        ("einops", "einops"),
        ("triton", "triton"),
    ]
    
    for name, import_name in packages:
        try:
            module = __import__(import_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"✓ {name}: {version}")
        except ImportError:
            print(f"✗ {name}: 未安装")


def main():
    print("\n" + "=" * 50)
    print("Python 环境检查工具")
    print("=" * 50)
    print(f"Python 版本: {sys.version}")
    print(f"Python 路径: {sys.executable}")
    
    # 检查torch
    torch_ok = check_torch()
    
    # 检查flash attention
    fa_ok = check_flash_attention()
    
    # 检查xformers
    xf_ok = check_xformers()
    
    # 检查其他依赖
    check_other_dependencies()
    
    # 总结
    print("\n" + "=" * 50)
    print("检查总结")
    print("=" * 50)
    
    if torch_ok:
        import torch
        cuda_status = "可用" if torch.cuda.is_available() else "不可用"
        print(f"PyTorch: ✓ (CUDA {cuda_status})")
    else:
        print("PyTorch: ✗")
        
    print(f"Flash Attention: {'✓' if fa_ok else '✗'}")
    print(f"xFormers: {'✓' if xf_ok else '✗'}")
    
    if fa_ok or xf_ok:
        print("\n✓ 至少有一个高效 attention 实现可用")
    else:
        print("\n⚠ 没有高效 attention 实现可用，建议使用 pip install flash-attn 或 pip install xformers")
    
    print("\n" + "=" * 50)


if __name__ == "__main__":
    main()
