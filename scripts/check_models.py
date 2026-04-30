#!/usr/bin/env python3
"""
VersTTS 模型路径检查脚本
用于验证离线部署时所有模型文件是否存在
"""

import os
import sys
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.absolute()

# 颜色定义
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"

def print_status(name: str, exists: bool, path: str, optional: bool = False):
    """打印状态信息"""
    if exists:
        print(f"{GREEN}[✓]{RESET} {name}: {path}")
    elif optional:
        print(f"{YELLOW}[○]{RESET} {name} (可选): {path}")
    else:
        print(f"{RED}[✗]{RESET} {name} (缺失): {path}")

def check_directory(path: Path, name: str, optional: bool = False) -> bool:
    """检查目录是否存在"""
    exists = path.exists() and path.is_dir()
    print_status(name, exists, str(path), optional)
    return exists

def check_file(path: Path, name: str, optional: bool = False) -> bool:
    """检查文件是否存在"""
    exists = path.exists() and path.is_file()
    print_status(name, exists, str(path), optional)
    return exists

def check_chattts() -> dict:
    """检查 ChatTTS 模型"""
    print(f"\n{BLUE}=== ChatTTS ==={RESET}")
    model_dir = PROJECT_ROOT / "algorithms" / "ChatTTS" / "models"
    
    checks = {
        "模型目录": check_directory(model_dir, "模型目录"),
    }
    
    # 检查具体的模型文件
    if model_dir.exists():
        files = list(model_dir.rglob("*.pt")) + list(model_dir.rglob("*.bin")) + list(model_dir.rglob("*.safetensors"))
        if files:
            print(f"  找到 {len(files)} 个模型文件")
            for f in files[:5]:  # 只显示前5个
                print(f"    - {f.name}")
            if len(files) > 5:
                print(f"    ... 还有 {len(files) - 5} 个文件")
    
    return checks

def check_cosyvoice() -> dict:
    """检查 CosyVoice 模型"""
    print(f"\n{BLUE}=== CosyVoice ==={RESET}")
    model_dir = PROJECT_ROOT / "algorithms" / "CosyVoice" / "models"
    
    checks = {
        "模型目录": check_directory(model_dir, "模型目录"),
    }
    
    return checks

def check_f5tts() -> dict:
    """检查 F5-TTS 模型"""
    print(f"\n{BLUE}=== F5-TTS ==={RESET}")
    model_dir = PROJECT_ROOT / "algorithms" / "F5-TTS" / "models"
    
    checks = {
        "模型目录": check_directory(model_dir, "模型目录", optional=True),
    }
    
    return checks

def check_gptsovits() -> dict:
    """检查 GPT-SoVITS 模型"""
    print(f"\n{BLUE}=== GPT-SoVITS ==={RESET}")
    pretrained_dir = PROJECT_ROOT / "algorithms" / "GPT-SoVITS" / "GPT_SoVITS" / "pretrained_models"
    
    checks = {
        "预训练模型目录": check_directory(pretrained_dir, "预训练模型目录"),
    }
    
    # 检查关键模型文件
    if pretrained_dir.exists():
        bert_dir = pretrained_dir / "chinese-roberta-wwm-ext-large"
        hubert_dir = pretrained_dir / "chinese-hubert-base"
        v2_dir = pretrained_dir / "gsv-v2final-pretrained"
        
        checks["BERT模型"] = check_directory(bert_dir, "  ├─ BERT模型", optional=True)
        checks["HuBERT模型"] = check_directory(hubert_dir, "  ├─ HuBERT模型", optional=True)
        checks["V2模型"] = check_directory(v2_dir, "  └─ V2预训练模型", optional=True)
        
        # 检查具体文件
        if v2_dir.exists():
            gpt_file = v2_dir / "s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"
            sovits_file = v2_dir / "s2G2333k.pth"
            checks["GPT权重"] = check_file(gpt_file, "      ├─ GPT权重")
            checks["SoVITS权重"] = check_file(sovits_file, "      └─ SoVITS权重")
    
    return checks

def check_openvoice() -> dict:
    """检查 OpenVoice 模型"""
    print(f"\n{BLUE}=== OpenVoice ==={RESET}")
    v1_dir = PROJECT_ROOT / "algorithms" / "OpenVoice" / "checkpoints_v1"
    v2_dir = PROJECT_ROOT / "algorithms" / "OpenVoice" / "checkpoints_v2"
    
    checks = {
        "V1模型目录": check_directory(v1_dir, "V1模型目录"),
        "V2模型目录": check_directory(v2_dir, "V2模型目录"),
    }
    
    # 检查具体文件
    if v1_dir.exists():
        zh_tts = v1_dir / "checkpoints" / "base_speakers" / "ZH" / "checkpoint.pth"
        checks["中文TTS模型"] = check_file(zh_tts, "  └─ 中文TTS模型")
    
    if v2_dir.exists():
        converter = v2_dir / "checkpoints_v2" / "converter" / "checkpoint.pth"
        checks["V2 Converter"] = check_file(converter, "  └─ V2 Converter")
    
    return checks

def check_qwen3tts() -> dict:
    """检查 Qwen3-TTS 模型"""
    print(f"\n{BLUE}=== Qwen3-TTS ==={RESET}")
    model_dir = PROJECT_ROOT / "algorithms" / "Qwen3-TTS" / "models"
    
    checks = {
        "模型目录": check_directory(model_dir, "模型目录"),
    }
    
    # 检查具体模型
    base_model = model_dir / "Qwen" / "Qwen3-TTS-12Hz-1.7B-Base"
    if base_model.exists():
        print(f"  {GREEN}[✓]{RESET} 基础模型存在: Qwen3-TTS-12Hz-1.7B-Base")
    else:
        print(f"  {YELLOW}[○]{RESET} 基础模型: 未找到 (可选)")
    
    return checks

def check_voxcpm() -> dict:
    """检查 VoxCPM 模型"""
    print(f"\n{BLUE}=== VoxCPM ==={RESET}")
    model_dir = PROJECT_ROOT / "algorithms" / "VoxCPM" / "models" / "VoxCPM2"
    
    checks = {
        "VoxCPM2模型": check_directory(model_dir, "VoxCPM2模型"),
    }
    
    # 检查关键文件
    if model_dir.exists():
        config_file = model_dir / "config.json"
        model_file = model_dir / "model.safetensors"
        checks["配置文件"] = check_file(config_file, "  ├─ config.json")
        checks["模型权重"] = check_file(model_file, "  └─ model.safetensors")
    
    return checks

def check_indextts() -> dict:
    """检查 IndexTTS 模型"""
    print(f"\n{BLUE}=== IndexTTS ==={RESET}")
    model_dir = PROJECT_ROOT / "algorithms" / "IndexTTS" / "checkpoints"
    
    checks = {
        "模型目录": check_directory(model_dir, "模型目录"),
    }
    
    # 检查关键文件
    if model_dir.exists():
        config_file = model_dir / "config.yaml"
        gpt_file = model_dir / "gpt.pth"
        s2mel_file = model_dir / "s2mel.pth"
        
        checks["配置文件"] = check_file(config_file, "  ├─ config.yaml")
        checks["GPT权重"] = check_file(gpt_file, "  ├─ gpt.pth")
        checks["S2MEL权重"] = check_file(s2mel_file, "  ├─ s2mel.pth")
        
        # 检查离线部署需要的额外 HuggingFace 模型
        print(f"\n  {BLUE}HuggingFace 模型检查 (离线部署必需):{RESET}")
        w2v_bert = model_dir / "w2v-bert-2.0"
        semantic_codec = model_dir / "semantic_codec" / "model.safetensors"
        campplus = model_dir / "campplus_cn_common.bin"
        bigvgan = model_dir / "bigvgan"
        
        checks["w2v-bert-2.0"] = check_directory(w2v_bert, "  ├─ w2v-bert-2.0", optional=True)
        checks["semantic_codec"] = check_file(semantic_codec, "  ├─ semantic_codec/model.safetensors", optional=True)
        checks["campplus"] = check_file(campplus, "  ├─ campplus_cn_common.bin", optional=True)
        checks["bigvgan"] = check_directory(bigvgan, "  └─ bigvgan/", optional=True)
        
        # 统计 HuggingFace 模型
        hf_models_found = sum([
            w2v_bert.exists(),
            semantic_codec.exists(),
            campplus.exists(),
            bigvgan.exists()
        ])
        if hf_models_found < 4:
            print(f"\n  {YELLOW}⚠ 离线部署警告: 仅找到 {hf_models_found}/4 个 HuggingFace 模型{RESET}")
            print(f"  {YELLOW}  运行以下命令下载缺失的模型:{RESET}")
            if not w2v_bert.exists():
                print(f"    huggingface-cli download facebook/w2v-bert-2.0 --local-dir {model_dir}/w2v-bert-2.0")
            if not semantic_codec.exists():
                print(f"    huggingface-cli download amphion/MaskGCT semantic_codec/model.safetensors --local-dir {model_dir}")
            if not campplus.exists():
                print(f"    huggingface-cli download funasr/campplus campplus_cn_common.bin --local-dir {model_dir}")
            if not bigvgan.exists():
                print(f"    huggingface-cli download nvidia/bigvgan_v2_22khz_80band_256x --local-dir {model_dir}/bigvgan")
    
    return checks

def check_fireredtts() -> dict:
    """检查 FireRedTTS2 模型"""
    print(f"\n{BLUE}=== FireRedTTS2 ==={RESET}")
    model_dir = PROJECT_ROOT / "algorithms" / "FireRedTTS2" / "pretrained_models" / "FireRedTTS2"
    
    checks = {
        "模型目录": check_directory(model_dir, "模型目录"),
    }
    
    # 检查关键文件
    if model_dir.exists():
        llm_pretrain = model_dir / "llm_pretrain.pt"
        llm_posttrain = model_dir / "llm_posttrain.pt"
        codec = model_dir / "codec.pt"
        config_llm = model_dir / "config_llm.json"
        config_codec = model_dir / "config_codec.json"
        qwen_dir = model_dir / "Qwen2.5-1.5B"
        
        checks["LLM预训练权重"] = check_file(llm_pretrain, "  ├─ llm_pretrain.pt")
        checks["LLM微调权重"] = check_file(llm_posttrain, "  ├─ llm_posttrain.pt", optional=True)
        checks["Codec权重"] = check_file(codec, "  ├─ codec.pt")
        checks["LLM配置"] = check_file(config_llm, "  ├─ config_llm.json")
        checks["Codec配置"] = check_file(config_codec, "  ├─ config_codec.json")
        checks["Qwen模型"] = check_directory(qwen_dir, "  └─ Qwen2.5-1.5B")
    
    return checks

def check_hf_cache() -> dict:
    """检查 HuggingFace 缓存目录"""
    print(f"\n{BLUE}=== HuggingFace 缓存 ==={RESET}")
    
    hf_home = os.environ.get('HF_HOME', PROJECT_ROOT / "models" / "hf_cache")
    transformers_cache = os.environ.get('TRANSFORMERS_CACHE', PROJECT_ROOT / "models" / "transformers_cache")
    
    checks = {
        "HF_HOME": check_directory(Path(hf_home), f"HF_HOME: {hf_home}", optional=True),
        "TRANSFORMERS_CACHE": check_directory(Path(transformers_cache), f"TRANSFORMERS_CACHE: {transformers_cache}", optional=True),
    }
    
    return checks

def check_environment():
    """检查环境变量"""
    print(f"\n{BLUE}=== 环境变量 ==={RESET}")
    
    env_vars = [
        'HF_HUB_OFFLINE',
        'HF_DATASETS_OFFLINE', 
        'TRANSFORMERS_OFFLINE',
        'HF_HOME',
        'TRANSFORMERS_CACHE',
        'TORCH_HOME'
    ]
    
    for var in env_vars:
        value = os.environ.get(var)
        if value:
            print(f"{GREEN}[✓]{RESET} {var}={value}")
        else:
            print(f"{YELLOW}[○]{RESET} {var} (未设置)")

def main():
    """主函数"""
    print("=" * 60)
    print("VersTTS 模型路径检查工具")
    print("=" * 60)
    print(f"项目根目录: {PROJECT_ROOT}")
    print(f"Python版本: {sys.version}")
    
    # 检查各算法模型
    all_checks = {}
    all_checks.update(check_chattts())
    all_checks.update(check_cosyvoice())
    all_checks.update(check_f5tts())
    all_checks.update(check_gptsovits())
    all_checks.update(check_openvoice())
    all_checks.update(check_qwen3tts())
    all_checks.update(check_voxcpm())
    all_checks.update(check_indextts())
    all_checks.update(check_fireredtts())
    all_checks.update(check_hf_cache())
    
    # 检查环境变量
    check_environment()
    
    # 统计结果
    print("\n" + "=" * 60)
    print("检查总结")
    print("=" * 60)
    
    total = len(all_checks)
    passed = sum(1 for v in all_checks.values() if v)
    failed = total - passed
    
    print(f"总检查项: {total}")
    print(f"{GREEN}通过: {passed}{RESET}")
    if failed > 0:
        print(f"{RED}失败: {failed}{RESET}")
    
    # 离线部署建议
    print("\n" + "=" * 60)
    print("离线部署建议")
    print("=" * 60)
    
    if failed > 0:
        print(f"{YELLOW}警告: 有 {failed} 个必需模型未找到{RESET}")
        print("\n请按以下步骤准备模型文件:")
        print("1. 在有网络的环境中下载所有模型文件")
        print("2. 将模型文件打包传输到目标服务器")
        print("3. 解压到对应的 algorithms/<算法>/models/ 目录")
        print("\n详细步骤请参考 OFFLINE_DEPLOYMENT.md")
    else:
        print(f"{GREEN}所有必需模型文件已准备就绪！{RESET}")
        print("\n可以使用以下命令启动服务:")
        print("  ./start_server.sh start --offline")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
