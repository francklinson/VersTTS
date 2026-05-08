#!/bin/bash
#
# HuggingFace 缓存迁移脚本
# 只保留 VersTTS 项目需要的模型，删除不需要的以节省空间
#

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 项目路径
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HF_CACHE_DIR="$PROJECT_ROOT/models/hf_cache"
HUB_DIR="$HF_CACHE_DIR/hub"

echo "========================================"
echo "  VersTTS HuggingFace 缓存清理脚本"
echo "========================================"
echo ""

# 检查目录是否存在
if [ ! -d "$HUB_DIR" ]; then
    echo -e "${RED}错误: 缓存目录不存在: $HUB_DIR${NC}"
    exit 1
fi

# 定义需要的模型列表（这些模型是 VersTTS 各算法必需的）
declare -a REQUIRED_MODELS=(
    # GPT-SoVITS 必需
    "models--facebook--hubert-base-ls960"
    "models--hfl--chinese-roberta-wwm-ext-large"
    
    # F5-TTS 必需
    "models--SWivid--F5-TTS"
    
    # VoxCPM 必需
    "models--openbmb--VoxCPM2"
    
    # IndexTTS 必需
    "models--facebook--w2v-bert-2.0"
    "models--nvidia--bigvgan_v2_22khz_80band_256x"
    "models--amphion--MaskGCT"
    "models--funasr--campplus"
    
    # OpenVoice 可能需要的（用于语音转换）
    "models--microsoft--wavlm-base"
    
    # 其他可能被 TTS 算法使用的
    "models--facebook--wav2vec2-base"
    "models--bert-base-uncased"
)

echo -e "${BLUE}需要的模型:${NC}"
for model in "${REQUIRED_MODELS[@]}"; do
    if [ -d "$HUB_DIR/$model" ]; then
        size=$(du -sh "$HUB_DIR/$model" 2>/dev/null | cut -f1)
        echo -e "  ${GREEN}✓${NC} $model (${size})"
    else
        echo -e "  ${YELLOW}○${NC} $model (未下载)"
    fi
done

echo ""
echo -e "${BLUE}扫描不需要的模型...${NC}"

# 计算可以节省的空间
total_saved=0
models_to_remove=()

for dir in "$HUB_DIR"/models--*; do
    if [ -d "$dir" ]; then
        dirname=$(basename "$dir")
        is_needed=false
        
        for required in "${REQUIRED_MODELS[@]}"; do
            if [[ "$dirname" == *"$required"* ]]; then
                is_needed=true
                break
            fi
        done
        
        if [ "$is_needed" = false ]; then
            size=$(du -sb "$dir" 2>/dev/null | cut -f1)
            total_saved=$((total_saved + size))
            models_to_remove+=("$dirname")
            hsize=$(du -sh "$dir" 2>/dev/null | cut -f1)
            echo -e "  ${RED}✗${NC} $dirname (${hsize})"
        fi
    fi
done

# 转换字节为人类可读
saved_gb=$(echo "scale=2; $total_saved / 1024 / 1024 / 1024" | bc)

echo ""
echo "========================================"
echo -e "找到 ${#models_to_remove[@]} 个不需要的模型"
echo -e "可节省空间: ${GREEN}${saved_gb} GB${NC}"
echo "========================================"
echo ""

# 询问是否删除
if [ ${#models_to_remove[@]} -eq 0 ]; then
    echo -e "${GREEN}所有模型都是必需的，无需清理！${NC}"
    exit 0
fi

echo "选项:"
echo "  1) 删除不需要的模型 (节省 ${saved_gb} GB)"
echo "  2) 保留所有模型 (完整迁移，占用更多空间)"
echo "  3) 取消"
echo ""
read -p "请选择 [1/2/3]: " choice

case $choice in
    1)
        echo ""
        echo -e "${YELLOW}正在删除不需要的模型...${NC}"
        for model in "${models_to_remove[@]}"; do
            echo "  删除: $model"
            rm -rf "$HUB_DIR/$model"
        done
        echo ""
        echo -e "${GREEN}✓ 清理完成！已节省 ${saved_gb} GB 空间${NC}"
        ;;
    2)
        echo ""
        echo -e "${BLUE}保留所有模型，准备完整迁移...${NC}"
        ;;
    3)
        echo ""
        echo "已取消"
        exit 0
        ;;
    *)
        echo "无效选项"
        exit 1
        ;;
esac

echo ""
echo "========================================"
echo "  迁移准备完成!"
echo "========================================"
echo ""
echo "当前缓存大小:"
du -sh "$HF_CACHE_DIR"
echo ""
echo "服务器部署命令:"
echo "  tar -czvf verttts-deploy.tar.gz --exclude='.git' --exclude='.venv' --exclude='__pycache__' VersTTS/"
echo ""
