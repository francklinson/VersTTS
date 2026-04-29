#!/bin/bash
# 下载VoxCPM、IndexTTS、FireRedTTS2模型脚本

set -e

echo "=========================================="
echo "开始下载TTS模型"
echo "=========================================="

# 设置HF镜像加速
export HF_ENDPOINT="https://hf-mirror.com"

PROJECT_ROOT="/home/zhouchenghao/PycharmProjects/VersTTS"
cd "$PROJECT_ROOT"

# 检查并安装huggingface-hub
if ! command -v huggingface-cli &> /dev/null; then
    echo "安装 huggingface-hub..."
    pip install huggingface-hub[cli,hf_xet]
fi

# ===== VoxCPM 模型下载 =====
echo ""
echo "=========================================="
echo "[1/3] 下载 VoxCPM2 模型"
echo "=========================================="
VOXCPM_MODEL_DIR="$PROJECT_ROOT/algorithms/VoxCPM/models/VoxCPM2"
mkdir -p "$VOXCPM_MODEL_DIR"

if [ ! -f "$VOXCPM_MODEL_DIR/config.json" ]; then
    echo "正在下载 VoxCPM2 模型 (openbmb/VoxCPM2)..."
    huggingface-cli download openbmb/VoxCPM2 --local-dir "$VOXCPM_MODEL_DIR" --resume-download
    echo "VoxCPM2 模型下载完成"
else
    echo "VoxCPM2 模型已存在，跳过下载"
fi

# ===== IndexTTS 模型下载 =====
echo ""
echo "=========================================="
echo "[2/3] 下载 IndexTTS-2 模型"
echo "=========================================="
INDEXTTS_MODEL_DIR="$PROJECT_ROOT/algorithms/IndexTTS/checkpoints"
mkdir -p "$INDEXTTS_MODEL_DIR"

if [ ! -f "$INDEXTTS_MODEL_DIR/config.yaml" ]; then
    echo "正在下载 IndexTTS-2 模型 (IndexTeam/IndexTTS-2)..."
    huggingface-cli download IndexTeam/IndexTTS-2 --local-dir "$INDEXTTS_MODEL_DIR" --resume-download
    echo "IndexTTS-2 模型下载完成"
else
    echo "IndexTTS-2 模型已存在，跳过下载"
fi

# ===== FireRedTTS2 模型下载 =====
echo ""
echo "=========================================="
echo "[3/3] 下载 FireRedTTS2 模型"
echo "=========================================="
FIRERED_MODEL_DIR="$PROJECT_ROOT/algorithms/FireRedTTS2/pretrained_models/FireRedTTS2"
mkdir -p "$FIRERED_MODEL_DIR"

if [ ! -f "$FIRERED_MODEL_DIR/config.json" ]; then
    echo "正在下载 FireRedTTS2 模型 (FireRedTeam/FireRedTTS2)..."
    huggingface-cli download FireRedTeam/FireRedTTS2 --local-dir "$FIRERED_MODEL_DIR" --resume-download
    echo "FireRedTTS2 模型下载完成"
else
    echo "FireRedTTS2 模型已存在，跳过下载"
fi

echo ""
echo "=========================================="
echo "所有模型下载完成"
echo "=========================================="
echo "模型路径:"
echo "  - VoxCPM2: $VOXCPM_MODEL_DIR"
echo "  - IndexTTS-2: $INDEXTTS_MODEL_DIR"
echo "  - FireRedTTS2: $FIRERED_MODEL_DIR"
