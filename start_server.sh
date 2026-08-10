#!/bin/bash
#
# VersTTS 服务管理脚本
# 支持 start | stop | restart | status
# 使用 .venv 虚拟环境
#

set -e

# 获取脚本所在目录（项目根目录）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="$SCRIPT_DIR/.venv"
PID_FILE="$SCRIPT_DIR/.server.pid"
LOG_FILE="$SCRIPT_DIR/logs/server.log"
ENV_FILE="$SCRIPT_DIR/.env.offline"

# 服务配置
HOST="0.0.0.0"
PORT="8006"

# GPU 配置（使用 CUDA_VISIBLE_DEVICES 分配不同显卡）
# 格式: "0", "1", "0,1" 等，空字符串表示使用所有可用GPU
# 主服务 GPU 配置
MAIN_GPU="${MAIN_GPU:-0}"
# OmniVoice 独立服务 GPU 配置
OMNIVOICE_GPU="${OMNIVOICE_GPU:-0}"
# CosyVoice 独立服务 GPU 配置
COSYVOICE_GPU="${COSYVOICE_GPU:-0}"
# PilotTTS 独立服务 GPU 配置
PILOTTS_GPU="${PILOTTS_GPU:-0}"
# GPT-SoVITS 独立服务 GPU 配置
GPTSOVITS_GPU="${GPTSOVITS_GPU:-0}"
# Fish-Speech 独立服务 GPU 配置
FISHSPEECH_GPU="${FISHSPEECH_GPU:-0}"

# 任务队列并发配置
# 基于实际测试，双模型并行时速度下降50-100%，因此默认采用单模型串行
# 如需启用多模型并行，可调整此值（建议根据GPU数量和显存大小设置）
MAX_CONCURRENT_MODELS="${MAX_CONCURRENT_MODELS:-2}"

# OmniVoice 独立服务配置
OMNIVOICE_HOST="127.0.0.1"
OMNIVOICE_PORT="${OMNIVOICE_PORT:-8007}"
OMNIVOICE_PID_FILE="$SCRIPT_DIR/.omnivoice.pid"
OMNIVOICE_LOG_FILE="$SCRIPT_DIR/logs/omnivoice_service.log"
OMNIVOICE_SCRIPT="$SCRIPT_DIR/omnivoice_service.py"

# CosyVoice 独立服务配置
COSYVOICE_HOST="127.0.0.1"
COSYVOICE_PORT="${COSYVOICE_PORT:-8008}"
COSYVOICE_PID_FILE="$SCRIPT_DIR/.cosyvoice.pid"
COSYVOICE_LOG_FILE="$SCRIPT_DIR/logs/cosyvoice_service.log"
COSYVOICE_SCRIPT="$SCRIPT_DIR/cosyvoice_service.py"

# PilotTTS 独立服务配置
PILOTTS_HOST="127.0.0.1"
PILOTTS_PORT="${PILOTTS_PORT:-8009}"
PILOTTS_PID_FILE="$SCRIPT_DIR/.pilottts.pid"
PILOTTS_LOG_FILE="$SCRIPT_DIR/logs/pilottts_service.log"
PILOTTS_SCRIPT="$SCRIPT_DIR/pilottts_service.py"

# GPT-SoVITS 独立服务配置
GPTSOVITS_HOST="127.0.0.1"
GPTSOVITS_PORT="${GPTSOVITS_PORT:-8010}"
GPTSOVITS_PID_FILE="$SCRIPT_DIR/.gptsovits.pid"
GPTSOVITS_LOG_FILE="$SCRIPT_DIR/logs/gptsovits_service.log"
GPTSOVITS_SCRIPT="$SCRIPT_DIR/gptsovits_service.py"

# Fish-Speech 独立服务配置
FISHSPEECH_HOST="127.0.0.1"
FISHSPEECH_PORT="${FISHSPEECH_PORT:-8005}"
FISHSPEECH_PID_FILE="$SCRIPT_DIR/logs/fishspeech_service.pid"
FISHSPEECH_LOG_FILE="$SCRIPT_DIR/logs/fishspeech_service.log"
FISHSPEECH_SCRIPT="$SCRIPT_DIR/fishspeech_service.py"

# HTTPS 配置（留空则使用 HTTP）
# 生成自签名证书: openssl req -x509 -newkey rsa:2048 -keyout ssl/key.pem -out ssl/cert.pem -days 365 -nodes
SSL_CERT="ssl/cert.pem"  # SSL 证书路径，如 "ssl/cert.pem"
SSL_KEY="ssl/key.pem"   # SSL 私钥路径，如 "ssl/key.pem"

# 路径配置
MODELS_DIR="models"
OUTPUTS_DIR="outputs"
LOGS_DIR="logs"
SPEAKERS_DIR="speakers"

# 日志配置
LOG_LEVEL="INFO"
LOG_MAX_SIZE="50"
LOG_BACKUP_COUNT="5"

# 离线模式（1=启用，0=禁用）
TRANSFORMERS_OFFLINE="1"
HF_HUB_OFFLINE="1"
SKIP_CHECK=false
RELOAD=false
OFFLINE_MODE=false

# ========== 模型预加载配置 ==========
# 所有独立服务默认不预加载，首次调用时按需加载
# 空闲超时后自动卸载模型释放显存
PRELOAD_OMNIVOICE="${PRELOAD_OMNIVOICE:-0}"   # 1=启动时加载, 0=按需加载
PRELOAD_COSYVOICE="${PRELOAD_COSYVOICE:-0}"   # 1=启动时加载, 0=按需加载
PRELOAD_PILOTTS="${PRELOAD_PILOTTS:-0}"       # 1=启动时加载, 0=按需加载
PRELOAD_GPTSOVITS="${PRELOAD_GPTSOVITS:-0}"   # 1=启动时加载, 0=按需加载
PRELOAD_FISHSPEECH="${PRELOAD_FISHSPEECH:-0}"   # 1=启动时加载, 0=按需加载

# ========== 启动等待超时配置（秒）==========
# 子服务首次启动需加载大量依赖（torch/transformers 等），导入阶段即可达 20-30s，
# 故默认值留足余量。可通过环境变量覆盖，例如 START_WAIT_NO_PRELOAD=90 ./start_server.sh start
START_WAIT_NO_PRELOAD="${START_WAIT_NO_PRELOAD:-60}"   # 不预加载时的等待秒数（默认 60）
START_WAIT_PRELOAD="${START_WAIT_PRELOAD:-180}"        # 预加载时的等待秒数（默认 180）

# ========== 空闲超时与心跳配置 ==========
IDLE_TIMEOUT="${IDLE_TIMEOUT:-900}"           # 空闲超时秒数，默认 15 分钟
HEARTBEAT_INTERVAL="${HEARTBEAT_INTERVAL:-60}" # 心跳间隔秒数，默认 1 分钟

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_step() {
    echo -e "${CYAN}[→]${NC} $1"
}

# 检查服务是否真正就绪（端口监听 + /health 响应）。
# 与 is_*_running（仅判断进程存活）区分：进程可能活着但还没监听端口（启动中）或已僵死。
# 参数: $1=端口  返回: 0=就绪, 1=未就绪
is_service_ready() {
    local port="$1"
    curl -s -m 2 "http://127.0.0.1:${port}/health" >/dev/null 2>&1
}

# 帮助信息
usage() {
    echo "用法: $0 <命令> [选项]"
    echo ""
    echo "命令:"
    echo "  start      启动服务"
    echo "  stop       停止服务"
    echo "  restart    重启服务"
    echo "  status     查看服务状态"
    echo ""
    echo "OmniVoice 独立服务命令:"
    echo "  start-omnivoice      启动 OmniVoice 独立服务"
    echo "  stop-omnivoice       停止 OmniVoice 独立服务"
    echo "  restart-omnivoice    重启 OmniVoice 独立服务"
    echo "  status-omnivoice     查看 OmniVoice 服务状态"
    echo ""
    echo "CosyVoice 独立服务命令:"
    echo "  start-cosyvoice      启动 CosyVoice 独立服务"
    echo "  stop-cosyvoice       停止 CosyVoice 独立服务"
    echo "  restart-cosyvoice    重启 CosyVoice 独立服务"
    echo "  status-cosyvoice     查看 CosyVoice 服务状态"
    echo ""
    echo "PilotTTS 独立服务命令:"
    echo "  start-pilottts       启动 PilotTTS 独立服务"
    echo "  stop-pilottts        停止 PilotTTS 独立服务"
    echo "  restart-pilottts     重启 PilotTTS 独立服务"
    echo "  status-pilottts      查看 PilotTTS 服务状态"
    echo ""
    echo "GPT-SoVITS 独立服务命令:"
    echo "  start-gptsovits     启动 GPT-SoVITS 独立服务"
    echo "  stop-gptsovits      停止 GPT-SoVITS 独立服务"
    echo "  restart-gptsovits   重启 GPT-SoVITS 独立服务"
    echo "  status-gptsovits    查看 GPT-SoVITS 服务状态"
    echo ""
    echo "Fish-Speech 独立服务命令:"
    echo "  start-fishspeech    启动 Fish-Speech 独立服务"
    echo "  stop-fishspeech     停止 Fish-Speech 独立服务"
    echo "  restart-fishspeech  重启 Fish-Speech 独立服务"
    echo "  status-fishspeech   查看 Fish-Speech 服务状态"
    echo ""
    echo "选项 (仅 start/restart 有效):"
    echo "  -h, --help       显示帮助信息"
    echo "  --host <地址>    服务主机地址 (默认: 0.0.0.0)"
    echo "  --port <端口>    服务端口 (默认: 8000)"
    echo "  --skip-check     跳过环境检查"
    echo "  --reload         开发模式(自动重载)"
    echo "  --offline        离线模式(禁用HuggingFace等外部资源访问)"
    echo ""
    echo "示例:"
    echo "  $0 start                     # 默认启动"
    echo "  $0 start --port 8080         # 指定端口启动"
    echo "  $0 start --reload            # 开发模式启动"
    echo "  $0 start --offline           # 离线模式启动"
    echo "  $0 stop                      # 停止服务"
    echo "  $0 restart                   # 重启服务"
    echo "  $0 status                    # 查看状态"
    echo "  $0 start-omnivoice           # 启动 OmniVoice 服务"
    echo "  $0 stop-omnivoice            # 停止 OmniVoice 服务"
    echo "  $0 status-omnivoice          # 查看 OmniVoice 状态"
    exit 0
}

# 检查虚拟环境
check_venv() {
    print_step "检查虚拟环境..."
    if [ ! -d "$VENV_PATH" ]; then
        print_error "虚拟环境不存在: $VENV_PATH"
        echo "       请先创建虚拟环境: python -m venv .venv"
        exit 1
    fi
    print_success "虚拟环境存在: $VENV_PATH"
}

# 获取服务 PID
get_pid() {
    if [ -f "$PID_FILE" ]; then
        cat "$PID_FILE" 2>/dev/null
    fi
}

# 检查服务是否运行
is_running() {
    local pid
    pid=$(get_pid)
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        return 0
    fi
    return 1
}

# 检查 transformers 多版本环境
check_transformers_versions() {
    print_step "检查 transformers 多版本环境..."

    local tf_errors=0
    local tf_warnings=0

    # --- 辅助：用 Python 做版本比较 ---
    _version_check() {
        # 用法: _version_check <actual> <op> <expected>
        # 返回 0 表示满足条件，1 表示不满足
        python -c "from packaging.version import Version; v=Version('$1'); exit(0 if v $2 Version('$3') else 1)" 2>/dev/null
    }

    # ================================================================
    # 1. 全局 transformers — 宽松检查: >= 4.57.0, < 5.0.0
    #    Qwen3TTS 运行时也会检查，这里只是提前预警
    # ================================================================
    local ACTUAL_GLOBAL=$(python -c "import transformers; print(transformers.__version__)" 2>/dev/null)
    if [ -z "$ACTUAL_GLOBAL" ]; then
        print_error "全局 transformers: 未安装或导入失败"
        tf_errors=$((tf_errors + 1))
    elif _version_check "$ACTUAL_GLOBAL" ">=" "5.0.0"; then
        print_error "全局 transformers: $ACTUAL_GLOBAL (不允许 5.x — 会破坏 Qwen3TTS/VoxCPM 等)"
        print_info "请降级: pip install 'transformers>=4.57.0,<5.0.0'"
        tf_errors=$((tf_errors + 1))
    elif _version_check "$ACTUAL_GLOBAL" ">=" "4.57.0"; then
        print_success "全局 transformers: $ACTUAL_GLOBAL (主服务 — Qwen3TTS/VoxCPM/ChatTTS等)"
    else
        print_warn "全局 transformers: $ACTUAL_GLOBAL (建议 >= 4.57.0 — Qwen3TTS 需要 4.57+)"
        tf_warnings=$((tf_warnings + 1))
    fi

    # ================================================================
    # 2. lib/transformers4 — 严格锁定: == 4.51.3
    #    PilotTTS 上限 4.52.4，CosyVoice tokenizer 在 4.52+ 出问题
    # ================================================================
    local EXPECTED_TF4="4.51.3"
    local TF4_PATH="$SCRIPT_DIR/lib/transformers4"
    if [ ! -d "$TF4_PATH" ]; then
        print_error "lib/transformers4: 目录不存在 (CosyVoice/PilotTTS/GPT-SoVITS 需要)"
        print_info "请执行: pip install --target $TF4_PATH transformers==$EXPECTED_TF4"
        tf_errors=$((tf_errors + 1))
    else
        local ACTUAL_TF4=$(PYTHONPATH="$TF4_PATH" python -c "import transformers; print(transformers.__version__)" 2>/dev/null)
        if [ -z "$ACTUAL_TF4" ]; then
            print_error "lib/transformers4: 无法加载 transformers 模块"
            tf_errors=$((tf_errors + 1))
        elif [ "$ACTUAL_TF4" != "$EXPECTED_TF4" ]; then
            if _version_check "$ACTUAL_TF4" ">" "$EXPECTED_TF4"; then
                print_error "lib/transformers4: $ACTUAL_TF4 (期望 $EXPECTED_TF4 — >= 4.52 会破坏 CosyVoice tokenizer)"
                print_info "请降级: pip install --target $TF4_PATH transformers==$EXPECTED_TF4 --force-reinstall"
                tf_errors=$((tf_errors + 1))
            else
                print_warn "lib/transformers4: $ACTUAL_TF4 (期望 $EXPECTED_TF4 — 版本偏低，可能缺失功能)"
                tf_warnings=$((tf_warnings + 1))
            fi
        else
            print_success "lib/transformers4: $ACTUAL_TF4 (CosyVoice/PilotTTS/GPT-SoVITS)"
        fi
    fi

    # ================================================================
    # 3. lib/transformers5 — 宽松检查: >= 5.3.0
    #    OmniVoice 对 5.x 子版本不敏感
    # ================================================================
    local EXPECTED_TF5_MIN="5.3.0"
    local TF5_PATH="$SCRIPT_DIR/lib/transformers5"
    if [ ! -d "$TF5_PATH" ]; then
        print_error "lib/transformers5: 目录不存在 (OmniVoice 需要)"
        print_info "请执行: pip install --target $TF5_PATH 'transformers>=5.3.0'"
        tf_errors=$((tf_errors + 1))
    else
        local ACTUAL_TF5=$(PYTHONPATH="$TF5_PATH" python -c "import transformers; print(transformers.__version__)" 2>/dev/null)
        if [ -z "$ACTUAL_TF5" ]; then
            print_error "lib/transformers5: 无法加载 transformers 模块"
            tf_errors=$((tf_errors + 1))
        elif _version_check "$ACTUAL_TF5" ">=" "$EXPECTED_TF5_MIN"; then
            print_success "lib/transformers5: $ACTUAL_TF5 (OmniVoice)"
        else
            print_warn "lib/transformers5: $ACTUAL_TF5 (建议 >= $EXPECTED_TF5_MIN — OmniVoice 需要 5.x)"
            tf_warnings=$((tf_warnings + 1))
        fi
    fi

    # ================================================================
    # 汇总
    # ================================================================
    if [ $tf_errors -gt 0 ] || [ $tf_warnings -gt 0 ]; then
        echo ""
    fi
    if [ $tf_errors -gt 0 ]; then
        print_error "transformers 版本环境存在 $tf_errors 个错误，相关服务将无法启动"
        echo ""
    elif [ $tf_warnings -gt 0 ]; then
        print_warn "transformers 版本环境存在 $tf_warnings 个警告，部分功能可能异常"
        echo ""
    else
        print_success "transformers 多版本环境检查通过"
    fi
}

# 启动服务
do_start() {
    echo ""
    echo "========================================"
    echo "      VersTTS 服务启动"
    echo "========================================"
    echo ""

    # 检查是否已在运行
    if is_running; then
        local current_pid=$(get_pid)
        print_warn "服务已在运行中"
        print_info "当前PID: $current_pid"
        print_info "访问地址: http://$HOST:$PORT"
        echo ""
        echo "如需重启，请使用: $0 restart"
        exit 0
    fi

    print_step "检查运行环境..."
    check_venv

    print_step "激活虚拟环境..."
    source "$VENV_PATH/bin/activate"
    print_success "虚拟环境已激活"

    # 加载离线模式环境变量
    if [ "$OFFLINE_MODE" = true ] && [ -f "$ENV_FILE" ]; then
        print_step "加载离线模式环境变量..."
        source "$ENV_FILE"
        print_success "离线模式已启用 (HuggingFace离线)"
    fi

    # 导出环境变量供 Python 后端使用
    export HOST
    export PORT
    export OMNIVOICE_HOST
    export OMNIVOICE_PORT
    export COSYVOICE_HOST
    export COSYVOICE_PORT
    export PILOTTS_HOST
    export PILOTTS_PORT
    export GPTSOVITS_HOST
    export GPTSOVITS_PORT
    export FISHSPEECH_HOST
    export FISHSPEECH_PORT
    export MODELS_DIR
    export OUTPUTS_DIR
    export LOGS_DIR
    export SPEAKERS_DIR
    export LOG_LEVEL
    export LOG_MAX_SIZE
    export LOG_BACKUP_COUNT
    export TRANSFORMERS_OFFLINE
    export HF_HUB_OFFLINE
    export PRELOAD_MODELS
    export PRELOAD_OMNIVOICE
    export PRELOAD_COSYVOICE
    export PRELOAD_PILOTTS
    export PRELOAD_GPTSOVITS
    export PRELOAD_FISHSPEECH
    export IDLE_TIMEOUT
    export HEARTBEAT_INTERVAL

    # 主服务地址（供独立服务注册/心跳/驱逐使用）
    export MAIN_HOST="127.0.0.1"
    export MAIN_PORT="$PORT"

    # 检查 Python
    PYTHON_VERSION=$(python --version 2>&1)
    print_info "Python 版本: $PYTHON_VERSION"

    # 检查 CUDA
    print_step "检查 GPU 状态..."
    python -c "
import torch
if torch.cuda.is_available():
    print(f'  CUDA 可用: ✓')
    print(f'  CUDA 版本: {torch.version.cuda}')
    print(f'  GPU 设备: {torch.cuda.get_device_name(0)}')
    print(f'  GPU 数量: {torch.cuda.device_count()}')
    print(f'  GPU 显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
else:
    print('  CUDA 可用: ✗')
    print('  警告: CUDA 不可用，将使用 CPU 模式')
" 2>/dev/null || print_warn "CUDA 检查失败"

    # ========== transformers 版本检查 ==========
    check_transformers_versions

    # ========== 模型文件检查 ==========
    print_step "检查模型文件..."

    local MODELS_ABS_DIR="$SCRIPT_DIR/$MODELS_DIR"
    local model_errors=0

    # --- GPT-SoVITS ---
    local GS_DIR="$MODELS_ABS_DIR/GPT-SoVITS"
    local gs_ok=true
    # BERT & HuBERT
    [ -f "$GS_DIR/chinese-roberta-wwm-ext-large/config.json" ] || { print_warn "缺失: $GS_DIR/chinese-roberta-wwm-ext-large/"; gs_ok=false; }
    [ -f "$GS_DIR/chinese-hubert-base/config.json" ] || { print_warn "缺失: $GS_DIR/chinese-hubert-base/"; gs_ok=false; }
    # v1
    [ -f "$GS_DIR/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt" ] || { print_warn "缺失: $GS_DIR/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"; gs_ok=false; }
    [ -f "$GS_DIR/s2G488k.pth" ] || { print_warn "缺失: $GS_DIR/s2G488k.pth"; gs_ok=false; }
    # v2
    [ -f "$GS_DIR/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt" ] || { print_warn "缺失: $GS_DIR/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"; gs_ok=false; }
    [ -f "$GS_DIR/gsv-v2final-pretrained/s2G2333k.pth" ] || { print_warn "缺失: $GS_DIR/gsv-v2final-pretrained/s2G2333k.pth"; gs_ok=false; }
    # v2Pro/v2ProPlus/v3/v4 共用 s1v3.ckpt
    [ -f "$GS_DIR/s1v3.ckpt" ] || { print_warn "缺失: $GS_DIR/s1v3.ckpt"; gs_ok=false; }
    [ -f "$GS_DIR/v2Pro/s2Gv2Pro.pth" ] || { print_warn "缺失: $GS_DIR/v2Pro/s2Gv2Pro.pth"; gs_ok=false; }
    [ -f "$GS_DIR/v2Pro/s2Gv2ProPlus.pth" ] || { print_warn "缺失: $GS_DIR/v2Pro/s2Gv2ProPlus.pth"; gs_ok=false; }
    [ -f "$GS_DIR/s2Gv3.pth" ] || { print_warn "缺失: $GS_DIR/s2Gv3.pth"; gs_ok=false; }
    [ -f "$GS_DIR/gsv-v4-pretrained/s2Gv4.pth" ] || { print_warn "缺失: $GS_DIR/gsv-v4-pretrained/s2Gv4.pth"; gs_ok=false; }
    # SV 模型
    [ -f "$GS_DIR/sv/pretrained_eres2netv2w24s4ep4.ckpt" ] || { print_warn "缺失: $GS_DIR/sv/pretrained_eres2netv2w24s4ep4.ckpt"; gs_ok=false; }
    # G2PW 多音字模型（中文必需，缺失会触发联网下载，离线环境会失败）
    [ -d "$GS_DIR/G2PWModel" ] || { print_warn "缺失: $GS_DIR/G2PWModel/ (中文多音字模型，缺失将联网下载)"; gs_ok=false; }
    if $gs_ok; then print_success "GPT-SoVITS: 模型完整 (v1/v2/v2Pro/v2ProPlus/v3/v4)"; else model_errors=$((model_errors + 1)); fi

    # --- OmniVoice ---
    local OV_DIR="$MODELS_ABS_DIR/OmniVoice"
    if [ -f "$OV_DIR/config.json" ] && [ -f "$OV_DIR/model.safetensors" ]; then
        print_success "OmniVoice: 模型完整"
    else
        print_warn "OmniVoice: 模型不完整"
        [ ! -f "$OV_DIR/config.json" ] && print_warn "缺失: $OV_DIR/config.json"
        [ ! -f "$OV_DIR/model.safetensors" ] && print_warn "缺失: $OV_DIR/model.safetensors"
        model_errors=$((model_errors + 1))
    fi

    # --- CosyVoice ---
    local CV_DIR="$MODELS_ABS_DIR/CosyVoice/Fun-CosyVoice3-0.5B"
    if [ -d "$CV_DIR" ] && [ -f "$CV_DIR/config.json" ]; then
        print_success "CosyVoice: 模型完整"
    else
        print_warn "CosyVoice: 模型不完整"
        [ ! -d "$CV_DIR" ] && print_warn "缺失: $CV_DIR/"
        [ -d "$CV_DIR" ] && [ ! -f "$CV_DIR/config.json" ] && print_warn "缺失: $CV_DIR/config.json"
        model_errors=$((model_errors + 1))
    fi

    # --- PilotTTS ---
    local PT_DIR="$MODELS_ABS_DIR/PilotTTS"
    local pt_ok=true
    [ -f "$PT_DIR/pilot_tts.pt" ] || { print_warn "缺失: $PT_DIR/pilot_tts.pt"; pt_ok=false; }
    [ -d "$PT_DIR/Qwen3-0.6B" ] || { print_warn "缺失: $PT_DIR/Qwen3-0.6B/"; pt_ok=false; }
    [ -d "$PT_DIR/w2v-bert-2.0" ] || { print_warn "缺失: $PT_DIR/w2v-bert-2.0/"; pt_ok=false; }
    if $pt_ok; then print_success "PilotTTS: 模型完整"; else model_errors=$((model_errors + 1)); fi

    # --- wenet ASR ---
    local WN_DIR="$MODELS_ABS_DIR/wenet/wenetspeech"
    if [ -f "$WN_DIR/final.pt" ] && [ -f "$WN_DIR/train.yaml" ]; then
        print_success "wenet ASR: 模型完整"
    else
        print_warn "wenet ASR: 模型不完整 (不影响核心TTS功能)"
        [ ! -f "$WN_DIR/final.pt" ] && print_warn "缺失: $WN_DIR/final.pt"
        [ ! -f "$WN_DIR/train.yaml" ] && print_warn "缺失: $WN_DIR/train.yaml"
    fi

    if [ $model_errors -gt 0 ]; then
        print_warn "部分模型文件缺失，对应服务可能无法正常推理"
    fi

    # 创建必要目录
    print_step "创建必要目录..."
    mkdir -p "$SCRIPT_DIR/outputs"
    mkdir -p "$SCRIPT_DIR/uploads"
    mkdir -p "$SCRIPT_DIR/logs"
    mkdir -p "$SCRIPT_DIR/records"
    print_success "目录检查完成"

    # 构建启动命令
    local protocol="http"
    local cmd=(python -m uvicorn backend.main:app --host "$HOST" --port "$PORT")
    
    # HTTPS 配置
    if [ -n "$SSL_CERT" ] && [ -n "$SSL_KEY" ]; then
        if [ ! -f "$SSL_CERT" ]; then
            print_error "SSL 证书不存在: $SSL_CERT"
            print_info "生成自签名证书: openssl req -x509 -newkey rsa:2048 -keyout $SSL_KEY -out $SSL_CERT -days 365 -nodes"
            exit 1
        fi
        if [ ! -f "$SSL_KEY" ]; then
            print_error "SSL 私钥不存在: $SSL_KEY"
            exit 1
        fi
        cmd+=(--ssl-certfile "$SSL_CERT" --ssl-keyfile "$SSL_KEY")
        protocol="https"
        print_info "HTTPS 模式: 已启用"
        print_info "证书文件: $SSL_CERT"
    fi
    
    if [ "$RELOAD" = true ]; then
        cmd+=(--reload)
        print_info "开发模式: 已启用自动重载"
    fi
    
    # 导出协议变量和并发配置
    export SERVER_PROTOCOL="$protocol"
    export MAX_CONCURRENT_MODELS="$MAX_CONCURRENT_MODELS"

    echo ""
    print_step "启动 Uvicorn 服务..."
    print_info "命令: ${cmd[*]}"
    print_info "GPU设备: $MAIN_GPU"
    print_info "日志文件: $LOG_FILE"
    print_info "预加载模型: $PRELOAD_MODELS"
    print_info "最大并发模型数: $MAX_CONCURRENT_MODELS"
    echo ""

    # 后台启动并记录 PID
    cd "$SCRIPT_DIR"
    CUDA_VISIBLE_DEVICES="$MAIN_GPU" GPU_ID="$MAIN_GPU" nohup "${cmd[@]}" >> "$LOG_FILE" 2>&1 &
    local new_pid=$!
    echo "$new_pid" > "$PID_FILE"

    print_step "等待服务启动 (PID: $new_pid)..."
    echo ""

    # 构建健康检查 URL
    local health_url="${protocol}://$HOST:$PORT/health"
    local curl_opts="-s"
    if [ "$protocol" = "https" ]; then
        curl_opts="-sk"  # -k: 跳过 SSL 证书验证（自签名证书）
    fi

    # 等待服务启动
    local count=0
    local spin='⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'
    while [ $count -lt 60 ]; do
        if kill -0 "$new_pid" 2>/dev/null; then
            if curl $curl_opts "$health_url" >/dev/null 2>&1; then
                echo ""
                echo ""
                echo "========================================"
                echo "      服务启动成功"
                echo "========================================"
                print_success "主服务状态: 运行中 (PID: $new_pid)"
                print_info "服务地址: ${protocol}://$HOST:$PORT"
                print_info "API文档:  ${protocol}://$HOST:$PORT/docs"
                print_info "前端页面: ${protocol}://$HOST:$PORT"
                print_info "日志文件: $LOG_FILE"
                echo ""

                # 自动启动 OmniVoice 独立服务
                echo "----------------------------------------"
                print_step "正在检查 OmniVoice 独立服务..."
                
                if is_omnivoice_running; then
                    local ov_pid=$(cat "$OMNIVOICE_PID_FILE" 2>/dev/null)
                    print_success "OmniVoice 已在运行 (PID: $ov_pid)"
                else
                    if [ ! -f "$OMNIVOICE_SCRIPT" ]; then
                        print_warn "OmniVoice 服务脚本不存在，跳过"
                    else
                        print_step "正在启动 OmniVoice 独立服务 (端口: $OMNIVOICE_PORT, GPU: $OMNIVOICE_GPU, 预加载: $PRELOAD_OMNIVOICE)..."
                        
                        cd "$SCRIPT_DIR"
                        CUDA_VISIBLE_DEVICES="$OMNIVOICE_GPU" GPU_ID="$OMNIVOICE_GPU" nohup python "$OMNIVOICE_SCRIPT" >> "$OMNIVOICE_LOG_FILE" 2>&1 &
                        local ov_pid=$!
                        echo "$ov_pid" > "$OMNIVOICE_PID_FILE"
                        
                        # 根据预加载设置调整等待时间
                        if [ "$PRELOAD_OMNIVOICE" = "1" ]; then
                            local ov_max_wait=$START_WAIT_PRELOAD  # 预加载需要更长时间
                        else
                            local ov_max_wait=$START_WAIT_NO_PRELOAD  # 不预加载启动更快
                        fi
                        
                        local ov_count=0
                        local ov_spin='⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'
                        local ov_ready=0
                        while [ $ov_count -lt $ov_max_wait ]; do
                            if kill -0 "$ov_pid" 2>/dev/null; then
                                if curl -s "http://127.0.0.1:$OMNIVOICE_PORT/health" >/dev/null 2>&1; then
                                    print_success "OmniVoice 已启动 (PID: $ov_pid, 端口: $OMNIVOICE_PORT)"
                                    ov_ready=1
                                    break
                                fi
                            else
                                print_warn "OmniVoice 启动失败，请检查日志: $OMNIVOICE_LOG_FILE"
                                rm -f "$OMNIVOICE_PID_FILE"
                                break
                            fi
                            sleep 1
                            ov_count=$((ov_count + 1))
                            printf "\r  %s  OmniVoice 启动中... %d/%d 秒" "${ov_spin:$((ov_count % 10)):1}" "$ov_count" "$ov_max_wait"
                        done
                        # 等满超时仍未就绪：进程还活着说明仍在后台启动（多服务并发冷启动时会互相拖慢），
                        # 不判死，提示用户稍后用 status 确认，避免误报失败。
                        if [ "$ov_ready" = "0" ] && kill -0 "$ov_pid" 2>/dev/null; then
                            echo ""
                            print_warn "OmniVoice ${ov_max_wait}s 内未就绪，进程仍在后台启动中"
                            print_info "稍后用 '$0 status' 确认；若长期未就绪请检查日志: $OMNIVOICE_LOG_FILE"
                            OMNIVOICE_START_OK=0
                        elif [ "$ov_ready" = "0" ]; then
                            # 进程已退出
                            OMNIVOICE_START_OK=0
                        else
                            OMNIVOICE_START_OK=1
                        fi
                    fi
                fi

                # 自动启动 PilotTTS 独立服务
                echo "----------------------------------------"
                print_step "正在检查 PilotTTS 独立服务..."

                if is_pilottts_running; then
                    local pt_pid=$(cat "$PILOTTS_PID_FILE" 2>/dev/null)
                    print_success "PilotTTS 已在运行 (PID: $pt_pid)"
                else
                    if [ ! -f "$PILOTTS_SCRIPT" ]; then
                        print_warn "PilotTTS 服务脚本不存在，跳过"
                    else
                        print_step "正在启动 PilotTTS 独立服务 (端口: $PILOTTS_PORT, GPU: $PILOTTS_GPU, 预加载: $PRELOAD_PILOTTS)..."

                        cd "$SCRIPT_DIR"
                        CUDA_VISIBLE_DEVICES="$PILOTTS_GPU" GPU_ID="$PILOTTS_GPU" nohup python "$PILOTTS_SCRIPT" >> "$PILOTTS_LOG_FILE" 2>&1 &
                        local pt_pid=$!
                        echo "$pt_pid" > "$PILOTTS_PID_FILE"

                        if [ "$PRELOAD_PILOTTS" = "1" ]; then
                            local pt_max_wait=$START_WAIT_PRELOAD
                        else
                            local pt_max_wait=$START_WAIT_NO_PRELOAD
                        fi

                        local pt_count=0
                        local pt_spin='⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'
                        local pt_ready=0
                        while [ $pt_count -lt $pt_max_wait ]; do
                            if kill -0 "$pt_pid" 2>/dev/null; then
                                if curl -s "http://127.0.0.1:$PILOTTS_PORT/health" >/dev/null 2>&1; then
                                    print_success "PilotTTS 已启动 (PID: $pt_pid, 端口: $PILOTTS_PORT)"
                                    pt_ready=1
                                    break
                                fi
                            else
                                print_warn "PilotTTS 启动失败，请检查日志: $PILOTTS_LOG_FILE"
                                rm -f "$PILOTTS_PID_FILE"
                                break
                            fi
                            sleep 1
                            pt_count=$((pt_count + 1))
                            printf "\r  %s  PilotTTS 启动中... %d/%d 秒" "${pt_spin:$((pt_count % 10)):1}" "$pt_count" "$pt_max_wait"
                        done
                        # 等满超时仍未就绪：进程还活着说明仍在后台启动，不判死，提示用 status 确认
                        if [ "$pt_ready" = "0" ] && kill -0 "$pt_pid" 2>/dev/null; then
                            echo ""
                            print_warn "PilotTTS ${pt_max_wait}s 内未就绪，进程仍在后台启动中"
                            print_info "稍后用 '$0 status' 确认；若长期未就绪请检查日志: $PILOTTS_LOG_FILE"
                            PILOTTS_START_OK=0
                        elif [ "$pt_ready" = "0" ]; then
                            PILOTTS_START_OK=0
                        else
                            PILOTTS_START_OK=1
                        fi
                    fi
                fi

                # 自动启动 GPT-SoVITS 独立服务
                echo "----------------------------------------"
                print_step "正在检查 GPT-SoVITS 独立服务..."

                if is_gptsovits_running; then
                    local gs_pid=$(cat "$GPTSOVITS_PID_FILE" 2>/dev/null)
                    print_success "GPT-SoVITS 已在运行 (PID: $gs_pid)"
                else
                    if [ ! -f "$GPTSOVITS_SCRIPT" ]; then
                        print_warn "GPT-SoVITS 服务脚本不存在，跳过"
                    else
                        print_step "正在启动 GPT-SoVITS 独立服务 (端口: $GPTSOVITS_PORT, GPU: $GPTSOVITS_GPU, 预加载: $PRELOAD_GPTSOVITS)..."

                        cd "$SCRIPT_DIR"
                        CUDA_VISIBLE_DEVICES="$GPTSOVITS_GPU" GPU_ID="$GPTSOVITS_GPU" nohup python "$GPTSOVITS_SCRIPT" >> "$GPTSOVITS_LOG_FILE" 2>&1 &
                        local gs_pid=$!
                        echo "$gs_pid" > "$GPTSOVITS_PID_FILE"

                        if [ "$PRELOAD_GPTSOVITS" = "1" ]; then
                            local gs_max_wait=$START_WAIT_PRELOAD
                        else
                            local gs_max_wait=$START_WAIT_NO_PRELOAD
                        fi

                        local gs_count=0
                        local gs_spin='⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'
                        local gs_ready=0
                        while [ $gs_count -lt $gs_max_wait ]; do
                            if kill -0 "$gs_pid" 2>/dev/null; then
                                if curl -s "http://127.0.0.1:$GPTSOVITS_PORT/health" >/dev/null 2>&1; then
                                    print_success "GPT-SoVITS 已启动 (PID: $gs_pid, 端口: $GPTSOVITS_PORT)"
                                    gs_ready=1
                                    break
                                fi
                            else
                                print_warn "GPT-SoVITS 启动失败，请检查日志: $GPTSOVITS_LOG_FILE"
                                rm -f "$GPTSOVITS_PID_FILE"
                                break
                            fi
                            sleep 1
                            gs_count=$((gs_count + 1))
                            printf "\r  %s  GPT-SoVITS 启动中... %d/%d 秒" "${gs_spin:$((gs_count % 10)):1}" "$gs_count" "$gs_max_wait"
                        done
                        # 等满超时仍未就绪：进程还活着说明仍在后台启动，不判死，提示用 status 确认
                        if [ "$gs_ready" = "0" ] && kill -0 "$gs_pid" 2>/dev/null; then
                            echo ""
                            print_warn "GPT-SoVITS ${gs_max_wait}s 内未就绪，进程仍在后台启动中"
                            print_info "稍后用 '$0 status' 确认；若长期未就绪请检查日志: $GPTSOVITS_LOG_FILE"
                            GPTSOVITS_START_OK=0
                        elif [ "$gs_ready" = "0" ]; then
                            GPTSOVITS_START_OK=0
                        else
                            GPTSOVITS_START_OK=1
                        fi
                    fi
                fi

                # 自动启动 Fish-Speech 独立服务
                echo "----------------------------------------"
                print_step "正在检查 Fish-Speech 独立服务..."

                if is_fishspeech_running; then
                    local fs_pid=$(cat "$FISHSPEECH_PID_FILE" 2>/dev/null)
                    print_success "Fish-Speech 已在运行 (PID: $fs_pid)"
                else
                    if [ ! -f "$FISHSPEECH_SCRIPT" ]; then
                        print_warn "Fish-Speech 服务脚本不存在，跳过"
                    else
                        print_step "正在启动 Fish-Speech 独立服务 (端口: $FISHSPEECH_PORT, GPU: $FISHSPEECH_GPU, 预加载: $PRELOAD_FISHSPEECH)..."

                        cd "$SCRIPT_DIR"
                        CUDA_VISIBLE_DEVICES="$FISHSPEECH_GPU" GPU_ID="$FISHSPEECH_GPU" nohup python "$FISHSPEECH_SCRIPT" >> "$FISHSPEECH_LOG_FILE" 2>&1 &
                        local fs_pid=$!
                        echo "$fs_pid" > "$FISHSPEECH_PID_FILE"

                        if [ "$PRELOAD_FISHSPEECH" = "1" ]; then
                            local fs_max_wait=$START_WAIT_PRELOAD
                        else
                            local fs_max_wait=$START_WAIT_NO_PRELOAD
                        fi

                        local fs_count=0
                        local fs_spin='⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'
                        local fs_ready=0
                        while [ $fs_count -lt $fs_max_wait ]; do
                            if kill -0 "$fs_pid" 2>/dev/null; then
                                if curl -s "http://127.0.0.1:$FISHSPEECH_PORT/health" >/dev/null 2>&1; then
                                    print_success "Fish-Speech 已启动 (PID: $fs_pid, 端口: $FISHSPEECH_PORT)"
                                    fs_ready=1
                                    break
                                fi
                            else
                                print_warn "Fish-Speech 启动失败，请检查日志: $FISHSPEECH_LOG_FILE"
                                rm -f "$FISHSPEECH_PID_FILE"
                                break
                            fi
                            sleep 1
                            fs_count=$((fs_count + 1))
                            printf "\r  %s  Fish-Speech 启动中... %d/%d 秒" "${fs_spin:$((fs_count % 10)):1}" "$fs_count" "$fs_max_wait"
                        done
                        if [ "$fs_ready" = "0" ] && kill -0 "$fs_pid" 2>/dev/null; then
                            echo ""
                            print_warn "Fish-Speech ${fs_max_wait}s 内未就绪，进程仍在后台启动中"
                            print_info "稍后用 '$0 status' 确认；若长期未就绪请检查日志: $FISHSPEECH_LOG_FILE"
                            FISHSPEECH_START_OK=0
                        elif [ "$fs_ready" = "0" ]; then
                            FISHSPEECH_START_OK=0
                        else
                            FISHSPEECH_START_OK=1
                        fi
                    fi
                fi

                # 自动启动 CosyVoice 独立服务
                echo "----------------------------------------"
                print_step "正在检查 CosyVoice 独立服务..."

                if is_cosyvoice_running; then
                    local cv_pid=$(cat "$COSYVOICE_PID_FILE" 2>/dev/null)
                    print_success "CosyVoice 已在运行 (PID: $cv_pid)"
                else
                    if [ ! -f "$COSYVOICE_SCRIPT" ]; then
                        print_warn "CosyVoice 服务脚本不存在，跳过"
                    else
                        print_step "正在启动 CosyVoice 独立服务 (端口: $COSYVOICE_PORT, GPU: $COSYVOICE_GPU, 预加载: $PRELOAD_COSYVOICE)..."

                        cd "$SCRIPT_DIR"
                        CUDA_VISIBLE_DEVICES="$COSYVOICE_GPU" GPU_ID="$COSYVOICE_GPU" nohup python "$COSYVOICE_SCRIPT" >> "$COSYVOICE_LOG_FILE" 2>&1 &
                        local cv_pid=$!
                        echo "$cv_pid" > "$COSYVOICE_PID_FILE"

                        # 根据预加载设置调整等待时间
                        if [ "$PRELOAD_COSYVOICE" = "1" ]; then
                            local cv_max_wait=$START_WAIT_PRELOAD  # 预加载需要更长时间
                        else
                            local cv_max_wait=$START_WAIT_NO_PRELOAD   # 不预加载启动更快
                        fi

                        local cv_count=0
                        local cv_spin='⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'
                        local cv_ready=0
                        while [ $cv_count -lt $cv_max_wait ]; do
                            if kill -0 "$cv_pid" 2>/dev/null; then
                                if curl -s "http://127.0.0.1:$COSYVOICE_PORT/health" >/dev/null 2>&1; then
                                    print_success "CosyVoice 已启动 (PID: $cv_pid, 端口: $COSYVOICE_PORT)"
                                    cv_ready=1
                                    break
                                fi
                            else
                                print_warn "CosyVoice 启动失败，请检查日志: $COSYVOICE_LOG_FILE"
                                rm -f "$COSYVOICE_PID_FILE"
                                break
                            fi
                            sleep 1
                            cv_count=$((cv_count + 1))
                            printf "\r  %s  CosyVoice 启动中... %d/%d 秒" "${cv_spin:$((cv_count % 10)):1}" "$cv_count" "$cv_max_wait"
                        done
                        # 等满超时仍未就绪：进程还活着说明仍在后台启动（多服务并发冷启动时会互相拖慢），
                        # 不判死，提示用户稍后用 status 确认，避免误报失败。
                        if [ "$cv_ready" = "0" ] && kill -0 "$cv_pid" 2>/dev/null; then
                            echo ""
                            print_warn "CosyVoice ${cv_max_wait}s 内未就绪，进程仍在后台启动中"
                            print_info "稍后用 '$0 status' 确认；若长期未就绪请检查日志: $COSYVOICE_LOG_FILE"
                            COSYVOICE_START_OK=0
                        elif [ "$cv_ready" = "0" ]; then
                            # 进程已退出
                            COSYVOICE_START_OK=0
                        else
                            COSYVOICE_START_OK=1
                        fi
                    fi
                fi

                echo ""
                # 汇总各服务实际就绪状态（健康检查通过=1，超时/失败=0）
                # 任意子服务未就绪时，明确提示而非打印「全部成功」误导
                local _all_ok=1
                for _v in OMNIVOICE_START_OK COSYVOICE_START_OK PILOTTS_START_OK GPTSOVITS_START_OK FISHSPEECH_START_OK; do
                    eval "_val=\${$_v:-1}"
                    [ "$_val" = "0" ] && _all_ok=0
                done

                echo "========================================"
                if [ "$_all_ok" = "1" ]; then
                    echo "      全部服务启动成功"
                else
                    echo "      部分服务未就绪，请查看上方提示"
                fi
                echo "========================================"
                print_success "主服务状态: 运行中 (PID: $new_pid, 端口: $PORT, GPU: $MAIN_GPU)"
                # 子服务状态按实际就绪标记打印（运行中 / 未就绪）
                if [ "${OMNIVOICE_START_OK:-1}" = "1" ]; then
                    print_info "OmniVoice 服务: 运行中 (PID: $ov_pid, 端口: $OMNIVOICE_PORT, GPU: $OMNIVOICE_GPU)"
                else
                    print_error "OmniVoice 服务: 未就绪（启动超时或失败，见上方提示）"
                fi
                if [ "${COSYVOICE_START_OK:-1}" = "1" ]; then
                    print_info "CosyVoice 服务: 运行中 (PID: $cv_pid, 端口: $COSYVOICE_PORT, GPU: $COSYVOICE_GPU)"
                else
                    print_error "CosyVoice 服务: 未就绪（启动超时或失败，见上方提示）"
                fi
                if [ "${PILOTTS_START_OK:-1}" = "1" ]; then
                    print_info "PilotTTS 服务: 运行中 (PID: $pt_pid, 端口: $PILOTTS_PORT, GPU: $PILOTTS_GPU)"
                else
                    print_error "PilotTTS 服务: 未就绪（启动超时或失败，见上方提示）"
                fi
                if [ "${GPTSOVITS_START_OK:-1}" = "1" ]; then
                    print_info "GPT-SoVITS 服务: 运行中 (PID: $gs_pid, 端口: $GPTSOVITS_PORT, GPU: $GPTSOVITS_GPU)"
                else
                    print_error "GPT-SoVITS 服务: 未就绪（启动超时或失败，见上方提示）"
                fi
                if [ "${FISHSPEECH_START_OK:-1}" = "1" ]; then
                    print_info "Fish-Speech 服务: 运行中 (PID: $fs_pid, 端口: $FISHSPEECH_PORT, GPU: $FISHSPEECH_GPU)"
                else
                    print_error "Fish-Speech 服务: 未就绪（启动超时或失败，见上方提示）"
                fi
                print_info "前端页面: ${protocol}://$HOST:$PORT"
                print_info "API文档:  ${protocol}://$HOST:$PORT/docs"
                echo ""
                print_info "常用命令:"
                echo "  查看状态: $0 status"
                echo "  查看日志: tail -f $LOG_FILE"
                echo "  停止服务: $0 stop"
                echo ""
                print_info "OmniVoice 独立服务:"
                echo "  单独启动: $0 start-omnivoice"
                echo "  单独停止: $0 stop-omnivoice"
                echo "  查看状态: $0 status-omnivoice"
                echo "  查看日志: tail -f $OMNIVOICE_LOG_FILE"
                echo ""
                print_info "CosyVoice 独立服务:"
                echo "  单独启动: $0 start-cosyvoice"
                echo "  单独停止: $0 stop-cosyvoice"
                echo "  查看状态: $0 status-cosyvoice"
                echo "  查看日志: tail -f $COSYVOICE_LOG_FILE"
                echo ""
                print_info "PilotTTS 独立服务:"
                echo "  单独启动: $0 start-pilottts"
                echo "  单独停止: $0 stop-pilottts"
                echo "  查看状态: $0 status-pilottts"
                echo "  查看日志: tail -f $PILOTTS_LOG_FILE"
                echo ""
                print_info "GPT-SoVITS 独立服务:"
                echo "  单独启动: $0 start-gptsovits"
                echo "  单独停止: $0 stop-gptsovits"
                echo "  查看状态: $0 status-gptsovits"
                echo "  查看日志: tail -f $GPTSOVITS_LOG_FILE"
                echo "========================================"
                exit 0
            fi
        else
            echo ""
            echo ""
            print_error "服务启动失败 - 进程已退出"
            print_info "查看错误日志:"
            echo "  tail -n 50 $LOG_FILE"
            rm -f "$PID_FILE"
            exit 1
        fi
        sleep 1
        count=$((count + 1))
        printf "\r  %s  等待中... %d/60 秒" "${spin:$((count % 10)):1}" "$count"
    done

    echo ""
    echo ""
    print_warn "服务启动超时 (30秒)"
    print_info "可能原因:"
    echo "  1. 模型加载时间较长"
    echo "  2. 端口被占用: $PORT"
    echo "  3. 依赖包缺失"
    echo ""
    print_info "查看详细日志:"
    echo "  tail -n 100 $LOG_FILE"
    exit 1
}
do_stop() {
    echo ""
    echo "========================================"
    echo "      VersTTS 服务停止"
    echo "========================================"
    echo ""

    # 停止主服务
    local pid
    pid=$(get_pid)

    if [ -z "$pid" ]; then
        print_info "主服务未运行"
    else
        if ! kill -0 "$pid" 2>/dev/null; then
            print_warn "主服务未运行 (PID: $pid 已失效)"
            rm -f "$PID_FILE"
        else
            print_info "发现主服务运行中 (PID: $pid)"
            print_step "正在停止主服务..."

            kill "$pid" 2>/dev/null || true

            local count=0
            local spin='⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'
            while [ $count -lt 10 ]; do
                if ! kill -0 "$pid" 2>/dev/null; then
                    echo ""
                    print_success "主服务已停止"
                    rm -f "$PID_FILE"
                    break
                fi
                sleep 1
                count=$((count + 1))
                printf "\r  %s  等待中... %d/10 秒" "${spin:$((count % 10)):1}" "$count"
            done

            if kill -0 "$pid" 2>/dev/null; then
                echo ""
                print_warn "优雅终止超时，执行强制停止..."
                kill -9 "$pid" 2>/dev/null || true
                sleep 1
                print_success "主服务已强制停止"
                rm -f "$PID_FILE"
            fi
        fi
    fi

    echo ""
    # 停止 OmniVoice 独立服务
    do_stop_omnivoice

    echo ""
    # 停止 PilotTTS 独立服务
    do_stop_pilottts

    echo ""
    # 停止 GPT-SoVITS 独立服务
    do_stop_gptsovits

    echo ""
    # 停止 CosyVoice 独立服务
    do_stop_cosyvoice

    echo ""
    # 停止 Fish-Speech 独立服务
    do_stop_fishspeech
}

# 重启服务
do_restart() {
    echo ""
    echo "========================================"
    echo "      VersTTS 服务重启"
    echo "========================================"
    echo ""

    local pid
    pid=$(get_pid)

    echo "【阶段 1/7】停止主服务"
    echo "----------------------------------------"

    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        print_info "发现运行中的主服务 (PID: $pid)"
        print_step "正在停止主服务..."
        kill "$pid" 2>/dev/null || true

        local count=0
        while [ $count -lt 10 ]; do
            if ! kill -0 "$pid" 2>/dev/null; then
                print_success "主服务已停止"
                break
            fi
            sleep 1
            count=$((count + 1))
            echo -n "."
        done

        if kill -0 "$pid" 2>/dev/null; then
            echo ""
            print_warn "执行强制停止..."
            kill -9 "$pid" 2>/dev/null || true
            print_success "主服务已强制停止"
        fi

        rm -f "$PID_FILE"
        echo ""
        print_step "等待 2 秒确保端口释放..."
        sleep 2
    else
        if [ -n "$pid" ]; then
            print_warn "发现残留PID文件 (PID: $pid)，进程已不存在"
            rm -f "$PID_FILE"
        else
            print_info "没有运行中的主服务"
        fi
    fi

    print_success "主服务停止完成"
    echo ""

    # 阶段1.5: 停止 OmniVoice
    echo "【阶段 2/7】停止 OmniVoice 服务"
    echo "----------------------------------------"
    do_stop_omnivoice

    # 阶段1.6: 停止 PilotTTS
    echo "【阶段 3/7】停止 PilotTTS 服务"
    echo "----------------------------------------"
    do_stop_pilottts

    # 阶段1.7: 停止 CosyVoice
    echo "【阶段 4/7】停止 CosyVoice 服务"
    echo "----------------------------------------"
    do_stop_cosyvoice

    # 阶段1.8: 停止 GPT-SoVITS
    echo "【阶段 5/7】停止 GPT-SoVITS 服务"
    echo "----------------------------------------"
    do_stop_gptsovits

    # 阶段1.9: 停止 Fish-Speech
    echo "【阶段 6/7】停止 Fish-Speech 服务"
    echo "----------------------------------------"
    do_stop_fishspeech

    echo ""
    print_success "全部服务停止完成"
    echo ""

    # 阶段2: 启动服务（会自动启动 OmniVoice、PilotTTS、CosyVoice、GPT-SoVITS 和 Fish-Speech）
    echo "【阶段 7/7】启动全部服务"
    echo "----------------------------------------"
    do_start
}

# 查看状态
do_status() {
    echo ""
    echo "========================================"
    echo "      VersTTS 服务状态"
    echo "========================================"
    echo ""

    local pid
    pid=$(get_pid)

    if [ -z "$pid" ]; then
        print_warn "服务未运行"
        print_info "PID 文件: 不存在"
        echo ""
        print_info "启动服务:"
        echo "  $0 start"
        exit 0
    fi

    if kill -0 "$pid" 2>/dev/null; then
        # 检测是否启用 HTTPS（检查 uvicorn 进程参数）
        local proc_protocol="http"
        if ps -p "$pid" -o args= | grep -q "\-\-ssl"; then
            proc_protocol="https"
        fi
        
        local status_curl_opts="-s"
        if [ "$proc_protocol" = "https" ]; then
            status_curl_opts="-sk"
        fi
        local status_health_url="${proc_protocol}://$HOST:$PORT/health"

        print_success "主服务运行中"
        print_info "进程PID:  $pid"
        print_info "启动时间: $(ps -o lstart= -p "$pid" 2>/dev/null || echo "未知")"
        print_info "运行时长: $(ps -o etime= -p "$pid" 2>/dev/null || echo "未知")"
        echo ""

        print_step "执行主服务健康检查..."
        local health
        health=$(curl $status_curl_opts "$status_health_url" 2>/dev/null || echo "")
        if [ -n "$health" ]; then
            print_success "主服务健康检查通过"
            echo ""
            echo "$health" | python -m json.tool 2>/dev/null || echo "$health"
        else
            print_error "主服务健康检查失败"
            print_info "服务进程存在但无法响应请求"
        fi
        echo ""
        print_info "主服务地址:"
        echo "  前端页面: http://$HOST:$PORT"
        echo "  API文档:  http://$HOST:$PORT/docs"
        echo "  健康检查: http://$HOST:$PORT/health"
        print_info "主服务日志: $LOG_FILE"
        echo ""
    else
        print_error "主服务未运行 (PID 文件残留)"
        print_warn "PID 文件中记录的进程已失效: $pid"
        print_step "清理残留的PID文件..."
        rm -f "$PID_FILE"
        print_success "清理完成"
        echo ""
        print_info "启动主服务:"
        echo "  $0 start"
    fi

    # 检查 OmniVoice 服务状态
    echo "----------------------------------------"
    print_step "OmniVoice 独立服务状态:"
    echo "----------------------------------------"
    if is_omnivoice_running; then
        local ov_pid=$(cat "$OMNIVOICE_PID_FILE" 2>/dev/null)
        print_success "OmniVoice 运行中 (PID: $ov_pid, 端口: $OMNIVOICE_PORT)"
        
        local ov_health
        ov_health=$(curl -s "http://127.0.0.1:$OMNIVOICE_PORT/health" 2>/dev/null || echo "")
        if [ -n "$ov_health" ]; then
            print_success "OmniVoice 健康检查通过"
            echo "$ov_health" | python -m json.tool 2>/dev/null || echo "$ov_health"
        else
            print_error "OmniVoice 健康检查失败"
        fi
        print_info "OmniVoice 日志: $OMNIVOICE_LOG_FILE"
    else
        print_warn "OmniVoice 未运行"
    fi

    # 检查 PilotTTS 服务状态
    echo "----------------------------------------"
    print_step "PilotTTS 独立服务状态:"
    echo "----------------------------------------"
    if is_pilottts_running; then
        local pt_pid=$(cat "$PILOTTS_PID_FILE" 2>/dev/null)
        print_success "PilotTTS 运行中 (PID: $pt_pid, 端口: $PILOTTS_PORT)"

        local pt_health
        pt_health=$(curl -s "http://127.0.0.1:$PILOTTS_PORT/health" 2>/dev/null || echo "")
        if [ -n "$pt_health" ]; then
            print_success "PilotTTS 健康检查通过"
            echo "$pt_health" | python -m json.tool 2>/dev/null || echo "$pt_health"
        else
            print_error "PilotTTS 健康检查失败"
        fi
        print_info "PilotTTS 日志: $PILOTTS_LOG_FILE"
    else
        print_warn "PilotTTS 未运行"
    fi

    # 检查 CosyVoice 服务状态
    echo "----------------------------------------"
    print_step "CosyVoice 独立服务状态:"
    echo "----------------------------------------"
    if is_cosyvoice_running; then
        local cv_pid=$(cat "$COSYVOICE_PID_FILE" 2>/dev/null)
        print_success "CosyVoice 运行中 (PID: $cv_pid, 端口: $COSYVOICE_PORT)"

        local cv_health
        cv_health=$(curl -s "http://127.0.0.1:$COSYVOICE_PORT/health" 2>/dev/null || echo "")
        if [ -n "$cv_health" ]; then
            print_success "CosyVoice 健康检查通过"
            echo "$cv_health" | python -m json.tool 2>/dev/null || echo "$cv_health"
        else
            print_error "CosyVoice 健康检查失败"
        fi
        print_info "CosyVoice 日志: $COSYVOICE_LOG_FILE"
    else
        print_warn "CosyVoice 未运行"
    fi

    # 检查 GPT-SoVITS 服务状态
    echo "----------------------------------------"
    print_step "GPT-SoVITS 独立服务状态:"
    echo "----------------------------------------"
    if is_gptsovits_running; then
        local gs_pid=$(cat "$GPTSOVITS_PID_FILE" 2>/dev/null)
        print_success "GPT-SoVITS 运行中 (PID: $gs_pid, 端口: $GPTSOVITS_PORT)"

        local gs_health
        gs_health=$(curl -s "http://127.0.0.1:$GPTSOVITS_PORT/health" 2>/dev/null || echo "")
        if [ -n "$gs_health" ]; then
            print_success "GPT-SoVITS 健康检查通过"
            echo "$gs_health" | python -m json.tool 2>/dev/null || echo "$gs_health"
        else
            print_error "GPT-SoVITS 健康检查失败"
        fi
        print_info "GPT-SoVITS 日志: $GPTSOVITS_LOG_FILE"
    else
        print_warn "GPT-SoVITS 未运行"
    fi

    # 检查 Fish-Speech 服务状态
    echo "----------------------------------------"
    print_step "Fish-Speech 独立服务状态:"
    echo "----------------------------------------"
    if is_fishspeech_running; then
        local fs_pid=$(cat "$FISHSPEECH_PID_FILE" 2>/dev/null)
        print_success "Fish-Speech 运行中 (PID: $fs_pid, 端口: $FISHSPEECH_PORT)"

        local fs_health
        fs_health=$(curl -s "http://127.0.0.1:$FISHSPEECH_PORT/health" 2>/dev/null || echo "")
        if [ -n "$fs_health" ]; then
            print_success "Fish-Speech 健康检查通过"
            echo "$fs_health" | python -m json.tool 2>/dev/null || echo "$fs_health"
        else
            print_error "Fish-Speech 健康检查失败"
        fi
        print_info "Fish-Speech 日志: $FISHSPEECH_LOG_FILE"
    else
        print_warn "Fish-Speech 未运行"
    fi
}

# ============ OmniVoice 独立服务管理 ============

# 检查 OmniVoice 服务是否运行
is_omnivoice_running() {
    if [ -f "$OMNIVOICE_PID_FILE" ]; then
        local pid=$(cat "$OMNIVOICE_PID_FILE" 2>/dev/null)
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            return 0
        fi
    fi
    return 1
}

# 启动 OmniVoice 独立服务
do_start_omnivoice() {
    echo ""
    echo "========================================"
    echo "      OmniVoice 独立服务启动"
    echo "========================================"
    echo ""

    # 检查是否已在运行
    if is_omnivoice_running; then
        local current_pid=$(cat "$OMNIVOICE_PID_FILE" 2>/dev/null)
        print_warn "OmniVoice 服务已在运行中"
        print_info "当前PID: $current_pid"
        print_info "服务端口: $OMNIVOICE_PORT"
        print_info "健康检查: http://127.0.0.1:$OMNIVOICE_PORT/health"
        exit 0
    fi

    # 检查服务脚本是否存在
    if [ ! -f "$OMNIVOICE_SCRIPT" ]; then
        print_error "OmniVoice 服务脚本不存在: $OMNIVOICE_SCRIPT"
        exit 1
    fi

    # 检查 transformers5 目录
    if [ ! -d "$SCRIPT_DIR/lib/transformers5" ]; then
        print_error "transformers5 目录不存在: $SCRIPT_DIR/lib/transformers5"
        print_info "请先安装: pip install --target $SCRIPT_DIR/lib/transformers5 transformers>=5.3.0"
        exit 1
    fi

    print_step "激活虚拟环境..."
    source "$VENV_PATH/bin/activate"
    print_success "虚拟环境已激活"

    # 加载离线模式环境变量
    if [ "$OFFLINE_MODE" = true ] && [ -f "$ENV_FILE" ]; then
        print_step "加载离线模式环境变量..."
        source "$ENV_FILE"
        print_success "离线模式已启用"
    fi

    # 导出环境变量供 OmniVoice 服务使用
    export OMNIVOICE_HOST
    export OMNIVOICE_PORT
    export TRANSFORMERS_OFFLINE
    export HF_HUB_OFFLINE
    export HF_HOME
    export HUGGINGFACE_HUB_CACHE
    export TRANSFORMERS_CACHE
    export PRELOAD_OMNIVOICE
    export IDLE_TIMEOUT
    export HEARTBEAT_INTERVAL
    export MAIN_HOST="127.0.0.1"
    export MAIN_PORT="$PORT"

    mkdir -p "$SCRIPT_DIR/logs"

    print_info "Python 版本: $(python --version 2>&1)"
    echo ""
    print_step "启动 OmniVoice 独立服务..."
    print_info "服务端口: $OMNIVOICE_PORT"
    print_info "GPU设备: $OMNIVOICE_GPU"
    print_info "日志文件: $OMNIVOICE_LOG_FILE"
    echo ""

    # 后台启动
    cd "$SCRIPT_DIR"
    CUDA_VISIBLE_DEVICES="$OMNIVOICE_GPU" GPU_ID="$OMNIVOICE_GPU" nohup python "$OMNIVOICE_SCRIPT" >> "$OMNIVOICE_LOG_FILE" 2>&1 &
    local new_pid=$!
    echo "$new_pid" > "$OMNIVOICE_PID_FILE"

    print_step "等待服务启动 (PID: $new_pid)..."
    echo ""

    # 等待服务启动
    # 等待服务启动：超时按预加载配置取值，可通过环境变量覆盖
    if [ "$PRELOAD_OMNIVOICE" = "1" ]; then
        local ov_max_wait=$START_WAIT_PRELOAD
    else
        local ov_max_wait=$START_WAIT_NO_PRELOAD
    fi
    local count=0
    local spin='⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'
    while [ $count -lt $ov_max_wait ]; do
        if kill -0 "$new_pid" 2>/dev/null; then
            if curl -s "http://127.0.0.1:$OMNIVOICE_PORT/health" >/dev/null 2>&1; then
                echo ""
                echo ""
                echo "========================================"
                echo "      OmniVoice 服务启动成功"
                echo "========================================"
                print_success "服务状态: 运行中"
                print_info "进程PID:  $new_pid"
                print_info "服务端口: $OMNIVOICE_PORT"
                print_info "健康检查: http://127.0.0.1:$OMNIVOICE_PORT/health"
                print_info "日志文件: $OMNIVOICE_LOG_FILE"
                echo ""
                print_info "常用命令:"
                echo "  查看状态: $0 status-omnivoice"
                echo "  查看日志: tail -f $OMNIVOICE_LOG_FILE"
                echo "  停止服务: $0 stop-omnivoice"
                echo "========================================"
                exit 0
            fi
        else
            echo ""
            echo ""
            print_error "OmniVoice 服务启动失败 - 进程已退出"
            print_info "查看错误日志:"
            echo "  tail -n 50 $OMNIVOICE_LOG_FILE"
            rm -f "$OMNIVOICE_PID_FILE"
            exit 1
        fi
        sleep 1
        count=$((count + 1))
        printf "\r  %s  等待中... %d/%d 秒" "${spin:$((count % 10)):1}" "$count" "$ov_max_wait"
    done

    echo ""
    echo ""
    print_warn "OmniVoice 服务启动超时 (${ov_max_wait}秒)"
    print_info "可能原因:"
    echo "  1. 模型加载时间较长 (OmniVoice 模型约 2.3GB)"
    echo "  2. 端口被占用: $OMNIVOICE_PORT"
    echo "  3. transformers5 依赖不完整"
    echo ""
    print_info "查看详细日志:"
    echo "  tail -n 100 $OMNIVOICE_LOG_FILE"
    exit 1
}

# 停止 OmniVoice 独立服务
do_stop_omnivoice() {
    echo "----------------------------------------"
    echo "      OmniVoice 独立服务停止"
    echo "----------------------------------------"

    local pid=""
    
    # 先尝试从 PID 文件获取 PID
    if is_omnivoice_running; then
        pid=$(cat "$OMNIVOICE_PID_FILE" 2>/dev/null)
        print_info "发现运行中的 OmniVoice 服务 (PID: $pid)"
    else
        # PID 文件不存在或进程已失效，尝试查找实际运行的进程
        pid=$(pgrep -f "python.*omnivoice_service.py" 2>/dev/null | head -1)
        if [ -n "$pid" ]; then
            print_warn "PID 文件不存在，但找到运行中的 OmniVoice 进程 (PID: $pid)"
        else
            print_info "OmniVoice 服务未运行"
            # 清理残留 PID 文件
            if [ -f "$OMNIVOICE_PID_FILE" ]; then
                rm -f "$OMNIVOICE_PID_FILE"
            fi
            return 0
        fi
    fi

    print_step "正在停止服务..."

    # 优雅终止
    kill "$pid" 2>/dev/null || true

    local count=0
    while [ $count -lt 10 ]; do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo ""
            print_success "OmniVoice 服务已停止"
            rm -f "$OMNIVOICE_PID_FILE"
            return 0
        fi
        sleep 1
        count=$((count + 1))
        echo -n "."
    done

    echo ""
    print_warn "优雅终止超时，执行强制停止..."
    kill -9 "$pid" 2>/dev/null || true
    sleep 1

    if ! kill -0 "$pid" 2>/dev/null; then
        print_success "OmniVoice 服务已强制停止"
        rm -f "$OMNIVOICE_PID_FILE"
    else
        print_error "无法停止 OmniVoice 服务 (PID: $pid)"
    fi
}

# 查看 OmniVoice 服务状态
do_status_omnivoice() {
    echo ""
    echo "========================================"
    echo "      OmniVoice 独立服务状态"
    echo "========================================"
    echo ""

    if ! is_omnivoice_running; then
        print_warn "OmniVoice 服务未运行"
        if [ -f "$OMNIVOICE_PID_FILE" ]; then
            rm -f "$OMNIVOICE_PID_FILE"
        fi
        echo ""
        print_info "启动服务:"
        echo "  $0 start-omnivoice"
        exit 0
    fi

    local pid=$(cat "$OMNIVOICE_PID_FILE" 2>/dev/null)
    print_success "OmniVoice 服务运行中"
    print_info "进程PID:  $pid"
    print_info "服务端口: $OMNIVOICE_PORT"
    echo ""

    print_step "执行健康检查..."
    local health
    health=$(curl -s "http://127.0.0.1:$OMNIVOICE_PORT/health" 2>/dev/null || echo "")
    if [ -n "$health" ]; then
        print_success "健康检查通过"
        echo ""
        echo "$health" | python -m json.tool 2>/dev/null || echo "$health"
    else
        print_error "健康检查失败"
        print_info "服务进程存在但无法响应请求"
    fi
    echo ""
    print_info "日志文件: $OMNIVOICE_LOG_FILE"
    print_info "查看日志: tail -f $OMNIVOICE_LOG_FILE"
}

# ============ CosyVoice 独立服务管理 ============

# 检查 CosyVoice 服务是否运行
is_cosyvoice_running() {
    if [ -f "$COSYVOICE_PID_FILE" ]; then
        local pid=$(cat "$COSYVOICE_PID_FILE" 2>/dev/null)
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            return 0
        fi
    fi
    return 1
}

# 启动 CosyVoice 独立服务
do_start_cosyvoice() {
    echo ""
    echo "========================================"
    echo "      CosyVoice 独立服务启动"
    echo "========================================"
    echo ""

    # 检查是否已在运行
    if is_cosyvoice_running; then
        local current_pid=$(cat "$COSYVOICE_PID_FILE" 2>/dev/null)
        print_warn "CosyVoice 服务已在运行中"
        print_info "当前PID: $current_pid"
        print_info "服务端口: $COSYVOICE_PORT"
        print_info "健康检查: http://127.0.0.1:$COSYVOICE_PORT/health"
        exit 0
    fi

    # 检查服务脚本是否存在
    if [ ! -f "$COSYVOICE_SCRIPT" ]; then
        print_error "CosyVoice 服务脚本不存在: $COSYVOICE_SCRIPT"
        exit 1
    fi

    # 检查 transformers4 目录
    if [ ! -d "$SCRIPT_DIR/lib/transformers4" ]; then
        print_error "transformers4 目录不存在: $SCRIPT_DIR/lib/transformers4"
        print_info "请先安装: pip install --target $SCRIPT_DIR/lib/transformers4 transformers==4.51.3"
        exit 1
    fi

    print_step "激活虚拟环境..."
    source "$VENV_PATH/bin/activate"
    print_success "虚拟环境已激活"

    # 加载离线模式环境变量
    if [ "$OFFLINE_MODE" = true ] && [ -f "$ENV_FILE" ]; then
        print_step "加载离线模式环境变量..."
        source "$ENV_FILE"
        print_success "离线模式已启用"
    fi

    # 导出环境变量供 CosyVoice 服务使用
    export COSYVOICE_HOST
    export COSYVOICE_PORT
    export TRANSFORMERS_OFFLINE
    export HF_HUB_OFFLINE
    export HF_HOME
    export HUGGINGFACE_HUB_CACHE
    export TRANSFORMERS_CACHE
    export PRELOAD_COSYVOICE
    export IDLE_TIMEOUT
    export HEARTBEAT_INTERVAL
    export MAIN_HOST="127.0.0.1"
    export MAIN_PORT="$PORT"

    mkdir -p "$SCRIPT_DIR/logs"

    print_info "Python 版本: $(python --version 2>&1)"
    echo ""
    print_step "启动 CosyVoice 独立服务..."
    print_info "服务端口: $COSYVOICE_PORT"
    print_info "GPU设备: $COSYVOICE_GPU"
    print_info "日志文件: $COSYVOICE_LOG_FILE"
    echo ""

    # 后台启动
    cd "$SCRIPT_DIR"
    CUDA_VISIBLE_DEVICES="$COSYVOICE_GPU" GPU_ID="$COSYVOICE_GPU" nohup python "$COSYVOICE_SCRIPT" >> "$COSYVOICE_LOG_FILE" 2>&1 &
    local new_pid=$!
    echo "$new_pid" > "$COSYVOICE_PID_FILE"

    print_step "等待服务启动 (PID: $new_pid)..."
    echo ""

    # 等待服务启动：超时按预加载配置取值，可通过环境变量覆盖
    if [ "$PRELOAD_COSYVOICE" = "1" ]; then
        local cv_max_wait=$START_WAIT_PRELOAD
    else
        local cv_max_wait=$START_WAIT_NO_PRELOAD
    fi
    local count=0
    local spin='⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'
    while [ $count -lt $cv_max_wait ]; do
        if kill -0 "$new_pid" 2>/dev/null; then
            if curl -s "http://127.0.0.1:$COSYVOICE_PORT/health" >/dev/null 2>&1; then
                echo ""
                echo ""
                echo "========================================"
                echo "      CosyVoice 服务启动成功"
                echo "========================================"
                print_success "服务状态: 运行中"
                print_info "进程PID:  $new_pid"
                print_info "服务端口: $COSYVOICE_PORT"
                print_info "健康检查: http://127.0.0.1:$COSYVOICE_PORT/health"
                print_info "日志文件: $COSYVOICE_LOG_FILE"
                echo ""
                print_info "常用命令:"
                echo "  查看状态: $0 status-cosyvoice"
                echo "  查看日志: tail -f $COSYVOICE_LOG_FILE"
                echo "  停止服务: $0 stop-cosyvoice"
                echo "========================================"
                exit 0
            fi
        else
            echo ""
            echo ""
            print_error "CosyVoice 服务启动失败 - 进程已退出"
            print_info "查看错误日志:"
            echo "  tail -n 50 $COSYVOICE_LOG_FILE"
            rm -f "$COSYVOICE_PID_FILE"
            exit 1
        fi
        sleep 1
        count=$((count + 1))
        printf "\r  %s  等待中... %d/%d 秒" "${spin:$((count % 10)):1}" "$count" "$cv_max_wait"
    done

    echo ""
    echo ""
    print_warn "CosyVoice 服务启动超时 (${cv_max_wait}秒)"
    print_info "可能原因:"
    echo "  1. 模型加载时间较长 (CosyVoice 模型较大)"
    echo "  2. 端口被占用: $COSYVOICE_PORT"
    echo "  3. transformers4 依赖不完整"
    echo ""
    print_info "查看详细日志:"
    echo "  tail -n 100 $COSYVOICE_LOG_FILE"
    exit 1
}

# 停止 CosyVoice 独立服务
do_stop_cosyvoice() {
    echo "----------------------------------------"
    echo "      CosyVoice 独立服务停止"
    echo "----------------------------------------"

    local pid=""
    
    # 先尝试从 PID 文件获取 PID
    if is_cosyvoice_running; then
        pid=$(cat "$COSYVOICE_PID_FILE" 2>/dev/null)
        print_info "发现运行中的 CosyVoice 服务 (PID: $pid)"
    else
        # PID 文件不存在或进程已失效，尝试查找实际运行的进程
        pid=$(pgrep -f "python.*cosyvoice_service.py" 2>/dev/null | head -1)
        if [ -n "$pid" ]; then
            print_warn "PID 文件不存在，但找到运行中的 CosyVoice 进程 (PID: $pid)"
        else
            print_info "CosyVoice 服务未运行"
            # 清理残留 PID 文件
            if [ -f "$COSYVOICE_PID_FILE" ]; then
                rm -f "$COSYVOICE_PID_FILE"
            fi
            return 0
        fi
    fi

    print_step "正在停止服务..."

    # 优雅终止
    kill "$pid" 2>/dev/null || true

    local count=0
    while [ $count -lt 10 ]; do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo ""
            print_success "CosyVoice 服务已停止"
            rm -f "$COSYVOICE_PID_FILE"
            return 0
        fi
        sleep 1
        count=$((count + 1))
        echo -n "."
    done

    echo ""
    print_warn "优雅终止超时，执行强制停止..."
    kill -9 "$pid" 2>/dev/null || true
    sleep 1

    if ! kill -0 "$pid" 2>/dev/null; then
        print_success "CosyVoice 服务已强制停止"
        rm -f "$COSYVOICE_PID_FILE"
    else
        print_error "无法停止 CosyVoice 服务 (PID: $pid)"
    fi
}

# 查看 CosyVoice 服务状态
do_status_cosyvoice() {
    echo ""
    echo "========================================"
    echo "      CosyVoice 独立服务状态"
    echo "========================================"
    echo ""

    if ! is_cosyvoice_running; then
        print_warn "CosyVoice 服务未运行"
        if [ -f "$COSYVOICE_PID_FILE" ]; then
            rm -f "$COSYVOICE_PID_FILE"
        fi
        echo ""
        print_info "启动服务:"
        echo "  $0 start-cosyvoice"
        exit 0
    fi

    local pid=$(cat "$COSYVOICE_PID_FILE" 2>/dev/null)
    print_success "CosyVoice 服务运行中"
    print_info "进程PID:  $pid"
    print_info "服务端口: $COSYVOICE_PORT"
    echo ""

    print_step "执行健康检查..."
    local health
    health=$(curl -s "http://127.0.0.1:$COSYVOICE_PORT/health" 2>/dev/null || echo "")
    if [ -n "$health" ]; then
        print_success "健康检查通过"
        echo ""
        echo "$health" | python -m json.tool 2>/dev/null || echo "$health"
    else
        print_error "健康检查失败"
        print_info "服务进程存在但无法响应请求"
    fi
    echo ""
    print_info "日志文件: $COSYVOICE_LOG_FILE"
    print_info "查看日志: tail -f $COSYVOICE_LOG_FILE"
}

# ============ PilotTTS 独立服务管理 ============

# 检查 PilotTTS 服务是否运行
is_pilottts_running() {
    if [ -f "$PILOTTS_PID_FILE" ]; then
        local pid=$(cat "$PILOTTS_PID_FILE" 2>/dev/null)
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            return 0
        fi
    fi
    return 1
}

# 启动 PilotTTS 独立服务
do_start_pilottts() {
    echo ""
    echo "========================================"
    echo "      PilotTTS 独立服务启动"
    echo "========================================"
    echo ""

    # 检查是否已在运行
    if is_pilottts_running; then
        local current_pid=$(cat "$PILOTTS_PID_FILE" 2>/dev/null)
        print_warn "PilotTTS 服务已在运行中"
        print_info "当前PID: $current_pid"
        print_info "服务端口: $PILOTTS_PORT"
        print_info "健康检查: http://127.0.0.1:$PILOTTS_PORT/health"
        exit 0
    fi

    # 检查服务脚本是否存在
    if [ ! -f "$PILOTTS_SCRIPT" ]; then
        print_error "PilotTTS 服务脚本不存在: $PILOTTS_SCRIPT"
        exit 1
    fi

    # 检查 transformers4 目录
    if [ ! -d "$SCRIPT_DIR/lib/transformers4" ]; then
        print_error "transformers4 目录不存在: $SCRIPT_DIR/lib/transformers4"
        print_info "请先安装: pip install --target $SCRIPT_DIR/lib/transformers4 transformers==4.51.3"
        exit 1
    fi

    print_step "激活虚拟环境..."
    source "$VENV_PATH/bin/activate"
    print_success "虚拟环境已激活"

    # 加载离线模式环境变量
    if [ "$OFFLINE_MODE" = true ] && [ -f "$ENV_FILE" ]; then
        print_step "加载离线模式环境变量..."
        source "$ENV_FILE"
        print_success "离线模式已启用"
    fi

    # 导出环境变量供 PilotTTS 服务使用
    export PILOTTS_HOST
    export PILOTTS_PORT
    export TRANSFORMERS_OFFLINE
    export HF_HUB_OFFLINE
    export HF_HOME
    export HUGGINGFACE_HUB_CACHE
    export TRANSFORMERS_CACHE
    export PRELOAD_PILOTTS
    export IDLE_TIMEOUT
    export HEARTBEAT_INTERVAL
    export MAIN_HOST="127.0.0.1"
    export MAIN_PORT="$PORT"

    mkdir -p "$SCRIPT_DIR/logs"

    print_info "Python 版本: $(python --version 2>&1)"
    echo ""
    print_step "启动 PilotTTS 独立服务..."
    print_info "服务端口: $PILOTTS_PORT"
    print_info "GPU设备: $PILOTTS_GPU"
    print_info "日志文件: $PILOTTS_LOG_FILE"
    echo ""

    # 后台启动
    cd "$SCRIPT_DIR"
    CUDA_VISIBLE_DEVICES="$PILOTTS_GPU" GPU_ID="$PILOTTS_GPU" nohup python "$PILOTTS_SCRIPT" >> "$PILOTTS_LOG_FILE" 2>&1 &
    local new_pid=$!
    echo "$new_pid" > "$PILOTTS_PID_FILE"

    print_step "等待服务启动 (PID: $new_pid)..."
    echo ""

    # 等待服务启动：超时按预加载配置取值，可通过环境变量覆盖
    if [ "$PRELOAD_PILOTTS" = "1" ]; then
        local pt_max_wait=$START_WAIT_PRELOAD
    else
        local pt_max_wait=$START_WAIT_NO_PRELOAD
    fi
    local count=0
    local spin='⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'
    while [ $count -lt $pt_max_wait ]; do
        if kill -0 "$new_pid" 2>/dev/null; then
            if curl -s "http://127.0.0.1:$PILOTTS_PORT/health" >/dev/null 2>&1; then
                echo ""
                echo ""
                echo "========================================"
                echo "      PilotTTS 服务启动成功"
                echo "========================================"
                print_success "服务状态: 运行中"
                print_info "进程PID:  $new_pid"
                print_info "服务端口: $PILOTTS_PORT"
                print_info "健康检查: http://127.0.0.1:$PILOTTS_PORT/health"
                print_info "日志文件: $PILOTTS_LOG_FILE"
                echo ""
                print_info "常用命令:"
                echo "  查看状态: $0 status-pilottts"
                echo "  查看日志: tail -f $PILOTTS_LOG_FILE"
                echo "  停止服务: $0 stop-pilottts"
                echo "========================================"
                exit 0
            fi
        else
            echo ""
            echo ""
            print_error "PilotTTS 服务启动失败 - 进程已退出"
            print_info "查看错误日志:"
            echo "  tail -n 50 $PILOTTS_LOG_FILE"
            rm -f "$PILOTTS_PID_FILE"
            exit 1
        fi
        sleep 1
        count=$((count + 1))
        printf "\r  %s  等待中... %d/%d 秒" "${spin:$((count % 10)):1}" "$count" "$pt_max_wait"
    done

    echo ""
    echo ""
    print_warn "PilotTTS 服务启动超时 (${pt_max_wait}秒)"
    print_info "可能原因:"
    echo "  1. 模型加载时间较长 (PilotTTS 模型约 6GB)"
    echo "  2. 端口被占用: $PILOTTS_PORT"
    echo "  3. transformers4 依赖不完整"
    echo ""
    print_info "查看详细日志:"
    echo "  tail -n 100 $PILOTTS_LOG_FILE"
    exit 1
}

# 停止 PilotTTS 独立服务
do_stop_pilottts() {
    echo "----------------------------------------"
    echo "      PilotTTS 独立服务停止"
    echo "----------------------------------------"

    local pid=""

    # 先尝试从 PID 文件获取 PID
    if is_pilottts_running; then
        pid=$(cat "$PILOTTS_PID_FILE" 2>/dev/null)
        print_info "发现运行中的 PilotTTS 服务 (PID: $pid)"
    else
        # PID 文件不存在或进程已失效，尝试查找实际运行的进程
        pid=$(pgrep -f "python.*pilottts_service.py" 2>/dev/null | head -1)
        if [ -n "$pid" ]; then
            print_warn "PID 文件不存在，但找到运行中的 PilotTTS 进程 (PID: $pid)"
        else
            print_info "PilotTTS 服务未运行"
            if [ -f "$PILOTTS_PID_FILE" ]; then
                rm -f "$PILOTTS_PID_FILE"
            fi
            return 0
        fi
    fi

    print_step "正在停止服务..."

    # 优雅终止
    kill "$pid" 2>/dev/null || true

    local count=0
    while [ $count -lt 10 ]; do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo ""
            print_success "PilotTTS 服务已停止"
            rm -f "$PILOTTS_PID_FILE"
            return 0
        fi
        sleep 1
        count=$((count + 1))
        echo -n "."
    done

    echo ""
    print_warn "优雅终止超时，执行强制停止..."
    kill -9 "$pid" 2>/dev/null || true
    sleep 1

    if ! kill -0 "$pid" 2>/dev/null; then
        print_success "PilotTTS 服务已强制停止"
        rm -f "$PILOTTS_PID_FILE"
    else
        print_error "无法停止 PilotTTS 服务 (PID: $pid)"
    fi
}

# 查看 PilotTTS 服务状态
do_status_pilottts() {
    echo ""
    echo "========================================"
    echo "      PilotTTS 独立服务状态"
    echo "========================================"
    echo ""

    if ! is_pilottts_running; then
        print_warn "PilotTTS 服务未运行"
        if [ -f "$PILOTTS_PID_FILE" ]; then
            rm -f "$PILOTTS_PID_FILE"
        fi
        echo ""
        print_info "启动服务:"
        echo "  $0 start-pilottts"
        exit 0
    fi

    local pid=$(cat "$PILOTTS_PID_FILE" 2>/dev/null)
    print_success "PilotTTS 服务运行中"
    print_info "进程PID:  $pid"
    print_info "服务端口: $PILOTTS_PORT"
    echo ""

    print_step "执行健康检查..."
    local health
    health=$(curl -s "http://127.0.0.1:$PILOTTS_PORT/health" 2>/dev/null || echo "")
    if [ -n "$health" ]; then
        print_success "健康检查通过"
        echo ""
        echo "$health" | python -m json.tool 2>/dev/null || echo "$health"
    else
        print_error "健康检查失败"
        print_info "服务进程存在但无法响应请求"
    fi
    echo ""
    print_info "日志文件: $PILOTTS_LOG_FILE"
    print_info "查看日志: tail -f $PILOTTS_LOG_FILE"
}

# ============ GPT-SoVITS 独立服务管理 ============

# 检查 GPT-SoVITS 服务是否在运行
is_gptsovits_running() {
    if [ -f "$GPTSOVITS_PID_FILE" ]; then
        local pid=$(cat "$GPTSOVITS_PID_FILE" 2>/dev/null)
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            return 0
        fi
    fi
    return 1
}

# 启动 GPT-SoVITS 独立服务
do_start_gptsovits() {
    echo ""
    echo "========================================"
    echo "      GPT-SoVITS 独立服务启动"
    echo "========================================"
    echo ""

    if is_gptsovits_running; then
        local current_pid=$(cat "$GPTSOVITS_PID_FILE" 2>/dev/null)
        print_warn "GPT-SoVITS 服务已在运行中"
        print_info "当前PID: $current_pid"
        print_info "服务端口: $GPTSOVITS_PORT"
        print_info "健康检查: http://127.0.0.1:$GPTSOVITS_PORT/health"
        exit 0
    fi

    if [ ! -f "$GPTSOVITS_SCRIPT" ]; then
        print_error "GPT-SoVITS 服务脚本不存在: $GPTSOVITS_SCRIPT"
        exit 1
    fi

    if [ ! -d "$SCRIPT_DIR/lib/transformers4" ]; then
        print_error "transformers4 目录不存在: $SCRIPT_DIR/lib/transformers4"
        print_info "请先安装: pip install --target $SCRIPT_DIR/lib/transformers4 transformers==4.51.3"
        exit 1
    fi

    print_step "激活虚拟环境..."
    source "$VENV_PATH/bin/activate"
    print_success "虚拟环境已激活"

    if [ "$OFFLINE_MODE" = true ] && [ -f "$ENV_FILE" ]; then
        print_step "加载离线模式环境变量..."
        source "$ENV_FILE"
        print_success "离线模式已启用"
    fi

    export GPTSOVITS_HOST
    export GPTSOVITS_PORT
    export TRANSFORMERS_OFFLINE
    export HF_HUB_OFFLINE
    export HF_HOME
    export HUGGINGFACE_HUB_CACHE
    export TRANSFORMERS_CACHE
    export PRELOAD_GPTSOVITS
    export IDLE_TIMEOUT
    export HEARTBEAT_INTERVAL
    export MAIN_HOST="127.0.0.1"
    export MAIN_PORT="$PORT"

    mkdir -p "$SCRIPT_DIR/logs"

    print_info "Python 版本: $(python --version 2>&1)"
    echo ""
    print_step "启动 GPT-SoVITS 独立服务..."
    print_info "服务端口: $GPTSOVITS_PORT"
    print_info "GPU设备: $GPTSOVITS_GPU"
    print_info "日志文件: $GPTSOVITS_LOG_FILE"
    echo ""

    cd "$SCRIPT_DIR"
    CUDA_VISIBLE_DEVICES="$GPTSOVITS_GPU" GPU_ID="$GPTSOVITS_GPU" nohup python "$GPTSOVITS_SCRIPT" >> "$GPTSOVITS_LOG_FILE" 2>&1 &
    local new_pid=$!
    echo "$new_pid" > "$GPTSOVITS_PID_FILE"

    print_step "等待服务启动 (PID: $new_pid)..."
    echo ""

    # 等待服务启动：超时按预加载配置取值，可通过环境变量覆盖
    if [ "$PRELOAD_GPTSOVITS" = "1" ]; then
        local gs_max_wait=$START_WAIT_PRELOAD
    else
        local gs_max_wait=$START_WAIT_NO_PRELOAD
    fi
    local count=0
    local spin='⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'
    while [ $count -lt $gs_max_wait ]; do
        if kill -0 "$new_pid" 2>/dev/null; then
            if curl -s "http://127.0.0.1:$GPTSOVITS_PORT/health" >/dev/null 2>&1; then
                echo ""
                echo ""
                echo "========================================"
                echo "      GPT-SoVITS 服务启动成功"
                echo "========================================"
                print_success "服务状态: 运行中"
                print_info "进程PID:  $new_pid"
                print_info "服务端口: $GPTSOVITS_PORT"
                print_info "健康检查: http://127.0.0.1:$GPTSOVITS_PORT/health"
                print_info "日志文件: $GPTSOVITS_LOG_FILE"
                echo ""
                print_info "常用命令:"
                echo "  查看状态: $0 status-gptsovits"
                echo "  查看日志: tail -f $GPTSOVITS_LOG_FILE"
                echo "  停止服务: $0 stop-gptsovits"
                echo "========================================"
                exit 0
            fi
        else
            echo ""
            echo ""
            print_error "GPT-SoVITS 服务启动失败 - 进程已退出"
            print_info "查看错误日志:"
            echo "  tail -n 50 $GPTSOVITS_LOG_FILE"
            rm -f "$GPTSOVITS_PID_FILE"
            exit 1
        fi
        sleep 1
        count=$((count + 1))
        printf "\r  %s  等待中... %d/%d 秒" "${spin:$((count % 10)):1}" "$count" "$gs_max_wait"
    done

    echo ""
    print_warn "GPT-SoVITS 服务启动超时 (${gs_max_wait}秒)"
    print_info "服务可能仍在初始化，请检查日志: tail -f $GPTSOVITS_LOG_FILE"
}

# 停止 GPT-SoVITS 独立服务
do_stop_gptsovits() {
    echo "----------------------------------------"
    echo "      GPT-SoVITS 独立服务停止"
    echo "----------------------------------------"

    local pid=""

    if is_gptsovits_running; then
        pid=$(cat "$GPTSOVITS_PID_FILE" 2>/dev/null)
        print_info "发现运行中的 GPT-SoVITS 服务 (PID: $pid)"
    else
        pid=$(pgrep -f "python.*gptsovits_service.py" 2>/dev/null | head -1)
        if [ -n "$pid" ]; then
            print_warn "PID 文件不存在，但找到运行中的 GPT-SoVITS 进程 (PID: $pid)"
        else
            print_info "GPT-SoVITS 服务未运行"
            if [ -f "$GPTSOVITS_PID_FILE" ]; then
                rm -f "$GPTSOVITS_PID_FILE"
            fi
            return 0
        fi
    fi

    print_step "正在停止服务..."
    kill "$pid" 2>/dev/null || true

    local count=0
    while [ $count -lt 10 ]; do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo ""
            print_success "GPT-SoVITS 服务已停止"
            rm -f "$GPTSOVITS_PID_FILE"
            return 0
        fi
        sleep 1
        count=$((count + 1))
        echo -n "."
    done

    echo ""
    print_warn "优雅终止超时，执行强制停止..."
    kill -9 "$pid" 2>/dev/null || true
    sleep 1

    if ! kill -0 "$pid" 2>/dev/null; then
        print_success "GPT-SoVITS 服务已强制停止"
        rm -f "$GPTSOVITS_PID_FILE"
    else
        print_error "无法停止 GPT-SoVITS 服务 (PID: $pid)"
    fi
}

# 查看 GPT-SoVITS 服务状态
do_status_gptsovits() {
    echo ""
    echo "========================================"
    echo "      GPT-SoVITS 独立服务状态"
    echo "========================================"
    echo ""

    if ! is_gptsovits_running; then
        print_warn "GPT-SoVITS 服务未运行"
        if [ -f "$GPTSOVITS_PID_FILE" ]; then
            rm -f "$GPTSOVITS_PID_FILE"
        fi
        echo ""
        print_info "启动服务:"
        echo "  $0 start-gptsovits"
        exit 0
    fi

    local pid=$(cat "$GPTSOVITS_PID_FILE" 2>/dev/null)
    print_success "GPT-SoVITS 服务运行中"
    print_info "进程PID:  $pid"
    print_info "服务端口: $GPTSOVITS_PORT"
    echo ""

    local health=$(curl -s "http://127.0.0.1:$GPTSOVITS_PORT/health" 2>/dev/null)
    if [ -n "$health" ]; then
        print_success "健康检查通过"
        echo ""
        echo "$health" | python -m json.tool 2>/dev/null || echo "$health"
    else
        print_error "健康检查失败"
    fi
    echo ""
    print_info "日志文件: $GPTSOVITS_LOG_FILE"
    print_info "查看日志: tail -f $GPTSOVITS_LOG_FILE"
}

# ============ Fish-Speech 独立服务管理 ============

# 检查 Fish-Speech 服务是否在运行
is_fishspeech_running() {
    if [ -f "$FISHSPEECH_PID_FILE" ]; then
        local pid=$(cat "$FISHSPEECH_PID_FILE" 2>/dev/null)
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            return 0
        fi
    fi
    return 1
}

# 启动 Fish-Speech 独立服务
do_start_fishspeech() {
    echo ""
    echo "========================================"
    echo "      Fish-Speech 独立服务启动"
    echo "========================================"
    echo ""

    if is_fishspeech_running; then
        local current_pid=$(cat "$FISHSPEECH_PID_FILE" 2>/dev/null)
        print_warn "Fish-Speech 服务已在运行中"
        print_info "当前PID: $current_pid"
        print_info "服务端口: $FISHSPEECH_PORT"
        print_info "健康检查: http://127.0.0.1:$FISHSPEECH_PORT/health"
        exit 0
    fi

    if [ ! -f "$FISHSPEECH_SCRIPT" ]; then
        print_error "Fish-Speech 服务脚本不存在: $FISHSPEECH_SCRIPT"
        exit 1
    fi

    print_step "激活虚拟环境..."
    source "$VENV_PATH/bin/activate"
    print_success "虚拟环境已激活"

    if [ "$OFFLINE_MODE" = true ] && [ -f "$ENV_FILE" ]; then
        print_step "加载离线模式环境变量..."
        source "$ENV_FILE"
        print_success "离线模式已启用"
    fi

    export FISHSPEECH_HOST
    export FISHSPEECH_PORT
    export TRANSFORMERS_OFFLINE
    export HF_HUB_OFFLINE
    export HF_HOME
    export HUGGINGFACE_HUB_CACHE
    export TRANSFORMERS_CACHE
    export PRELOAD_FISHSPEECH
    export IDLE_TIMEOUT
    export HEARTBEAT_INTERVAL
    export MAIN_HOST="127.0.0.1"
    export MAIN_PORT="$PORT"

    mkdir -p "$SCRIPT_DIR/logs"

    print_info "Python 版本: $(python --version 2>&1)"
    echo ""
    print_step "启动 Fish-Speech 独立服务..."
    print_info "服务端口: $FISHSPEECH_PORT"
    print_info "GPU设备: $FISHSPEECH_GPU"
    print_info "日志文件: $FISHSPEECH_LOG_FILE"
    echo ""

    cd "$SCRIPT_DIR"
    CUDA_VISIBLE_DEVICES="$FISHSPEECH_GPU" GPU_ID="$FISHSPEECH_GPU" nohup python "$FISHSPEECH_SCRIPT" >> "$FISHSPEECH_LOG_FILE" 2>&1 &
    local new_pid=$!
    echo "$new_pid" > "$FISHSPEECH_PID_FILE"

    print_step "等待服务启动 (PID: $new_pid)..."
    echo ""

    # 等待服务启动：超时按预加载配置取值，可通过环境变量覆盖
    if [ "$PRELOAD_FISHSPEECH" = "1" ]; then
        local fs_max_wait=$START_WAIT_PRELOAD
    else
        local fs_max_wait=$START_WAIT_NO_PRELOAD
    fi
    local count=0
    local spin='⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'
    while [ $count -lt $fs_max_wait ]; do
        if kill -0 "$new_pid" 2>/dev/null; then
            if curl -s "http://127.0.0.1:$FISHSPEECH_PORT/health" >/dev/null 2>&1; then
                echo ""
                echo ""
                echo "========================================"
                echo "      Fish-Speech 服务启动成功"
                echo "========================================"
                print_success "服务状态: 运行中"
                print_info "进程PID:  $new_pid"
                print_info "服务端口: $FISHSPEECH_PORT"
                print_info "健康检查: http://127.0.0.1:$FISHSPEECH_PORT/health"
                print_info "日志文件: $FISHSPEECH_LOG_FILE"
                echo ""
                print_info "常用命令:"
                echo "  查看状态: $0 status-fishspeech"
                echo "  查看日志: tail -f $FISHSPEECH_LOG_FILE"
                echo "  停止服务: $0 stop-fishspeech"
                echo "========================================"
                exit 0
            fi
        else
            echo ""
            echo ""
            print_error "Fish-Speech 服务启动失败 - 进程已退出"
            print_info "查看错误日志:"
            echo "  tail -n 50 $FISHSPEECH_LOG_FILE"
            rm -f "$FISHSPEECH_PID_FILE"
            exit 1
        fi
        sleep 1
        count=$((count + 1))
        printf "\r  %s  等待中... %d/%d 秒" "${spin:$((count % 10)):1}" "$count" "$fs_max_wait"
    done

    echo ""
    print_warn "Fish-Speech 服务启动超时 (${fs_max_wait}秒)"
    print_info "服务可能仍在初始化，请检查日志: tail -f $FISHSPEECH_LOG_FILE"
}

# 停止 Fish-Speech 独立服务
do_stop_fishspeech() {
    echo "----------------------------------------"
    echo "      Fish-Speech 独立服务停止"
    echo "----------------------------------------"

    local pid=""

    if is_fishspeech_running; then
        pid=$(cat "$FISHSPEECH_PID_FILE" 2>/dev/null)
        print_info "发现运行中的 Fish-Speech 服务 (PID: $pid)"
    else
        pid=$(pgrep -f "python.*fishspeech_service.py" 2>/dev/null | head -1)
        if [ -n "$pid" ]; then
            print_warn "PID 文件不存在，但找到运行中的 Fish-Speech 进程 (PID: $pid)"
        else
            print_info "Fish-Speech 服务未运行"
            if [ -f "$FISHSPEECH_PID_FILE" ]; then
                rm -f "$FISHSPEECH_PID_FILE"
            fi
            return 0
        fi
    fi

    print_step "正在停止服务..."
    kill "$pid" 2>/dev/null || true

    local count=0
    while [ $count -lt 10 ]; do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo ""
            print_success "Fish-Speech 服务已停止"
            rm -f "$FISHSPEECH_PID_FILE"
            return 0
        fi
        sleep 1
        count=$((count + 1))
        echo -n "."
    done

    echo ""
    print_warn "优雅终止超时，执行强制停止..."
    kill -9 "$pid" 2>/dev/null || true
    sleep 1

    if ! kill -0 "$pid" 2>/dev/null; then
        print_success "Fish-Speech 服务已强制停止"
        rm -f "$FISHSPEECH_PID_FILE"
    else
        print_error "无法停止 Fish-Speech 服务 (PID: $pid)"
    fi
}

# 查看 Fish-Speech 服务状态
do_status_fishspeech() {
    echo ""
    echo "========================================"
    echo "      Fish-Speech 独立服务状态"
    echo "========================================"
    echo ""

    if ! is_fishspeech_running; then
        print_warn "Fish-Speech 服务未运行"
        if [ -f "$FISHSPEECH_PID_FILE" ]; then
            rm -f "$FISHSPEECH_PID_FILE"
        fi
        echo ""
        print_info "启动服务:"
        echo "  $0 start-fishspeech"
        exit 0
    fi

    local pid=$(cat "$FISHSPEECH_PID_FILE" 2>/dev/null)
    print_success "Fish-Speech 服务运行中"
    print_info "进程PID:  $pid"
    print_info "服务端口: $FISHSPEECH_PORT"
    echo ""

    local health=$(curl -s "http://127.0.0.1:$FISHSPEECH_PORT/health" 2>/dev/null)
    if [ -n "$health" ]; then
        print_success "健康检查通过"
        echo ""
        echo "$health" | python -m json.tool 2>/dev/null || echo "$health"
    else
        print_error "健康检查失败"
    fi
    echo ""
    print_info "日志文件: $FISHSPEECH_LOG_FILE"
    print_info "查看日志: tail -f $FISHSPEECH_LOG_FILE"
}

# ============ 主逻辑 ============

# 解析命令
COMMAND=""
if [ $# -eq 0 ]; then
    usage
fi

COMMAND="$1"
shift

# 解析选项
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --skip-check)
            SKIP_CHECK=true
            shift
            ;;
        --reload)
            RELOAD=true
            shift
            ;;
        --offline)
            OFFLINE_MODE=true
            shift
            ;;
        *)
            print_error "未知选项: $1"
            usage
            ;;
    esac
done

# 执行命令
case "$COMMAND" in
    start)
        do_start
        ;;
    stop)
        do_stop
        ;;
    restart)
        do_restart
        ;;
    status)
        do_status
        ;;
    start-omnivoice)
        do_start_omnivoice
        ;;
    stop-omnivoice)
        do_stop_omnivoice
        ;;
    restart-omnivoice)
        do_stop_omnivoice
        sleep 2
        do_start_omnivoice
        ;;
    status-omnivoice)
        do_status_omnivoice
        ;;
    start-cosyvoice)
        do_start_cosyvoice
        ;;
    stop-cosyvoice)
        do_stop_cosyvoice
        ;;
    restart-cosyvoice)
        do_stop_cosyvoice
        sleep 2
        do_start_cosyvoice
        ;;
    status-cosyvoice)
        do_status_cosyvoice
        ;;
    start-pilottts)
        do_start_pilottts
        ;;
    stop-pilottts)
        do_stop_pilottts
        ;;
    restart-pilottts)
        do_stop_pilottts
        sleep 2
        do_start_pilottts
        ;;
    status-pilottts)
        do_status_pilottts
        ;;
    start-gptsovits)
        do_start_gptsovits
        ;;
    stop-gptsovits)
        do_stop_gptsovits
        ;;
    restart-gptsovits)
        do_stop_gptsovits
        sleep 2
        do_start_gptsovits
        ;;
    status-gptsovits)
        do_status_gptsovits
        ;;
    start-fishspeech)
        do_start_fishspeech
        ;;
    stop-fishspeech)
        do_stop_fishspeech
        ;;
    restart-fishspeech)
        do_stop_fishspeech
        sleep 2
        do_start_fishspeech
        ;;
    status-fishspeech)
        do_status_fishspeech
        ;;
    *)
        print_error "未知命令: $COMMAND"
        usage
        ;;
esac
