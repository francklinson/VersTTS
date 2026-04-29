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

# 默认配置
HOST="0.0.0.0"
PORT="8000"
SKIP_CHECK=false
RELOAD=false
OFFLINE_MODE=false

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

    # 创建必要目录
    print_step "创建必要目录..."
    mkdir -p "$SCRIPT_DIR/output"
    mkdir -p "$SCRIPT_DIR/uploads"
    mkdir -p "$SCRIPT_DIR/logs"
    mkdir -p "$SCRIPT_DIR/records"
    print_success "目录检查完成"

    # 构建启动命令
    local cmd=(python -m uvicorn backend.api_server:app --host "$HOST" --port "$PORT")
    if [ "$RELOAD" = true ]; then
        cmd+=(--reload)
        print_info "开发模式: 已启用自动重载"
    fi

    echo ""
    print_step "启动 Uvicorn 服务..."
    print_info "命令: ${cmd[*]}"
    print_info "日志文件: $LOG_FILE"
    echo ""

    # 后台启动并记录 PID
    cd "$SCRIPT_DIR"
    nohup "${cmd[@]}" >> "$LOG_FILE" 2>&1 &
    local new_pid=$!
    echo "$new_pid" > "$PID_FILE"

    print_step "等待服务启动 (PID: $new_pid)..."
    echo ""

    # 等待服务启动
    local count=0
    local spin='⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'
    while [ $count -lt 30 ]; do
        if kill -0 "$new_pid" 2>/dev/null; then
            if curl -s "http://$HOST:$PORT/health" >/dev/null 2>&1; then
                echo ""
                echo ""
                echo "========================================"
                echo "      服务启动成功"
                echo "========================================"
                print_success "服务状态: 运行中"
                print_info "进程PID:  $new_pid"
                print_info "服务地址: http://$HOST:$PORT"
                print_info "API文档:  http://$HOST:$PORT/docs"
                print_info "前端页面: http://$HOST:$PORT"
                print_info "日志文件: $LOG_FILE"
                echo ""
                print_info "常用命令:"
                echo "  查看状态: $0 status"
                echo "  查看日志: tail -f $LOG_FILE"
                echo "  停止服务: $0 stop"
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
        printf "\r  %s  等待中... %d/30 秒" "${spin:$((count % 10)):1}" "$count"
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

# 停止服务
do_stop() {
    echo ""
    echo "========================================"
    echo "      VersTTS 服务停止"
    echo "========================================"
    echo ""

    local pid
    pid=$(get_pid)

    if [ -z "$pid" ]; then
        print_warn "服务未运行 (PID 文件不存在)"
        print_info "可能情况:"
        echo "  1. 服务从未启动过"
        echo "  2. 服务异常退出，PID文件被清理"
        exit 0
    fi

    if ! kill -0 "$pid" 2>/dev/null; then
        print_warn "服务未运行 (PID: $pid 已失效)"
        print_step "清理残留的PID文件..."
        rm -f "$PID_FILE"
        print_success "清理完成"
        exit 0
    fi

    print_info "发现运行中的服务 (PID: $pid)"
    print_step "正在停止服务 (优雅终止)..."

    # 先尝试优雅终止
    kill "$pid" 2>/dev/null || true

    local count=0
    local spin='⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'
    while [ $count -lt 10 ]; do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo ""
            print_success "服务已停止"
            rm -f "$PID_FILE"
            exit 0
        fi
        sleep 1
        count=$((count + 1))
        printf "\r  %s  等待中... %d/10 秒" "${spin:$((count % 10)):1}" "$count"
    done

    echo ""
    print_warn "优雅终止超时，执行强制停止..."
    kill -9 "$pid" 2>/dev/null || true
    sleep 1

    if ! kill -0 "$pid" 2>/dev/null; then
        print_success "服务已强制停止"
        rm -f "$PID_FILE"
    else
        print_error "无法停止服务 (PID: $pid)"
        print_info "请手动终止: kill -9 $pid"
        exit 1
    fi
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

    # 阶段1: 停止服务
    echo "【阶段 1/2】停止现有服务"
    echo "----------------------------------------"

    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        print_info "发现运行中的服务 (PID: $pid)"
        print_step "正在停止服务..."
        kill "$pid" 2>/dev/null || true

        local count=0
        while [ $count -lt 10 ]; do
            if ! kill -0 "$pid" 2>/dev/null; then
                print_success "服务已停止"
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
            print_success "服务已强制停止"
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
            print_info "没有运行中的服务"
        fi
    fi

    print_success "停止阶段完成"
    echo ""

    # 阶段2: 启动服务
    echo "【阶段 2/2】启动新服务"
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
        print_success "服务运行中"
        print_info "进程PID:  $pid"
        print_info "启动时间: $(ps -o lstart= -p "$pid" 2>/dev/null || echo "未知")"
        print_info "运行时长: $(ps -o etime= -p "$pid" 2>/dev/null || echo "未知")"
        echo ""

        print_step "执行健康检查..."
        local health
        health=$(curl -s "http://$HOST:$PORT/health" 2>/dev/null || echo "")
        if [ -n "$health" ]; then
            print_success "健康检查通过"
            echo ""
            echo "$health" | python -m json.tool 2>/dev/null || echo "$health"
        else
            print_error "健康检查失败"
            print_info "服务进程存在但无法响应请求"
        fi
        echo ""
        print_info "访问地址:"
        echo "  前端页面: http://$HOST:$PORT"
        echo "  API文档:  http://$HOST:$PORT/docs"
        echo "  健康检查: http://$HOST:$PORT/health"
        print_info "日志文件: $LOG_FILE"
    else
        print_error "服务未运行 (PID 文件残留)"
        print_warn "PID 文件中记录的进程已失效: $pid"
        print_step "清理残留的PID文件..."
        rm -f "$PID_FILE"
        print_success "清理完成"
        echo ""
        print_info "启动服务:"
        echo "  $0 start"
    fi
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
    *)
        print_error "未知命令: $COMMAND"
        usage
        ;;
esac
