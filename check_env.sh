#!/bin/bash
# 使用虚拟环境检查Python环境

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="$SCRIPT_DIR/.venv"

if [ ! -d "$VENV_PATH" ]; then
    echo "错误: 虚拟环境不存在: $VENV_PATH"
    exit 1
fi

source "$VENV_PATH/bin/activate"
python "$SCRIPT_DIR/check_env.py"
