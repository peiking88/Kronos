#!/bin/bash
#
# Kronos 一键启动脚本
# 功能: 环境检查 → 依赖安装 → 启动 WebUI (前后端一体)
#
# 用法:
#   ./start.sh              前台启动 (Ctrl+C 停止)
#   ./start.sh -d           后台启动
#   ./start.sh -p 8080      指定端口
#   ./start.sh stop         停止后台服务
#   ./start.sh status       查看状态
#

set -euo pipefail

# ── 项目路径 ──────────────────────────────────────────
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"
WEBUI_DIR="$PROJECT_DIR/webui"
LOG_FILE="$PROJECT_DIR/logs/webui.log"
PID_FILE="$PROJECT_DIR/logs/webui.pid"
HOST="0.0.0.0"
PORT=7070
DAEMON=false

# ── 颜色 ──────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; }

# ── 解析参数 ──────────────────────────────────────────
usage() {
    echo "用法: $0 [选项] [命令]"
    echo ""
    echo "命令:"
    echo "  (无)         前台启动 WebUI"
    echo "  stop         停止后台服务"
    echo "  status       查看运行状态"
    echo "  restart      重启服务"
    echo ""
    echo "选项:"
    echo "  -d           后台启动 (守护进程模式)"
    echo "  -p PORT      指定端口 (默认 7070)"
    echo "  -h           显示帮助"
}

while getopts "dp:h" opt; do
    case $opt in
        d) DAEMON=true ;;
        p) PORT="$OPTARG" ;;
        h) usage; exit 0 ;;
        *) usage; exit 1 ;;
    esac
done
shift $((OPTIND - 1))

COMMAND="${1:-}"

# ── 辅助函数 ──────────────────────────────────────────
get_pid() {
    if [ -f "$PID_FILE" ]; then
        cat "$PID_FILE" 2>/dev/null
    fi
}

is_running() {
    local pid
    pid=$(get_pid)
    [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null
}

do_stop() {
    if is_running; then
        local pid
        pid=$(get_pid)
        info "停止 Kronos WebUI (PID=$pid)..."
        kill "$pid" 2>/dev/null || true
        # 等待进程退出
        for i in $(seq 1 10); do
            if ! kill -0 "$pid" 2>/dev/null; then
                break
            fi
            sleep 0.5
        done
        # 强制终止
        if kill -0 "$pid" 2>/dev/null; then
            warn "进程未响应，强制终止..."
            kill -9 "$pid" 2>/dev/null || true
        fi
        rm -f "$PID_FILE"
        info "已停止"
    else
        warn "服务未运行"
        rm -f "$PID_FILE"
    fi
}

do_status() {
    if is_running; then
        local pid
        pid=$(get_pid)
        info "Kronos WebUI 运行中 (PID=$pid)"
        info "访问: http://localhost:$PORT"
        if [ -f "$LOG_FILE" ]; then
            echo ""
            echo "最近日志:"
            tail -5 "$LOG_FILE"
        fi
    else
        warn "Kronos WebUI 未运行"
    fi
}

activate_venv() {
    if [ ! -f "$VENV_DIR/bin/activate" ]; then
        error "虚拟环境不存在: $VENV_DIR"
        error "请先创建: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
        exit 1
    fi
    # 恢复 python 符号链接（迁移后常见问题）
    local python_bin
    python_bin=$(head -1 "$VENV_DIR/bin/pip" | sed 's|^#!||' | sed 's|/pip$|/python3|')
    local system_python
    system_python=$(readlink -f /usr/bin/python3 2>/dev/null || echo "/usr/bin/python3")
    if [ ! -x "$python_bin" ]; then
        local venv_python="$VENV_DIR/bin/python3"
        warn "venv python 链接缺失，正在修复..."
        ln -sf "$system_python" "$venv_python" 2>/dev/null || true
    fi
    source "$VENV_DIR/bin/activate"
}

check_dependencies() {
    info "检查依赖..."
    local missing=()
    for mod in flask flask_cors pandas numpy plotly torch; do
        if ! python -c "import $mod" 2>/dev/null; then
            missing+=("$mod")
        fi
    done

    if [ ${#missing[@]} -gt 0 ]; then
        warn "缺少依赖: ${missing[*]}"
        info "安装依赖 (使用阿里云镜像)..."
        pip install -i https://mirrors.aliyun.com/pypi/simple/ \
            flask==2.3.3 flask-cors==4.0.0 plotly \
            -r "$PROJECT_DIR/requirements.txt" \
            -r "$WEBUI_DIR/requirements.txt" \
            2>&1 | tail -5
        info "依赖安装完成"
    else
        info "依赖完整"
    fi
}

check_port() {
    if ss -tlnp 2>/dev/null | grep -q ":$PORT "; then
        error "端口 $PORT 已被占用"
        ss -tlnp 2>/dev/null | grep ":$PORT "
        echo ""
        error "请用 -p 指定其他端口，或先停止占用进程"
        exit 1
    fi
}

# ── 命令分发 ──────────────────────────────────────────
case "$COMMAND" in
    stop)
        do_stop
        exit 0
        ;;
    status)
        do_status
        exit 0
        ;;
    restart)
        do_stop
        ;;
esac

# ── 启动前检查 ────────────────────────────────────────
echo "======================================"
echo "  Kronos WebUI 一键启动"
echo "======================================"
echo ""

# 1. venv
info "项目目录: $PROJECT_DIR"
activate_venv
info "Python: $(python --version)"

# 2. 依赖
check_dependencies

# 3. 模型库
if python -c "from model import Kronos" 2>/dev/null; then
    info "Kronos 模型库: 可用"
else
    warn "Kronos 模型库不可用 (将使用模拟数据)"
fi

# 4. 端口
check_port

# ── 启动 ──────────────────────────────────────────────
mkdir -p "$(dirname "$LOG_FILE")"

if $DAEMON; then
    # 后台模式
    info "后台启动 (端口=$PORT)..."
    nohup python "$WEBUI_DIR/app.py" --host "$HOST" --port "$PORT" \
        > "$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"

    # 等待启动
    for i in $(seq 1 15); do
        if curl -s -o /dev/null "http://localhost:$PORT" 2>/dev/null; then
            info "启动成功!"
            info "访问: http://localhost:$PORT"
            info "日志: $LOG_FILE"
            info "停止: $0 stop"
            exit 0
        fi
        sleep 1
    done
    error "启动超时，请检查日志: $LOG_FILE"
    exit 1
else
    # 前台模式
    echo ""
    info "前台启动 (端口=$PORT, Ctrl+C 停止)"
    info "访问: http://localhost:$PORT"
    echo ""
    export PYTHONPATH="$PROJECT_DIR"
    cd "$WEBUI_DIR"
    exec python app.py
fi
