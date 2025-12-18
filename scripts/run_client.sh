#!/bin/bash
# 启动推理客户端
# 在机器人侧环境中运行

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# 检查是否已生成 proto 代码
if [ ! -f "src/generated/lerobot_inference_pb2.py" ]; then
    echo "未找到 proto 生成代码，正在生成..."
    bash scripts/generate_proto.sh
fi

# 默认参数
SERVER="${LEROBOT_SERVER:-localhost:50051}"
EPISODE="${LEROBOT_EPISODE:-0}"
CONTROL_FREQ="${LEROBOT_CONTROL_FREQ:-30}"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --server)
            SERVER="$2"
            shift 2
            ;;
        --episode)
            EPISODE="$2"
            shift 2
            ;;
        --control-freq)
            CONTROL_FREQ="$2"
            shift 2
            ;;
        *)
            # 传递给 Python 脚本
            break
            ;;
    esac
done

echo "=== 启动 Astribot 推理客户端 ==="
echo "Server: $SERVER"
echo "Episode: $EPISODE"
echo "Control Freq: $CONTROL_FREQ Hz"
echo "==============================="

exec python3 -m src.client.inference_client \
    --server "$SERVER" \
    --episode "$EPISODE" \
    --control-freq "$CONTROL_FREQ" \
    "$@"


