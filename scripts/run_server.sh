#!/bin/bash
# 启动推理服务器
# 在 lerobot 环境中运行

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
HOST="${LEROBOT_SERVER_HOST:-0.0.0.0}"
PORT="${LEROBOT_SERVER_PORT:-50051}"
DATASET="${LEROBOT_DATASET_PATH:-}"
MODEL="${LEROBOT_MODEL_PATH:-}"
DEVICE="${LEROBOT_DEVICE:-cuda}"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

echo "=== 启动 LeRobot 推理服务器 ==="
echo "Host: $HOST"
echo "Port: $PORT"

CMD="python3 -m src.server.inference_server --host $HOST --port $PORT --device $DEVICE"

if [ -n "$DATASET" ]; then
    echo "Dataset: $DATASET"
    CMD="$CMD --dataset $DATASET"
elif [ -n "$MODEL" ]; then
    echo "Model: $MODEL"
    CMD="$CMD --model $MODEL"
else
    echo "警告: 未指定 dataset 或 model，服务器将以空闲模式运行"
fi

echo "运行命令: $CMD"
echo "==============================="

exec $CMD


