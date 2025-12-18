#!/bin/bash
# 生成 gRPC Python 代码
# 在两个环境中都需要运行此脚本

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

PROTO_DIR="$PROJECT_ROOT/proto"
OUTPUT_DIR="$PROJECT_ROOT/src/generated"

echo "=== 生成 gRPC Python 代码 ==="
echo "Proto 目录: $PROTO_DIR"
echo "输出目录: $OUTPUT_DIR"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 检查 grpcio-tools 是否安装
if ! python3 -c "import grpc_tools.protoc" 2>/dev/null; then
    echo "正在安装 grpcio-tools..."
    pip install grpcio-tools
fi

# 生成代码
python3 -m grpc_tools.protoc \
    -I"$PROTO_DIR" \
    --python_out="$OUTPUT_DIR" \
    --grpc_python_out="$OUTPUT_DIR" \
    "$PROTO_DIR/lerobot_inference.proto"

# 创建 __init__.py
cat > "$OUTPUT_DIR/__init__.py" << 'EOF'
# -*- coding: utf-8 -*-
"""Generated gRPC code"""
from .lerobot_inference_pb2 import *
from .lerobot_inference_pb2_grpc import *
EOF

# 修复导入路径问题
# Python gRPC 生成的代码有相对导入问题
sed -i 's/import lerobot_inference_pb2/from . import lerobot_inference_pb2/g' "$OUTPUT_DIR/lerobot_inference_pb2_grpc.py" 2>/dev/null || \
sed -i '' 's/import lerobot_inference_pb2/from . import lerobot_inference_pb2/g' "$OUTPUT_DIR/lerobot_inference_pb2_grpc.py"

echo "=== 生成完成 ==="
echo "生成的文件:"
ls -la "$OUTPUT_DIR"


