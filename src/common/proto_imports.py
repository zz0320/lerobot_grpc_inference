# -*- coding: utf-8 -*-
"""
Shared protobuf import logic.

Server 和 Client 都需要导入 protobuf 生成文件，
且都使用相同的 try/except fallback 处理不同工作目录的情况。
此模块统一管理导入逻辑。
"""

try:
    from src.generated import lerobot_inference_pb2 as pb2
    from src.generated import lerobot_inference_pb2_grpc as pb2_grpc
except ImportError:
    try:
        from generated import lerobot_inference_pb2 as pb2
        from generated import lerobot_inference_pb2_grpc as pb2_grpc
    except ImportError:
        pb2 = None
        pb2_grpc = None
        print("警告: 未找到 protobuf 生成文件，请先运行 scripts/generate_proto.sh")
