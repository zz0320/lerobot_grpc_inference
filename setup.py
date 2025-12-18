#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LeRobot gRPC Inference - 安装脚本
"""

from setuptools import setup, find_packages

setup(
    name="lerobot_grpc_inference",
    version="1.0.0",
    description="LeRobot gRPC 推理框架 for Astribot",
    author="Astribot",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "grpcio>=1.50.0",
        "grpcio-tools>=1.50.0",
        "numpy>=1.20.0",
    ],
    extras_require={
        "server": [
            "pandas>=1.5.0",
            "pyarrow>=10.0.0",
            "grpcio-reflection>=1.50.0",
        ],
        "client": [],
    },
    entry_points={
        "console_scripts": [
            "lerobot-server=src.server.inference_server:main",
            "lerobot-client=src.client.inference_client:main",
        ],
    },
)


