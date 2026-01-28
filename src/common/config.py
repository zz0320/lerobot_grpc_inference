# -*- coding: utf-8 -*-
"""
配置管理 (精简版)
"""

from dataclasses import dataclass, field
from typing import Optional, List

from .constants import (
    DEFAULT_CONTROL_FREQ,
    DEFAULT_GRPC_PORT,
    DEFAULT_GRPC_HOST,
    LEROBOT_ACTION_DIM_NO_CHASSIS,
    LEROBOT_ACTION_DIM_WITH_CHASSIS,
)


@dataclass
class ActionConfig:
    """
    Action 配置

    分离三个概念:
    1. state_dim: 输入 state 的维度 (22 或 25)，由机器人状态采集决定
    2. action_dim: 模型/数据集输出的 action 维度 (22 或 25)，由训练时决定
    3. enable_chassis: 执行时是否控制底盘 (只影响发送给机器人的命令)

    字段命名与 proto ActionOutputConfig 保持一致 (enable_*)
    """
    # ========== 输入配置 ==========
    # 输入 state 是否包含底盘 (影响 Client 采集的状态维度)
    state_includes_chassis: bool = False

    # ========== 执行配置 ==========
    # 执行 action 时是否控制底盘 (即使 action 有 25 维，也可以选择不执行底盘)
    enable_chassis: bool = False
    # 是否启用头部控制
    enable_head: bool = True
    # 是否启用腰部控制
    enable_torso: bool = True

    @property
    def state_dim(self) -> int:
        """输入 state 维度"""
        # 基础: arm_left(7) + arm_right(7) + gripper_left(1) + gripper_right(1) + head(2) + torso(4) = 22
        return 25 if self.state_includes_chassis else 22


@dataclass
class ServerConfig:
    """
    Server 配置
    
    注意: Server 以空闲模式启动，等待 Client 通过 Configure() 指定模型/数据集
    """
    host: str = DEFAULT_GRPC_HOST
    port: int = DEFAULT_GRPC_PORT
    max_workers: int = 10
    
    # 推理设备 (默认值，Client 可覆盖)
    device: str = "cuda"
    
    # 推理配置
    fps: float = DEFAULT_CONTROL_FREQ
    
    # Action 配置 (默认值，Client 可覆盖)
    action_config: ActionConfig = field(default_factory=ActionConfig)


@dataclass
class ClientConfig:
    """Client 配置"""
    server_host: str = "localhost"
    server_port: int = DEFAULT_GRPC_PORT
    timeout: float = 10.0
    
    # 策略配置 (Client 端指定)
    model_path: Optional[str] = None  # 模型路径或 HuggingFace repo id
    device: str = "cuda"  # 推理设备
    policy_type: Optional[str] = None  # 策略类型 (可选)
    
    # 控制配置
    control_freq: float = DEFAULT_CONTROL_FREQ
    control_way: str = "direct"  # "direct" or "filter"
    
    # 平滑配置
    smooth_window: int = 0  # 0 = 不平滑
    max_velocity: float = 0.0  # 0 = 不限制
    
    # Action 配置
    action_config: ActionConfig = field(default_factory=ActionConfig)
    
    @property
    def server_address(self) -> str:
        return f"{self.server_host}:{self.server_port}"
    
    @property
    def mode(self) -> str:
        """推理模式"""
        if self.model_path:
            return "model"
        return "none"
