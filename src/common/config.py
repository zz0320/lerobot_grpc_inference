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
    3. execute_chassis: 执行时是否控制底盘 (只影响发送给机器人的命令)
    """
    # ========== 输入配置 ==========
    # 输入 state 是否包含底盘 (影响 Client 采集的状态维度)
    state_includes_chassis: bool = False
    
    # ========== 执行配置 ==========
    # 执行 action 时是否控制底盘 (即使 action 有 25 维，也可以选择不执行底盘)
    execute_chassis: bool = False
    # 是否执行头部控制
    execute_head: bool = True
    # 是否执行腰部控制
    execute_torso: bool = True
    
    @property
    def state_dim(self) -> int:
        """输入 state 维度"""
        # 基础: arm_left(7) + arm_right(7) + gripper_left(1) + gripper_right(1) + head(2) + torso(4) = 22
        return 25 if self.state_includes_chassis else 22
    
    # ========== 兼容旧接口 ==========
    @property
    def enable_chassis(self) -> bool:
        """兼容旧接口"""
        return self.execute_chassis
    
    @property
    def enable_head(self) -> bool:
        """兼容旧接口"""
        return self.execute_head
    
    @property
    def enable_torso(self) -> bool:
        """兼容旧接口"""
        return self.execute_torso


@dataclass
class TemporalEnsembleConfig:
    """
    Temporal Ensemble 配置
    
    实现类似 ACT 的 temporal ensemble 机制:
    - 缓存多个 action chunk 预测
    - 使用指数加权平均融合当前时间步的 actions
    """
    # 是否启用 temporal ensemble
    enabled: bool = False
    
    # 每次推理后执行多少步再重新推理 (类似 ACT 的 n_action_steps)
    # 设为 1 表示每步都推理，设为 chunk_size 表示用完整个 chunk 再推理
    n_action_steps: int = 1
    
    # 指数衰减系数 (越大表示越重视新预测)
    # w = exp(-coeff * age)，age 是 chunk 的年龄
    temporal_ensemble_coeff: float = 0.01
    
    # 最大缓存的 chunk 数量 (0 表示不限制，根据 chunk_size 自动管理)
    max_chunks: int = 0


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
    
    # Temporal Ensemble 配置
    temporal_ensemble_config: TemporalEnsembleConfig = field(default_factory=TemporalEnsembleConfig)


@dataclass
class ClientConfig:
    """Client 配置"""
    server_host: str = "localhost"
    server_port: int = DEFAULT_GRPC_PORT
    timeout: float = 10.0
    
    # 策略配置 (Client 端指定)
    model_path: Optional[str] = None  # 模型路径或 HuggingFace repo id
    dataset_path: Optional[str] = None  # 数据集路径 (回放模式)
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
        elif self.dataset_path:
            return "dataset"
        return "none"
