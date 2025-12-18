# -*- coding: utf-8 -*-
"""
配置管理
"""

import os
import json
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List

from .constants import (
    DEFAULT_CONTROL_FREQ,
    DEFAULT_GRPC_PORT,
    DEFAULT_GRPC_HOST,
    LEROBOT_ACTION_DIM_V2,
    LEROBOT_ACTION_DIM_V2_NO_CHASSIS,
)


@dataclass
class ActionConfig:
    """Action 输出配置"""
    # 是否包含底盘控制 (可选)
    enable_chassis: bool = False
    # 是否包含头部控制
    enable_head: bool = True
    # 是否包含腰部控制
    enable_torso: bool = True
    
    @property
    def action_dim(self) -> int:
        """计算 action 维度"""
        dim = 14 + 2  # arm_left(7) + arm_right(7) + gripper_left(1) + gripper_right(1)
        if self.enable_head:
            dim += 2
        if self.enable_torso:
            dim += 4
        if self.enable_chassis:
            dim += 3
        return dim
    
    @property
    def enabled_parts(self) -> List[str]:
        """获取启用的部件列表"""
        parts = ['arm_left', 'arm_right', 'gripper_left', 'gripper_right']
        if self.enable_head:
            parts.append('head')
        if self.enable_torso:
            parts.append('torso')
        if self.enable_chassis:
            parts.append('chassis')
        return parts


@dataclass
class ServerConfig:
    """Server 配置"""
    host: str = DEFAULT_GRPC_HOST
    port: int = DEFAULT_GRPC_PORT
    max_workers: int = 10
    
    # 模型配置
    model_path: Optional[str] = None
    dataset_path: Optional[str] = None
    device: str = "cuda"
    
    # 推理配置
    fps: float = DEFAULT_CONTROL_FREQ
    
    # Action 配置
    action_config: ActionConfig = field(default_factory=ActionConfig)


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
    
    # 规划配置
    planning_frames: int = 5
    planning_duration: float = 3.0
    
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


@dataclass
class Config:
    """
    统一配置类
    
    支持从文件、环境变量、命令行参数加载
    """
    server: ServerConfig = field(default_factory=ServerConfig)
    client: ClientConfig = field(default_factory=ClientConfig)
    
    # 运行模式
    mode: str = "client"  # "server" or "client"
    
    # 日志配置
    log_level: str = "INFO"
    log_file: Optional[str] = None
    
    @classmethod
    def from_file(cls, path: str) -> "Config":
        """从 JSON 文件加载配置"""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """从字典加载配置"""
        config = cls()
        
        if 'server' in data:
            server_data = data['server'].copy()
            # 处理嵌套的 action_config
            if 'action_config' in server_data:
                server_data['action_config'] = ActionConfig(**server_data['action_config'])
            config.server = ServerConfig(**server_data)
        if 'client' in data:
            client_data = data['client'].copy()
            # 处理嵌套的 action_config
            if 'action_config' in client_data:
                client_data['action_config'] = ActionConfig(**client_data['action_config'])
            config.client = ClientConfig(**client_data)
        if 'mode' in data:
            config.mode = data['mode']
        if 'log_level' in data:
            config.log_level = data['log_level']
        if 'log_file' in data:
            config.log_file = data['log_file']
        
        return config
    
    @classmethod
    def from_env(cls) -> "Config":
        """从环境变量加载配置"""
        config = cls()
        
        # Server 配置
        if os.getenv('LEROBOT_SERVER_HOST'):
            config.server.host = os.getenv('LEROBOT_SERVER_HOST')
        if os.getenv('LEROBOT_SERVER_PORT'):
            config.server.port = int(os.getenv('LEROBOT_SERVER_PORT'))
        if os.getenv('LEROBOT_MODEL_PATH'):
            config.server.model_path = os.getenv('LEROBOT_MODEL_PATH')
        if os.getenv('LEROBOT_DATASET_PATH'):
            config.server.dataset_path = os.getenv('LEROBOT_DATASET_PATH')
        if os.getenv('LEROBOT_DEVICE'):
            config.server.device = os.getenv('LEROBOT_DEVICE')
        
        # Client 配置
        if os.getenv('LEROBOT_CLIENT_HOST'):
            config.client.server_host = os.getenv('LEROBOT_CLIENT_HOST')
        if os.getenv('LEROBOT_CLIENT_PORT'):
            config.client.server_port = int(os.getenv('LEROBOT_CLIENT_PORT'))
        if os.getenv('LEROBOT_CONTROL_FREQ'):
            config.client.control_freq = float(os.getenv('LEROBOT_CONTROL_FREQ'))
        
        # 通用配置
        if os.getenv('LEROBOT_MODE'):
            config.mode = os.getenv('LEROBOT_MODE')
        if os.getenv('LEROBOT_LOG_LEVEL'):
            config.log_level = os.getenv('LEROBOT_LOG_LEVEL')
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'server': asdict(self.server),
            'client': asdict(self.client),
            'mode': self.mode,
            'log_level': self.log_level,
            'log_file': self.log_file
        }
    
    def save(self, path: str):
        """保存配置到文件"""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

