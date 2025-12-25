# -*- coding: utf-8 -*-
"""
通用工具类 (精简版 - 仅支持 V2.0)
"""

import logging
import sys
from typing import List, Optional
import numpy as np

from .constants import (
    ARM_INDICES,
    LEROBOT_ACTION_DIM_NO_CHASSIS,
    LEROBOT_ACTION_DIM_WITH_CHASSIS,
)


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """
    设置日志
    
    Args:
        level: 日志级别
        log_file: 日志文件路径 (可选)
    
    Returns:
        配置好的 logger
    """
    logger = logging.getLogger("lerobot_inference")
    logger.setLevel(getattr(logging, level.upper()))
    
    # 清除已有的 handlers
    logger.handlers = []
    
    # 控制台输出
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件输出 (如果指定)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class ActionSmoother:
    """
    动作平滑器 - 使用简单移动平均消除抖动
    """
    
    def __init__(self, window_size: int = 5):
        """
        Args:
            window_size: 平滑窗口大小
        """
        self.window_size = window_size
        self._history: List[np.ndarray] = []
    
    def smooth(self, action: List[float]) -> List[float]:
        """平滑一个 action"""
        action_arr = np.array(action, dtype=np.float64)
        
        self._history.append(action_arr)
        if len(self._history) > self.window_size:
            self._history.pop(0)
        
        smoothed = np.mean(self._history, axis=0)
        return smoothed.tolist()
    
    def reset(self):
        """重置状态"""
        self._history = []


class VelocityLimiter:
    """
    速度限制器 - 限制相邻帧之间的最大变化量
    
    注意：只对手臂关节限制，不对夹爪限制
    """
    
    def __init__(self, max_delta: float = 0.05):
        """
        Args:
            max_delta: 每帧最大角度变化 (弧度)，默认 0.05 rad ≈ 2.9°
        """
        self.max_delta = max_delta
        self._last_action: Optional[np.ndarray] = None
    
    def limit(self, action: List[float]) -> List[float]:
        """限制速度 (只限制手臂，不限制夹爪)"""
        action_arr = np.array(action, dtype=np.float64)
        
        if self._last_action is None:
            self._last_action = action_arr.copy()
            return action_arr.tolist()
        
        limited = action_arr.copy()
        
        # 只对手臂关节进行速度限制
        for i in ARM_INDICES:
            if i < len(action_arr):
                delta = action_arr[i] - self._last_action[i]
                delta = np.clip(delta, -self.max_delta, self.max_delta)
                limited[i] = self._last_action[i] + delta
        
        self._last_action = limited.copy()
        return limited.tolist()
    
    def reset(self):
        """重置状态"""
        self._last_action = None


def lerobot_action_to_waypoint(action: List[float], include_chassis: bool = False) -> List[List[float]]:
    """
    将 LeRobot V2.0 格式的 action 转换为 Astribot waypoint 格式
    
    V2.0 格式:
    - 22维 (不含底盘): [arm_left(7), arm_right(7), gripper_left(1), gripper_right(1), head(2), torso(4)]
    - 25维 (含底盘): [arm_left(7), arm_right(7), gripper_left(1), gripper_right(1), head(2), torso(4), chassis(3)]
    
    Args:
        action: LeRobot action 数组 (22或25维)
        include_chassis: 是否包含底盘控制
        
    Returns:
        waypoint: [torso(4), arm_left(7), gripper_left(1), arm_right(7), gripper_right(1), head(2), chassis(3)?]
    """
    if isinstance(action, np.ndarray):
        action = action.tolist()
    
    action_len = len(action)
    
    # 确保 action 长度至少为 22
    if action_len < LEROBOT_ACTION_DIM_NO_CHASSIS:
        raise ValueError(f"Action 长度必须至少为 {LEROBOT_ACTION_DIM_NO_CHASSIS}，当前为 {action_len}")
    
    # 构建 waypoint
    # V2.0 格式: [arm_left(7), arm_right(7), gripper_left(1), gripper_right(1), head(2), torso(4), chassis(3)?]
    waypoint = [
        action[18:22],         # torso (4)
        action[0:7],           # arm_left (7)
        [action[14]],          # gripper_left (1)
        action[7:14],          # arm_right (7)
        [action[15]],          # gripper_right (1)
        action[16:18],         # head (2)
    ]
    
    # 添加底盘 (如果需要且 action 包含底盘数据)
    if include_chassis and action_len >= LEROBOT_ACTION_DIM_WITH_CHASSIS:
        waypoint.append(action[22:25])  # chassis (3)
    
    return waypoint


def waypoint_to_lerobot_action(waypoint: List[List[float]], include_chassis: bool = False) -> List[float]:
    """
    将 Astribot waypoint 格式转换为 LeRobot V2.0 格式的 action
    
    Args:
        waypoint: [torso(4), arm_left(7), gripper_left(1), arm_right(7), gripper_right(1), head(2), chassis(3)?]
        include_chassis: 是否包含底盘
        
    Returns:
        action: LeRobot action 数组 (22或25维)
    """
    torso = waypoint[0]         # 4
    arm_left = waypoint[1]      # 7
    gripper_left = waypoint[2]  # 1
    arm_right = waypoint[3]     # 7
    gripper_right = waypoint[4] # 1
    head = waypoint[5] if len(waypoint) > 5 else [0.0, 0.0]  # 2
    
    # V2.0 格式: [arm_left, arm_right, gripper_left, gripper_right, head, torso, chassis?]
    action = arm_left + arm_right + gripper_left + gripper_right + head + torso
    
    if include_chassis and len(waypoint) > 6:
        chassis = waypoint[6]  # 3
        action = action + chassis
    
    return action
