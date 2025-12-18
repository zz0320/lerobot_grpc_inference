# -*- coding: utf-8 -*-
"""
通用工具类
"""

import logging
import sys
from typing import List, Optional
import numpy as np

from .constants import (
    ARM_INDICES,
    LEROBOT_ACTION_DIM_V2,
    LEROBOT_ACTION_DIM_V2_NO_CHASSIS,
    ACTION_INDEX_CONFIG,
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
    动作平滑器 - 使用滑动平均消除抖动
    
    支持两种模式:
    1. EMA (指数移动平均): 对最近的值给予更高权重
    2. SMA (简单移动平均): 对窗口内所有值平均
    """
    
    def __init__(self, window_size: int = 5, mode: str = "sma"):
        """
        Args:
            window_size: 平滑窗口大小 (SMA) 或 alpha=2/(window_size+1) (EMA)
            mode: "sma" (简单移动平均) 或 "ema" (指数移动平均)
        """
        self.window_size = window_size
        self.mode = mode
        self.alpha = 2.0 / (window_size + 1)  # EMA 系数
        self._history: List[np.ndarray] = []
        self._last_smoothed: Optional[np.ndarray] = None
    
    def smooth(self, action: List[float]) -> List[float]:
        """平滑一个 action"""
        action_arr = np.array(action, dtype=np.float64)
        
        if self.mode == "ema":
            # EMA 模式
            if self._last_smoothed is None:
                self._last_smoothed = action_arr.copy()
                return action_arr.tolist()
            
            smoothed = self.alpha * action_arr + (1 - self.alpha) * self._last_smoothed
            self._last_smoothed = smoothed.copy()
            return smoothed.tolist()
        
        else:
            # SMA 模式 (默认)
            self._history.append(action_arr)
            if len(self._history) > self.window_size:
                self._history.pop(0)
            
            smoothed = np.mean(self._history, axis=0)
            return smoothed.tolist()
    
    def reset(self):
        """重置状态"""
        self._history = []
        self._last_smoothed = None


class VelocityLimiter:
    """
    速度限制器 - 限制相邻帧之间的最大变化量
    
    防止因为噪声导致的突然跳变
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
        
        # 夹爪 (14, 15) 不限制，直接使用原始值
        
        self._last_action = limited.copy()
        return limited.tolist()
    
    def reset(self):
        """重置状态"""
        self._last_action = None


def lerobot_action_to_waypoint(action: List[float], include_chassis: bool = False) -> List[List[float]]:
    """
    将 LeRobot 格式的 action 转换为 Astribot waypoint 格式
    
    支持两种 action 格式:
    - V1.0 (16维): [arm_left(7), arm_right(7), gripper_left(1), gripper_right(1)]
    - V2.0 (22/25维): [arm_left(7), arm_right(7), gripper_left(1), gripper_right(1), 
                       head(2), torso(4), chassis(3, 可选)]
    
    Args:
        action: LeRobot action 数组
        include_chassis: 是否包含底盘控制
        
    Returns:
        waypoint: [torso(4), arm_left(7), gripper_left(1), arm_right(7), gripper_right(1), head(2), chassis(3)?]
    """
    if isinstance(action, np.ndarray):
        action = action.tolist()
    
    action_len = len(action)
    
    # V1.0 格式 (16维): 只有手臂和夹爪
    if action_len == 16:
        waypoint = [
            [0.0, 0.0, 0.0, 0.0],  # torso: 零值
            action[0:7],           # arm_left
            [action[14]],          # gripper_left
            action[7:14],          # arm_right
            [action[15]],          # gripper_right
            [0.0, 0.0]             # head: 零值
        ]
        return waypoint
    
    # V2.0 格式 (22维): 不含底盘
    # [arm_left(7), arm_right(7), gripper_left(1), gripper_right(1), head(2), torso(4)]
    if action_len == 22:
        waypoint = [
            action[18:22],         # torso (4)
            action[0:7],           # arm_left (7)
            [action[14]],          # gripper_left (1)
            action[7:14],          # arm_right (7)
            [action[15]],          # gripper_right (1)
            action[16:18],         # head (2)
        ]
        return waypoint
    
    # V2.0 格式 (25维): 含底盘
    # [arm_left(7), arm_right(7), gripper_left(1), gripper_right(1), head(2), torso(4), chassis(3)]
    if action_len >= 25:
        waypoint = [
            action[18:22],         # torso (4)
            action[0:7],           # arm_left (7)
            [action[14]],          # gripper_left (1)
            action[7:14],          # arm_right (7)
            [action[15]],          # gripper_right (1)
            action[16:18],         # head (2)
        ]
        if include_chassis:
            waypoint.append(action[22:25])  # chassis (3)
        return waypoint
    
    # 默认处理: 补齐到 16 维
    while len(action) < 16:
        action.append(0.0)
    
    waypoint = [
        [0.0, 0.0, 0.0, 0.0],  # torso: 零值
        action[0:7],           # arm_left
        [action[14]],          # gripper_left
        action[7:14],          # arm_right
        [action[15]],          # gripper_right
        [0.0, 0.0]             # head: 零值
    ]
    
    return waypoint


def waypoint_to_lerobot_action(waypoint: List[List[float]], version: str = "v2") -> List[float]:
    """
    将 Astribot waypoint 格式转换为 LeRobot 格式的 action
    
    Args:
        waypoint: [torso(4), arm_left(7), gripper_left(1), arm_right(7), gripper_right(1), head(2), chassis(3)?]
        version: "v1" 返回 16 维, "v2" 返回 22 维, "v2_chassis" 返回 25 维
        
    Returns:
        action: LeRobot action 数组
    """
    torso = waypoint[0]         # 4
    arm_left = waypoint[1]      # 7
    gripper_left = waypoint[2]  # 1
    arm_right = waypoint[3]     # 7
    gripper_right = waypoint[4] # 1
    head = waypoint[5] if len(waypoint) > 5 else [0.0, 0.0]  # 2
    chassis = waypoint[6] if len(waypoint) > 6 else [0.0, 0.0, 0.0]  # 3
    
    if version == "v1":
        # V1.0: [arm_left(7), arm_right(7), gripper_left(1), gripper_right(1)]
        action = arm_left + arm_right + gripper_left + gripper_right
    elif version == "v2_chassis":
        # V2.0 含底盘: [arm_left, arm_right, gripper_left, gripper_right, head, torso, chassis]
        action = arm_left + arm_right + gripper_left + gripper_right + head + torso + chassis
    else:
        # V2.0 不含底盘: [arm_left, arm_right, gripper_left, gripper_right, head, torso]
        action = arm_left + arm_right + gripper_left + gripper_right + head + torso
    
    return action


