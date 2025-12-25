# -*- coding: utf-8 -*-
"""
常量定义 (精简版 - 仅支持 V2.0)
"""

# ============================================================================
# V2.0 数据集维度配置
# ============================================================================

# V2.0 数据集 (不含底盘): [arm_left(7), arm_right(7), gripper_left(1), gripper_right(1), head(2), torso(4)]
LEROBOT_ACTION_DIM_NO_CHASSIS = 22

# V2.0 数据集 (含底盘): [arm_left(7), arm_right(7), gripper_left(1), gripper_right(1), head(2), torso(4), chassis(3)]
LEROBOT_ACTION_DIM_WITH_CHASSIS = 25

# 默认使用不含底盘的维度
LEROBOT_ACTION_DIM = LEROBOT_ACTION_DIM_NO_CHASSIS

# ============================================================================
# Action 各部件维度配置
# ============================================================================

ACTION_DIM_CONFIG = {
    'arm_left': 7,
    'arm_right': 7,
    'gripper_left': 1,
    'gripper_right': 1,
    'head': 2,
    'torso': 4,
    'chassis': 3,
}

# 各部件在 action 数组中的起始索引 (V2.0 格式)
ACTION_INDEX_CONFIG = {
    'arm_left': (0, 7),        # [0:7]
    'arm_right': (7, 14),      # [7:14]
    'gripper_left': (14, 15),  # [14:15]
    'gripper_right': (15, 16), # [15:16]
    'head': (16, 18),          # [16:18]
    'torso': (18, 22),         # [18:22]
    'chassis': (22, 25),       # [22:25]
}

# ============================================================================
# 关节索引
# ============================================================================

ARM_LEFT_INDICES = list(range(0, 7))
ARM_RIGHT_INDICES = list(range(7, 14))
ARM_INDICES = ARM_LEFT_INDICES + ARM_RIGHT_INDICES

# 夹爪索引
GRIPPER_LEFT_INDEX = 14
GRIPPER_RIGHT_INDEX = 15

# 头部索引
HEAD_INDICES = list(range(16, 18))

# 腰部索引
TORSO_INDICES = list(range(18, 22))

# 底盘索引
CHASSIS_INDICES = list(range(22, 25))

# ============================================================================
# Astribot 部件配置
# ============================================================================

# 不含底盘的部件列表 (默认使用)
ASTRIBOT_NAMES_LIST = [
    'astribot_torso',
    'astribot_arm_left',
    'astribot_gripper_left',
    'astribot_arm_right',
    'astribot_gripper_right',
    'astribot_head',
]

# 含底盘的部件列表
ASTRIBOT_NAMES_LIST_WITH_CHASSIS = [
    'astribot_torso',
    'astribot_arm_left',
    'astribot_gripper_left',
    'astribot_arm_right',
    'astribot_gripper_right',
    'astribot_head',
    'astribot_chassis',
]

ASTRIBOT_DOF_CONFIG = {
    'astribot_torso': 4,
    'astribot_arm_left': 7,
    'astribot_gripper_left': 1,
    'astribot_arm_right': 7,
    'astribot_gripper_right': 1,
    'astribot_head': 2,
    'astribot_chassis': 3,
}

# ============================================================================
# 默认配置
# ============================================================================

DEFAULT_CONTROL_FREQ = 30.0
DEFAULT_GRPC_PORT = 50051
DEFAULT_GRPC_HOST = "0.0.0.0"

# gRPC 配置
GRPC_MAX_MESSAGE_LENGTH = 50 * 1024 * 1024  # 50MB
GRPC_KEEPALIVE_TIME_MS = 10000
GRPC_KEEPALIVE_TIMEOUT_MS = 5000
