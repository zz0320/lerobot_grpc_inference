# -*- coding: utf-8 -*-
"""
常量定义
"""

# ============================================================================
# 数据集版本配置
# ============================================================================

# V1.0 数据集: [arm_left(7), arm_right(7), gripper_left(1), gripper_right(1)]
LEROBOT_ACTION_DIM_V1 = 16

# V2.0 数据集: [arm_left(7), arm_right(7), gripper_left(1), gripper_right(1), 
#               head(2), torso(4), chassis(3)]
LEROBOT_ACTION_DIM_V2 = 25

# V2.0 无底盘: [arm_left(7), arm_right(7), gripper_left(1), gripper_right(1),
#               head(2), torso(4)]
LEROBOT_ACTION_DIM_V2_NO_CHASSIS = 22

# 默认使用 V2.0
LEROBOT_ACTION_DIM = LEROBOT_ACTION_DIM_V2

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
# 手臂关节索引 (兼容 V1.0)
# ============================================================================

ARM_LEFT_INDICES = list(range(0, 7))
ARM_RIGHT_INDICES = list(range(7, 14))
ARM_INDICES = ARM_LEFT_INDICES + ARM_RIGHT_INDICES

# 夹爪索引
GRIPPER_LEFT_INDEX = 14
GRIPPER_RIGHT_INDEX = 15

# 头部索引 (V2.0)
HEAD_INDICES = list(range(16, 18))

# 腰部索引 (V2.0)
TORSO_INDICES = list(range(18, 22))

# 底盘索引 (V2.0)
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
# 数据集列名配置 (用于解析 parquet 文件)
# ============================================================================

# V2.0 数据集 action 列名
DATASET_ACTION_COLUMNS_V2 = [
    'action.arm_left',
    'action.arm_right',
    'action.gripper_left',
    'action.gripper_right',
    'action.head',
    'action.torso',
    'action.chassis',
]

# V2.0 数据集 observation state 列名
DATASET_STATE_COLUMNS_V2 = [
    'observation.state.arm_left.position',
    'observation.state.arm_right.position',
    'observation.state.gripper_left.position',
    'observation.state.gripper_right.position',
    'observation.state.head.position',
    'observation.state.torso.position',
    'observation.state.chassis.position',
]

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


