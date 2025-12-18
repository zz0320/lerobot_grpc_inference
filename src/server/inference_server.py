# -*- coding: utf-8 -*-
"""
LeRobot 推理服务器
运行环境: Python 3.10+ (lerobot 环境)

直接复用 LeRobot 的推理逻辑:
- 使用 get_policy_class 和 from_pretrained 加载模型
- 使用 make_pre_post_processors 创建预处理器和后处理器
- 使用 policy.predict_action_chunk 或 policy.select_action 进行推理

支持两种模式:
1. 模型推理模式: 使用训练好的策略模型
2. 数据集回放模式: 从数据集读取 action
"""

import os
import sys
import time
import signal
import logging
from concurrent import futures
from typing import Optional, Iterator, Dict, Any, List
import numpy as np

import grpc

# 添加项目路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

# 导入生成的 protobuf 代码
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

# 导入通用模块
from src.common.config import Config, ServerConfig, ActionConfig
from src.common.utils import setup_logging
from src.common.constants import (
    LEROBOT_ACTION_DIM,
    LEROBOT_ACTION_DIM_V2,
    LEROBOT_ACTION_DIM_V2_NO_CHASSIS,
    ACTION_DIM_CONFIG,
    ACTION_INDEX_CONFIG,
    DATASET_ACTION_COLUMNS_V2,
    DATASET_STATE_COLUMNS_V2,
)

# ============================================================================
# LeRobot 推理相关导入
# ============================================================================
HAS_LEROBOT = False
HAS_TORCH = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    pass

try:
    # LeRobot 核心推理组件
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors
    from lerobot.policies.pretrained import PreTrainedPolicy
    from lerobot.processor import PolicyAction, PolicyProcessorPipeline
    HAS_LEROBOT = True
except ImportError:
    pass

# Pandas/PyArrow 导入 (用于数据集回放)
HAS_PANDAS = False
try:
    import pandas as pd
    import pyarrow.parquet as pq
    HAS_PANDAS = True
except ImportError:
    pass

logger = logging.getLogger("lerobot_inference.server")


# ============================================================================
# 数据集加载器 (用于回放模式)
# ============================================================================
class DatasetLoader:
    """
    LeRobot 数据集加载器
    
    支持两种数据集格式:
    - V1.0: action 列为单一数组 [arm_left(7), arm_right(7), gripper_left(1), gripper_right(1)]
    - V2.0: action 列可以是单一数组，也可以是分开的列 (action.arm_left, action.head, 等)
    """
    
    def __init__(self, dataset_path: str, action_config: Optional[ActionConfig] = None):
        self.dataset_path = dataset_path
        self.df = None
        self.info = None
        self.action_config = action_config or ActionConfig()
        self.dataset_version = "v1.0"  # 默认 V1.0
        self.has_separate_action_columns = False
        self._load()
    
    def _load(self):
        """加载数据集"""
        if not HAS_PANDAS:
            raise ImportError("需要安装 pandas 和 pyarrow: pip install pandas pyarrow")
        
        # 加载 info
        import json
        info_path = os.path.join(self.dataset_path, 'meta', 'info.json')
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                self.info = json.load(f)
            logger.info(f"数据集信息: {self.info}")
            
            # 检测数据集版本
            self._detect_dataset_version()
        
        # 加载数据
        data_dir = os.path.join(self.dataset_path, 'data')
        parquet_files = []
        
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.parquet'):
                    parquet_files.append(os.path.join(root, file))
        
        if not parquet_files:
            raise FileNotFoundError(f"未找到数据文件: {data_dir}")
        
        dfs = []
        for pf in sorted(parquet_files):
            df = pd.read_parquet(pf)
            dfs.append(df)
        
        self.df = pd.concat(dfs, ignore_index=True)
        logger.info(f"加载了 {len(self.df)} 帧数据")
        
        # 检查是否有分开的 action 列
        self._check_action_columns()
    
    def _detect_dataset_version(self):
        """检测数据集版本"""
        if self.info and 'features' in self.info:
            features = self.info['features']
            # 检查是否有头部/腰部/底盘特征
            if 'action.head' in features or 'action.torso' in features or 'action.chassis' in features:
                self.dataset_version = "v2.0"
                logger.info(f"检测到数据集版本: V2.0 (包含头部/腰部/底盘)")
            elif 'observation.state.head.position' in features:
                self.dataset_version = "v2.0"
                logger.info(f"检测到数据集版本: V2.0 (基于 observation 特征)")
            else:
                self.dataset_version = "v1.0"
                logger.info(f"检测到数据集版本: V1.0")
    
    def _check_action_columns(self):
        """检查 action 列结构"""
        columns = self.df.columns.tolist()
        
        # 检查是否有分开的 action 列
        self.has_separate_action_columns = 'action.arm_left' in columns
        
        if self.has_separate_action_columns:
            logger.info("数据集使用分开的 action 列 (action.arm_left, action.head, 等)")
        else:
            logger.info("数据集使用合并的 action 列")
    
    def get_action(self, episode: int, frame: int, 
                   action_config: Optional[ActionConfig] = None) -> Optional[np.ndarray]:
        """
        获取指定帧的 action
        
        Args:
            episode: episode 索引
            frame: 帧索引
            action_config: action 配置，控制输出哪些部件
            
        Returns:
            action 数组，根据配置返回对应维度
        """
        config = action_config or self.action_config
        
        ep_df = self.df[self.df['episode_index'] == episode]
        ep_df = ep_df.sort_values('frame_index').reset_index(drop=True)
        
        if frame >= len(ep_df):
            return None
        
        row = ep_df.iloc[frame]
        
        if self.has_separate_action_columns:
            # V2.0 格式: 从分开的列组装 action
            return self._assemble_action_from_columns(row, config)
        else:
            # V1.0 或合并的 V2.0 格式
            action = row['action']
            if isinstance(action, np.ndarray):
                return self._filter_action(action, config)
            return self._filter_action(np.array(action, dtype=np.float32), config)
    
    def _assemble_action_from_columns(self, row, config: ActionConfig) -> np.ndarray:
        """从分开的列组装 action"""
        action_parts = []
        
        # 基础部件 (总是包含)
        action_parts.append(np.array(row['action.arm_left'], dtype=np.float32))
        action_parts.append(np.array(row['action.arm_right'], dtype=np.float32))
        
        # 夹爪
        gripper_left = row['action.gripper_left']
        gripper_right = row['action.gripper_right']
        action_parts.append(np.array([gripper_left], dtype=np.float32).flatten())
        action_parts.append(np.array([gripper_right], dtype=np.float32).flatten())
        
        # 头部 (可选)
        if config.enable_head and 'action.head' in row.index:
            action_parts.append(np.array(row['action.head'], dtype=np.float32))
        
        # 腰部 (可选)
        if config.enable_torso and 'action.torso' in row.index:
            action_parts.append(np.array(row['action.torso'], dtype=np.float32))
        
        # 底盘 (可选)
        if config.enable_chassis and 'action.chassis' in row.index:
            action_parts.append(np.array(row['action.chassis'], dtype=np.float32))
        
        return np.concatenate(action_parts)
    
    def _filter_action(self, action: np.ndarray, config: ActionConfig) -> np.ndarray:
        """根据配置过滤 action"""
        # 如果 action 维度是 V2.0 (25维)，需要根据配置过滤
        if len(action) == LEROBOT_ACTION_DIM_V2:
            parts = []
            # 基础部件: arm_left(7) + arm_right(7) + gripper_left(1) + gripper_right(1)
            parts.append(action[0:16])
            
            # 头部 [16:18]
            if config.enable_head:
                parts.append(action[16:18])
            
            # 腰部 [18:22]
            if config.enable_torso:
                parts.append(action[18:22])
            
            # 底盘 [22:25]
            if config.enable_chassis:
                parts.append(action[22:25])
            
            return np.concatenate(parts)
        
        # V1.0 格式或不需要过滤，直接返回
        return action
    
    def get_full_action(self, episode: int, frame: int) -> Optional[np.ndarray]:
        """获取完整的 action (不过滤)"""
        ep_df = self.df[self.df['episode_index'] == episode]
        ep_df = ep_df.sort_values('frame_index').reset_index(drop=True)
        
        if frame >= len(ep_df):
            return None
        
        row = ep_df.iloc[frame]
        
        if self.has_separate_action_columns:
            # 返回完整的 25 维 action
            config = ActionConfig(enable_chassis=True, enable_head=True, enable_torso=True)
            return self._assemble_action_from_columns(row, config)
        else:
            action = row['action']
            if isinstance(action, np.ndarray):
                return action
            return np.array(action, dtype=np.float32)
    
    def get_episode_length(self, episode: int) -> int:
        """获取 episode 的长度"""
        ep_df = self.df[self.df['episode_index'] == episode]
        return len(ep_df)
    
    def get_total_episodes(self) -> int:
        """获取总 episode 数"""
        return self.df['episode_index'].nunique()
    
    def get_fps(self) -> float:
        """获取帧率"""
        if self.info:
            return float(self.info.get('fps', 30))
        return 30.0
    
    def get_action_dim(self) -> int:
        """获取 action 维度"""
        if self.info and 'features' in self.info:
            if 'action' in self.info['features']:
                shape = self.info['features']['action'].get('shape', [16])
                return shape[0] if isinstance(shape, list) else shape
        
        # 从数据推断
        if self.df is not None and len(self.df) > 0:
            if self.has_separate_action_columns:
                return LEROBOT_ACTION_DIM_V2
            elif 'action' in self.df.columns:
                sample_action = self.df.iloc[0]['action']
                if hasattr(sample_action, 'shape'):
                    return sample_action.shape[0]
        
        return LEROBOT_ACTION_DIM


# ============================================================================
# LeRobot 模型推理器
# ============================================================================
class LeRobotModelInference:
    """
    LeRobot 模型推理器
    
    直接复用 LeRobot 的推理逻辑:
    - 使用 get_policy_class 获取策略类
    - 使用 from_pretrained 加载预训练模型
    - 使用 make_pre_post_processors 创建处理器
    - 使用 predict_action_chunk 或 select_action 进行推理
    """
    
    def __init__(
        self, 
        pretrained_path: str, 
        device: str = "cuda",
        policy_type: Optional[str] = None
    ):
        """
        Args:
            pretrained_path: 预训练模型路径 (本地路径或 HuggingFace repo id)
            device: 推理设备 ("cuda", "cpu", "mps")
            policy_type: 策略类型 ("act", "diffusion", "pi0" 等)，如果为 None 则自动检测
        """
        if not HAS_LEROBOT:
            raise ImportError("需要安装 lerobot")
        if not HAS_TORCH:
            raise ImportError("需要安装 torch")
        
        self.pretrained_path = pretrained_path
        self.device = device
        self.policy_type = policy_type
        
        self.policy: Optional[PreTrainedPolicy] = None
        self.preprocessor: Optional[PolicyProcessorPipeline] = None
        self.postprocessor: Optional[PolicyProcessorPipeline] = None
        
        self._load()
    
    def _load(self):
        """加载模型和处理器"""
        logger.info(f"加载 LeRobot 模型: {self.pretrained_path}")
        logger.info(f"设备: {self.device}")
        
        start_time = time.time()
        
        # 1. 加载配置确定策略类型
        from lerobot.configs.policies import PreTrainedConfig
        config = PreTrainedConfig.from_pretrained(self.pretrained_path)
        
        if self.policy_type is None:
            self.policy_type = config.type
        
        logger.info(f"策略类型: {self.policy_type}")
        
        # 2. 获取策略类并加载模型
        policy_class = get_policy_class(self.policy_type)
        self.policy = policy_class.from_pretrained(self.pretrained_path)
        self.policy.to(self.device)
        self.policy.eval()
        
        # 3. 创建预处理器和后处理器
        device_override = {"device": self.device}
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config,
            pretrained_path=self.pretrained_path,
            preprocessor_overrides={"device_processor": device_override},
            postprocessor_overrides={"device_processor": device_override},
        )
        
        load_time = time.time() - start_time
        logger.info(f"模型加载完成，耗时: {load_time:.2f}s")
        
        # 打印模型信息
        if hasattr(self.policy.config, 'chunk_size'):
            logger.info(f"Action chunk size: {self.policy.config.chunk_size}")
        if hasattr(self.policy.config, 'n_action_steps'):
            logger.info(f"N action steps: {self.policy.config.n_action_steps}")
    
    def reset(self):
        """重置策略状态 (清除 action queue 等)"""
        if self.policy:
            self.policy.reset()
    
    def predict(self, observation: Dict[str, Any]) -> np.ndarray:
        """
        执行单步推理
        
        Args:
            observation: 观测数据字典，包含:
                - "observation.state": 关节状态 tensor
                - "observation.images.*": 图像 tensor (可选)
        
        Returns:
            action: numpy array，形状 (action_dim,)
        """
        if self.policy is None:
            raise RuntimeError("模型未加载")
        
        # 1. 预处理
        obs_processed = self.preprocessor(observation)
        
        # 2. 推理 - 使用 select_action (内部处理 action chunk 和 queue)
        with torch.inference_mode():
            action_tensor = self.policy.select_action(obs_processed)
        
        # 3. 后处理
        action_tensor = self.postprocessor(action_tensor)
        
        # 4. 转换为 numpy
        if isinstance(action_tensor, torch.Tensor):
            action = action_tensor.cpu().numpy()
        else:
            action = np.array(action_tensor)
        
        # 确保是 1D
        if action.ndim > 1:
            action = action.squeeze()
        
        return action
    
    def predict_chunk(self, observation: Dict[str, Any]) -> np.ndarray:
        """
        预测完整的 action chunk
        
        Args:
            observation: 观测数据字典
        
        Returns:
            action_chunk: numpy array，形状 (chunk_size, action_dim)
        """
        if self.policy is None:
            raise RuntimeError("模型未加载")
        
        # 1. 预处理
        obs_processed = self.preprocessor(observation)
        
        # 2. 推理 - 直接获取 action chunk
        with torch.inference_mode():
            action_chunk = self.policy.predict_action_chunk(obs_processed)
        
        # 3. 后处理每个 action
        if action_chunk.ndim == 2:
            action_chunk = action_chunk.unsqueeze(0)  # 添加 batch 维度
        
        _, chunk_size, _ = action_chunk.shape
        processed_actions = []
        for i in range(chunk_size):
            single_action = action_chunk[:, i, :]
            processed_action = self.postprocessor(single_action)
            processed_actions.append(processed_action)
        
        # Stack 并转换为 numpy
        action_chunk = torch.stack(processed_actions, dim=1).squeeze(0)
        
        if isinstance(action_chunk, torch.Tensor):
            return action_chunk.cpu().numpy()
        return np.array(action_chunk)


# ============================================================================
# gRPC 服务实现
# ============================================================================
class LeRobotInferenceServicer(pb2_grpc.LeRobotInferenceServiceServicer):
    """
    LeRobot 推理服务实现
    
    支持两种初始化方式:
    1. Server 启动时指定 (--model 或 --dataset)
    2. Client 连接时通过 Configure() 动态指定 (推荐)
    
    支持 V2.0 数据集格式:
    - 包含头部、腰部、底盘控制
    - 底盘控制可通过配置开关
    """
    
    def __init__(self, config: ServerConfig):
        self.config = config
        self.is_ready = False
        
        # 推理模式
        self.mode = "none"  # "model", "dataset", "none"
        self.model_inference: Optional[LeRobotModelInference] = None
        self.dataset_loader: Optional[DatasetLoader] = None
        
        # Action 配置 (控制输出哪些部件)
        self.action_config = config.action_config if hasattr(config, 'action_config') else ActionConfig()
        
        # 状态
        self.current_episode = 0
        self.current_frame = 0
        
        # 如果启动时指定了路径，则初始化
        self._initialize_from_config()
    
    def _initialize_from_config(self):
        """从启动配置初始化 (可选)"""
        try:
            if self.config.model_path:
                logger.info(f"从启动配置初始化模型: {self.config.model_path}")
                self._load_model(self.config.model_path, self.config.device)
                
            elif self.config.dataset_path:
                logger.info(f"从启动配置初始化数据集: {self.config.dataset_path}")
                self._load_dataset(self.config.dataset_path)
                
            else:
                logger.info("Server 以空闲模式启动，等待 Client 配置...")
                self.mode = "none"
                self.is_ready = False
                
        except Exception as e:
            logger.error(f"初始化失败: {e}")
            import traceback
            traceback.print_exc()
            self.is_ready = False
    
    def _load_model(self, model_path: str, device: str = "cuda", policy_type: str = None):
        """加载模型"""
        logger.info(f"加载模型: {model_path} (device={device})")
        self.model_inference = LeRobotModelInference(
            pretrained_path=model_path,
            device=device,
            policy_type=policy_type if policy_type else None
        )
        self.dataset_loader = None
        self.mode = "model"
        self.is_ready = True
        self.current_frame = 0
        logger.info("模型加载成功")
    
    def _load_dataset(self, dataset_path: str, action_config: Optional[ActionConfig] = None):
        """加载数据集"""
        logger.info(f"加载数据集: {dataset_path}")
        
        # 使用传入的配置或默认配置
        config = action_config or self.action_config
        
        self.dataset_loader = DatasetLoader(dataset_path, action_config=config)
        self.model_inference = None
        self.mode = "dataset"
        self.is_ready = True
        self.current_frame = 0
        
        # 更新 action 配置
        if action_config:
            self.action_config = action_config
        
        logger.info(f"数据集加载成功 (版本: {self.dataset_loader.dataset_version})")
        logger.info(f"Action 配置: 头部={config.enable_head}, 腰部={config.enable_torso}, 底盘={config.enable_chassis}")
    
    def Configure(self, request: pb2.PolicyConfig, context) -> pb2.ServiceStatus:
        """
        Client 端配置策略
        
        这是推荐的方式：Client 连接时指定要使用的模型或数据集
        
        支持配置:
        - mode: "model" 或 "dataset"
        - action_config: Action 输出配置 (可选)
            - enable_chassis: 是否包含底盘控制
            - enable_head: 是否包含头部控制
            - enable_torso: 是否包含腰部控制
        """
        logger.info(f"收到 Client 配置请求: mode={request.mode}")
        
        try:
            # 解析 action 配置
            action_config = self._parse_action_config(request)
            
            if request.mode == "model":
                if not request.model_path:
                    return self._get_status("错误: 未指定 model_path")
                
                device = request.device if request.device else "cuda"
                policy_type = request.policy_type if request.policy_type else None
                
                # 更新 action 配置
                if action_config:
                    self.action_config = action_config
                
                self._load_model(request.model_path, device, policy_type)
                return self._get_status(f"已加载模型: {request.model_path}")
                
            elif request.mode == "dataset":
                if not request.dataset_path:
                    return self._get_status("错误: 未指定 dataset_path")
                
                self._load_dataset(request.dataset_path, action_config)
                return self._get_status(f"已加载数据集: {request.dataset_path}")
                
            else:
                return self._get_status(f"错误: 未知模式 {request.mode}")
                
        except Exception as e:
            logger.error(f"配置失败: {e}")
            import traceback
            traceback.print_exc()
            return self._get_status(f"配置失败: {e}")
    
    def _parse_action_config(self, request: pb2.PolicyConfig) -> Optional[ActionConfig]:
        """解析 action 配置"""
        if hasattr(request, 'action_config') and request.HasField('action_config'):
            ac = request.action_config
            return ActionConfig(
                enable_chassis=ac.enable_chassis if hasattr(ac, 'enable_chassis') else False,
                enable_head=ac.enable_head if hasattr(ac, 'enable_head') else True,
                enable_torso=ac.enable_torso if hasattr(ac, 'enable_torso') else True,
            )
        return None
    
    def _build_observation_dict(self, request: pb2.Observation) -> Dict[str, Any]:
        """
        将 gRPC 请求转换为 LeRobot 观测字典格式
        
        LeRobot 期望的格式:
        - "observation.state": (1, state_dim) tensor
        - "observation.images.xxx": (1, C, H, W) tensor
        """
        obs_dict = {}
        
        # 关节状态
        if request.joint_positions:
            state = torch.tensor(
                list(request.joint_positions),
                dtype=torch.float32
            ).unsqueeze(0)  # (1, state_dim)
            obs_dict["observation.state"] = state
        
        # 图像 (如果有)
        for img_data in request.images:
            # 解码图像
            # TODO: 根据实际需求实现图像解码
            pass
        
        return obs_dict
    
    def _get_action(self, observation: pb2.Observation) -> Optional[np.ndarray]:
        """获取 action"""
        if self.mode == "dataset":
            # 从数据集获取 (使用当前的 action 配置)
            action = self.dataset_loader.get_action(
                observation.episode_id,
                observation.frame_index,
                action_config=self.action_config
            )
            return action
            
        elif self.mode == "model":
            # 模型推理
            obs_dict = self._build_observation_dict(observation)
            action = self.model_inference.predict(obs_dict)
            
            # 根据配置过滤 action
            if len(action) == LEROBOT_ACTION_DIM_V2:
                action = self._filter_action(action)
            
            return action
        
        else:
            # 模拟模式：返回输入
            return np.array(observation.joint_positions, dtype=np.float32)
    
    def _filter_action(self, action: np.ndarray) -> np.ndarray:
        """根据 action 配置过滤 action"""
        if len(action) != LEROBOT_ACTION_DIM_V2:
            return action
        
        parts = []
        # 基础部件: arm_left(7) + arm_right(7) + gripper_left(1) + gripper_right(1) = 16
        parts.append(action[0:16])
        
        # 头部 [16:18]
        if self.action_config.enable_head:
            parts.append(action[16:18])
        
        # 腰部 [18:22]
        if self.action_config.enable_torso:
            parts.append(action[18:22])
        
        # 底盘 [22:25]
        if self.action_config.enable_chassis:
            parts.append(action[22:25])
        
        return np.concatenate(parts)
    
    def Predict(self, request: pb2.Observation, context) -> pb2.Action:
        """单次推理"""
        if not self.is_ready:
            return pb2.Action(
                status=pb2.NOT_READY,
                error_message="服务未就绪"
            )
        
        try:
            action = self._get_action(request)
            
            if action is None:
                return pb2.Action(
                    status=pb2.EPISODE_END,
                    is_terminal=True
                )
            
            self.current_frame = request.frame_index + 1
            
            return pb2.Action(
                values=action.tolist(),
                is_terminal=False,
                status=pb2.OK,
                server_frame_index=self.current_frame
            )
            
        except Exception as e:
            logger.error(f"推理错误: {e}")
            import traceback
            traceback.print_exc()
            return pb2.Action(
                status=pb2.ERROR,
                error_message=str(e)
            )
    
    def StreamPredict(
        self,
        request_iterator: Iterator[pb2.Observation],
        context
    ) -> Iterator[pb2.Action]:
        """流式推理"""
        logger.info("开始流式推理")
        
        for obs in request_iterator:
            if context.is_active():
                action_response = self.Predict(obs, context)
                yield action_response
                
                if action_response.is_terminal:
                    break
            else:
                break
        
        logger.info("流式推理结束")
    
    def PredictBatch(
        self,
        request_iterator: Iterator[pb2.Observation],
        context
    ) -> Iterator[pb2.Action]:
        """批量推理"""
        return self.StreamPredict(request_iterator, context)
    
    def Control(self, request: pb2.ControlCommand, context) -> pb2.ServiceStatus:
        """控制命令"""
        cmd_type = request.type
        params = dict(request.params)
        
        if cmd_type == pb2.CMD_START:
            self.is_ready = True
            return self._get_status("服务已启动")
            
        elif cmd_type == pb2.CMD_STOP:
            self.is_ready = False
            return self._get_status("服务已停止")
            
        elif cmd_type == pb2.CMD_RESET:
            self.current_frame = 0
            if self.model_inference:
                self.model_inference.reset()
            return self._get_status("已重置")
            
        elif cmd_type == pb2.CMD_SET_EPISODE:
            ep = int(params.get("episode", "0"))
            self.current_episode = ep
            self.current_frame = 0
            if self.model_inference:
                self.model_inference.reset()
            return self._get_status(f"切换到 episode {ep}")
            
        elif cmd_type == pb2.CMD_LOAD_MODEL:
            model_path = params.get("path", "")
            if model_path:
                try:
                    self.model_inference = LeRobotModelInference(
                        pretrained_path=model_path,
                        device=self.config.device
                    )
                    self.mode = "model"
                    self.is_ready = True
                    return self._get_status(f"已加载模型: {model_path}")
                except Exception as e:
                    return self._get_status(f"加载模型失败: {e}")
            return self._get_status("未指定模型路径")
            
        elif cmd_type == pb2.CMD_LOAD_DATASET:
            dataset_path = params.get("path", "")
            if dataset_path:
                try:
                    self.dataset_loader = DatasetLoader(dataset_path)
                    self.mode = "dataset"
                    self.is_ready = True
                    return self._get_status(f"已加载数据集: {dataset_path}")
                except Exception as e:
                    return self._get_status(f"加载数据集失败: {e}")
            return self._get_status("未指定数据集路径")
        
        return self._get_status("未知命令")
    
    def GetStatus(self, request: pb2.Empty, context) -> pb2.ServiceStatus:
        """获取状态"""
        return self._get_status()
    
    def Reset(self, request: pb2.Empty, context) -> pb2.ServiceStatus:
        """重置"""
        self.current_frame = 0
        if self.model_inference:
            self.model_inference.reset()
        return self._get_status("已重置")
    
    def Ping(self, request: pb2.Heartbeat, context) -> pb2.HeartbeatResponse:
        """心跳检测"""
        return pb2.HeartbeatResponse(
            client_timestamp=request.client_timestamp,
            server_timestamp=int(time.time() * 1000),
            is_alive=self.is_ready
        )
    
    def _get_status(self, message: str = "") -> pb2.ServiceStatus:
        """构建状态响应"""
        total_frames = 0
        fps = self.config.fps
        
        if self.mode == "dataset" and self.dataset_loader:
            total_frames = self.dataset_loader.get_episode_length(self.current_episode)
            fps = self.dataset_loader.get_fps()
        
        model_name = "none"
        if self.mode == "model" and self.config.model_path:
            model_name = os.path.basename(self.config.model_path)
        elif self.mode == "dataset" and self.config.dataset_path:
            model_name = os.path.basename(self.config.dataset_path)
        
        return pb2.ServiceStatus(
            is_ready=self.is_ready,
            model_name=model_name,
            current_episode=self.current_episode,
            current_frame=self.current_frame,
            total_frames=total_frames,
            fps=fps,
            message=message,
            mode=self.mode
        )


# ============================================================================
# 服务器类
# ============================================================================
class InferenceServer:
    """推理服务器"""
    
    def __init__(self, config: ServerConfig):
        self.config = config
        self.server = None
        self.servicer = None
        self._stopped = False
    
    def start(self):
        """启动服务器"""
        if pb2 is None or pb2_grpc is None:
            raise RuntimeError("未找到 protobuf 生成文件，请先运行 scripts/generate_proto.sh")
        
        # 创建 gRPC 服务器
        self.server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=self.config.max_workers),
            options=[
                ('grpc.max_send_message_length', 50 * 1024 * 1024),
                ('grpc.max_receive_message_length', 50 * 1024 * 1024),
                ('grpc.keepalive_time_ms', 10000),
                ('grpc.keepalive_timeout_ms', 5000),
            ]
        )
        
        # 创建服务实现
        self.servicer = LeRobotInferenceServicer(self.config)
        pb2_grpc.add_LeRobotInferenceServiceServicer_to_server(
            self.servicer, self.server
        )
        
        # 添加反射服务 (方便调试)
        try:
            from grpc_reflection.v1alpha import reflection
            SERVICE_NAMES = (
                pb2.DESCRIPTOR.services_by_name['LeRobotInferenceService'].full_name,
                reflection.SERVICE_NAME,
            )
            reflection.enable_server_reflection(SERVICE_NAMES, self.server)
        except ImportError:
            logger.warning("grpc_reflection 未安装，跳过反射服务")
        
        # 绑定端口
        address = f'{self.config.host}:{self.config.port}'
        self.server.add_insecure_port(address)
        
        # 启动
        self.server.start()
        logger.info(f"gRPC 服务器已启动: {address}")
        logger.info(f"运行模式: {self.servicer.mode}")
        
        return self
    
    def wait_for_termination(self, timeout: Optional[float] = None):
        """等待服务器终止"""
        if self.server:
            self.server.wait_for_termination(timeout)
    
    def stop(self, grace: float = 5.0):
        """停止服务器"""
        if self.server and not self._stopped:
            logger.info("正在停止服务器...")
            self.server.stop(grace)
            self._stopped = True
            logger.info("服务器已停止")


def run_server(config: ServerConfig):
    """运行推理服务器"""
    setup_logging("INFO")
    
    server = InferenceServer(config)
    
    def signal_handler(signum, frame):
        logger.info(f"收到信号 {signum}，正在停止...")
        server.stop()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        server.start()
        server.wait_for_termination()
    except KeyboardInterrupt:
        server.stop()


def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='LeRobot 推理服务器',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 数据集回放模式 (V2.0 数据集，包含头部、腰部、底盘)
  python inference_server.py --dataset /path/to/dataset_v2
  
  # 数据集回放模式，启用底盘控制
  python inference_server.py --dataset /path/to/dataset_v2 --enable-chassis
  
  # 数据集回放模式，禁用头部和腰部控制
  python inference_server.py --dataset /path/to/dataset_v2 --no-head --no-torso
  
  # 模型推理模式 (使用 LeRobot 训练的模型)
  python inference_server.py --model /path/to/trained_model
  
  # 从 HuggingFace Hub 加载模型
  python inference_server.py --model lerobot/act_aloha_sim_insertion_human
  
  # 指定端口和设备
  python inference_server.py --model /path/to/model --port 50052 --device cuda:1
        """
    )
    
    parser.add_argument('--host', default='0.0.0.0', help='监听地址 (默认: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=50051, help='监听端口 (默认: 50051)')
    parser.add_argument('--model', type=str, help='模型路径或 HuggingFace repo id')
    parser.add_argument('--dataset', type=str, help='数据集路径 (数据集回放模式)')
    parser.add_argument('--device', default='cuda', help='推理设备 (默认: cuda)')
    parser.add_argument('--workers', type=int, default=10, help='工作线程数 (默认: 10)')
    parser.add_argument('--fps', type=float, default=30.0, help='目标帧率 (默认: 30)')
    
    # Action 配置参数
    parser.add_argument('--enable-chassis', action='store_true', 
                        help='启用底盘控制 (默认: 禁用)')
    parser.add_argument('--no-head', action='store_true',
                        help='禁用头部控制 (默认: 启用)')
    parser.add_argument('--no-torso', action='store_true',
                        help='禁用腰部控制 (默认: 启用)')
    
    args = parser.parse_args()
    
    # 构建 Action 配置
    action_config = ActionConfig(
        enable_chassis=args.enable_chassis,
        enable_head=not args.no_head,
        enable_torso=not args.no_torso
    )
    
    config = ServerConfig(
        host=args.host,
        port=args.port,
        max_workers=args.workers,
        model_path=args.model,
        dataset_path=args.dataset,
        device=args.device,
        fps=args.fps,
        action_config=action_config
    )
    
    # 打印配置信息
    logger.info(f"Action 配置:")
    logger.info(f"  - 头部控制: {'启用' if action_config.enable_head else '禁用'}")
    logger.info(f"  - 腰部控制: {'启用' if action_config.enable_torso else '禁用'}")
    logger.info(f"  - 底盘控制: {'启用' if action_config.enable_chassis else '禁用'}")
    logger.info(f"  - 输出维度: {action_config.action_dim}")
    
    run_server(config)


if __name__ == '__main__':
    main()
