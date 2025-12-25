# -*- coding: utf-8 -*-
"""
LeRobot 推理服务器 (精简版)
运行环境: Python 3.10+ (lerobot 环境)

直接复用 LeRobot 的推理逻辑:
- 使用 get_policy_class 和 from_pretrained 加载模型
- 使用 make_pre_post_processors 创建预处理器和后处理器
- 使用 policy.predict_action_chunk 或 policy.select_action 进行推理

支持两种模式:
1. 模型推理模式: 使用训练好的策略模型
2. 数据集回放模式: 从数据集读取 action

配置方式: Server 以空闲模式启动，等待 Client 通过 Configure() 指定模型/数据集
"""

import os
import sys
import time
import signal
import logging
from concurrent import futures
from typing import Optional, Iterator, Dict, Any
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
from src.common.config import ServerConfig, ActionConfig
from src.common.utils import setup_logging
from src.common.constants import (
    LEROBOT_ACTION_DIM,
    LEROBOT_ACTION_DIM_NO_CHASSIS,
    LEROBOT_ACTION_DIM_WITH_CHASSIS,
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
# 数据集加载器 (V2.0 格式)
# ============================================================================
class DatasetLoader:
    """
    LeRobot 数据集加载器 (仅支持 V2.0 格式)
    
    V2.0 格式:
    - 22维: [arm_left(7), arm_right(7), gripper_left(1), gripper_right(1), head(2), torso(4)]
    - 25维: [arm_left(7), arm_right(7), gripper_left(1), gripper_right(1), head(2), torso(4), chassis(3)]
    """
    
    def __init__(self, dataset_path: str, action_config: Optional[ActionConfig] = None):
        self.dataset_path = dataset_path
        self.df = None
        self.info = None
        self.action_config = action_config or ActionConfig()
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
    
    def _check_action_columns(self):
        """检查 action 列结构"""
        columns = self.df.columns.tolist()
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
            action 数组 (22或25维)
        """
        config = action_config or self.action_config
        
        ep_df = self.df[self.df['episode_index'] == episode]
        ep_df = ep_df.sort_values('frame_index').reset_index(drop=True)
        
        if frame >= len(ep_df):
            return None
        
        row = ep_df.iloc[frame]
        
        if self.has_separate_action_columns:
            return self._assemble_action_from_columns(row, config)
        else:
            action = row['action']
            if isinstance(action, np.ndarray):
                return self._filter_action(action, config)
            return self._filter_action(np.array(action, dtype=np.float32), config)
    
    def _assemble_action_from_columns(self, row, config: ActionConfig) -> np.ndarray:
        """从分开的列组装 action (V2.0 格式)"""
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
        """根据配置过滤 action (V2.0 格式)"""
        if len(action) == LEROBOT_ACTION_DIM_WITH_CHASSIS:
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
        
        return action
    
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
                shape = self.info['features']['action'].get('shape', [22])
                return shape[0] if isinstance(shape, list) else shape
        
        # 从数据推断
        if self.df is not None and len(self.df) > 0:
            if self.has_separate_action_columns:
                return LEROBOT_ACTION_DIM_WITH_CHASSIS
            elif 'action' in self.df.columns:
                sample_action = self.df.iloc[0]['action']
                if hasattr(sample_action, 'shape'):
                    return sample_action.shape[0]
        
        return LEROBOT_ACTION_DIM_NO_CHASSIS


# ============================================================================
# LeRobot 模型推理器
# ============================================================================
class LeRobotModelInference:
    """
    LeRobot 模型推理器
    
    直接复用 LeRobot 的推理逻辑
    """
    
    def __init__(
        self, 
        pretrained_path: str, 
        device: str = "cuda",
        policy_type: Optional[str] = None
    ):
        if not HAS_LEROBOT:
            raise ImportError("需要安装 lerobot")
        if not HAS_TORCH:
            raise ImportError("需要安装 torch")
        
        self.pretrained_path = pretrained_path
        self.device = device
        self.policy_type = policy_type
        
        self.policy = None
        self.preprocessor = None
        self.postprocessor = None
        
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
        
        if hasattr(self.policy.config, 'chunk_size'):
            logger.info(f"Action chunk size: {self.policy.config.chunk_size}")
    
    def reset(self):
        """重置策略状态"""
        if self.policy:
            self.policy.reset()
    
    def predict(self, observation: Dict[str, Any]) -> np.ndarray:
        """执行单步推理"""
        if self.policy is None:
            raise RuntimeError("模型未加载")
        
        obs_processed = self.preprocessor(observation)
        
        with torch.inference_mode():
            action_tensor = self.policy.select_action(obs_processed)
        
        action_tensor = self.postprocessor(action_tensor)
        
        if isinstance(action_tensor, torch.Tensor):
            action = action_tensor.cpu().numpy()
        else:
            action = np.array(action_tensor)
        
        if action.ndim > 1:
            action = action.squeeze()
        
        return action


# ============================================================================
# gRPC 服务实现
# ============================================================================
class LeRobotInferenceServicer(pb2_grpc.LeRobotInferenceServiceServicer):
    """
    LeRobot 推理服务实现 (精简版)
    
    配置方式: Server 以空闲模式启动，等待 Client 通过 Configure() 指定模型/数据集
    """
    
    def __init__(self, config: ServerConfig):
        self.config = config
        self.is_ready = False
        
        # 推理模式
        self.mode = "none"  # "model", "dataset", "none"
        self.model_inference: Optional[LeRobotModelInference] = None
        self.dataset_loader: Optional[DatasetLoader] = None
        
        # Action 配置
        self.action_config = config.action_config
        
        # 状态
        self.current_episode = 0
        self.current_frame = 0
        self.model_name = "none"
        
        logger.info("Server 以空闲模式启动，等待 Client 配置...")
    
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
        self.model_name = os.path.basename(model_path)
        self.is_ready = True
        self.current_frame = 0
        logger.info("模型加载成功")
    
    def _load_dataset(self, dataset_path: str, action_config: Optional[ActionConfig] = None):
        """加载数据集"""
        logger.info(f"加载数据集: {dataset_path}")
        
        config = action_config or self.action_config
        self.dataset_loader = DatasetLoader(dataset_path, action_config=config)
        self.model_inference = None
        self.mode = "dataset"
        self.model_name = os.path.basename(dataset_path)
        self.is_ready = True
        self.current_frame = 0
        
        if action_config:
            self.action_config = action_config
        
        logger.info(f"数据集加载成功")
        logger.info(f"Action 配置: 头部={config.enable_head}, 腰部={config.enable_torso}, 底盘={config.enable_chassis}")
    
    def Configure(self, request: pb2.PolicyConfig, context) -> pb2.ServiceStatus:
        """Client 端配置策略"""
        logger.info(f"收到 Client 配置请求: mode={request.mode}")
        
        try:
            # 解析 action 配置
            action_config = self._parse_action_config(request)
            
            if request.mode == "model":
                if not request.model_path:
                    return self._get_status("错误: 未指定 model_path")
                
                device = request.device if request.device else "cuda"
                policy_type = request.policy_type if request.policy_type else None
                
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
        - "observation.images.{camera_name}": (1, C, H, W) tensor, 值域 [0, 1]
        """
        obs_dict = {}
        
        # 处理关节状态
        if request.joint_positions:
            state = torch.tensor(
                list(request.joint_positions),
                dtype=torch.float32
            ).unsqueeze(0)  # (1, state_dim)
            obs_dict["observation.state"] = state
        
        # 处理图像数据
        for img_data in request.images:
            if img_data.data:
                try:
                    image_tensor = self._decode_image(img_data)
                    if image_tensor is not None:
                        key = f"observation.images.{img_data.camera_name}"
                        obs_dict[key] = image_tensor
                        logger.debug(f"处理图像 {img_data.camera_name}: shape={image_tensor.shape}")
                except Exception as e:
                    logger.warning(f"图像解码失败 ({img_data.camera_name}): {e}")
        
        return obs_dict
    
    def _decode_image(self, img_data: pb2.ImageData) -> Optional[torch.Tensor]:
        """
        解码图像数据为 LeRobot 期望的 tensor 格式
        
        Args:
            img_data: gRPC ImageData 消息
            
        Returns:
            torch.Tensor: shape (1, C, H, W), 值域 [0, 1]
        """
        import io
        from PIL import Image
        
        encoding = img_data.encoding.lower()
        
        if encoding in ['jpeg', 'jpg', 'png']:
            # 解码压缩图像
            image = Image.open(io.BytesIO(img_data.data))
            image = image.convert('RGB')
        elif encoding == 'raw':
            # 原始 RGB 数据
            if img_data.width > 0 and img_data.height > 0:
                image = Image.frombytes('RGB', (img_data.width, img_data.height), img_data.data)
            else:
                logger.warning("raw 格式需要指定 width 和 height")
                return None
        else:
            logger.warning(f"不支持的图像编码格式: {encoding}")
            return None
        
        # 转换为 tensor: (H, W, C) -> (C, H, W), 归一化到 [0, 1]
        image_array = np.array(image, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)  # (C, H, W)
        image_tensor = image_tensor.unsqueeze(0)  # (1, C, H, W)
        
        return image_tensor
    
    def _get_action(self, observation: pb2.Observation) -> Optional[np.ndarray]:
        """获取 action"""
        if self.mode == "dataset":
            action = self.dataset_loader.get_action(
                observation.episode_id,
                observation.frame_index,
                action_config=self.action_config
            )
            return action
            
        elif self.mode == "model":
            obs_dict = self._build_observation_dict(observation)
            action = self.model_inference.predict(obs_dict)
            
            # 根据配置过滤 action
            if len(action) == LEROBOT_ACTION_DIM_WITH_CHASSIS:
                action = self._filter_action(action)
            
            return action
        
        else:
            return np.array(observation.joint_positions, dtype=np.float32)
    
    def _filter_action(self, action: np.ndarray) -> np.ndarray:
        """根据 action 配置过滤 action"""
        if len(action) != LEROBOT_ACTION_DIM_WITH_CHASSIS:
            return action
        
        parts = []
        parts.append(action[0:16])
        
        if self.action_config.enable_head:
            parts.append(action[16:18])
        
        if self.action_config.enable_torso:
            parts.append(action[18:22])
        
        if self.action_config.enable_chassis:
            parts.append(action[22:25])
        
        return np.concatenate(parts)
    
    def Predict(self, request: pb2.Observation, context) -> pb2.Action:
        """单次推理"""
        if not self.is_ready:
            return pb2.Action(
                status=pb2.NOT_READY,
                error_message="服务未就绪，请先调用 Configure"
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
    
    def Control(self, request: pb2.ControlCommand, context) -> pb2.ServiceStatus:
        """控制命令 (精简: 只支持 RESET 和 SET_EPISODE)"""
        cmd_type = request.type
        params = dict(request.params)
        
        if cmd_type == pb2.CMD_RESET:
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
    
    def _get_status(self, message: str = "") -> pb2.ServiceStatus:
        """构建状态响应"""
        total_frames = 0
        fps = self.config.fps
        
        if self.mode == "dataset" and self.dataset_loader:
            total_frames = self.dataset_loader.get_episode_length(self.current_episode)
            fps = self.dataset_loader.get_fps()
        
        return pb2.ServiceStatus(
            is_ready=self.is_ready,
            model_name=self.model_name,
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
        
        self.server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=self.config.max_workers),
            options=[
                ('grpc.max_send_message_length', 50 * 1024 * 1024),
                ('grpc.max_receive_message_length', 50 * 1024 * 1024),
                ('grpc.keepalive_time_ms', 10000),
                ('grpc.keepalive_timeout_ms', 5000),
            ]
        )
        
        self.servicer = LeRobotInferenceServicer(self.config)
        pb2_grpc.add_LeRobotInferenceServiceServicer_to_server(
            self.servicer, self.server
        )
        
        address = f'{self.config.host}:{self.config.port}'
        self.server.add_insecure_port(address)
        
        self.server.start()
        logger.info(f"gRPC 服务器已启动: {address}")
        logger.info(f"等待 Client 配置...")
        
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
        description='LeRobot 推理服务器 (精简版)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 启动服务器 (等待 Client 配置)
  python inference_server.py --port 50051
  
  # 指定设备
  python inference_server.py --port 50051 --device cuda:0
        """
    )
    
    parser.add_argument('--host', default='0.0.0.0', help='监听地址 (默认: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=50051, help='监听端口 (默认: 50051)')
    parser.add_argument('--device', default='cuda', help='推理设备 (默认: cuda)')
    parser.add_argument('--workers', type=int, default=10, help='工作线程数 (默认: 10)')
    parser.add_argument('--fps', type=float, default=30.0, help='目标帧率 (默认: 30)')
    
    args = parser.parse_args()
    
    config = ServerConfig(
        host=args.host,
        port=args.port,
        max_workers=args.workers,
        device=args.device,
        fps=args.fps,
    )
    
    run_server(config)


if __name__ == '__main__':
    main()
