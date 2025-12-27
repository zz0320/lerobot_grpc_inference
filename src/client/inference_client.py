# -*- coding: utf-8 -*-
"""
LeRobot 推理客户端 (精简版)
运行环境: 机器人侧 (Astribot SDK 环境)

通过 gRPC 连接远程推理服务器获取 action
配置方式: Client 连接时通过 Configure() 指定模型/数据集
"""

import os
import sys
import time
import signal
import logging
from collections import deque
from typing import List, Optional, Iterator, Callable, Dict
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
from src.common.config import ClientConfig, ActionConfig
from src.common.utils import (
    setup_logging, 
    ActionSmoother, 
    VelocityLimiter,
    lerobot_action_to_waypoint
)
from src.common.constants import (
    ASTRIBOT_NAMES_LIST,
    ASTRIBOT_NAMES_LIST_WITH_CHASSIS,
    LEROBOT_ACTION_DIM,
    LEROBOT_ACTION_DIM_NO_CHASSIS,
    LEROBOT_ACTION_DIM_WITH_CHASSIS,
    GRPC_MAX_MESSAGE_LENGTH,
    READY_POSITION_22,
    READY_POSITION_25,
)

# Astribot SDK
HAS_ASTRIBOT = False
try:
    from core.astribot_api.astribot_client import Astribot
    HAS_ASTRIBOT = True
except ImportError:
    pass

# ROS 相机
HAS_ROS = False
try:
    import rospy
    from sensor_msgs.msg import CompressedImage
    HAS_ROS = True
except ImportError:
    pass

logger = logging.getLogger("lerobot_inference.client")


# ============================================================================
# Astribot 图像话题配置
# ============================================================================

# ROS 图像话题 -> 相机名称
ASTRIBOT_IMAGE_TOPICS = {
    '/astribot_camera/head_rgbd/color_compress/compressed': 'head',
    '/astribot_camera/left_wrist_rgbd/color_compress/compressed': 'wrist_left',
    '/astribot_camera/right_wrist_rgbd/color_compress/compressed': 'wrist_right',
    '/astribot_camera/torso_rgbd/color_compress/compressed': 'torso',
}

# 图像尺寸 (H, W, C)
ASTRIBOT_IMAGE_SHAPES = {
    'head': (720, 1280, 3),
    'wrist_left': (360, 640, 3),
    'wrist_right': (360, 640, 3),
    'torso': (720, 1280, 3),
}


class AstribotCameraSubscriber:
    """
    Astribot ROS 图像话题订阅器
    
    从 ROS 话题获取压缩图像数据
    
    Example:
        >>> camera = AstribotCameraSubscriber()
        >>> camera.start()
        >>> 
        >>> # 获取图像 (返回 JPEG bytes)
        >>> head_img = camera.get_image('head')
        >>> wrist_left_img = camera.get_image('wrist_left')
        >>> 
        >>> # 获取所有图像
        >>> all_images = camera.get_all_images()
        >>> 
        >>> camera.stop()
    """
    
    def __init__(self, camera_names: List[str] = None):
        """
        Args:
            camera_names: 要订阅的相机名称列表，默认全部 ['head', 'wrist_left', 'wrist_right', 'torso']
        """
        if not HAS_ROS:
            raise ImportError("需要安装 ROS: rospy, sensor_msgs")
        
        self.camera_names = camera_names or list(ASTRIBOT_IMAGE_SHAPES.keys())
        self._images: Dict[str, bytes] = {}
        self._subscribers = []
        self._initialized = False
    
    def start(self, init_node: bool = True):
        """
        启动订阅
        
        Args:
            init_node: 是否初始化 ROS 节点 (如果已有节点运行则设为 False)
        """
        if init_node:
            try:
                rospy.init_node('astribot_camera_subscriber', anonymous=True)
            except rospy.exceptions.ROSException:
                pass  # 节点已初始化
        
        # 订阅话题
        for topic, cam_name in ASTRIBOT_IMAGE_TOPICS.items():
            if cam_name in self.camera_names:
                sub = rospy.Subscriber(
                    topic,
                    CompressedImage,
                    self._callback,
                    callback_args=cam_name,
                    queue_size=1
                )
                self._subscribers.append(sub)
                logger.info(f"订阅相机: {cam_name} <- {topic}")
        
        self._initialized = True
        logger.info(f"相机订阅器已启动，共 {len(self._subscribers)} 个相机")
    
    def _callback(self, msg: "CompressedImage", cam_name: str):
        """ROS 回调函数"""
        self._images[cam_name] = bytes(msg.data)
    
    def get_image(self, camera_name: str) -> Optional[bytes]:
        """
        获取指定相机的图像 (JPEG/PNG bytes)
        
        Args:
            camera_name: 相机名称 ('head', 'wrist_left', 'wrist_right', 'torso')
            
        Returns:
            图像数据 (bytes)，如果没有数据则返回 None
        """
        return self._images.get(camera_name)
    
    def get_all_images(self) -> Dict[str, bytes]:
        """
        获取所有相机的图像
        
        Returns:
            {camera_name: image_bytes} 字典
        """
        return {k: v for k, v in self._images.items() if v is not None}
    
    def get_images_for_inference(self, client: "InferenceClient") -> List[dict]:
        """
        获取用于推理的编码图像列表
        
        Args:
            client: InferenceClient 实例 (用于 encode_image)
            
        Returns:
            可直接传递给 predict() 的图像列表
        """
        images = []
        for cam_name, img_bytes in self._images.items():
            if img_bytes:
                images.append({
                    'name': cam_name,
                    'data': img_bytes,
                    'width': ASTRIBOT_IMAGE_SHAPES[cam_name][1],
                    'height': ASTRIBOT_IMAGE_SHAPES[cam_name][0],
                    'encoding': 'jpeg'  # ROS CompressedImage 通常是 JPEG
                })
        return images
    
    def wait_for_images(self, timeout: float = 5.0) -> bool:
        """
        等待所有相机图像就绪
        
        Args:
            timeout: 超时时间 (秒)
            
        Returns:
            是否所有图像都已就绪
        """
        start = time.time()
        while time.time() - start < timeout:
            if all(cam in self._images for cam in self.camera_names):
                return True
            time.sleep(0.1)
        return False
    
    def stop(self):
        """停止订阅"""
        for sub in self._subscribers:
            sub.unregister()
        self._subscribers = []
        self._images = {}
        self._initialized = False
        logger.info("相机订阅器已停止")

# 全局中断标志
_interrupted = False

def _signal_handler(signum, frame):
    global _interrupted
    _interrupted = True
    logger.warning(f"收到中断信号 {signum}")

signal.signal(signal.SIGINT, _signal_handler)


class ActionChunkManager:
    """
    Action Chunk 管理器
    
    在 Client 端管理 action queue，实现：
    1. 从 Server 获取完整的 action chunk
    2. 在本地逐步消费 action
    3. 当 queue 用完时，自动请求新的 chunk
    
    这种方式减少了网络调用频率，适用于 action chunking 策略 (ACT, Diffusion 等)
    
    Example:
        >>> # 创建 chunk 管理器
        >>> chunk_manager = ActionChunkManager(
        ...     client=inference_client,
        ...     n_action_steps=50  # 每个 chunk 使用的 action 数量
        ... )
        >>> 
        >>> # 在控制循环中使用
        >>> for frame_idx in range(1000):
        ...     # 获取下一个 action (自动管理 chunk)
        ...     action = chunk_manager.get_action(
        ...         joint_positions=current_state,
        ...         images=images,
        ...         episode_id=0,
        ...         frame_index=frame_idx
        ...     )
        ...     if action is None:
        ...         break
        ...     # 发送到机器人...
    """
    
    def __init__(
        self,
        client: "InferenceClient",
        n_action_steps: Optional[int] = None,
        auto_refill_threshold: float = 0.0
    ):
        """
        初始化 Action Chunk 管理器
        
        Args:
            client: InferenceClient 实例
            n_action_steps: 每个 chunk 实际使用的 action 数量
                           如果为 None，使用 Server 返回的完整 chunk
                           如果小于 chunk_size，只使用前 n_action_steps 个 action
            auto_refill_threshold: 自动补充阈值 (0.0-1.0)
                                   当 queue 剩余比例低于此值时提前获取新 chunk
                                   0.0 表示用完才请求
        """
        self.client = client
        self.n_action_steps = n_action_steps
        self.auto_refill_threshold = auto_refill_threshold
        
        # Action queue
        self._action_queue: deque = deque()
        self._chunk_size = 0  # Server 返回的 chunk 大小
        self._action_dim = 0
        
        # 状态
        self._current_chunk_start_frame = 0
        self._actions_consumed = 0  # 当前 chunk 已消费的 action 数
        self._total_actions_consumed = 0  # 总共消费的 action 数
        self._is_terminal = False
        
        logger.info(f"ActionChunkManager 初始化: n_action_steps={n_action_steps}, "
                   f"auto_refill_threshold={auto_refill_threshold}")
    
    @property
    def queue_size(self) -> int:
        """当前 queue 中的 action 数量"""
        return len(self._action_queue)
    
    @property
    def chunk_size(self) -> int:
        """Server 返回的 chunk 大小"""
        return self._chunk_size
    
    @property
    def action_dim(self) -> int:
        """Action 维度"""
        return self._action_dim
    
    @property
    def is_empty(self) -> bool:
        """Queue 是否为空"""
        return len(self._action_queue) == 0
    
    @property
    def is_terminal(self) -> bool:
        """Episode 是否结束"""
        return self._is_terminal and self.is_empty
    
    def reset(self):
        """重置状态"""
        self._action_queue.clear()
        self._actions_consumed = 0
        self._total_actions_consumed = 0
        self._is_terminal = False
        self._current_chunk_start_frame = 0
        logger.debug("ActionChunkManager 已重置")
    
    def _should_refill(self) -> bool:
        """检查是否需要补充 action"""
        if self._is_terminal:
            return False
        if self._chunk_size == 0:
            return True  # 首次请求
        
        effective_size = self.n_action_steps or self._chunk_size
        remaining_ratio = len(self._action_queue) / effective_size
        return remaining_ratio <= self.auto_refill_threshold
    
    def _fetch_chunk(
        self,
        joint_positions: List[float],
        episode_id: int,
        frame_index: int,
        images: Optional[List[dict]] = None
    ) -> bool:
        """
        从 Server 获取新的 action chunk
        
        Returns:
            是否成功获取
        """
        logger.debug(f"请求新的 action chunk, frame_index={frame_index}")
        
        try:
            chunk_response = self.client.predict_chunk(
                joint_positions=joint_positions,
                episode_id=episode_id,
                frame_index=frame_index,
                images=images
            )
            
            if chunk_response.status == pb2.EPISODE_END:
                logger.info("Episode 结束")
                self._is_terminal = True
                return False
            
            if chunk_response.status != pb2.OK:
                logger.error(f"获取 chunk 失败: {chunk_response.error_message}")
                return False
            
            # 更新 chunk 信息
            self._chunk_size = chunk_response.chunk_size
            self._action_dim = chunk_response.action_dim
            self._current_chunk_start_frame = frame_index
            
            # 清空旧的 queue 并添加新的 actions
            self._action_queue.clear()
            self._actions_consumed = 0
            
            # 确定实际使用的 action 数量
            n_to_use = self.n_action_steps if self.n_action_steps else self._chunk_size
            n_to_use = min(n_to_use, len(chunk_response.actions))
            
            for i in range(n_to_use):
                action = list(chunk_response.actions[i].values)
                self._action_queue.append(action)
            
            logger.debug(f"获取到 chunk: size={self._chunk_size}, "
                        f"使用={n_to_use}, dim={self._action_dim}")
            
            return True
            
        except Exception as e:
            logger.error(f"获取 chunk 异常: {e}")
            return False
    
    def get_action(
        self,
        joint_positions: List[float],
        episode_id: int = 0,
        frame_index: int = 0,
        images: Optional[List[dict]] = None
    ) -> Optional[List[float]]:
        """
        获取下一个 action
        
        自动管理 chunk 请求:
        - 如果 queue 为空，请求新的 chunk
        - 如果到达 auto_refill_threshold，提前请求
        
        Args:
            joint_positions: 当前关节位置
            episode_id: episode 索引
            frame_index: 当前帧索引
            images: 图像列表
        
        Returns:
            action 列表，如果 episode 结束返回 None
        """
        # 检查是否需要获取新 chunk
        if self.is_empty or self._should_refill():
            # 计算请求 chunk 的起始帧
            chunk_frame = self._current_chunk_start_frame + self._actions_consumed
            if self.is_empty:
                chunk_frame = frame_index
            
            if not self._fetch_chunk(joint_positions, episode_id, chunk_frame, images):
                if self._is_terminal and not self.is_empty:
                    pass  # 还有剩余 action，继续消费
                elif self.is_empty:
                    return None
        
        # 从 queue 中取出 action
        if self.is_empty:
            return None
        
        action = self._action_queue.popleft()
        self._actions_consumed += 1
        self._total_actions_consumed += 1
        
        return action
    
    def peek_action(self) -> Optional[List[float]]:
        """查看下一个 action (不移除)"""
        if self.is_empty:
            return None
        return list(self._action_queue[0])
    
    def get_remaining_actions(self) -> List[List[float]]:
        """获取所有剩余的 actions"""
        return [list(a) for a in self._action_queue]


class InferenceClient:
    """
    gRPC 推理客户端
    
    负责与远程推理服务器通信
    """
    
    def __init__(
        self, 
        server_address: str = "localhost:50051", 
        timeout: float = 10.0
    ):
        self.server_address = server_address
        self.timeout = timeout
        self.channel = None
        self.stub = None
        self._connected = False
        
        self._connect()
    
    def _connect(self):
        """连接到服务器"""
        if pb2 is None or pb2_grpc is None:
            raise RuntimeError("未找到 protobuf 生成文件")
        
        logger.info(f"连接推理服务器: {self.server_address}")
        
        self.channel = grpc.insecure_channel(
            self.server_address,
            options=[
                ('grpc.max_send_message_length', GRPC_MAX_MESSAGE_LENGTH),
                ('grpc.max_receive_message_length', GRPC_MAX_MESSAGE_LENGTH),
                ('grpc.keepalive_time_ms', 10000),
                ('grpc.keepalive_timeout_ms', 5000),
            ]
        )
        self.stub = pb2_grpc.LeRobotInferenceServiceStub(self.channel)
        
        try:
            grpc.channel_ready_future(self.channel).result(timeout=self.timeout)
            self._connected = True
            logger.info(f"已连接到推理服务器")
        except grpc.FutureTimeoutError:
            raise ConnectionError(f"无法连接到服务器: {self.server_address}")
    
    @property
    def is_connected(self) -> bool:
        return self._connected
    
    def get_status(self) -> "pb2.ServiceStatus":
        """获取服务状态"""
        return self.stub.GetStatus(pb2.Empty())
    
    def configure(
        self,
        mode: str,
        model_path: str = "",
        dataset_path: str = "",
        device: str = "cuda",
        policy_type: str = "",
        action_config: Optional[ActionConfig] = None
    ) -> "pb2.ServiceStatus":
        """
        配置 Server 使用的模型/数据集
        
        Args:
            mode: "model" 或 "dataset"
            model_path: 模型路径或 HuggingFace repo id (mode="model" 时)
            dataset_path: 数据集路径 (mode="dataset" 时)
            device: 推理设备
            policy_type: 策略类型 (可选)
            action_config: Action 输出配置
        """
        config = pb2.PolicyConfig(
            mode=mode,
            model_path=model_path,
            dataset_path=dataset_path,
            device=device,
            policy_type=policy_type
        )
        
        if action_config:
            config.action_config.CopyFrom(pb2.ActionOutputConfig(
                enable_chassis=action_config.enable_chassis,
                enable_head=action_config.enable_head,
                enable_torso=action_config.enable_torso
            ))
        
        return self.stub.Configure(config)
    
    @staticmethod
    def encode_image(image, camera_name: str = "cam", encoding: str = "jpeg", quality: int = 85) -> dict:
        """
        将图像编码为可发送的格式
        
        Args:
            image: PIL Image, numpy array (H, W, C), 或 bytes
            camera_name: 相机名称 (e.g., "cam_left", "cam_right", "cam_wrist")
            encoding: 编码格式 ("jpeg", "png", "raw")
            quality: JPEG 质量 (1-100)
            
        Returns:
            dict: 可传递给 predict() 的图像字典
            
        Example:
            >>> from PIL import Image
            >>> img = Image.open("camera.jpg")
            >>> encoded = client.encode_image(img, "cam_left", "jpeg")
            >>> action = client.predict(joint_positions, images=[encoded])
        """
        import io
        from PIL import Image as PILImage
        
        # 转换为 PIL Image
        if isinstance(image, np.ndarray):
            # numpy array (H, W, C) -> PIL
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            pil_image = PILImage.fromarray(image)
        elif isinstance(image, bytes):
            # 已经是编码后的数据
            return {
                'name': camera_name,
                'data': image,
                'width': 0,
                'height': 0,
                'encoding': encoding
            }
        elif hasattr(image, 'mode'):  # PIL Image
            pil_image = image
        else:
            raise ValueError(f"不支持的图像类型: {type(image)}")
        
        width, height = pil_image.size
        
        # 编码
        if encoding.lower() in ['jpeg', 'jpg']:
            buffer = io.BytesIO()
            pil_image.convert('RGB').save(buffer, format='JPEG', quality=quality)
            data = buffer.getvalue()
        elif encoding.lower() == 'png':
            buffer = io.BytesIO()
            pil_image.save(buffer, format='PNG')
            data = buffer.getvalue()
        elif encoding.lower() == 'raw':
            data = pil_image.convert('RGB').tobytes()
        else:
            raise ValueError(f"不支持的编码格式: {encoding}")
        
        return {
            'name': camera_name,
            'data': data,
            'width': width,
            'height': height,
            'encoding': encoding
        }
    
    def predict(
        self,
        joint_positions: List[float],
        episode_id: int = 0,
        frame_index: int = 0,
        images: Optional[List[dict]] = None,
        extra_state: str = ""
    ) -> "pb2.Action":
        """
        单次推理
        
        Args:
            joint_positions: 当前关节位置
            episode_id: episode 索引
            frame_index: 帧索引
            images: 图像列表，使用 encode_image() 编码
            extra_state: 额外状态信息 (JSON)
        
        Returns:
            Action 响应
            
        Example:
            >>> # 不带图像
            >>> action = client.predict(joint_positions=[0.0] * 22)
            >>> 
            >>> # 带图像
            >>> img_left = client.encode_image(cam_left_frame, "cam_left")
            >>> img_right = client.encode_image(cam_right_frame, "cam_right")
            >>> action = client.predict(joint_positions, images=[img_left, img_right])
        """
        obs = pb2.Observation(
            joint_positions=joint_positions,
            timestamp=time.time(),
            episode_id=episode_id,
            frame_index=frame_index,
            extra_state=extra_state
        )
        
        if images:
            for img in images:
                obs.images.append(pb2.ImageData(
                    camera_name=img.get('name', 'cam'),
                    data=img.get('data', b''),
                    width=img.get('width', 0),
                    height=img.get('height', 0),
                    encoding=img.get('encoding', 'jpeg')
                ))
        
        return self.stub.Predict(obs)
    
    def predict_chunk(
        self,
        joint_positions: List[float],
        episode_id: int = 0,
        frame_index: int = 0,
        images: Optional[List[dict]] = None,
        extra_state: str = ""
    ) -> "pb2.ActionChunk":
        """
        Chunk 推理 - 一次性获取完整的 action chunk
        
        适用于 action chunking 策略 (ACT, Diffusion 等)
        Server 返回完整的 action chunk，Client 在本地消费
        
        Args:
            joint_positions: 当前关节位置
            episode_id: episode 索引
            frame_index: 帧索引 (chunk 的起始帧)
            images: 图像列表，使用 encode_image() 编码
            extra_state: 额外状态信息 (JSON)
        
        Returns:
            ActionChunk 响应，包含多个 action
            
        Example:
            >>> # 获取 action chunk
            >>> chunk = client.predict_chunk(joint_positions=[0.0] * 22, frame_index=0)
            >>> print(f"Chunk size: {chunk.chunk_size}, Action dim: {chunk.action_dim}")
            >>> 
            >>> # 逐步消费 chunk 中的 action
            >>> for action_step in chunk.actions:
            ...     action = list(action_step.values)
            ...     # 发送到机器人...
        """
        obs = pb2.Observation(
            joint_positions=joint_positions,
            timestamp=time.time(),
            episode_id=episode_id,
            frame_index=frame_index,
            extra_state=extra_state
        )
        
        if images:
            for img in images:
                obs.images.append(pb2.ImageData(
                    camera_name=img.get('name', 'cam'),
                    data=img.get('data', b''),
                    width=img.get('width', 0),
                    height=img.get('height', 0),
                    encoding=img.get('encoding', 'jpeg')
                ))
        
        return self.stub.PredictChunk(obs)
    
    def stream_predict(
        self,
        observation_generator: Callable[[], Optional["pb2.Observation"]]
    ) -> Iterator["pb2.Action"]:
        """
        流式推理
        
        Args:
            observation_generator: 观测数据生成器函数
        
        Yields:
            Action 响应
        """
        def obs_iter():
            while True:
                obs = observation_generator()
                if obs is None:
                    break
                yield obs
        
        return self.stub.StreamPredict(obs_iter())
    
    def reset(self) -> "pb2.ServiceStatus":
        """重置推理状态"""
        return self.stub.Reset(pb2.Empty())
    
    def set_episode(self, episode: int) -> "pb2.ServiceStatus":
        """设置当前 episode"""
        cmd = pb2.ControlCommand(
            type=pb2.CMD_SET_EPISODE,
            params={"episode": str(episode)}
        )
        return self.stub.Control(cmd)
    
    def close(self):
        """关闭连接"""
        if self.channel:
            self.channel.close()
            self._connected = False
        logger.info("已断开推理服务器连接")


class AstribotController:
    """
    Astribot 机器人控制器
    
    整合 gRPC 客户端、机器人 SDK 和相机订阅
    
    支持两种推理模式:
    1. 单步模式 (use_chunk=False): 每次请求获取一个 action
    2. Chunk 模式 (use_chunk=True): 一次获取完整 action chunk，本地消费
    """
    
    def __init__(
        self, 
        config: ClientConfig, 
        enable_camera: bool = False, 
        camera_names: List[str] = None,
        use_chunk: bool = False,
        n_action_steps: Optional[int] = None
    ):
        """
        初始化控制器
        
        Args:
            config: 客户端配置
            enable_camera: 是否启用相机订阅 (用于视觉策略)
            camera_names: 要订阅的相机名称列表，默认 ['head', 'wrist_left', 'wrist_right']
            use_chunk: 是否使用 chunk 模式
            n_action_steps: chunk 模式下每个 chunk 使用的 action 数量
        """
        self.config = config
        self._use_chunk = use_chunk
        self._n_action_steps = n_action_steps
        
        logger.info(f"初始化 AstribotController")
        logger.info(f"  - 服务器: {config.server_address}")
        logger.info(f"  - 控制频率: {config.control_freq} Hz")
        logger.info(f"  - 推理模式: {'Chunk' if use_chunk else '单步'}")
        
        # 初始化推理客户端
        self.inference_client = InferenceClient(
            server_address=config.server_address,
            timeout=config.timeout
        )
        
        # 配置 Server
        self._configure_server()
        
        # 控制参数
        self.control_freq = self.config.control_freq
        self.control_period = 1.0 / self.config.control_freq
        self.control_way = self.config.control_way
        
        # 平滑器和速度限制器
        self.smoother = None
        self.velocity_limiter = None
        
        if self.config.smooth_window > 0:
            self.smoother = ActionSmoother(window_size=self.config.smooth_window)
            logger.info(f"  - 平滑窗口: {self.config.smooth_window}")
        
        if self.config.max_velocity > 0:
            self.velocity_limiter = VelocityLimiter(max_delta=self.config.max_velocity)
            logger.info(f"  - 速度限制: {self.config.max_velocity} rad/frame")
        
        # 初始化 Astribot SDK
        self.astribot = None
        if HAS_ASTRIBOT:
            self.astribot = Astribot(freq=self.config.control_freq)
            logger.info("Astribot SDK 已初始化")
        else:
            logger.warning("Astribot SDK 不可用，将以模拟模式运行")
        
        # 初始化相机订阅器
        self.camera_subscriber = None
        self._enable_camera = enable_camera
        if enable_camera:
            if HAS_ROS:
                cam_names = camera_names or ['head', 'wrist_left', 'wrist_right']
                self.camera_subscriber = AstribotCameraSubscriber(cam_names)
                self.camera_subscriber.start(init_node=True)
                logger.info(f"  - 相机订阅: {cam_names}")
            else:
                logger.warning("ROS 不可用，无法启用相机订阅")
                self._enable_camera = False
        
        # 状态
        self._current_waypoint = None
        self._episode_id = 0
        self._frame_index = 0
        self._use_wbc = False
        
        # 维度配置 (分离输入和执行)
        self._state_includes_chassis = self.config.action_config.state_includes_chassis  # 输入 state 是否含底盘
        self._execute_chassis = self.config.action_config.execute_chassis  # 执行时是否控制底盘
        
        # Chunk 模式管理器
        self._chunk_manager: Optional[ActionChunkManager] = None
        if self._use_chunk:
            self._chunk_manager = ActionChunkManager(
                client=self.inference_client,
                n_action_steps=self._n_action_steps,
                auto_refill_threshold=0.0  # 用完再请求
            )
            logger.info(f"  - Chunk 模式: n_action_steps={self._n_action_steps}")
        
        logger.info(f"  - 输入 state 维度: {22 if not self._state_includes_chassis else 25}")
        logger.info(f"  - 执行底盘控制: {self._execute_chassis}")
    
    def _get_names_list(self, include_chassis: bool = None) -> List[str]:
        """
        根据配置获取正确的部件名称列表
        
        Args:
            include_chassis: 是否包含底盘，None 表示使用 _execute_chassis 配置
        """
        use_chassis = include_chassis if include_chassis is not None else self._execute_chassis
        if use_chassis:
            return ASTRIBOT_NAMES_LIST_WITH_CHASSIS
        return ASTRIBOT_NAMES_LIST
    
    def _configure_server(self):
        """配置 Server 端使用的模型/数据集"""
        if self.config.model_path:
            logger.info(f"配置 Server 使用模型: {self.config.model_path}")
            status = self.inference_client.configure(
                mode="model",
                model_path=self.config.model_path,
                device=self.config.device,
                policy_type=self.config.policy_type or "",
                action_config=self.config.action_config
            )
            logger.info(f"Server 配置结果: {status.message}")
            
        elif self.config.dataset_path:
            logger.info(f"配置 Server 使用数据集: {self.config.dataset_path}")
            status = self.inference_client.configure(
                mode="dataset",
                dataset_path=self.config.dataset_path,
                action_config=self.config.action_config
            )
            logger.info(f"Server 配置结果: {status.message}")
            
        else:
            status = self.inference_client.get_status()
            if status.is_ready:
                logger.info(f"Server 已就绪，模式: {status.mode}")
            else:
                logger.warning("未指定模型或数据集，且 Server 未就绪")
    
    def get_current_joint_positions(self) -> List[float]:
        """
        获取当前关节位置 (state)
        
        Returns:
            关节位置列表，维度由 state_includes_chassis 决定:
            - state_includes_chassis=False: 22维
            - state_includes_chassis=True: 25维
        """
        if self._current_waypoint:
            # 从 waypoint 转换回 lerobot action 格式
            from src.common.utils import waypoint_to_lerobot_action
            return waypoint_to_lerobot_action(
                self._current_waypoint, 
                include_chassis=self._state_includes_chassis  # 输入维度由此控制
            )
        
        # 返回零向量，维度由 state_includes_chassis 决定
        dim = LEROBOT_ACTION_DIM_WITH_CHASSIS if self._state_includes_chassis else LEROBOT_ACTION_DIM_NO_CHASSIS
        return [0.0] * dim
    
    def move_to_home(self):
        """移动到 home 位置"""
        logger.info("移动到 home 位置...")
        if self.astribot:
            self.astribot.move_to_home()
        logger.info("已到达 home 位置")
    
    def move_to_ready_position(self, duration: float = 5.0) -> bool:
        """
        使用路径规划移动到预设的准备位置 (Ready Position)
        
        准备位置定义在 constants.py 中的 READY_POSITION_22/25
        
        Args:
            duration: 移动时间 (秒)
        
        Returns:
            是否成功
        """
        logger.info("=" * 60)
        logger.info("阶段1: 移动到准备位置 (Ready Position)")
        logger.info("=" * 60)
        
        # 选择准备位置 (根据是否控制底盘)
        if self._execute_chassis:
            ready_position = READY_POSITION_25
            logger.info("使用 25 维准备位置 (含底盘)")
        else:
            ready_position = READY_POSITION_22
            logger.info("使用 22 维准备位置 (不含底盘)")
        
        logger.info(f"目标位置 (前5维): {ready_position[:5]}")
        logger.info(f"规划时间: {duration}s")
        
        # 转换为 waypoint
        waypoint = lerobot_action_to_waypoint(
            ready_position, 
            include_chassis=self._execute_chassis
        )
        
        # 执行路径规划
        if self.astribot:
            logger.info("开始路径规划移动...")
            self.astribot.move_joints_waypoints(
                self._get_names_list(),
                [waypoint],
                [duration],
                use_wbc=self._use_wbc
            )
        else:
            logger.info(f"模拟模式: 等待 {duration}s...")
            time.sleep(duration)
        
        self._current_waypoint = waypoint
        
        logger.info("✓ 已到达准备位置")
        logger.info("=" * 60)
        return True
    
    def move_to_initial_position(self, duration: float = 3.0) -> bool:
        """
        [已弃用] 使用 move_to_ready_position 代替
        """
        return self.move_to_ready_position(duration)
    
    def move_waypoints(
        self, 
        actions: List[List[float]], 
        time_list: List[float]
    ):
        """
        使用路径规划执行多个路径点
        
        Args:
            actions: action 列表 (可以是 22 或 25 维)
            time_list: 时间列表
        """
        if not actions:
            return
        
        action_has_chassis = len(actions[0]) >= LEROBOT_ACTION_DIM_WITH_CHASSIS
        
        logger.info(f"路径规划执行 {len(actions)} 个路径点...")
        logger.info(f"  - Action 维度: {len(actions[0])} (含底盘: {action_has_chassis})")
        logger.info(f"  - 执行底盘控制: {self._execute_chassis}")
        
        # 转换为 waypoint (根据 execute_chassis 决定)
        waypoints = [
            lerobot_action_to_waypoint(a, include_chassis=self._execute_chassis) 
            for a in actions
        ]
        
        if self.astribot:
            self.astribot.move_joints_waypoints(
                self._get_names_list(),
                waypoints,
                time_list,
                use_wbc=self._use_wbc
            )
        else:
            time.sleep(sum(time_list))
        
        if waypoints:
            self._current_waypoint = waypoints[-1]
        logger.info("路径规划完成")
    
    def get_current_images(self) -> Optional[List[dict]]:
        """
        获取当前相机图像
        
        Returns:
            图像列表 (用于 predict)，如果相机未启用则返回 None
        """
        if self.camera_subscriber and self._enable_camera:
            return self.camera_subscriber.get_images_for_inference(self.inference_client)
        return None
    
    def step(self, with_images: bool = None) -> bool:
        """
        执行一步推理和控制
        
        支持两种模式:
        1. 单步模式: 每次请求获取一个 action
        2. Chunk 模式: 从本地 action queue 获取，queue 空时请求新 chunk
        
        Args:
            with_images: 是否发送图像，None 表示根据 enable_camera 自动决定
        
        Returns:
            True 继续, False 结束
        """
        # 获取本体状态 (关节位置)
        joint_positions = self.get_current_joint_positions()
        
        # 获取图像 (如果启用)
        images = None
        send_images = with_images if with_images is not None else self._enable_camera
        if send_images:
            images = self.get_current_images()
        
        # 根据模式获取 action
        if self._use_chunk and self._chunk_manager is not None:
            # Chunk 模式: 从本地 queue 获取 action
            action = self._chunk_manager.get_action(
                joint_positions=joint_positions,
                images=images,
                episode_id=self._episode_id,
                frame_index=self._frame_index
            )
            
            if action is None:
                return False
        else:
            # 单步模式: 每次请求一个 action
            response = self.inference_client.predict(
                joint_positions=joint_positions,
                images=images,
                episode_id=self._episode_id,
                frame_index=self._frame_index
            )
            
            if response.status == pb2.EPISODE_END or response.is_terminal:
                return False
            
            if response.status != pb2.OK:
                logger.error(f"推理错误: {response.error_message}")
                return False
            
            action = list(response.values)
        
        # 应用速度限制
        if self.velocity_limiter:
            action = self.velocity_limiter.limit(action)
        
        # 应用平滑
        if self.smoother:
            action = self.smoother.smooth(action)
        
        # 发送到机器人 (根据 execute_chassis 决定是否控制底盘)
        waypoint = lerobot_action_to_waypoint(action, include_chassis=self._execute_chassis)
        
        if self.astribot:
            self.astribot.set_joints_position(
                self._get_names_list(),
                waypoint,
                control_way=self.control_way,
                use_wbc=self._use_wbc
            )
        
        self._current_waypoint = waypoint
        self._frame_index += 1
        
        return True
    
    def set_episode(self, episode: int):
        """设置当前 episode"""
        self._episode_id = episode
        self._frame_index = 0
        self.inference_client.set_episode(episode)
        
        if self.smoother:
            self.smoother.reset()
        if self.velocity_limiter:
            self.velocity_limiter.reset()
        if self._chunk_manager:
            self._chunk_manager.reset()
    
    def close(self):
        """关闭控制器"""
        if self.camera_subscriber:
            self.camera_subscriber.stop()
        self.inference_client.close()
        logger.info("控制器已关闭")


def run_inference_loop(
    controller: AstribotController,
    episode: int = 0,
    max_frames: int = 10000,
    move_to_ready: bool = True,
    ready_move_duration: float = 5.0
):
    """
    运行推理控制循环 (两阶段: 移动到准备位置 + 实时推理)
    
    阶段1: 移动到准备位置 (Ready Position)
        - 使用路径规划移动到预设的固定位置
        - 准备位置定义在 constants.py
    
    阶段2: 实时推理控制
        - 从 frame_index=0 开始推理
        - 按控制频率执行
    
    Args:
        controller: 控制器
        episode: episode 索引
        max_frames: 最大帧数
        move_to_ready: 是否先移动到准备位置
        ready_move_duration: 移动到准备位置的时间 (秒)
    """
    global _interrupted
    _interrupted = False
    
    logger.info("=" * 60)
    logger.info(f"开始推理控制 (episode={episode})")
    logger.info(f"  - 移动到准备位置: {move_to_ready} ({ready_move_duration}s)")
    logger.info(f"  - 控制频率: {controller.control_freq} Hz")
    logger.info("=" * 60)
    
    # 设置 episode
    controller.set_episode(episode)
    
    # 获取服务状态
    status = controller.inference_client.get_status()
    total_frames = status.total_frames or max_frames
    logger.info(f"服务状态: {'就绪' if status.is_ready else '未就绪'}")
    logger.info(f"模式: {status.mode}, 总帧数: {total_frames}")
    
    # ==================== 等待图像就绪 ====================
    if controller._enable_camera and controller.camera_subscriber:
        logger.info("等待相机图像就绪...")
        if controller.camera_subscriber.wait_for_images(timeout=10.0):
            logger.info("所有相机图像已就绪")
        else:
            logger.warning("部分相机图像未就绪，继续执行...")
    
    # ==================== 阶段1: 移动到准备位置 ====================
    if move_to_ready:
        if not controller.move_to_ready_position(duration=ready_move_duration):
            logger.error("移动到准备位置失败，中止推理")
            return
        
        if _interrupted:
            logger.warning("用户中断")
            return
    
    # ==================== 阶段2: 实时推理控制 ====================
    logger.info("=" * 60)
    logger.info(f"阶段2: 实时推理控制 ({total_frames} 帧 @ {controller.control_freq}Hz)")
    logger.info("=" * 60)
    
    # 从 frame_index=0 开始
    controller._frame_index = 0
    
    control_period = controller.control_period
    
    start_time = time.time()
    frame_count = 0
    
    while not _interrupted and frame_count < total_frames:
        loop_start = time.time()
        
        if not controller.step():
            logger.info("Episode 结束")
            break
        
        frame_count += 1
        
        # 打印进度
        if frame_count % 100 == 0:
            elapsed = time.time() - start_time
            actual_freq = frame_count / elapsed if elapsed > 0 else 0
            progress = frame_count / total_frames * 100
            logger.info(f"进度: {frame_count}/{total_frames} "
                       f"({progress:.1f}%) | 实际频率: {actual_freq:.1f}Hz")
        
        # 频率控制
        elapsed = time.time() - loop_start
        if elapsed < control_period:
            time.sleep(control_period - elapsed)
    
    total_time = time.time() - start_time
    if frame_count > 0:
        logger.info(f"推理完成! 帧数: {frame_count}, 耗时: {total_time:.2f}s, "
                   f"平均频率: {frame_count/total_time:.1f}Hz")


def main():
    """命令行入口"""
    import argparse
    global _interrupted
    
    parser = argparse.ArgumentParser(
        description='Astribot 推理控制客户端 (精简版)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用数据集回放
  python inference_client.py --server localhost:50051 \\
      --dataset /path/to/lerobot_dataset --episode 0
  
  # 使用模型推理
  python inference_client.py --server localhost:50051 \\
      --model /path/to/trained_model --device cuda
  
  # 启用底盘控制
  python inference_client.py --server localhost:50051 \\
      --dataset /path/to/dataset --enable-chassis
  
  # 开启平滑
  python inference_client.py --server localhost:50051 \\
      --dataset /path/to/dataset --smooth 5 --max-velocity 0.05
        """
    )
    
    # Server 连接
    parser.add_argument('--server', type=str, default='localhost:50051',
                        help='推理服务器地址 (默认: localhost:50051)')
    parser.add_argument('--timeout', type=float, default=10.0,
                        help='连接超时 (默认: 10s)')
    
    # 策略配置
    parser.add_argument('--model', type=str, default=None,
                        help='模型路径或 HuggingFace repo id')
    parser.add_argument('--dataset', type=str, default=None,
                        help='数据集路径 (数据集回放模式)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='推理设备 (默认: cuda)')
    parser.add_argument('--policy-type', type=str, default=None,
                        help='策略类型 (可选)')
    
    # State 输入配置
    parser.add_argument('--state-with-chassis', action='store_true',
                        help='输入 state 包含底盘 (25维)，默认不包含 (22维)')
    
    # Action 执行配置
    parser.add_argument('--execute-chassis', action='store_true',
                        help='执行 action 时控制底盘 (默认: 不控制)')
    parser.add_argument('--no-head', action='store_true',
                        help='执行时禁用头部控制 (默认: 启用)')
    parser.add_argument('--no-torso', action='store_true',
                        help='执行时禁用腰部控制 (默认: 启用)')
    
    # 相机配置
    parser.add_argument('--enable-camera', action='store_true',
                        help='启用相机订阅，发送图像到 Server (视觉策略需要)')
    parser.add_argument('--cameras', type=str, default='head,wrist_left,wrist_right',
                        help='要订阅的相机列表 (逗号分隔，默认: head,wrist_left,wrist_right)')
    
    # 回放配置
    parser.add_argument('--episode', type=int, default=0,
                        help='Episode 索引 (默认: 0)')
    parser.add_argument('--max-frames', type=int, default=10000,
                        help='最大帧数 (默认: 10000)')
    
    # 控制配置
    parser.add_argument('--control-freq', type=float, default=30.0,
                        help='控制频率 Hz (默认: 30)')
    parser.add_argument('--control-way', type=str, default='direct',
                        choices=['filter', 'direct'],
                        help='控制方式 (默认: direct)')
    parser.add_argument('--smooth', type=int, default=0,
                        help='平滑窗口大小 (0=不平滑)')
    parser.add_argument('--max-velocity', type=float, default=0.0,
                        help='最大速度 rad/frame (0=不限制)')
    
    # Chunk 模式配置
    parser.add_argument('--use-chunk', action='store_true',
                        help='使用 chunk 模式: 一次获取完整 action chunk，本地消费')
    parser.add_argument('--n-action-steps', type=int, default=None,
                        help='Chunk 模式下每个 chunk 使用的 action 数量 (默认: 使用完整 chunk)')
    
    # 准备位置配置
    parser.add_argument('--move-to-ready', action='store_true', default=True,
                        help='启动时先移动到准备位置 (默认: 开启)')
    parser.add_argument('--no-move-to-ready', action='store_true',
                        help='禁用移动到准备位置，直接开始推理')
    parser.add_argument('--ready-duration', type=float, default=5.0,
                        help='移动到准备位置的时间 (默认: 5.0s)')
    
    args = parser.parse_args()
    
    setup_logging("INFO")
    
    # 解析服务器地址
    if ':' in args.server:
        host, port = args.server.rsplit(':', 1)
        port = int(port)
    else:
        host = args.server
        port = 50051
    
    # 构建 Action 配置 (分离输入和执行)
    action_config = ActionConfig(
        state_includes_chassis=args.state_with_chassis,  # 输入 state 是否含底盘
        execute_chassis=args.execute_chassis,            # 执行时是否控制底盘
        execute_head=not args.no_head,
        execute_torso=not args.no_torso
    )
    
    # 构建配置
    config = ClientConfig(
        server_host=host,
        server_port=port,
        timeout=args.timeout,
        model_path=args.model,
        dataset_path=args.dataset,
        device=args.device,
        policy_type=args.policy_type,
        control_freq=args.control_freq,
        control_way=args.control_way,
        smooth_window=args.smooth,
        max_velocity=args.max_velocity,
        action_config=action_config
    )
    
    controller = None
    
    # 解析相机列表
    camera_names = [c.strip() for c in args.cameras.split(',') if c.strip()]
    
    # 是否移动到准备位置
    move_to_ready = args.move_to_ready and not args.no_move_to_ready
    
    try:
        controller = AstribotController(
            config,
            enable_camera=args.enable_camera,
            camera_names=camera_names,
            use_chunk=args.use_chunk,
            n_action_steps=args.n_action_steps
        )
        
        run_inference_loop(
            controller,
            episode=args.episode,
            max_frames=args.max_frames,
            move_to_ready=move_to_ready,
            ready_move_duration=args.ready_duration
        )
        
        if not _interrupted:
            logger.info("返回 home 位置...")
            controller.move_to_home()
        
    except KeyboardInterrupt:
        logger.warning("程序被中断")
    except Exception as e:
        logger.error(f"错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if controller:
            controller.close()
        logger.info("程序退出")


if __name__ == '__main__':
    main()
