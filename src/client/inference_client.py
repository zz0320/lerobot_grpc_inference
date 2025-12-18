# -*- coding: utf-8 -*-
"""
LeRobot 推理客户端
运行环境: 机器人侧 (Astribot SDK 环境)

通过 gRPC 连接远程推理服务器获取 action
"""

import os
import sys
import time
import signal
import logging
from typing import List, Optional, Iterator, Callable
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
from src.common.config import Config, ClientConfig
from src.common.utils import (
    setup_logging, 
    ActionSmoother, 
    VelocityLimiter,
    lerobot_action_to_waypoint
)
from src.common.constants import (
    ASTRIBOT_NAMES_LIST,
    ASTRIBOT_NAMES_LIST_WITH_CHASSIS,
    ASTRIBOT_DOF_CONFIG,
    LEROBOT_ACTION_DIM,
    LEROBOT_ACTION_DIM_V2,
    LEROBOT_ACTION_DIM_V2_NO_CHASSIS,
    GRPC_MAX_MESSAGE_LENGTH
)

# Astribot SDK
HAS_ASTRIBOT = False
try:
    from core.astribot_api.astribot_client import Astribot
    HAS_ASTRIBOT = True
except ImportError:
    pass

# ROS
HAS_ROSPY = False
try:
    import rospy
    HAS_ROSPY = True
except ImportError:
    pass

logger = logging.getLogger("lerobot_inference.client")

# 全局中断标志
_interrupted = False

def _signal_handler(signum, frame):
    global _interrupted
    _interrupted = True
    logger.warning(f"收到中断信号 {signum}")

signal.signal(signal.SIGINT, _signal_handler)


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
        
        # 等待连接就绪
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
    
    def ping(self) -> bool:
        """心跳检测"""
        try:
            response = self.stub.Ping(pb2.Heartbeat(
                client_timestamp=int(time.time() * 1000),
                client_id="astribot"
            ))
            return response.is_alive
        except Exception:
            return False
    
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
            joint_positions: 当前关节位置 [16]
            episode_id: episode 索引
            frame_index: 帧索引
            images: 图像列表 (可选)
            extra_state: 额外状态信息 (JSON)
        
        Returns:
            Action 响应
        """
        obs = pb2.Observation(
            joint_positions=joint_positions,
            timestamp=time.time(),
            episode_id=episode_id,
            frame_index=frame_index,
            extra_state=extra_state
        )
        
        # 添加图像
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
    
    def start_service(self) -> "pb2.ServiceStatus":
        """启动服务"""
        cmd = pb2.ControlCommand(type=pb2.CMD_START)
        return self.stub.Control(cmd)
    
    def stop_service(self) -> "pb2.ServiceStatus":
        """停止服务"""
        cmd = pb2.ControlCommand(type=pb2.CMD_STOP)
        return self.stub.Control(cmd)
    
    def configure_model(
        self, 
        model_path: str, 
        device: str = "cuda",
        policy_type: str = ""
    ) -> "pb2.ServiceStatus":
        """
        配置 Server 使用指定的模型
        
        Args:
            model_path: 模型路径或 HuggingFace repo id
            device: 推理设备 ("cuda", "cpu", "mps")
            policy_type: 策略类型 (可选，自动检测)
        """
        config = pb2.PolicyConfig(
            mode="model",
            model_path=model_path,
            device=device,
            policy_type=policy_type
        )
        return self.stub.Configure(config)
    
    def configure_dataset(self, dataset_path: str) -> "pb2.ServiceStatus":
        """
        配置 Server 使用指定的数据集 (回放模式)
        
        Args:
            dataset_path: 数据集路径
        """
        config = pb2.PolicyConfig(
            mode="dataset",
            dataset_path=dataset_path
        )
        return self.stub.Configure(config)
    
    def close(self):
        """关闭连接"""
        if self.channel:
            self.channel.close()
            self._connected = False
        logger.info("已断开推理服务器连接")


class AstribotController:
    """
    Astribot 机器人控制器
    
    整合 gRPC 客户端和机器人 SDK
    """
    
    def __init__(self, config: ClientConfig):
        """
        初始化控制器
        
        Args:
            config: 客户端配置
        """
        self.config = config
        
        logger.info(f"初始化 AstribotController")
        logger.info(f"  - 服务器: {config.server_address}")
        logger.info(f"  - 控制频率: {config.control_freq} Hz")
        
        # 初始化推理客户端
        self.inference_client = InferenceClient(
            server_address=config.server_address,
            timeout=config.timeout
        )
        
        # 配置 Server (Client 端指定模型/数据集)
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
        
        # 状态
        self._current_waypoint = None
        self._episode_id = 0
        self._frame_index = 0
        self._use_wbc = False
        self._include_chassis = False  # 是否包含底盘控制
    
    def _get_names_list(self) -> List[str]:
        """根据配置获取正确的部件名称列表"""
        if self._include_chassis:
            return ASTRIBOT_NAMES_LIST_WITH_CHASSIS
        return ASTRIBOT_NAMES_LIST
    
    def _configure_server(self):
        """配置 Server 端使用的模型/数据集"""
        if self.config.model_path:
            logger.info(f"配置 Server 使用模型: {self.config.model_path}")
            status = self.inference_client.configure_model(
                model_path=self.config.model_path,
                device=self.config.device,
                policy_type=self.config.policy_type or ""
            )
            logger.info(f"Server 配置结果: {status.message}")
            
        elif self.config.dataset_path:
            logger.info(f"配置 Server 使用数据集: {self.config.dataset_path}")
            status = self.inference_client.configure_dataset(self.config.dataset_path)
            logger.info(f"Server 配置结果: {status.message}")
            
        else:
            # 检查 Server 是否已经配置好了
            status = self.inference_client.get_status()
            if status.is_ready:
                logger.info(f"Server 已就绪，模式: {status.mode}")
            else:
                logger.warning("未指定模型或数据集，且 Server 未就绪")
    
    def get_current_joint_positions(self) -> List[float]:
        """
        获取当前关节位置
        
        Returns:
            [arm_left(7), arm_right(7), gripper_left(1), gripper_right(1)]
        """
        # TODO: 从 Astribot SDK 获取实际关节状态
        # 这里需要根据实际 SDK 接口实现
        if self._current_waypoint:
            arm_left = self._current_waypoint[1]
            arm_right = self._current_waypoint[3]
            gripper_left = self._current_waypoint[2][0]
            gripper_right = self._current_waypoint[4][0]
            return arm_left + arm_right + [gripper_left, gripper_right]
        return [0.0] * LEROBOT_ACTION_DIM
    
    def move_to_home(self):
        """移动到 home 位置"""
        logger.info("移动到 home 位置...")
        if self.astribot:
            self.astribot.move_to_home()
        logger.info("已到达 home 位置")
    
    def move_to_initial_position(self, duration: float = 3.0) -> bool:
        """
        使用路径规划移动到初始位置
        
        Args:
            duration: 移动时间 (秒)
        
        Returns:
            是否成功
        """
        # 从服务器获取第一帧 action
        response = self.inference_client.predict(
            joint_positions=self.get_current_joint_positions(),
            episode_id=self._episode_id,
            frame_index=0
        )
        
        if response.status != pb2.OK:
            logger.error(f"获取初始位置失败: {response.error_message}")
            return False
        
        action = list(response.values)
        
        # 根据 action 长度判断是否包含底盘
        self._include_chassis = len(action) >= 25
        
        waypoint = lerobot_action_to_waypoint(action, include_chassis=self._include_chassis)
        
        logger.info(f"移动到初始位置 (耗时 {duration}s)...")
        logger.info(f"  - Action 维度: {len(action)}, 包含底盘: {self._include_chassis}")
        
        if self.astribot:
            self.astribot.move_joints_waypoints(
                self._get_names_list(),
                [waypoint],
                [duration],
                use_wbc=self._use_wbc
            )
        else:
            # 模拟模式
            time.sleep(duration)
        
        self._current_waypoint = waypoint
        self._frame_index = 1
        logger.info("已到达初始位置")
        return True
    
    def move_waypoints(
        self, 
        actions: List[List[float]], 
        time_list: List[float]
    ):
        """
        使用路径规划执行多个路径点
        
        Args:
            actions: action 列表
            time_list: 时间列表
        """
        # 根据第一个 action 长度判断是否包含底盘
        if actions:
            self._include_chassis = len(actions[0]) >= 25
        
        waypoints = [lerobot_action_to_waypoint(a, include_chassis=self._include_chassis) for a in actions]
        
        logger.info(f"路径规划执行 {len(waypoints)} 个路径点...")
        logger.info(f"  - 包含底盘: {self._include_chassis}")
        
        if self.astribot:
            self.astribot.move_joints_waypoints(
                self._get_names_list(),
                waypoints,
                time_list,
                use_wbc=self._use_wbc
            )
        else:
            # 模拟模式
            time.sleep(sum(time_list))
        
        if waypoints:
            self._current_waypoint = waypoints[-1]
        logger.info("路径规划完成")
    
    def step(self) -> bool:
        """
        执行一步推理和控制
        
        Returns:
            True 继续, False 结束
        """
        # 获取当前观测
        joint_positions = self.get_current_joint_positions()
        
        # 从服务器获取 action
        response = self.inference_client.predict(
            joint_positions=joint_positions,
            episode_id=self._episode_id,
            frame_index=self._frame_index
        )
        
        if response.status == pb2.EPISODE_END or response.is_terminal:
            return False
        
        if response.status != pb2.OK:
            logger.error(f"推理错误: {response.error_message}")
            return False
        
        # 获取 action
        action = list(response.values)
        
        # 应用速度限制
        if self.velocity_limiter:
            action = self.velocity_limiter.limit(action)
        
        # 应用平滑
        if self.smoother:
            action = self.smoother.smooth(action)
        
        # 发送到机器人
        waypoint = lerobot_action_to_waypoint(action, include_chassis=self._include_chassis)
        
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
        
        # 重置滤波器
        if self.smoother:
            self.smoother.reset()
        if self.velocity_limiter:
            self.velocity_limiter.reset()
    
    def get_rate(self):
        """获取频率控制器"""
        if HAS_ROSPY:
            return rospy.Rate(self.control_freq)
        return None
    
    def close(self):
        """关闭控制器"""
        self.inference_client.close()
        logger.info("控制器已关闭")


def run_inference_loop(
    controller: AstribotController,
    episode: int = 0,
    planning_frames: int = 5,
    planning_duration: float = 3.0,
    max_frames: int = 10000
):
    """
    运行推理控制循环
    
    Args:
        controller: 控制器
        episode: episode 索引
        planning_frames: 前几帧使用路径规划
        planning_duration: 路径规划的总时间
        max_frames: 最大帧数
    """
    global _interrupted
    _interrupted = False
    
    logger.info("=" * 60)
    logger.info(f"开始推理控制 (episode={episode})")
    logger.info(f"  - 前 {planning_frames} 帧: 路径规划")
    logger.info(f"  - 控制频率: {controller.control_freq} Hz")
    logger.info("=" * 60)
    
    # 设置 episode
    controller.set_episode(episode)
    
    # 获取服务状态
    status = controller.inference_client.get_status()
    total_frames = status.total_frames or max_frames
    logger.info(f"服务状态: {'就绪' if status.is_ready else '未就绪'}")
    logger.info(f"模式: {status.mode}, 总帧数: {total_frames}")
    
    # ==================== 阶段1: 路径规划 ====================
    if planning_frames > 0:
        logger.info(f"=== 阶段1: 路径规划 ({planning_frames} 帧) ===")
        
        # 获取前 N 帧的 action
        planning_actions = []
        for i in range(planning_frames):
            response = controller.inference_client.predict(
                joint_positions=controller.get_current_joint_positions(),
                episode_id=episode,
                frame_index=i
            )
            if response.status == pb2.OK:
                planning_actions.append(list(response.values))
            else:
                break
        
        if planning_actions:
            # 构建时间列表
            time_per_frame = planning_duration / len(planning_actions)
            time_list = [time_per_frame * (i + 1) for i in range(len(planning_actions))]
            
            controller.move_waypoints(planning_actions, time_list)
            controller._frame_index = len(planning_actions)
    
    if _interrupted:
        logger.warning("用户中断")
        return
    
    # ==================== 阶段2: 实时控制 ====================
    realtime_start = controller._frame_index
    realtime_count = total_frames - realtime_start
    
    if realtime_count <= 0:
        logger.info("回放完成! (全部使用路径规划)")
        return
    
    logger.info(f"=== 阶段2: 实时控制 ({realtime_count} 帧 @ {controller.control_freq}Hz) ===")
    
    # 获取频率控制器
    rate = controller.get_rate()
    control_period = controller.control_period
    
    start_time = time.time()
    frame_count = 0
    
    while not _interrupted and frame_count < realtime_count:
        loop_start = time.time()
        
        # 执行一步
        if not controller.step():
            logger.info("Episode 结束")
            break
        
        frame_count += 1
        
        # 打印进度
        if frame_count % 100 == 0:
            elapsed = time.time() - start_time
            actual_freq = frame_count / elapsed if elapsed > 0 else 0
            progress = (realtime_start + frame_count) / total_frames * 100
            logger.info(f"进度: {realtime_start + frame_count}/{total_frames} "
                       f"({progress:.1f}%) | 实际频率: {actual_freq:.1f}Hz")
        
        # 频率控制
        if rate:
            rate.sleep()
        else:
            elapsed = time.time() - loop_start
            if elapsed < control_period:
                time.sleep(control_period - elapsed)
    
    total_time = time.time() - start_time
    logger.info(f"实时控制完成! 帧数: {frame_count}, 耗时: {total_time:.2f}s, "
               f"平均频率: {frame_count/total_time:.1f}Hz")


def main():
    """命令行入口"""
    import argparse
    global _interrupted
    
    parser = argparse.ArgumentParser(
        description='Astribot 推理控制客户端',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用数据集回放 (Client 端指定数据集路径)
  python inference_client.py --server localhost:50051 \\
      --dataset /path/to/lerobot_dataset --episode 0
  
  # 使用模型推理 (Client 端指定模型路径)
  python inference_client.py --server localhost:50051 \\
      --model /path/to/trained_model --device cuda
  
  # 从 HuggingFace Hub 加载模型
  python inference_client.py --server localhost:50051 \\
      --model lerobot/act_aloha_sim_insertion_human
  
  # 不指定模型/数据集 (使用 Server 启动时的配置)
  python inference_client.py --server localhost:50051 --episode 0
  
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
    
    # 策略配置 (Client 端指定)
    parser.add_argument('--model', type=str, default=None,
                        help='模型路径或 HuggingFace repo id (模型推理模式)')
    parser.add_argument('--dataset', type=str, default=None,
                        help='数据集路径 (数据集回放模式)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='推理设备 (默认: cuda)')
    parser.add_argument('--policy-type', type=str, default=None,
                        help='策略类型 (可选，自动检测)')
    
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
    parser.add_argument('--planning-frames', type=int, default=5,
                        help='前几帧使用路径规划 (默认: 5)')
    parser.add_argument('--planning-duration', type=float, default=3.0,
                        help='路径规划的总时间 (默认: 3.0s)')
    parser.add_argument('--smooth', type=int, default=0,
                        help='平滑窗口大小 (0=不平滑)')
    parser.add_argument('--max-velocity', type=float, default=0.0,
                        help='最大速度 rad/frame (0=不限制)')
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging("INFO")
    
    # 解析服务器地址
    if ':' in args.server:
        host, port = args.server.rsplit(':', 1)
        port = int(port)
    else:
        host = args.server
        port = 50051
    
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
        planning_frames=args.planning_frames,
        planning_duration=args.planning_duration
    )
    
    controller = None
    
    try:
        # 创建控制器
        controller = AstribotController(config)
        
        # 运行推理循环
        run_inference_loop(
            controller,
            episode=args.episode,
            planning_frames=config.planning_frames,
            planning_duration=config.planning_duration,
            max_frames=args.max_frames
        )
        
        # 返回 home
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

