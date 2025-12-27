# LeRobot gRPC Inference

基于 gRPC 的 LeRobot 推理框架，支持 Astribot 机器人控制。

## 架构概览

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                  系统架构                                            │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  ┌───────────────────────────────┐         ┌───────────────────────────────────┐   │
│  │      Client (机器人侧)          │  gRPC   │        Server (LeRobot)            │   │
│  │      Astribot SDK 环境         │ ◄─────► │        Python 3.10+               │   │
│  └───────────────────────────────┘         └───────────────────────────────────┘   │
│                                                                                     │
│  ┌───────────────────────────────┐         ┌───────────────────────────────────┐   │
│  │ • AstribotCameraSubscriber    │  ────►  │ • LeRobotModelInference           │   │
│  │   - ROS 话题订阅               │  图像   │   - 模型加载 (from_pretrained)     │   │
│  │   - 图像采集                   │  ────►  │   - 预处理/后处理                   │   │
│  │                               │         │   - select_action 推理             │   │
│  │ • InferenceClient             │  ◄────  │                                    │   │
│  │   - gRPC 通信                  │ Action │ • DatasetLoader                    │   │
│  │   - 图像编码                   │  ◄────  │   - Parquet 数据加载               │   │
│  │   - 配置服务端                 │         │   - V2.0 格式支持                  │   │
│  │                               │         │                                    │   │
│  │ • AstribotController          │         │ • LeRobotInferenceServicer        │   │
│  │   - 两阶段控制                 │         │   - Configure/Predict/Reset       │   │
│  │   - 动作平滑/速度限制          │         │                                    │   │
│  └───────────────────────────────┘         └───────────────────────────────────┘   │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

## 特性

- ✅ **V2.0 数据格式** (22/25维)
- ✅ **输入/执行分离配置** (输入维度与执行底盘独立控制)
- ✅ **图像传输支持** (JPEG/PNG 压缩，支持视觉策略)
- ✅ **Client 动态配置** (Server 以空闲模式启动)
- ✅ **两阶段控制** (路径规划 + 实时控制)
- ✅ **ROS 相机集成** (AstribotCameraSubscriber)

## 数据流

```
┌────────────────────────────────────────────────────────────────────────────────────┐
│                              推理数据流                                             │
├────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                    │
│   Client                              gRPC                              Server     │
│                                                                                    │
│   ┌────────────────────────┐                                                       │
│   │     Observation        │                                                       │
│   │  ┌──────────────────┐  │                                                       │
│   │  │ state (22/25维)  │  │  ← 本体状态 (关节角度)                                 │
│   │  └──────────────────┘  │    维度由 state_includes_chassis 控制                  │
│   │  ┌──────────────────┐  │                                                       │
│   │  │ images[]         │  │  ← 观测图像 (JPEG)                                     │
│   │  │  • head          │  │                                                       │
│   │  │  • wrist_left    │  │                                                       │
│   │  │  • wrist_right   │  │                                                       │
│   │  └──────────────────┘  │                                                       │
│   └────────────────────────┘                                                       │
│              │                                                                     │
│              │  Predict()                                                          │
│              ▼                                                                     │
│   ┌────────────────────────┐     ┌──────────────────────────────────────────────┐  │
│   │      Action            │     │  Server 处理:                                 │  │
│   │  ┌──────────────────┐  │     │  1. 解码 state → tensor                      │  │
│   │  │ values (22/25维) │  │ ←── │  2. 解码 images → tensor                     │  │
│   │  └──────────────────┘  │     │  3. policy.select_action(obs)                │  │
│   └────────────────────────┘     │  4. 返回 action                              │  │
│              │                   └──────────────────────────────────────────────┘  │
│              ▼                                                                     │
│   ┌────────────────────────┐                                                       │
│   │     执行到机器人        │                                                       │
│   │                        │                                                       │
│   │  execute_chassis=False │  → 只执行前 22 维 (忽略底盘)                           │
│   │  execute_chassis=True  │  → 执行全部 25 维 (包含底盘)                           │
│   └────────────────────────┘                                                       │
│                                                                                    │
└────────────────────────────────────────────────────────────────────────────────────┘
```

## 维度配置 (重要)

**输入维度**和**执行底盘**是独立配置的：

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `state_includes_chassis` | 输入 state 是否包含底盘 (22 vs 25维) | `False` |
| `execute_chassis` | 执行 action 时是否控制底盘 | `False` |

### 典型场景

| 场景 | state_includes_chassis | execute_chassis | 说明 |
|------|------------------------|-----------------|------|
| 固定底座 | `False` (22维) | `False` | 不采集底盘状态，不控制底盘 |
| 移动机器人 | `True` (25维) | `True` | 采集底盘状态，控制底盘 |
| 有底盘但锁定 | `True` (25维) | `False` | 采集底盘状态，但不发送底盘命令 |

### V2.0 数据格式

```
22维 (不含底盘):
┌─────────────────────────────────────────────────────────────────────────┐
│ arm_left(7) │ arm_right(7) │ gripper_L(1) │ gripper_R(1) │ head(2) │ torso(4) │
└─────────────────────────────────────────────────────────────────────────┘
     0-6           7-13            14             15          16-17      18-21

25维 (含底盘):
┌──────────────────────────────────────────────────────────────────────────────────┐
│ arm_left(7) │ arm_right(7) │ gripper_L(1) │ gripper_R(1) │ head(2) │ torso(4) │ chassis(3) │
└──────────────────────────────────────────────────────────────────────────────────┘
     0-6           7-13            14             15          16-17      18-21       22-24
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt

# 生成 gRPC 代码
bash scripts/generate_proto.sh
```

### 2. 启动 Server (LeRobot 环境)

```bash
# Server 以空闲模式启动，等待 Client 配置
python -m src.server.inference_server --port 50051 --device cuda
```

### 3. 启动 Client (机器人侧)

```bash
# 数据集回放 (22维, 不控制底盘)
python -m src.client.inference_client \
    --server 192.168.1.100:50051 \
    --dataset /path/to/lerobot_dataset \
    --episode 0

# 模型推理 (22维, 不控制底盘)
python -m src.client.inference_client \
    --server 192.168.1.100:50051 \
    --model /path/to/trained_model

# 视觉策略 (带图像)
python -m src.client.inference_client \
    --server 192.168.1.100:50051 \
    --model /path/to/vision_policy \
    --enable-camera \
    --cameras head,wrist_left,wrist_right

# 移动机器人 (25维输入, 控制底盘)
python -m src.client.inference_client \
    --server 192.168.1.100:50051 \
    --model /path/to/mobile_policy \
    --state-with-chassis \
    --execute-chassis

# 开启平滑和速度限制
python -m src.client.inference_client \
    --server 192.168.1.100:50051 \
    --dataset /path/to/dataset \
    --smooth 5 \
    --max-velocity 0.05
```

## 配置参数

### Server 配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--host` | 0.0.0.0 | 监听地址 |
| `--port` | 50051 | 监听端口 |
| `--device` | cuda | 推理设备 |
| `--workers` | 10 | 工作线程数 |
| `--fps` | 30.0 | 目标帧率 |

### Client 配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--server` | localhost:50051 | Server 地址 |
| `--model` | - | 模型路径 (推理模式) |
| `--dataset` | - | 数据集路径 (回放模式) |
| `--device` | cuda | 推理设备 |
| `--episode` | 0 | Episode 索引 |

**输入配置:**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--state-with-chassis` | false | 输入 state 包含底盘 (25维) |

**执行配置:**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--execute-chassis` | false | 执行时控制底盘 |
| `--no-head` | false | 禁用头部控制 |
| `--no-torso` | false | 禁用腰部控制 |

**相机配置:**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--enable-camera` | false | 启用相机订阅 (视觉策略) |
| `--cameras` | head,wrist_left,wrist_right | 相机列表 |

**控制配置:**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--control-freq` | 30.0 | 控制频率 (Hz) |
| `--move-to-ready` | true | 启动时先移动到准备位置 |
| `--ready-duration` | 5.0 | 移动到准备位置的时间 (秒) |
| `--smooth` | 0 | 平滑窗口 (0=不平滑) |
| `--max-velocity` | 0.0 | 最大速度 rad/frame |

## gRPC 接口

```protobuf
service LeRobotInferenceService {
    // Client 配置 Server 使用的模型/数据集
    rpc Configure(PolicyConfig) returns (ServiceStatus);
    
    // 单次推理 (发送 state + images, 返回 action)
    rpc Predict(Observation) returns (Action);
    
    // 流式推理 (双向流，高频控制)
    rpc StreamPredict(stream Observation) returns (stream Action);
    
    // 控制命令 (RESET / SET_EPISODE)
    rpc Control(ControlCommand) returns (ServiceStatus);
    
    // 获取状态
    rpc GetStatus(Empty) returns (ServiceStatus);
    
    // 重置
    rpc Reset(Empty) returns (ServiceStatus);
}
```

## Python API

### 基础推理 (不带图像)

```python
from src.client.inference_client import InferenceClient
from src.common.config import ActionConfig

# 创建客户端
client = InferenceClient("192.168.1.100:50051")

# 配置 Server
client.configure(
    mode="model",
    model_path="/path/to/model",
    device="cuda",
    action_config=ActionConfig(
        state_includes_chassis=False,  # 输入 22 维
        execute_chassis=False          # 不控制底盘
    )
)

# 推理
response = client.predict(
    joint_positions=[0.0] * 22,  # 当前状态
    episode_id=0,
    frame_index=0
)

action = list(response.values)  # 获取 action
print(f"Action dim: {len(action)}, values: {action[:5]}...")

client.close()
```

### 视觉策略推理 (带图像)

```python
from src.client.inference_client import InferenceClient, AstribotCameraSubscriber
from src.common.config import ActionConfig

# 创建相机订阅器
camera = AstribotCameraSubscriber(['head', 'wrist_left', 'wrist_right'])
camera.start()
camera.wait_for_images(timeout=5.0)

# 创建客户端
client = InferenceClient("192.168.1.100:50051")
client.configure(
    mode="model",
    model_path="/path/to/vision_policy",
    action_config=ActionConfig(state_includes_chassis=False)
)

# 推理循环
for frame_idx in range(1000):
    # 获取图像
    images = camera.get_images_for_inference(client)
    
    # 推理 (state + images → action)
    response = client.predict(
        joint_positions=current_state,
        images=images,
        frame_index=frame_idx
    )
    
    action = list(response.values)
    # 发送到机器人...

camera.stop()
client.close()
```

### 高级控制器

```python
from src.client.inference_client import AstribotController
from src.common.config import ClientConfig, ActionConfig

# 配置
config = ClientConfig(
    server_host="192.168.1.100",
    server_port=50051,
    model_path="/path/to/model",
    control_freq=30.0,
    smooth_window=5,
    max_velocity=0.05,
    action_config=ActionConfig(
        state_includes_chassis=False,  # 输入 22 维
        execute_chassis=False          # 不控制底盘
    )
)

# 创建控制器 (带相机)
controller = AstribotController(
    config,
    enable_camera=True,
    camera_names=['head', 'wrist_left', 'wrist_right']
)

# 设置 episode
controller.set_episode(0)

# 移动到准备位置 (路径规划)
controller.move_to_ready_position(duration=5.0)

# 实时控制循环
while True:
    if not controller.step():  # step() 会发送 state + images
        break
    time.sleep(1.0 / 30)

controller.close()
```

## ROS 相机话题

| 相机名称 | ROS 话题 | 尺寸 |
|----------|----------|------|
| `head` | `/astribot_camera/head_rgbd/color_compress/compressed` | 1280x720 |
| `wrist_left` | `/astribot_camera/left_wrist_rgbd/color_compress/compressed` | 640x360 |
| `wrist_right` | `/astribot_camera/right_wrist_rgbd/color_compress/compressed` | 640x360 |
| `torso` | `/astribot_camera/torso_rgbd/color_compress/compressed` | 1280x720 |

## 两阶段控制流程

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                           两阶段控制                                           │
├───────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ╔═════════════════════════════════════════════════════════════════════════╗  │
│  ║  阶段 1: 移动到准备位置 (Ready Position)                                   ║  │
│  ║                                                                         ║  │
│  ║  • 使用预设的 READY_POSITION_22/25 (constants.py)                       ║  │
│  ║  • 使用 move_joints_waypoints() 轨迹规划                                ║  │
│  ║  • 耗时: ready_duration 秒 (默认 5.0s)                                  ║  │
│  ╚═════════════════════════════════════════════════════════════════════════╝  │
│                                     │                                         │
│                                     ▼                                         │
│  ╔═════════════════════════════════════════════════════════════════════════╗  │
│  ║  阶段 2: 实时控制 (Real-time)                                             ║  │
│  ║                                                                         ║  │
│  ║  循环 @ control_freq Hz:                                                 ║  │
│  ║    1. 获取当前关节位置 (state)                                            ║  │
│  ║    2. 获取相机图像 (images) - 如果启用                                    ║  │
│  ║    3. 发送 Predict 请求 (state + images)                                 ║  │
│  ║    4. 应用速度限制 (VelocityLimiter)                                     ║  │
│  ║    5. 应用动作平滑 (ActionSmoother)                                      ║  │
│  ║    6. 发送 set_joints_position 到机器人                                  ║  │
│  ╚═════════════════════════════════════════════════════════════════════════╝  │
│                                                                               │
└───────────────────────────────────────────────────────────────────────────────┘
```

## 项目结构

```
lerobot_grpc_inference/
├── proto/                          # Protocol Buffers 定义
│   └── lerobot_inference.proto
├── src/
│   ├── common/                     # 共享模块
│   │   ├── config.py              # 配置管理 (ActionConfig, ServerConfig, ClientConfig)
│   │   ├── constants.py           # 常量定义 (READY_POSITION, 维度配置)
│   │   └── utils.py               # 工具函数 (平滑器, 速度限制器, 格式转换)
│   ├── server/                     # Server 端代码
│   │   └── inference_server.py    # gRPC 服务, DatasetLoader, ModelInference
│   ├── client/                     # Client 端代码
│   │   └── inference_client.py    # InferenceClient, AstribotController, CameraSubscriber
│   └── generated/                  # 自动生成的 gRPC 代码
├── scripts/                        # 脚本
│   ├── generate_proto.sh          # 生成 gRPC 代码
│   ├── run_server.sh              # 启动 Server
│   └── run_client.sh              # 启动 Client
├── requirements.txt
└── README.md
```

## 许可证

BSD 3-Clause License
