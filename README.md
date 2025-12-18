# LeRobot gRPC Inference

基于 gRPC 的 LeRobot 推理框架，支持 Astribot 机器人控制。

## 架构概览

```
┌─────────────────────────────────┐     gRPC      ┌─────────────────────────────────┐
│     Server (lerobot 环境)        │◄─────────────►│     Client (机器人侧)            │
│     Python 3.10.0               │               │     Astribot SDK 环境           │
│                                 │               │                                 │
│  - LeRobot 模型推理              │   Action      │  - Astribot SDK 控制             │
│  - 数据集回放                    │◄─────────────│  - 获取传感器数据                 │
│  - 生成 Action                  │──────────────►│  - 发送关节指令                   │
│                                 │  Observation  │                                 │
└─────────────────────────────────┘               └─────────────────────────────────┘
```

## 项目结构

```
lerobot_grpc_inference/
├── proto/                          # Protocol Buffers 定义
│   └── lerobot_inference.proto
├── src/
│   ├── common/                     # 共享模块
│   │   ├── config.py              # 配置管理
│   │   ├── constants.py           # 常量定义
│   │   └── utils.py               # 工具函数
│   ├── server/                     # Server 端代码
│   │   └── inference_server.py
│   ├── client/                     # Client 端代码
│   │   └── inference_client.py
│   └── generated/                  # 自动生成的 gRPC 代码
├── scripts/                        # 脚本
│   ├── generate_proto.sh          # 生成 gRPC 代码
│   ├── run_server.sh              # 启动 Server
│   └── run_client.sh              # 启动 Client
├── config/                         # 配置文件
│   └── default.json
├── requirements.txt                # 通用依赖
├── requirements-server.txt         # Server 额外依赖
├── requirements-client.txt         # Client 额外依赖
└── README.md
```

## 快速开始

### 1. 安装 (两个环境都执行)

```bash
# 克隆/复制项目到两个环境
# 环境1: lerobot 环境 (Python 3.10)
# 环境2: 机器人侧 (Astribot SDK 环境)

# 安装依赖
pip install -r requirements.txt

# 生成 gRPC 代码 (重要!)
bash scripts/generate_proto.sh
```

### 2. 启动 Server (lerobot 环境)

```bash
# 数据集回放模式
python -m src.server.inference_server \
    --host 0.0.0.0 \
    --port 50051 \
    --dataset /path/to/lerobot_dataset

# 模型推理模式 - 本地模型
python -m src.server.inference_server \
    --host 0.0.0.0 \
    --port 50051 \
    --model /path/to/trained_model

# 模型推理模式 - 从 HuggingFace Hub 加载
python -m src.server.inference_server \
    --model lerobot/act_aloha_sim_insertion_human \
    --device cuda

# 或使用脚本
bash scripts/run_server.sh --dataset /path/to/lerobot_dataset
```

### 3. 启动 Client (机器人侧) - **推荐方式：Client 指定模型/数据集**

```bash
# 数据集回放 (Client 端指定数据集路径)
python -m src.client.inference_client \
    --server 192.168.1.100:50051 \
    --dataset /path/to/lerobot_dataset \
    --episode 0

# 模型推理 (Client 端指定模型路径)
python -m src.client.inference_client \
    --server 192.168.1.100:50051 \
    --model /path/to/trained_model \
    --device cuda

# 从 HuggingFace Hub 加载模型
python -m src.client.inference_client \
    --server 192.168.1.100:50051 \
    --model lerobot/act_aloha_sim_insertion_human

# 不指定模型/数据集 (使用 Server 启动时的默认配置)
python -m src.client.inference_client \
    --server 192.168.1.100:50051 \
    --episode 0

# 开启平滑和速度限制
python -m src.client.inference_client \
    --server 192.168.1.100:50051 \
    --dataset /path/to/dataset \
    --smooth 5 \
    --max-velocity 0.05
```

## 详细配置

### Server 配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--host` | 0.0.0.0 | 监听地址 |
| `--port` | 50051 | 监听端口 |
| `--model` | - | 模型路径 (模型推理模式) |
| `--dataset` | - | 数据集路径 (数据集回放模式) |
| `--device` | cuda | 推理设备 |
| `--workers` | 10 | 工作线程数 |
| `--fps` | 30.0 | 目标帧率 |

### Client 配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--server` | localhost:50051 | Server 地址 |
| `--episode` | 0 | Episode 索引 |
| `--control-freq` | 30.0 | 控制频率 (Hz) |
| `--control-way` | direct | 控制方式 (direct/filter) |
| `--planning-frames` | 5 | 前 N 帧使用路径规划 |
| `--planning-duration` | 3.0 | 路径规划时间 (秒) |
| `--smooth` | 0 | 平滑窗口 (0=不平滑) |
| `--max-velocity` | 0.0 | 最大速度 rad/frame (0=不限制) |
| `--timeout` | 10.0 | 连接超时 (秒) |

### 环境变量

```bash
# Server 端
export LEROBOT_SERVER_HOST=0.0.0.0
export LEROBOT_SERVER_PORT=50051
export LEROBOT_DATASET_PATH=/path/to/dataset
export LEROBOT_MODEL_PATH=/path/to/model
export LEROBOT_DEVICE=cuda

# Client 端
export LEROBOT_SERVER=192.168.1.100:50051
export LEROBOT_EPISODE=0
export LEROBOT_CONTROL_FREQ=30
```

## API 使用

### Python API (Client 端)

```python
from src.client.inference_client import InferenceClient, AstribotController
from src.common.config import ClientConfig

# 方式1: 使用低级 API
client = InferenceClient(server_address="192.168.1.100:50051")

# 检查连接
status = client.get_status()
print(f"服务就绪: {status.is_ready}")

# 单次推理
action = client.predict(
    joint_positions=[0.0] * 16,
    episode_id=0,
    frame_index=0
)
print(f"Action: {list(action.values)}")

client.close()

# 方式2: 使用高级控制器
config = ClientConfig(
    server_host="192.168.1.100",
    server_port=50051,
    control_freq=30.0
)

controller = AstribotController(config)
controller.set_episode(0)
controller.move_to_initial_position(duration=3.0)

# 控制循环
while True:
    if not controller.step():
        break
    time.sleep(1.0 / 30)

controller.close()
```

### Python API (Server 端)

```python
from src.server.inference_server import InferenceServer
from src.common.config import ServerConfig

config = ServerConfig(
    host="0.0.0.0",
    port=50051,
    dataset_path="/path/to/dataset"
)

server = InferenceServer(config)
server.start()
server.wait_for_termination()
```

## gRPC 接口

### 消息类型

```protobuf
// 观测数据
message Observation {
    repeated float joint_positions = 1;  // [16] 关节位置
    repeated ImageData images = 2;        // 图像数据 (可选)
    double timestamp = 3;
    int32 episode_id = 4;
    int32 frame_index = 5;
}

// 动作指令
message Action {
    repeated float values = 1;   // [16] 关节目标位置
    bool is_terminal = 2;        // 是否结束
    StatusCode status = 3;       // 状态码
}
```

### 服务接口

```protobuf
service LeRobotInferenceService {
    rpc Predict(Observation) returns (Action);           // 单次推理
    rpc StreamPredict(stream Observation) returns (stream Action);  // 流式
    rpc GetStatus(Empty) returns (ServiceStatus);        // 获取状态
    rpc Reset(Empty) returns (ServiceStatus);            // 重置
    rpc Control(ControlCommand) returns (ServiceStatus); // 控制命令
}
```

## 数据格式

### LeRobot Action 格式 (16维)

```
[arm_left(7), arm_right(7), gripper_left(1), gripper_right(1)]

索引:
  0-6:   左臂 7 个关节
  7-13:  右臂 7 个关节
  14:    左夹爪
  15:    右夹爪
```

### Astribot Waypoint 格式

```
[torso(4), arm_left(7), gripper_left(1), arm_right(7), gripper_right(1), head(2)]
```

## LeRobot 推理流程

Server 端直接复用 LeRobot 的推理逻辑：

```python
# 1. 加载模型 (使用 LeRobot 的 factory)
from lerobot.policies.factory import get_policy_class, make_pre_post_processors

policy_class = get_policy_class("act")  # 或 "diffusion", "pi0" 等
policy = policy_class.from_pretrained("path/to/model")
policy.to("cuda")
policy.eval()

# 2. 创建预处理器和后处理器
preprocessor, postprocessor = make_pre_post_processors(
    policy.config,
    pretrained_path="path/to/model",
)

# 3. 推理
observation = preprocessor(raw_observation)  # 归一化、设备转移
with torch.inference_mode():
    action = policy.select_action(observation)  # 返回单个 action
    # 或
    action_chunk = policy.predict_action_chunk(observation)  # 返回 action chunk
action = postprocessor(action)  # 反归一化
```

### 支持的策略类型

| 策略 | 类型名 | 说明 |
|------|--------|------|
| ACT | `act` | Action Chunking Transformer |
| Diffusion | `diffusion` | Diffusion Policy |
| Pi0 | `pi0` | Physical Intelligence π0 |
| Pi05 | `pi05` | Physical Intelligence π0.5 |
| TDMPC | `tdmpc` | TD-MPC |
| VQ-BeT | `vqbet` | VQ-BeT |
| SmolVLA | `smolvla` | Small Vision-Language-Action |
| GR00T | `groot` | NVIDIA GR00T |

## 常见问题

### Q: 如何处理网络延迟?

使用流式推理 (`StreamPredict`) 可以减少每次请求的开销。同时确保 Server 和 Client 在同一局域网内。

### Q: 如何处理抖动?

启用平滑参数:
```bash
python -m src.client.inference_client --smooth 5 --max-velocity 0.05
```

### Q: 如何调试 gRPC 连接?

使用 `grpcurl` 工具:
```bash
# 安装 grpcurl
# 列出服务
grpcurl -plaintext localhost:50051 list

# 调用方法
grpcurl -plaintext localhost:50051 lerobot.LeRobotInferenceService/GetStatus
```

## 许可证

BSD 3-Clause License

