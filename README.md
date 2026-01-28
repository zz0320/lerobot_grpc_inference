# LeRobot gRPC Inference

gRPC 远程推理框架：Server 端加载 LeRobot 策略模型，Client 端在 Astribot 机器人上实时执行。

## 架构

```
 ┌─────────── 机器人侧 ───────────┐          ┌────────── GPU 服务器 ──────────┐
 │                                │          │                               │
 │  ROS Camera ──► 图像采集        │  gRPC    │  LeRobot Policy               │
 │  Astribot SDK ──► 关节读取      │ ◄──────► │  (ACT / Diffusion / pi0 ...)  │
 │  AstribotController ──► 执行    │          │  PyTorch + CUDA               │
 │  InferenceLogger ──► 日志       │          │                               │
 └────────────────────────────────┘          └───────────────────────────────┘
        Client (Python 3.8+)                     Server (Python 3.10+)
```

**核心设计：** Server 以空闲模式启动，Client 连接时通过 `Configure()` 指定模型路径，Server 动态加载。

## 快速开始

### 1. 安装

```bash
# Server 端 (GPU 服务器，需要 lerobot + torch 环境)
pip install -r requirements-server.txt

# Client 端 (机器人侧)
pip install -r requirements-client.txt

# 生成 gRPC 代码 (两端都要执行)
bash scripts/generate_proto.sh
```

### 2. 启动 Server

```bash
python -m src.server.inference_server --port 50051 --device cuda
```

Server 启动后处于空闲状态，等待 Client 配置。

### 3. 启动 Client

```bash
# 基础推理
python -m src.client.inference_client \
    --server 192.168.1.100:50051 \
    --model /path/to/model

# 视觉策略 + Chunk 模式 (推荐用于 ACT/Diffusion)
python -m src.client.inference_client \
    --server 192.168.1.100:50051 \
    --model /path/to/model \
    --enable-camera \
    --use-chunk --n-action-steps 30

# 带底盘 + 平滑
python -m src.client.inference_client \
    --server 192.168.1.100:50051 \
    --model /path/to/model \
    --state-with-chassis --execute-chassis \
    --smooth 5 --max-velocity 0.05
```

## 推理模式

### 单步模式 (默认)

每帧调用 `Predict()` 获取一个 action。适用于简单策略。

### Chunk 模式 (`--use-chunk`)

调用 `PredictChunk()` 一次获取完整 action chunk，Client 在本地逐步消费。用完后再请求新 chunk。

适用于 ACT、Diffusion 等 action chunking 策略。`--n-action-steps N` 控制每个 chunk 实际使用的 action 数量。

```
Client                                Server
  │                                     │
  │── PredictChunk(obs) ──────────────► │
  │                                     │── policy.select_action(obs)
  │◄── ActionChunk [a0, a1, ..., aN] ──│
  │                                     │
  │  本地逐步消费 a0, a1, a2 ...        │
  │  (无网络调用)                        │
  │                                     │
  │  queue 空, 再次请求                  │
  │── PredictChunk(obs) ──────────────► │
  │ ...                                 │
```

## Action 处理流水线

Client `step()` 中，模型输出经过以下处理后发给机器人：

```
模型原始输出 (raw_action)
    │
    ▼
部件过滤 (filtered_action)      ← enable_head / enable_torso / enable_chassis
    │                              禁用的部件替换为当前关节值
    ▼
速度限制 (velocity_limiter)     ← --max-velocity
    │
    ▼
移动平均平滑 (smoother)         ← --smooth
    │
    ▼
发送到机器人 (final_action)
```

推理日志中会记录完整流水线的每个阶段（仅记录与最终值不同的阶段），便于事后分析。

## V2.0 数据格式

```
22 维 (不含底盘):
[ arm_left(7) | arm_right(7) | grip_L(1) | grip_R(1) | head(2) | torso(4) ]
  idx 0-6       idx 7-13       idx 14      idx 15     idx 16-17  idx 18-21

25 维 (含底盘):
[ ... 同上 22 维 ... | chassis(3) ]
                       idx 22-24
```

输入维度和执行维度独立配置：

| 参数 | 作用 | 默认 |
|------|------|------|
| `state_includes_chassis` | Client 采集的 state 是否包含底盘 | `False` (22 维) |
| `enable_chassis` | 执行时是否控制底盘 | `False` |
| `enable_head` | 执行时是否控制头部 | `True` |
| `enable_torso` | 执行时是否控制腰部 | `True` |

禁用的部件在 Client 端过滤，替换为当前关节实际读数，不会发送给机器人。

## 命令行参数

### Server

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--host` | `0.0.0.0` | 监听地址 |
| `--port` | `50051` | 监听端口 |
| `--device` | `cuda` | 推理设备 |
| `--workers` | `10` | gRPC 工作线程数 |
| `--fps` | `30.0` | 目标帧率 |

### Client

**连接与模型：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--server` | `localhost:50051` | Server 地址 |
| `--model` | - | 模型路径或 HuggingFace repo |
| `--device` | `cuda` | 推理设备 |
| `--policy-type` | 自动检测 | 策略类型 (act, diffusion, pi0 等) |

**维度与部件控制：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--state-with-chassis` | `false` | 输入 state 包含底盘 (25 维) |
| `--execute-chassis` | `false` | 执行时控制底盘 |
| `--no-head` | `false` | 禁用头部控制 |
| `--no-torso` | `false` | 禁用腰部控制 |

**相机 (视觉策略)：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--enable-camera` | `false` | 启用 ROS 相机订阅 |
| `--cameras` | `head,wrist_left,wrist_right,torso` | 订阅的相机列表 |

**Chunk 模式：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--use-chunk` | `false` | 启用 chunk 模式 |
| `--n-action-steps` | 全部 | 每个 chunk 使用的 action 数量 |

**控制与平滑：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--control-freq` | `30.0` | 控制频率 (Hz) |
| `--control-way` | `direct` | 控制方式 (`direct` / `filter`) |
| `--smooth` | `0` | 移动平均窗口 (0 = 不平滑) |
| `--max-velocity` | `0.0` | 最大速度 rad/frame (0 = 不限制) |

**启动流程：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--move-to-ready` / `--no-move-to-ready` | 开启 | 是否先移动到准备位置 |
| `--ready-duration` | `5.0` | 移动到准备位置耗时 (秒) |
| `--initial-transition` | `0.0` | 初始过渡时间 (秒)，平滑过渡到第一帧 action |
| `--episode` | `0` | Episode 索引 |
| `--max-frames` | `10000` | 最大帧数 |

**推理日志：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--enable-logging` / `--no-logging` | 开启 | 是否记录推理日志 |
| `--log-dir` | `./inference_logs` | 日志保存目录 |
| `--log-session-name` | 自动时间戳 | 日志会话名称 |
| `--log-save-images` / `--no-log-save-images` | 保存 | 是否保存图像 |
| `--log-image-format` | `jpg` | 图像格式 (`jpg` / `png`) |

## 推理日志

默认开启。每次运行生成一个 session 目录：

```
inference_logs/
└── session_2025-01-09_12-30-45/
    ├── metadata.json          # 会话配置、统计 (fps, 延迟, 帧数)
    ├── inference_log.jsonl    # 逐帧数据 (JSONL, 每行一条)
    └── images/
        ├── frame_000000/      # 仅在推理帧保存图像
        │   ├── head.jpg
        │   ├── wrist_left.jpg
        │   └── wrist_right.jpg
        └── frame_000050/
            └── ...
```

每条 JSONL 记录包含：

| 字段 | 类型 | 说明 |
|------|------|------|
| `state` | `float[]` | 关节状态 (输入) |
| `action` | `float[]` | 最终发给机器人的 action |
| `raw_action` | `float[]?` | 模型原始输出 (与 action 不同时记录) |
| `filtered_action` | `float[]?` | 部件过滤后 (与 action 不同时记录) |
| `smoothed_action` | `float[]?` | 平滑/限速后 (与 action 不同时记录) |
| `latency_ms` | `float` | 推理延迟 (毫秒) |
| `image_paths` | `dict` | 图像文件相对路径 |
| `extra_info` | `dict` | 额外信息 (`is_inference_frame` 等) |

### 读取日志

```python
from src.client.inference_logger import InferenceLogReader

reader = InferenceLogReader("./inference_logs/session_2025-01-09_12-30-45")

# 读取元信息
metadata = reader.get_metadata()
print(f"总帧数: {metadata['total_frames']}, 平均 FPS: {metadata.get('avg_fps', 'N/A')}")

# 加载为 numpy 数组
states, actions = reader.load_as_arrays()

# 加载完整 action 处理流水线
pipeline = reader.load_action_pipeline()
# pipeline["raw_action"]      → 模型原始输出
# pipeline["filtered_action"] → 部件过滤后
# pipeline["smoothed_action"] → 平滑后
# pipeline["final_action"]    → 发给机器人的

# 加载延迟数据
latencies = reader.load_latencies()  # shape (N,), 单位 ms

# 加载图像
img = reader.load_image(frame_index=0, camera_name="head")
```

## Python API

### InferenceClient (底层)

```python
from src.client.inference_client import InferenceClient
from src.common.config import ActionConfig

client = InferenceClient("192.168.1.100:50051")

# 配置
client.configure(
    mode="model",
    model_path="/path/to/model",
    device="cuda",
    action_config=ActionConfig(enable_chassis=False)
)

# 单步推理
response = client.predict(joint_positions=[0.0] * 22, images=images)
action = list(response.values)

# Chunk 推理
chunk = client.predict_chunk(joint_positions=[0.0] * 22, images=images)
for step in chunk.actions:
    action = list(step.values)

client.close()
```

### AstribotController (推荐)

整合 gRPC 客户端、Astribot SDK、ROS 相机、action 后处理和日志。

```python
from src.client.inference_client import AstribotController, run_inference_loop
from src.client.inference_logger import InferenceLogger
from src.common.config import ClientConfig, ActionConfig

config = ClientConfig(
    server_host="192.168.1.100",
    server_port=50051,
    model_path="/path/to/model",
    control_freq=30.0,
    smooth_window=5,
    max_velocity=0.05,
    action_config=ActionConfig(enable_chassis=False)
)

inference_logger = InferenceLogger(log_dir="./inference_logs", save_images=True)

controller = AstribotController(
    config,
    enable_camera=True,
    camera_names=['head', 'wrist_left', 'wrist_right'],
    use_chunk=True,
    n_action_steps=30,
    inference_logger=inference_logger,
)

# 一键运行: 准备位置 → 实时推理循环
run_inference_loop(controller, episode=0, max_frames=10000)
controller.close()
```

### ActionChunkManager (独立使用)

```python
from src.client.inference_client import InferenceClient, ActionChunkManager

client = InferenceClient("192.168.1.100:50051")
client.configure(mode="model", model_path="/path/to/act_model")

chunk_mgr = ActionChunkManager(client, n_action_steps=50)

for frame in range(1000):
    action = chunk_mgr.get_action(
        joint_positions=current_state,
        images=images,
        frame_index=frame
    )
    if action is None:
        break
    send_to_robot(action)
```

## gRPC 接口

```protobuf
service LeRobotInferenceService {
    rpc Configure(PolicyConfig) returns (ServiceStatus);
    rpc Predict(Observation) returns (Action);
    rpc PredictChunk(Observation) returns (ActionChunk);
    rpc StreamPredict(stream Observation) returns (stream Action);
    rpc Control(ControlCommand) returns (ServiceStatus);
    rpc GetStatus(Empty) returns (ServiceStatus);
    rpc Reset(Empty) returns (ServiceStatus);
}
```

完整 proto 定义见 `proto/lerobot_inference.proto`。

## ROS 相机话题

| 相机名称 | ROS 话题 | 尺寸 |
|----------|----------|------|
| `head` | `/astribot_camera/head_rgbd/color_compress/compressed` | 1280x720 |
| `wrist_left` | `/astribot_camera/left_wrist_rgbd/color_compress/compressed` | 640x360 |
| `wrist_right` | `/astribot_camera/right_wrist_rgbd/color_compress/compressed` | 640x360 |
| `torso` | `/astribot_camera/torso_rgbd/color_compress/compressed` | 1280x720 |

## 项目结构

```
lerobot_grpc_inference/
├── proto/
│   └── lerobot_inference.proto     # gRPC 接口定义
├── src/
│   ├── common/
│   │   ├── constants.py            # 维度、索引、准备位置、gRPC 常量
│   │   ├── config.py               # ActionConfig, ServerConfig, ClientConfig
│   │   ├── utils.py                # 平滑器、速度限制器、格式转换、部件过滤
│   │   └── proto_imports.py        # 共享 protobuf 导入逻辑
│   ├── server/
│   │   └── inference_server.py     # gRPC 服务、LeRobotModelInference
│   ├── client/
│   │   ├── inference_client.py     # InferenceClient, AstribotController, ActionChunkManager
│   │   └── inference_logger.py     # InferenceLogger, InferenceLogReader
│   └── generated/                  # 自动生成的 gRPC 代码
├── scripts/
│   ├── generate_proto.sh
│   ├── run_server.sh
│   └── run_client.sh
├── config/
│   └── default.json
├── requirements.txt                # 公共依赖 (grpc, numpy)
├── requirements-server.txt         # Server 额外依赖 (torch, lerobot)
└── requirements-client.txt         # Client 额外依赖 (Astribot SDK)
```

## 许可证

BSD 3-Clause License
