# -*- coding: utf-8 -*-
"""
推理日志记录器

在 Client 侧记录全部推理信息:
- State (关节位置)
- Action (推理输出)
- Image (图像，保存为图片文件)

日志目录结构:
    inference_logs/
        session_2025-01-09_12-30-45/
            metadata.json          # 会话元信息
            inference_log.jsonl    # 推理数据 (每行一条 JSON 记录)
            images/
                frame_000000/
                    head.jpg
                    wrist_left.jpg
                    wrist_right.jpg
                frame_000001/
                    ...

Example:
    >>> logger = InferenceLogger(
    ...     log_dir="./inference_logs",
    ...     session_name="my_experiment",
    ...     save_images=True
    ... )
    >>> logger.start_session(episode_id=0, model_path="/path/to/model")
    >>> 
    >>> # 在推理循环中
    >>> for frame_idx in range(100):
    ...     state = get_current_state()
    ...     images = get_current_images()
    ...     action = inference_client.predict(state, images=images)
    ...     
    ...     # 记录推理信息
    ...     logger.log_step(
    ...         frame_index=frame_idx,
    ...         state=state,
    ...         action=list(action.values),
    ...         images=images,
    ...         extra_info={"confidence": action.confidence}
    ...     )
    >>> 
    >>> logger.end_session()
"""

import os
import sys
import json
import time
import logging
from datetime import datetime
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
import numpy as np

logger = logging.getLogger("lerobot_inference.client.logger")


@dataclass
class InferenceLogEntry:
    """
    单步推理日志条目

    记录完整的 action 处理流水线:
      raw_action   -> filtered_action -> smoothed_action -> final_action
      (模型输出)      (部件过滤)         (平滑+限速)        (发给机器人)

    若某阶段未改变 action，对应字段为 None，节省存储空间。
    """
    timestamp: float                            # Unix 时间戳
    episode_id: int                             # Episode ID
    frame_index: int                            # 帧索引
    state: List[float]                          # 关节状态 (输入)
    action: List[float]                         # 最终发给机器人的 action
    image_paths: Dict[str, str]                 # 图像文件路径 {camera_name: path}
    latency_ms: Optional[float] = None          # 推理延迟 (毫秒)
    raw_action: Optional[List[float]] = None    # 模型原始输出 (与 action 不同时才记录)
    filtered_action: Optional[List[float]] = None  # 部件过滤后 (与 raw/final 不同时才记录)
    smoothed_action: Optional[List[float]] = None  # 平滑/限速后 (与 filtered/final 不同时才记录)
    extra_info: Optional[Dict] = None           # 额外信息


class InferenceLogger:
    """
    推理日志记录器
    
    记录 Client 侧的全部推理信息，包括 state、action 和图像。
    """
    
    def __init__(
        self,
        log_dir: str = "./inference_logs",
        session_name: Optional[str] = None,
        save_images: bool = True,
        image_format: str = "jpg",
        image_quality: int = 95,
        flush_interval: int = 10,
        enabled: bool = True
    ):
        """
        初始化日志记录器
        
        Args:
            log_dir: 日志根目录
            session_name: 会话名称 (默认: 自动生成时间戳)
            save_images: 是否保存图像
            image_format: 图像格式 ("jpg", "png")
            image_quality: 图像质量 (仅 JPEG, 1-100)
            flush_interval: 每多少步刷新一次文件
            enabled: 是否启用日志记录
        """
        self.log_dir = log_dir
        self.save_images = save_images
        self.image_format = image_format.lower()
        self.image_quality = image_quality
        self.flush_interval = flush_interval
        self.enabled = enabled
        
        # 会话目录 (自动附加时间戳)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if session_name:
            self.session_name = f"{session_name}_{timestamp}"
        else:
            self.session_name = f"session_{timestamp}"
        
        self.session_dir = os.path.join(log_dir, self.session_name)
        self.images_dir = os.path.join(self.session_dir, "images")
        
        # 日志文件
        self.metadata_path = os.path.join(self.session_dir, "metadata.json")
        self.log_path = os.path.join(self.session_dir, "inference_log.jsonl")
        
        # 状态
        self._log_file = None
        self._session_started = False
        self._step_count = 0
        self._metadata = {}
        
        # 统计
        self._total_frames = 0
        self._total_images_saved = 0
    
    def start_session(
        self,
        episode_id: int = 0,
        model_path: Optional[str] = None,
        config: Optional[Dict] = None,
        **extra_metadata
    ):
        """
        开始新的日志会话
        
        Args:
            episode_id: Episode ID
            model_path: 模型路径
            config: 配置信息
            **extra_metadata: 额外的元信息
        """
        if not self.enabled:
            return
        
        # 创建目录
        os.makedirs(self.session_dir, exist_ok=True)
        if self.save_images:
            os.makedirs(self.images_dir, exist_ok=True)
        
        # 构建元信息
        self._metadata = {
            "session_name": self.session_name,
            "start_time": datetime.now().isoformat(),
            "start_timestamp": time.time(),
            "episode_id": episode_id,
            "model_path": model_path,
            "config": config or {},
            "save_images": self.save_images,
            "image_format": self.image_format,
            **extra_metadata
        }
        
        # 写入元信息
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self._metadata, f, indent=2, ensure_ascii=False)
        
        # 打开日志文件
        self._log_file = open(self.log_path, 'w', encoding='utf-8')
        
        self._session_started = True
        self._step_count = 0
        self._total_frames = 0
        self._total_images_saved = 0
        
        logger.info(f"推理日志会话已开始: {self.session_dir}")
        logger.info(f"  - 保存图像: {self.save_images}")
        logger.info(f"  - 图像格式: {self.image_format}")
    
    def log_step(
        self,
        frame_index: int,
        state: List[float],
        action: List[float],
        images: Optional[List[Dict]] = None,
        episode_id: int = 0,
        latency_ms: Optional[float] = None,
        raw_action: Optional[List[float]] = None,
        filtered_action: Optional[List[float]] = None,
        smoothed_action: Optional[List[float]] = None,
        extra_info: Optional[Dict] = None,
        save_images_this_step: bool = True
    ):
        """
        记录单步推理信息

        记录完整的 action 处理流水线:
          raw_action -> filtered_action -> smoothed_action -> action (final)

        为节省存储，当某阶段输出与最终 action 相同时，该字段记录为 None。

        Args:
            frame_index: 帧索引
            state: 关节状态 (float list)
            action: 最终发给机器人的 action (float list)
            images: 图像列表 (与 predict() 使用的格式相同)
                    [{'name': 'head', 'data': bytes, ...}, ...]
            episode_id: Episode ID
            latency_ms: 推理延迟 (毫秒)
            raw_action: 模型原始输出 (与 action 不同时传入)
            filtered_action: 部件过滤后的 action (与 action 不同时传入)
            smoothed_action: 平滑/限速后的 action (与 action 不同时传入)
            extra_info: 额外信息
            save_images_this_step: 是否在当前步保存图像 (默认 True)
                                   设为 False 可仅记录数据而不保存图像
        """
        if not self.enabled or not self._session_started:
            return

        # 保存图像 (只有当 save_images 和 save_images_this_step 都为 True 时才保存)
        image_paths = {}
        if self.save_images and images and save_images_this_step:
            image_paths = self._save_images(frame_index, images)

        def _to_list(v):
            if v is None:
                return None
            return list(v) if isinstance(v, np.ndarray) else list(v)

        final_action = _to_list(action)

        # 只记录与 final_action 不同的中间结果，节省存储
        raw_list = _to_list(raw_action)
        filtered_list = _to_list(filtered_action)
        smoothed_list = _to_list(smoothed_action)

        # 构建日志条目
        entry = InferenceLogEntry(
            timestamp=time.time(),
            episode_id=episode_id,
            frame_index=frame_index,
            state=list(state) if isinstance(state, np.ndarray) else state,
            action=final_action,
            image_paths=image_paths,
            latency_ms=latency_ms,
            raw_action=raw_list if raw_list != final_action else None,
            filtered_action=filtered_list if filtered_list != final_action else None,
            smoothed_action=smoothed_list if smoothed_list != final_action else None,
            extra_info=extra_info
        )
        
        # 写入 JSONL
        entry_dict = asdict(entry)
        self._log_file.write(json.dumps(entry_dict, ensure_ascii=False) + '\n')
        
        self._step_count += 1
        self._total_frames += 1
        
        # 定期刷新
        if self._step_count % self.flush_interval == 0:
            self._log_file.flush()
    
    def _save_images(self, frame_index: int, images: List[Dict]) -> Dict[str, str]:
        """
        保存图像到文件
        
        Args:
            frame_index: 帧索引
            images: 图像列表
        
        Returns:
            {camera_name: relative_path} 字典
        """
        image_paths = {}
        
        # 创建帧目录
        frame_dir = os.path.join(self.images_dir, f"frame_{frame_index:06d}")
        os.makedirs(frame_dir, exist_ok=True)
        
        for img_info in images:
            camera_name = img_info.get('name', 'unknown')
            img_data = img_info.get('data', b'')
            encoding = img_info.get('encoding', 'jpeg').lower()
            
            if not img_data:
                continue
            
            # 确定文件扩展名
            if encoding in ['jpeg', 'jpg']:
                ext = 'jpg'
            elif encoding == 'png':
                ext = 'png'
            else:
                # raw 或其他格式，需要转换
                ext = self.image_format
            
            # 文件路径
            filename = f"{camera_name}.{ext}"
            filepath = os.path.join(frame_dir, filename)
            relative_path = os.path.join("images", f"frame_{frame_index:06d}", filename)
            
            try:
                if encoding in ['jpeg', 'jpg', 'png']:
                    # 直接保存压缩图像
                    with open(filepath, 'wb') as f:
                        f.write(img_data)
                else:
                    # raw 格式，需要用 PIL 转换
                    self._save_raw_image(img_data, img_info, filepath)
                
                image_paths[camera_name] = relative_path
                self._total_images_saved += 1
                
            except Exception as e:
                logger.warning(f"保存图像失败 [{camera_name}]: {e}")
        
        return image_paths
    
    def _save_raw_image(self, img_data: bytes, img_info: Dict, filepath: str):
        """保存 raw 格式图像"""
        try:
            from PIL import Image
            
            width = img_info.get('width', 0)
            height = img_info.get('height', 0)
            
            if width <= 0 or height <= 0:
                logger.warning(f"无效的图像尺寸: {width}x{height}")
                return
            
            # 假设 RGB 格式
            img = Image.frombytes('RGB', (width, height), img_data)
            
            if filepath.endswith('.jpg'):
                img.save(filepath, 'JPEG', quality=self.image_quality)
            else:
                img.save(filepath, 'PNG')
                
        except ImportError:
            logger.warning("需要安装 PIL/Pillow 来保存 raw 格式图像")
        except Exception as e:
            logger.warning(f"保存 raw 图像失败: {e}")
    
    def end_session(self):
        """结束日志会话"""
        if not self.enabled or not self._session_started:
            return
        
        # 更新元信息
        self._metadata["end_time"] = datetime.now().isoformat()
        self._metadata["end_timestamp"] = time.time()
        self._metadata["total_frames"] = self._total_frames
        self._metadata["total_images_saved"] = self._total_images_saved
        
        duration = self._metadata["end_timestamp"] - self._metadata["start_timestamp"]
        self._metadata["duration_seconds"] = duration
        
        if duration > 0 and self._total_frames > 0:
            self._metadata["avg_fps"] = self._total_frames / duration
        
        # 写入更新后的元信息
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self._metadata, f, indent=2, ensure_ascii=False)
        
        # 关闭日志文件
        if self._log_file:
            self._log_file.flush()
            self._log_file.close()
            self._log_file = None
        
        self._session_started = False
        
        logger.info(f"推理日志会话已结束")
        logger.info(f"  - 总帧数: {self._total_frames}")
        logger.info(f"  - 保存图像数: {self._total_images_saved}")
        logger.info(f"  - 日志目录: {self.session_dir}")
    
    def get_session_dir(self) -> str:
        """获取当前会话目录"""
        return self.session_dir
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "session_name": self.session_name,
            "session_dir": self.session_dir,
            "total_frames": self._total_frames,
            "total_images_saved": self._total_images_saved,
            "session_started": self._session_started
        }
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出"""
        self.end_session()


class InferenceLogReader:
    """
    推理日志读取器
    
    读取和分析保存的推理日志
    
    Example:
        >>> reader = InferenceLogReader("./inference_logs/session_2025-01-09_12-30-45")
        >>> 
        >>> # 读取元信息
        >>> metadata = reader.get_metadata()
        >>> print(f"总帧数: {metadata['total_frames']}")
        >>> 
        >>> # 遍历日志条目
        >>> for entry in reader.iter_entries():
        ...     print(f"Frame {entry['frame_index']}: action={entry['action'][:3]}...")
        >>> 
        >>> # 读取所有数据为 numpy 数组
        >>> states, actions = reader.load_as_arrays()
        >>> print(f"States shape: {states.shape}")
        >>> print(f"Actions shape: {actions.shape}")
    """
    
    def __init__(self, session_dir: str):
        """
        初始化读取器
        
        Args:
            session_dir: 会话目录路径
        """
        self.session_dir = session_dir
        self.metadata_path = os.path.join(session_dir, "metadata.json")
        self.log_path = os.path.join(session_dir, "inference_log.jsonl")
        self.images_dir = os.path.join(session_dir, "images")
        
        if not os.path.exists(session_dir):
            raise FileNotFoundError(f"会话目录不存在: {session_dir}")
    
    def get_metadata(self) -> Dict:
        """读取元信息"""
        with open(self.metadata_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def iter_entries(self):
        """
        迭代日志条目
        
        Yields:
            dict: 日志条目
        """
        with open(self.log_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
    
    def load_entries(self) -> List[Dict]:
        """加载所有日志条目"""
        return list(self.iter_entries())
    
    def load_as_arrays(self) -> tuple:
        """
        将 state 和 action 加载为 numpy 数组

        Returns:
            (states, actions): numpy 数组元组
                states: shape (N, state_dim)
                actions: shape (N, action_dim)
        """
        states = []
        actions = []

        for entry in self.iter_entries():
            states.append(entry['state'])
            actions.append(entry['action'])

        return np.array(states), np.array(actions)

    def load_action_pipeline(self) -> Dict[str, np.ndarray]:
        """
        加载完整的 action 处理流水线数据

        Returns:
            dict 包含:
                "state": shape (N, state_dim)
                "raw_action": shape (N, action_dim) — 模型原始输出
                "filtered_action": shape (N, action_dim) — 部件过滤后
                "smoothed_action": shape (N, action_dim) — 平滑/限速后
                "final_action": shape (N, action_dim) — 最终发给机器人

            若某阶段与 final_action 相同，该阶段数组内容等于 final_action。
        """
        finals = []
        raws = []
        filtereds = []
        smootheds = []
        states = []

        for entry in self.iter_entries():
            final = entry['action']
            finals.append(final)
            states.append(entry['state'])
            # 若中间字段为 None，说明与 final 相同
            raws.append(entry.get('raw_action') or final)
            filtereds.append(entry.get('filtered_action') or final)
            smootheds.append(entry.get('smoothed_action') or final)

        return {
            "state": np.array(states),
            "raw_action": np.array(raws),
            "filtered_action": np.array(filtereds),
            "smoothed_action": np.array(smootheds),
            "final_action": np.array(finals),
        }

    def load_latencies(self) -> np.ndarray:
        """
        加载所有帧的推理延迟

        Returns:
            shape (N,) 的 numpy 数组 (毫秒)，None 值填充为 NaN
        """
        latencies = []
        for entry in self.iter_entries():
            v = entry.get('latency_ms')
            latencies.append(v if v is not None else float('nan'))
        return np.array(latencies)
    
    def get_image_path(self, frame_index: int, camera_name: str) -> Optional[str]:
        """
        获取指定帧的图像路径
        
        Args:
            frame_index: 帧索引
            camera_name: 相机名称
        
        Returns:
            图像文件绝对路径，如果不存在返回 None
        """
        # 尝试不同扩展名
        for ext in ['jpg', 'png']:
            path = os.path.join(
                self.images_dir,
                f"frame_{frame_index:06d}",
                f"{camera_name}.{ext}"
            )
            if os.path.exists(path):
                return path
        return None
    
    def load_image(self, frame_index: int, camera_name: str):
        """
        加载指定帧的图像
        
        Args:
            frame_index: 帧索引
            camera_name: 相机名称
        
        Returns:
            PIL Image 对象，如果不存在返回 None
        """
        try:
            from PIL import Image
            
            path = self.get_image_path(frame_index, camera_name)
            if path:
                return Image.open(path)
            return None
            
        except ImportError:
            logger.warning("需要安装 PIL/Pillow 来加载图像")
            return None


def list_sessions(log_dir: str = "./inference_logs") -> List[Dict]:
    """
    列出所有日志会话
    
    Args:
        log_dir: 日志根目录
    
    Returns:
        会话信息列表
    """
    sessions = []
    
    if not os.path.exists(log_dir):
        return sessions
    
    for name in sorted(os.listdir(log_dir), reverse=True):
        session_dir = os.path.join(log_dir, name)
        metadata_path = os.path.join(session_dir, "metadata.json")
        
        if os.path.isdir(session_dir) and os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                sessions.append({
                    "name": name,
                    "path": session_dir,
                    "start_time": metadata.get("start_time"),
                    "total_frames": metadata.get("total_frames", 0),
                    "duration_seconds": metadata.get("duration_seconds", 0),
                    "model_path": metadata.get("model_path"),
                })
            except Exception as e:
                logger.warning(f"读取会话元信息失败 [{name}]: {e}")
    
    return sessions

