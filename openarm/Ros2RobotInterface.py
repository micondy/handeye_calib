"""
ROS 2 Robot Interface

Interface class for communicating with ROS 2 robots through topics.
"""

import logging
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import Pose, PoseStamped, Twist
from tf2_ros import Buffer, ConnectivityException, ExtrapolationException, LookupException, TransformListener
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from rclpy.duration import Duration
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.publisher import Publisher
from rclpy.subscription import Subscription
from rclpy.time import Time

from .Ros2RobotConfig import Ros2RobotConfig

# ros_interface.py
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from control_msgs.msg import JointTrajectoryControllerState
from sensor_msgs.msg import CameraInfo, Image
from realsense2_camera_msgs.msg import RGBD

#对照表，订阅要消息类型
MSG_TYPE_MAP = {
    "sensor_msgs/msg/JointState": JointState,
    "std_msgs/msg/Float64MultiArray" : Float64MultiArray,
    "control_msgs/msg/JointTrajectoryControllerState" :JointTrajectoryControllerState,
    "sensor_msgs/msg/CameraInfo": CameraInfo,
    "sensor_msgs/msg/Image": Image,
    "realsense2_camera_msgs/msg/RGBD": RGBD,
    "geometry_msgs/msg/PoseStamped": PoseStamped,
}

from threading import Lock

logger = logging.getLogger(__name__)


def _transform_to_matrix(trans, rot):
    """
    将平移和四元数旋转转为4x4变换矩阵
    """
    import numpy as np
    # 平移（兼容geometry_msgs和array/list）
    T = np.eye(4)
    if hasattr(trans, 'x'):
        tx, ty, tz = trans.x, trans.y, trans.z
    else:
        tx, ty, tz = trans[0], trans[1], trans[2]
    T[0:3, 3] = [tx, ty, tz]
    # 四元数转旋转矩阵
    if hasattr(rot, 'x'):
        x, y, z, w = rot.x, rot.y, rot.z, rot.w
    else:
        x, y, z, w = rot[0], rot[1], rot[2], rot[3]
    R = np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x**2 + y**2)]
    ])
    T[0:3, 0:3] = R
    return T

def _matrix_to_pose(T):
    """
    4x4变换矩阵转为平移+四元数
    """
    import numpy as np
    from scipy.spatial.transform import Rotation as R
    pos = T[0:3, 3]
    rot = R.from_matrix(T[0:3, 0:3]).as_quat() # x,y,z,w
    return pos, rot

class Ros2RobotInterface:
    
    def __init__(self, config: Ros2RobotConfig):
        """
        ROS2机器人接口初始化
        """
        self._config = config
        self._connected: bool = False
        self._subscriptions = {} # [str, Subscription] 话题名和订阅者对象
        self._publishers = {}    # [str, Publisher] 话题名和发布者对象
        self._joints_names: List[str] = [] # 关节名称列表

        self._executor: SingleThreadedExecutor | None = None
        self._executor_thread: threading.Thread | None = None
        self._robot_node: Node | None = None

        self._cache_lock = Lock() # 锁
        # 回调函数缓存的最新消息和时间戳 str是 话题名 存储msg和消息的时间
        self._latest_msgs: Dict[str, Any] = {}
        self._latest_stamp: Dict[str, float] = {}
        self._tf_buffer: Optional[Buffer] = None
        self._tf_listener: Optional[TransformListener] = None

    @property
    def is_connected(self) -> bool:
        return self._connected and self._robot_node is not None
    
    def connect(self) -> None:
        """
        连接到ROS 2机器人，创建节点、订阅和发布者，并启动执行器线程。
        """ 
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} 已经连接")
        try:
            if not rclpy.ok():
                rclpy.init()

            self._robot_node = Node(
                node_name=self._config.robot_name+"_node",
            )

            self._tf_buffer = Buffer()
            self._tf_listener = TransformListener(self._tf_buffer, self._robot_node, spin_thread=False)

            # 创建订阅和发布者
            for name, cfg in self._config.topics_to_subscribe.items():
                msg_type = MSG_TYPE_MAP[cfg["type"]]
                topic_name = cfg["topic"]
                sub = self._robot_node.create_subscription(
                    msg_type,
                    topic_name,
                    lambda msg, n=name: self._generic_callback(n, msg),
                    10
                )
                self._subscriptions[name] = sub
                logger.info(f"已订阅话题: {topic_name} 类型: {cfg['type']}")

            for name, cfg in self._config.topics_to_publish.items():
                msg_type = MSG_TYPE_MAP[cfg["type"]]
                topic_name = cfg["topic"]
                pub = self._robot_node.create_publisher(
                    msg_type,
                    topic_name,
                    10
                )
                self._publishers[name] = pub
                logger.info(f"已发布话题: {topic_name} 类型: {cfg['type']}")

            # executor and thread,线程化
            self._executor = SingleThreadedExecutor()
            self._executor.add_node(self._robot_node)
            self._executor_thread = threading.Thread(
                target=self._executor.spin,
                daemon=True
            )
            self._executor_thread.start()

            time.sleep(1.0)
            self._connected = True
        except Exception as e:
            logger.error(f"连接ROS 2机器人接口失败: {e}")
            self.disconnect()
            raise

    def _stamp_to_time(self, stamp: Optional[Dict[str, int] | Tuple[int, int] | float]) -> Time:
        if stamp is None:
            return Time()

        if isinstance(stamp, dict):
            sec = int(stamp.get('sec', 0))
            nanosec = int(stamp.get('nanosec', 0))
            return Time(seconds=sec, nanoseconds=nanosec)

        if isinstance(stamp, tuple) and len(stamp) == 2:
            return Time(seconds=int(stamp[0]), nanoseconds=int(stamp[1]))

        stamp_float = float(stamp)
        sec = int(stamp_float)
        nanosec = int((stamp_float - sec) * 1e9)
        return Time(seconds=sec, nanoseconds=nanosec)

    def get_link_pose(
        self,
        link_name: str,
        reference_frame: str,
        stamp: Optional[Dict[str, int] | Tuple[int, int] | float] = None,
        max_time_diff: float = 0.05,
    ) -> Optional[Dict[str, Any]]:
        """
        获取指定link在reference_frame下的空间坐标（平移+四元数）
        优先使用 TF2 Buffer.lookup_transform 按时间戳查询。
        如果提供 stamp，则按该时刻查询（秒/纳秒或浮点秒）；未提供则查询最新。
        返回: {
            'position': np.array([x,y,z]),
            'orientation': np.array([x,y,z,w]),
            'pose_stamp': {'sec': int, 'nanosec': int},
        }
        """
        if self._tf_buffer is None:
            logger.warning("TF2 Buffer 未初始化，无法查询 link pose")
            return None

        try:
            query_time = self._stamp_to_time(stamp)

            timeout_sec = max(0.0, float(max_time_diff))
            transform = self._tf_buffer.lookup_transform(
                reference_frame.strip('/'),
                link_name.strip('/'),
                query_time,
                timeout=Duration(seconds=timeout_sec),
            )
        except (LookupException, ConnectivityException, ExtrapolationException) as exc:
            logger.warning(
                f"TF2 查询失败: {link_name} -> {reference_frame}, stamp={stamp}, error={exc}"
            )
            return None
        except Exception as exc:
            logger.warning(
                f"解析时间戳或查询TF失败: link={link_name}, reference={reference_frame}, stamp={stamp}, error={exc}"
            )
            return None

        trans = transform.transform.translation
        rot = transform.transform.rotation
        pose_stamp = {
            'sec': int(transform.header.stamp.sec),
            'nanosec': int(transform.header.stamp.nanosec),
        }
        return {
            'position': np.array([trans.x, trans.y, trans.z], dtype=np.float64),
            'orientation': np.array([rot.x, rot.y, rot.z, rot.w], dtype=np.float64),
            'pose_stamp': pose_stamp,
        }
    
    def _generic_callback(self, name: str, msg):
        """
        通用消息回调：只缓存最新一帧消息。
        适用于joint_states、图像等普通topic。
        """
        now = time.time()
        with self._cache_lock:
            self._latest_msgs[name] = msg
            self._latest_stamp[name] = now
        logger.debug(f"已缓存 {name} 的最新消息，时间戳: {now}")

    def get_joint_state(self ,topic_name: str = "joint_states") -> Dict[str, Any] | None:
        """
        获取关节状态，返回字典。topic_name默认为joint_states。
        """
        with self._cache_lock:
            msg = self._latest_msgs.get(topic_name)
        if msg is None:
            logger.warning(f"未获取到 {topic_name} 的关节状态消息")
            return None
        joint_state_dict={}
        for i,name in enumerate(msg.name):
            joint_state_dict[f"{name}.pos"]=msg.position[i]
            joint_state_dict[f"{name}.vel"]=msg.velocity[i]
            joint_state_dict[f"{name}.eff"]=msg.effort[i]
        logger.debug(f"已获取关节状态: {joint_state_dict}")
        return joint_state_dict

    def get_rgbd_data(self, topic_name: str = "camera_rgbd") -> Dict[str, Any] | None:
        """
        获取RGBD数据，包括RGB和深度图像以及相机信息
        """
        with self._cache_lock:
            msg = self._latest_msgs.get(topic_name)
        if msg is None:
            return None
        rgbd_data = {
            "header": {
                "stamp": {
                    "sec": msg.header.stamp.sec,
                    "nanosec": msg.header.stamp.nanosec,
                },
                "frame_id": msg.header.frame_id,
            },
            "rgb_camera_info": {
                "header": {
                    "stamp": {
                        "sec": msg.rgb_camera_info.header.stamp.sec,
                        "nanosec": msg.rgb_camera_info.header.stamp.nanosec,
                    },
                    "frame_id": msg.rgb_camera_info.header.frame_id,
                },
                "height": msg.rgb_camera_info.height,
                "width": msg.rgb_camera_info.width,
                "distortion_model": msg.rgb_camera_info.distortion_model,
                "d": list(msg.rgb_camera_info.d),
                "k": list(msg.rgb_camera_info.k),
                "r": list(msg.rgb_camera_info.r),
                "p": list(msg.rgb_camera_info.p),
                "binning_x": msg.rgb_camera_info.binning_x,
                "binning_y": msg.rgb_camera_info.binning_y,
                "roi": {
                    "x_offset": msg.rgb_camera_info.roi.x_offset,
                    "y_offset": msg.rgb_camera_info.roi.y_offset,
                    "height": msg.rgb_camera_info.roi.height,
                    "width": msg.rgb_camera_info.roi.width,
                    "do_rectify": msg.rgb_camera_info.roi.do_rectify,
                },
            },
            "depth_camera_info": {
                "header": {
                    "stamp": {
                        "sec": msg.depth_camera_info.header.stamp.sec,
                        "nanosec": msg.depth_camera_info.header.stamp.nanosec,
                    },
                    "frame_id": msg.depth_camera_info.header.frame_id,
                },
                "height": msg.depth_camera_info.height,
                "width": msg.depth_camera_info.width,
                "distortion_model": msg.depth_camera_info.distortion_model,
                "d": list(msg.depth_camera_info.d),
                "k": list(msg.depth_camera_info.k),
                "r": list(msg.depth_camera_info.r),
                "p": list(msg.depth_camera_info.p),
                "binning_x": msg.depth_camera_info.binning_x,
                "binning_y": msg.depth_camera_info.binning_y,
                "roi": {
                    "x_offset": msg.depth_camera_info.roi.x_offset,
                    "y_offset": msg.depth_camera_info.roi.y_offset,
                    "height": msg.depth_camera_info.roi.height,
                    "width": msg.depth_camera_info.roi.width,
                    "do_rectify": msg.depth_camera_info.roi.do_rectify,
                },
            },
            "rgb_image": {
                "header": {
                    "stamp": {
                        "sec": msg.rgb.header.stamp.sec,
                        "nanosec": msg.rgb.header.stamp.nanosec,
                    },
                    "frame_id": msg.rgb.header.frame_id,
                },
                "height": msg.rgb.height,
                "width": msg.rgb.width,
                "encoding": msg.rgb.encoding,
                "is_bigendian": msg.rgb.is_bigendian,
                "step": msg.rgb.step,
                "data": np.frombuffer(msg.rgb.data, dtype=np.uint8).reshape((msg.rgb.height, msg.rgb.width, 3)) if msg.rgb.encoding == "rgb8" else msg.rgb.data,
            },
            "depth_image": {
                "header": {
                    "stamp": {
                        "sec": msg.depth.header.stamp.sec,
                        "nanosec": msg.depth.header.stamp.nanosec,
                    },
                    "frame_id": msg.depth.header.frame_id,
                },
                "height": msg.depth.height,
                "width": msg.depth.width,
                "encoding": msg.depth.encoding,
                "is_bigendian": msg.depth.is_bigendian,
                "step": msg.depth.step,
                "data": np.frombuffer(msg.depth.data, dtype=np.uint16).reshape((msg.depth.height, msg.depth.width)) if msg.depth.encoding == "16UC1" else msg.depth.data,
            },
        }
        return rgbd_data

    def get_image(self, topic_name: str = "rgb_image") -> Dict[str, Any] | None:
        """
        获取图像数据,默认订阅"rgb_image"主题, 也可"depth_image"
        适用于sensor_msgs/msg/Image类型的消息
        """
        with self._cache_lock:
            msg = self._latest_msgs.get(topic_name)
        if msg is None:
            return None
        
        image_data = {
            "header": {
                "stamp": {
                    "sec": msg.header.stamp.sec,
                    "nanosec": msg.header.stamp.nanosec,
                },
                "frame_id": msg.header.frame_id,
            },
            "height": msg.height,
            "width": msg.width,
            "encoding": msg.encoding,
            "is_bigendian": msg.is_bigendian,
            "step": msg.step,
            "data": np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3)) if msg.encoding == "rgb8" else (
                np.frombuffer(msg.data, dtype=np.uint16).reshape((msg.height, msg.width)) if msg.encoding == "16UC1" else msg.data
            ),
        }
        return image_data
    
    def send_joint_commands(self, commands: Dict[str, float]) -> None:
        """
        发送关节命令，接收到的是 字典，joint1.pos,jonit2.pos，以及对应的值
        要根据joints={
        "left_forward_position_controller":["openarm_left_joint1","openarm_left_joint2","openarm_left_joint3","openarm_left_joint4","openarm_left_joint5","openarm_left_joint6","openarm_left_joint7",],
        "right_forward_position_controller":["openarm_right_joint1","openarm_right_joint2","openarm_right_joint3","openarm_right_joint4","openarm_right_joint5","openarm_right_joint6","openarm_right_joint7",]
        }
        发送命令 要按照对应的顺序组成
        /left_forward_position_controller/commands std_msgs/msg/Float64MultiArray "{data: [0.0, -0.5, 0.3, 0.0, 0.5, -0.2, 0.0]}"
        """
        for controller_name, joint_list in self._config.joints.items():
        # 按顺序取值
            data = []
            for joint_name in joint_list:
                key = f"{joint_name}.pos"
                if key not in commands:
                    raise KeyError(f"{key} 未在commands字典中找到")
                data.append(commands[key])
            # 生成消息
            msg = Float64MultiArray()
            msg.data = data
            logger.info(f"发送到 {controller_name} 的关节命令: {msg.data}")
            # 发布
            self._publishers[controller_name].publish(msg)

    def publish_message(self, topic_name: str, msg: Any) -> None:
        """
        通过 topics_to_publish 中配置的 topic_name 发布任意消息。
        例如：publish_message("target_pose", PoseStamped())
        """
        if topic_name not in self._publishers:
            raise KeyError(f"{topic_name} 未在 topics_to_publish 中配置")
        self._publishers[topic_name].publish(msg)
            
    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} 未连接。")

        self._connected = False
        self._tf_listener = None
        self._tf_buffer = None
        # Stop executor
        if self._executor:
            self._executor.shutdown()
            self._executor = None
        
        # Wait for thread to finish
        if self._executor_thread:
            self._executor_thread.join(timeout=2.0)
            self._executor_thread = None    
        
        # Destroy subscriptions and publishers
        for sub in self._subscriptions:
            self._robot_node.destroy_subscription(sub)
        self._subscriptions.clear()
        
        # Destroy node
        if self._robot_node:
            self._robot_node.destroy_node()
            self._robot_node = None

        # Shutdown rclpy
        try:
            rclpy.shutdown()
        except Exception as e:
            logger.warning(f"rclpy关闭时出错: {e}")
        
        logger.info("已断开与ROS 2机器人的连接")
