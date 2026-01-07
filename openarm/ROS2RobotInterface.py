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
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.publisher import Publisher
from rclpy.subscription import Subscription

from .ROS2RobotConfig import ROS2RobotConfig

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
}

from threading import Lock

logger = logging.getLogger(__name__)

class ROS2RobotInterface:
    
    def __init__(self, config: ROS2RobotConfig):
        """
        
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
        # 回调函数缓存的最新消息和时间戳 str是 话题名 存储msg和消息时间
        self._latest_msgs: Dict[str, Any] = {}
        self._latest_stamp: Dict[str, float] = {}

    @property
    def is_connected(self) -> bool:
        return self._connected and self._robot_node is not None
    
    def connect(self) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")
        try:
            if not rclpy.ok():
                rclpy.init()
            
            self._robot_node = Node(
                node_name=self._config.robot_name+"_node",
            )
             
            for name, cfg in self._config.topics_to_subscribe.items():      
                msg_type = MSG_TYPE_MAP[cfg["type"]]
                topic_name = cfg["topic"]
                sub = self._robot_node.create_subscription(
                    msg_type,
                    topic_name,
                    lambda msg, n=name: self._generic_callback(n, msg),
                    10
                )
                self._subscriptions[name]=sub
                logger.info(f"Subscribed to topic: {topic_name} with type {cfg['type']}")
            
            for name, cfg in self._config.topics_to_publish.items():
                msg_type = MSG_TYPE_MAP[cfg["type"]]
                topic_name = cfg["topic"]
                pub = self._robot_node.create_publisher(
                    msg_type,
                    topic_name,
                    10
                )
                self._publishers[name] = pub
                logger.info(f"Published to topic: {topic_name} with type {cfg['type']}")
            
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
            logger.error(f"Failed to connect to ROS 2 robot interface: {e}")
            self.disconnect()
            raise
    
    def _generic_callback(self, name: str, msg):
        """
        根据话题名称缓存最新的消息和时间戳
        """
        now = time.time()
        with self._cache_lock:
            self._latest_msgs[name] = msg
            self._latest_stamp[name] = now
            #print(self._latest_msgs[name])

    def get_joint_state(self ,topic_name: str = "joint_states") -> Dict[str, Any] | None:
        with self._cache_lock:
            msg = self._latest_msgs.get(topic_name)
        #print(msg)
        if msg is None:
            return None
        joint_state_dict={}
        for i,name in enumerate(msg.name):
            joint_state_dict[f"{name}.pos"]=msg.position[i]
            joint_state_dict[f"{name}.vel"]=msg.velocity[i]
            joint_state_dict[f"{name}.eff"]=msg.effort[i]
 
        return joint_state_dict

    def get_rgbd_data(self, topic_name: str = "camera_rgbd") -> Dict[str, Any] | None:
        """
        获取RGBD数据，包括RGB和深度图像以及相机信息
        """
        print("Getting RGBD data...")
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

    def get_camera_info(self, topic_name: str = "camera_rgbd") -> dict[str, tuple] | None:
        """
        返回RGB 和 Depth参数 (height, width, channels)

        """
        rgbd = self.get_rgbd_data(topic_name)
        if rgbd is None:
            return None

        # RGB
        rgb_height = rgbd["rgb_image"]["height"]
        rgb_width = rgbd["rgb_image"]["width"]
        rgb_channels = 3

        # Depth
        depth_height = rgbd["depth_image"]["height"]
        depth_width = rgbd["depth_image"]["width"]
        depth_channels = 1  # 单通道深度图

        return {
            f"rgb_image": (rgb_height, rgb_width, rgb_channels),
            f"depth_image": (depth_height, depth_width, depth_channels)
        }



    def get_image(self, topic_name: str = "image") -> Dict[str, Any] | None:
        """
        获取图像数据
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

    def get_rgb_image(self, topic_name: str = "rgb_image") -> Dict[str, Any] | None:
        """
        获取RGB图像数据
        """
        return self.get_image(topic_name)

    def get_depth_image(self, topic_name: str = "depth_image") -> Dict[str, Any] | None:
        """
        获取深度图像数据
        """
        return self.get_image(topic_name)


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
                    raise KeyError(f"{key} not found in commands dict")
                data.append(commands[key])
            # 生成消息
            msg = Float64MultiArray()
            msg.data = data
            print(f"Sending to {controller_name}: {msg.data}")
            # 发布
            self._publishers[controller_name].publish(msg)
            
    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self._connected = False
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
            logger.warning(f"Error during rclpy shutdown: {e}")
        
        logger.info("Disconnected from ROS 2 robot interface")
