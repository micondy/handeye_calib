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

from .config import ROS2RobotConfig

# ros_interface.py
from sensor_msgs.msg import JointState
from control_msgs.msg import JointTrajectoryControllerState

#对照表，订阅要消息类型
MSG_TYPE_MAP = {
    "sensor_msgs/JointState": JointState,
    "control_msgs/msg/JointTrajectoryControllerState" :JointTrajectoryControllerState,
}

from threading import Lock
@dataclass(frozen=True)
class JointStateSnapshot:
    position: Dict[str, float]
    velocity: Optional[Dict[str, float]]
    effort: Optional[Dict[str, float]]
    stamp: float  # local time.time()

logger = logging.getLogger(__name__)

class ROS2RobotInterface:
    
    def __init__(self, config: ROS2RobotConfig):
        """
        
        """
        self._config = config
        self._connected: bool = False
        self._subscriptions = []
        self._publishers = []

        self._executor: SingleThreadedExecutor | None = None
        self._executor_thread: threading.Thread | None = None
        self._robot_node: Node | None = None

        self._cache_lock = Lock()
        # topic_name -> latest snapshot
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
                self._subscriptions.append(sub)
                logger.info(f"Subscribed to topic: {topic_name} with type {cfg['type']}")
            
            for name, cfg in self._config.topics_to_publish.items():
                msg_type = MSG_TYPE_MAP[cfg["type"]]
                topic_name = cfg["topic"]
                pub = self._robot_node.create_publisher(
                    msg_type,
                    topic_name,
                    10
                )
                self._publishers.append(pub)
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
        now = time.time()
        with self._cache_lock:
            self._latest_msgs[name] = msg
            self._latest_stamp[name] = now
        pass

    def get_joint_state(self) -> Dict[str, Any] | None:
        with self._cache_lock:
            msg = self._latest_msgs.get("joint_state")

        if msg is None:
            return None

        assert isinstance(msg, JointState)

        return {
            "position": dict(zip(msg.name, msg.position)),
            "velocity": dict(zip(msg.name, msg.velocity)) if msg.velocity else None,
            "effort": dict(zip(msg.name, msg.effort)) if msg.effort else None,
        }
    def send_joint_commands(self, commands: Dict[str, float]) -> None:
        """
        发送关节命令
        """
        pass
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
