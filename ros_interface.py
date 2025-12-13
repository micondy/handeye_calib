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

logger = logging.getLogger(__name__)

class ROS2RobotInterface:
    
    def __init__(self, config: ROS2RobotConfig):
        """
        
        """
        self.config = config

        self._connected: bool = False
        self._subscriptions = []
        self._publishers = []
    
    @property
    def is_connected(self) -> bool:
        return self._connected and self.robot_node is not None
    
    def connect(self) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")
        try:
            if not rclpy.ok():
                rclpy.init()
            
            self.robot_node = Node(
                node_name=self.config.robot_name+"_node",
            )
             
            for name, cfg in self.config.topics_to_subscribe.items():
                msg_type = MSG_TYPE_MAP[cfg["type"]]
                topic_name = cfg["topic"]
                sub = self.robot_node.create_subscription(
                    msg_type,
                    topic_name,
                    lambda msg, n=name: self._generic_callback(n, msg),
                    10
                )
                self._subscriptions.append(sub)
                logger.info(f"Subscribed to topic: {topic_name} with type {cfg['type']}")
            # executor and thread,线程化
            self.executor = SingleThreadedExecutor()
            self.executor.add_node(self.robot_node)
            self.executor_thread = threading.Thread(
                target=self.executor.spin,
                daemon=True
            )
            self.executor_thread.start()
            

            time.sleep(1.0)            
            self._connected = True
        except Exception as e:
            logger.error(f"Failed to connect to ROS 2 robot interface: {e}")
            self.disconnect()
            raise
    
    def _generic_callback(self, name: str, msg):
        #print(f"[{name}] {msg}")
        pass

    def get_joint_state(self) -> Dict[str, Any] | None:
        pass
    
    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")


        self._connected = False
        # Stop executor
        if self.executor:
            self.executor.shutdown()
            self.executor = None
        
        # Wait for thread to finish
        if self.executor_thread:
            self.executor_thread.join(timeout=2.0)
            self.executor_thread = None
        
        # Destroy subscriptions and publishers
        for sub in self._subscriptions:
            self.robot_node.destroy_subscription(sub)
        self._subscriptions.clear()
        
        # Destroy node
        if self.robot_node:
            self.robot_node.destroy_node()
            self.robot_node = None

        # Shutdown rclpy
        try:
            rclpy.shutdown()
        except Exception as e:
            logger.warning(f"Error during rclpy shutdown: {e}")
        
        logger.info("Disconnected from ROS 2 robot interface")
