"""
ROS 2 Robot Implementation for LeRobot

This module provides a ROS 2 robot implementation that integrates with LeRobot,
supporting joint state monitoring and end-effector control through ROS 2 topics.
"""

import logging
import time
from functools import cached_property
from typing import Any

import numpy as np
from geometry_msgs.msg import Pose
# Import will be done dynamically to use the patched version
from lerobot.robots import Robot
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from .config import  ROS2RobotConfig
from .ros_interface import ROS2RobotInterface

logger = logging.getLogger(__name__)


class ROS2Robot(Robot):
    """ROS 2 robot implementation for LeRobot.
    
    This robot class interfaces with ROS 2 robots through topics:
    - Subscribes to /joint_states for joint state information
    - Subscribes to /left_current_pose for current end-effector pose
    - Publishes to /left_target for end-effector target commands
    
    Example:
        ```python
        from lerobot_ros2.robots import ROS2Robot, ROS2RobotConfig
        
        # Create configuration
        config = ROS2RobotConfig(
            id="my_robot",
            ros2_interface=ROS2RobotInterfaceConfig(
                joint_states_topic="/joint_states",
                end_effector_pose_topic="/left_current_pose",
                end_effector_target_topic="/left_target",
                control_type=ControlType.CARTESIAN_POSE
            )
        )
        
        # Create and connect robot
        robot = ROS2Robot(config)
        robot.connect()
        
        # Get observation
        obs = robot.get_observation()
        
        # Send action
        action = {"end_effector_pose": {...}}
        robot.send_action(action)
        
        # Disconnect when done
        robot.disconnect()
        ```
    """
    
    config_class = ROS2RobotConfig
    name = "ros2_robot"
    
    def __init__(self, config: ROS2RobotConfig ):

        super().__init__(config)
        self._config = config
        self._ros2_interface = ROS2RobotInterface(config)
    
    
    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        """Get the observation features for this robot.
        
        Returns:
            Dictionary mapping observation keys to their types.
        """
        features = {}
        
        # Joint positions
        for name ,joint_lists in self._config.joints.items():
            for joint_name in joint_lists:
                features[f"{joint_name}.pos"] = float
        
        # Joint velocities
        for name ,joint_lists in self._config.joints.items():
            for joint_name in joint_lists:
                features[f"{joint_name}.vel"] = float
        
        # Joint efforts (torques)
        for name ,joint_lists in self._config.joints.items():
            for joint_name in joint_lists:
                features[f"{joint_name}.effort"] = float 
        # # Camera features
        # for cam_name, cam_config in self.config.cameras.items():
        #     if cam_config.width and cam_config.height:
        #         features[cam_name] = (cam_config.height, cam_config.width, 3)
        #     else:
        #         features[cam_name] = (720, 1280, 3)  # Default resolution
        
        return features
    
    @cached_property
    def action_features(self) -> dict[str, type]:
        return self.observation_features
    
    @property
    def is_connected(self) -> bool:
        """Check if the robot is connected.
        
        Returns:
            True if robot and cameras are connected, False otherwise.
        """
        return self._ros2_interface.is_connected()
    
    def connect(self, calibrate: bool = True) -> None:
        """Connect to the robot.
        
        Args:
            calibrate: Whether to perform calibration (ignored for ROS 2 robots)
        """
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self.ros2_interface.connect()

    
    @property
    def is_calibrated(self) -> bool:
        pass
    
    def calibrate(self) -> None:
        pass
    
    def configure(self) -> None:
        pass
    
    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")
        
        obs_dict: dict[str, Any] = {}
        
        # Get joint states
        joint_state = self._ros2_interface.get_joint_state()
        


        joint_names = joint_state["names"]
        joint_positions = joint_state["positions"]
        joint_velocities = joint_state["velocities"]
        joint_efforts = joint_state["efforts"]
            
        for joint_name in self.config.ros2_interface.joint_names:
            try:
                idx = joint_names.index(joint_name)
                obs_dict[f"{joint_name}.pos"] = joint_positions[idx]
                obs_dict[f"{joint_name}.vel"] = joint_velocities[idx]
                obs_dict[f"{joint_name}.effort"] = joint_efforts[idx]
            except ValueError:
                logger.warning(f"Joint '{joint_name}' not found in joint state")
                obs_dict[f"{joint_name}.pos"] = 0.0
                obs_dict[f"{joint_name}.vel"] = 0.0
                obs_dict[f"{joint_name}.effort"] = 0.0
        return obs_dict
    
    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Send an action to the robot.
        
        Args:
            action: Dictionary containing the action to send
            
        Returns:
            The action that was actually sent to the robot
            
        Raises:
            DeviceNotConnectedError: If robot is not connected
            ValueError: If action format is invalid
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")
        
        # Create pose from action
        pose = Pose()
        pose.position.x = action.get("end_effector.position.x", 0.0)
        pose.position.y = action.get("end_effector.position.y", 0.0)
        pose.position.z = action.get("end_effector.position.z", 0.0)
        pose.orientation.x = action.get("end_effector.orientation.x", 0.0)
        pose.orientation.y = action.get("end_effector.orientation.y", 0.0)
        pose.orientation.z = action.get("end_effector.orientation.z", 0.0)
        pose.orientation.w = action.get("end_effector.orientation.w", 1.0)
        
        # Send the target pose
        self.ros2_interface.send_end_effector_target(pose)
        
        # Send gripper command if provided and gripper is enabled
        if self.config.ros2_interface.gripper_enabled and "gripper.position" in action:
            gripper_position = action["gripper.position"]
            self.ros2_interface.send_gripper_command(gripper_position)
            logger.debug(f"Sent gripper command: {gripper_position}")
        
        return action
    
    def disconnect(self) -> None:
        """Disconnect from the robot and cleanup resources."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")
        
        # Disconnect cameras
        for cam in self.cameras.values():
            cam.disconnect()
        
        # Disconnect ROS 2 interface
        self.ros2_interface.disconnect()
        
        logger.info(f"Disconnected {self}")
