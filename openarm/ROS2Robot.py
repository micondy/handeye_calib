#!/usr/bin/env python

from functools import cached_property
import logging
import time
from typing import Any, Dict

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.utils.errors import DeviceNotConnectedError, DeviceAlreadyConnectedError
from lerobot.robots import Robot
from lerobot.robots.utils import ensure_safe_goal_position

from .ROS2RobotInterface import ROS2RobotInterface
from .Ros2RobotConfig import Ros2RobotConfig

logger = logging.getLogger(__name__)


class ROS2Robot(Robot):
    
    config_class = Ros2RobotConfig
    name = "ros2_robot"


    def __init__(self, config: Ros2RobotConfig):
        super().__init__(config)
        self._config = config

        self._ros2_interface = ROS2RobotInterface(config)

        # 根据config设置关节
        self._joints = []
        for lst in (config.joints or {}).values():
            self._joints.extend(lst)

        # 设置摄像头,其实是没有摄像头的，从ros2接口获取图像，
        self._cameras = config.cameras

    @property
    def _joints_ft(self) -> Dict[str, type]:
        return {f"{j}.pos": float for j in self._joints}

    @property
    def _cameras_ft(self) -> dict[str, dict]:
        #返回 高 宽 通道数
        return { cam: (self._cameras[cam].height, self._cameras[cam].width, self._cameras[cam].channels) for cam in self._cameras        
    } 

    @property
    def observation_features(self) -> dict[str, dict]:
        """
        返回完整的观测特征字典，包含关节状态和摄像头信息
        """
        return {**self._joints_ft, **self._cameras_ft}

    @property
    def action_features(self) -> dict[str, type]:
        return self._joints_ft

    @property
    def is_connected(self) -> bool:
        # 连接状态由ros2接口决定
        return self._ros2_interface.is_connected
    
    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self._ros2_interface.connect()
        self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return False

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        return None

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        obs: dict[str, Any] = {}

        # 获取关节状态
        joint_state = self._ros2_interface.get_joint_state()
        if joint_state:
            obs.update(joint_state)

        start = time.perf_counter()
        obs["rgb_image"] = self._ros2_interface.get_rgbd_data()["rgb_image"]["data"] #读取data字段
        obs["depth_image"] = self._ros2_interface.get_rgbd_data()["depth_image"]["data"]
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read rgb_image: {dt_ms:.1f}ms")

        return obs

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        goal_pos = {key.removesuffix(".pos"): val for key, val in action.items()}

        self._ros2_interface.send_joint_commands(goal_pos)
        return goal_pos

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self._ros2_interface.disconnect()

        logger.info(f"{self} disconnected.")
