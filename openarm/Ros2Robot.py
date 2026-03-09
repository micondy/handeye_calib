#!/usr/bin/env python

from functools import cached_property
import logging
import time
from typing import Any, Dict

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.utils.errors import DeviceNotConnectedError, DeviceAlreadyConnectedError
from lerobot.robots import Robot
from lerobot.robots.utils import ensure_safe_goal_position

from .Ros2RobotInterface import Ros2RobotInterface
from .Ros2RobotConfig import Ros2RobotConfig

logger = logging.getLogger(__name__)


class Ros2Robot(Robot):

    name = "ros2_robot"
    config_class = Ros2RobotConfig

    def __init__(self, config: Ros2RobotConfig):
        super().__init__(config)
        self.config = config

        self._ros2_interface = Ros2RobotInterface(config)

        # 根据config设置关节
        self._joints = []
        for lst in (config.joints or {}).values():
            self._joints.extend(lst)

        # 设置摄像头,其实是没有摄像头的，从ros2接口获取图像，rgbd只是个名字

    @property
    def _joints_ft(self) -> Dict[str, type]:
        return {f"{j}.pos": float for j in self._joints}

    @property
    def _cameras_ft(self) -> dict[str, dict]:
        """
        Camera feature definitions for LeRobot dataset / pipeline.
        """
        info: dict[str, dict] = {}

        # RGB image
        info["rgb_image"] = {
            "dtype": "video",          
            "shape": [480, 640, 3],    # HWC
            "fps": 30,
        }

        # Depth image
        info["depth_image"] = {
            "dtype": "video",         
            "shape": [480, 640, 1],    # 单通道
            "fps": 30,
        }

        return info



    @cached_property
    def observation_features(self) -> dict[str, dict]:
        """
        返回完整的观测特征字典，包含关节状态和摄像头信息
        """
        features: dict[str, dict] = {}
        # 关节状态
        for j in self._joints:
            features[f"{j}.pos"] = {"dtype": "float", "shape": (1,)}
            features[f"{j}.vel"] = {"dtype": "float", "shape": (1,)}
            features[f"{j}.eff"] = {"dtype": "float", "shape": (1,)}
        # 摄像头
        features.update(self._cameras_ft)
        return features

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._joints_ft

    @property
    def is_connected(self) -> bool:
        #return self._ros2_interface.is_connected and all(cam.is_connected for cam in self.cameras.values()) # 暂时只检查ros2接口，无摄像头
        return self._ros2_interface.is_connected
    
    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

       
        self._ros2_interface.connect()

        # # connect cameras 暂时无摄像头
        # for cam in self.cameras.values():
        #     cam.connect()

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
        obs["rgb_image"] =self._ros2_interface.get_rgb_image()["data"]
        obs["depth_image"] =self._ros2_interface.get_depth_image()["data"]
        # 获取摄像头图像，暂无
        # for cam_key, cam in self.cameras.items():
        #     start = time.perf_counter()
        #     obs[cam_key] = cam.async_read()
        #     dt_ms = (time.perf_counter() - start) * 1e3
        #     logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # 'joint.pos' -> {'joint': val} 
        goal_pos: Dict[str, float] = {}
        for k, v in action.items():
            if k.endswith(".pos"):
                joint = k[:-4]
                goal_pos[joint] = float(v)

        # # Optionally clip relative movement
        # if self.config.max_relative_target is not None and goal_pos:
        #     # get present positions as a mapping joint->pos
        #     present_state = self._ros2_interface.get_joint_state() or {}
        #     present_pos = {k[:-4]: v for k, v in present_state.items() if k.endswith(".pos")}
        #     goal_present = {j: (g, present_pos.get(j, 0.0)) for j, g in goal_pos.items()}
        #     goal_pos = ensure_safe_goal_position(goal_present, self.config.max_relative_target)

        # Build command dict expected by Ros2RobotInterface: keys are 'joint.pos'
        commands = {f"{j}.pos": p for j, p in goal_pos.items()}
        self._ros2_interface.send_joint_commands(commands)

        # return the action actually sent (keys with suffix)
        return {f"{j}.pos": p for j, p in goal_pos.items()}

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self._ros2_interface.disconnect()

        # # disconnect cameras
        # for cam in self.cameras.values():
        #     cam.disconnect()

        logger.info(f"{self} disconnected.")
