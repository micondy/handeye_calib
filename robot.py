#!/usr/bin/env python

from functools import cached_property
import logging
import time
from typing import Any, Dict

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.utils.errors import DeviceNotConnectedError, DeviceAlreadyConnectedError
from lerobot.robots import Robot
from lerobot.utils import ensure_safe_goal_position

from .ros_interface import ROS2RobotInterface
from .config import ROS2RobotConfig

logger = logging.getLogger(__name__)


class ROS2Robot(Robot):

    name = "ros2_robot"
    config_class = ROS2RobotConfig

    def __init__(self, config: ROS2RobotConfig):
        super().__init__(config)
        self.config = config

        self._ros2_interface = ROS2RobotInterface(config)

        # flatten joint names from controllers -> list[str]
        self._joints = []
        for lst in (config.joints or {}).values():
            self._joints.extend(lst)

        # setup cameras
        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def _joints_ft(self) -> Dict[str, type]:
        return {f"{j}.pos": float for j in self._joints}

    @property
    def _cameras_ft(self) -> Dict[str, tuple]:
        return {cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras}

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        features: dict[str, type | tuple] = {}
        for j in self._joints:
            features[f"{j}.pos"] = float
            features[f"{j}.vel"] = float
            features[f"{j}.eff"] = float
        features.update(self._cameras_ft)
        return features

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._joints_ft

    @property
    def is_connected(self) -> bool:
        #return self._ros2_interface.is_connected and all(cam.is_connected for cam in self.cameras.values())
        return self._ros2_interface.is_connected
    
    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        # connect ROS2 interface
        self._ros2_interface.connect()

        # # connect cameras
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

        # get joint state from ros interface (returns keys like 'joint.pos')
        joint_state = self._ros2_interface.get_joint_state()
        if joint_state:
            obs.update(joint_state)

        # capture camera frames
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # extract goal positions: keys like 'joint.pos' -> {'joint': val}
        goal_pos: Dict[str, float] = {}
        for k, v in action.items():
            if k.endswith(".pos"):
                joint = k[:-4]
                goal_pos[joint] = float(v)

        # Optionally clip relative movement
        if self.config.max_relative_target is not None and goal_pos:
            # get present positions as a mapping joint->pos
            present_state = self._ros2_interface.get_joint_state() or {}
            present_pos = {k[:-4]: v for k, v in present_state.items() if k.endswith(".pos")}
            goal_present = {j: (g, present_pos.get(j, 0.0)) for j, g in goal_pos.items()}
            goal_pos = ensure_safe_goal_position(goal_present, self.config.max_relative_target)

        # Build command dict expected by ROS2RobotInterface: keys are 'joint.pos'
        commands = {f"{j}.pos": p for j, p in goal_pos.items()}
        self._ros2_interface.send_joint_commands(commands)

        # return the action actually sent (keys with suffix)
        return {f"{j}.pos": p for j, p in goal_pos.items()}

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # disconnect ROS interface
        self._ros2_interface.disconnect()

        # disconnect cameras
        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")
