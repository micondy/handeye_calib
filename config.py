from dataclasses import dataclass, field
from lerobot.cameras import CameraConfig
from lerobot.robots import RobotConfig


@RobotConfig.register_subclass("lerobot_ros2_openarm")
@dataclass
class ROS2RobotConfig(RobotConfig):
    """

    """
    robot_name: str   = field(default="lerobot_ros2_openarm")
    
    namespace: str    = field(default="")  

    joints: dict[str,list] = field(default_factory=dict)
    
    topics_to_subscribe: dict[str, dict] = field(default_factory=dict)

    topics_to_publish: dict[str, dict] = field(default_factory=dict)


    cameras: dict[str, CameraConfig] = field(default_factory=dict)

