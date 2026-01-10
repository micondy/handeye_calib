from dataclasses import dataclass, field
from lerobot.cameras import CameraConfig
from lerobot.robots import RobotConfig


@RobotConfig.register_subclass("lerobot_ros2_openarm")
@dataclass
class Ros2RobotConfig(RobotConfig):
    """
    """

    robot_name: str   = field(default="lerobot_ros2_openarm")# 会创建一个 lerobot_ros2_openarm_node 作为节点名称
    
    namespace: str    = field(default="")  

    joints: dict[str,list] = field(default_factory=dict)
    """ 
    左右臂关节列表
    "left_forward_position_controller": ["joint1","joint2"...] 
    """
    topics_to_subscribe: dict[str, dict] = field(default_factory=dict)
    """
    要订阅的主题列表 
            "joint_states": {
                "topic": "/joint_states",
                "type": "sensor_msgs/msg/JointState",
            },
            "camera_rgbd": {
                "topic": "/camera/camera/rgbd",
                "type": "realsense2_camera_msgs/msg/RGBD",
            },
            "rgb_image": {
                "topic": "/camera/camera/color/image_raw",
                "type": "sensor_msgs/msg/Image",
            },
            "depth_image": {
                "topic": "/camera/camera/aligned_depth_to_color/image_raw",
                "type": "sensor_msgs/msg/Image",
            }

    """
    topics_to_publish: dict[str, dict] = field(default_factory=dict)
    """
    要发布的主题列表
            "left_forward_position_controller": {
                "topic": "/left_forward_position_controller/commands",
                "type": "std_msgs/msg/Float64MultiArray",
            },
    """

    cameras: dict[str, CameraConfig] = field(default_factory=dict)

