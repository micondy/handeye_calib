from .config import ROS2RobotConfig
from .ros_interface import ROS2RobotInterface
import time
import logging

logging.basicConfig(level=logging.INFO)

def main():
    config = ROS2RobotConfig(
        robot_name="my_robot",
        joints=["joint1", "joint2", "joint3"],
        topics_to_subscribe={
            "joint_states": {
                "topic": "/joint_states",
                "type": "sensor_msgs/JointState",
            },
            "left_controller_state": {
                "topic": "/left_joint_trajectory_controller/controller_state",
                "type": "control_msgs/msg/JointTrajectoryControllerState",
            },
            "right_controller_state": {
                "topic": "/right_joint_trajectory_controller/controller_state",
                "type": "control_msgs/msg/JointTrajectoryControllerState",
            },
        }
    )
    ros2_robot_interface = ROS2RobotInterface(config)
    ros2_robot_interface.connect()
        # 阻塞，保持节点运行

        
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        ros2_robot_interface.disconnect()

if __name__ == "__main__":
    main()
