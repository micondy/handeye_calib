from .config import ROS2RobotConfig
from .ros_interface import ROS2RobotInterface
import time
import logging
import copy
import random
logging.basicConfig(level=logging.INFO)

def main():
    config = ROS2RobotConfig(
        robot_name="my_robot",
        joints={
        "left_forward_position_controller": ["openarm_left_joint1","openarm_left_joint2","openarm_left_joint3","openarm_left_joint4","openarm_left_joint5","openarm_left_joint6","openarm_left_joint7",],
        "right_forward_position_controller":["openarm_right_joint1","openarm_right_joint2","openarm_right_joint3","openarm_right_joint4","openarm_right_joint5","openarm_right_joint6","openarm_right_joint7",]
        },
        topics_to_subscribe={
            "joint_states": {
                "topic": "/joint_states",
                "type": "sensor_msgs/msg/JointState",
            },
        },
        topics_to_publish={
            "left_forward_position_controller": {
                "topic": "/left_forward_position_controller/commands",
                "type": "std_msgs/msg/Float64MultiArray",
            },
            "right_forward_position_controller": {
                "topic": "/right_forward_position_controller/commands",
                "type": "std_msgs/msg/Float64MultiArray",
            },
        }
    )
    ros2_robot_interface = ROS2RobotInterface(config)
    ros2_robot_interface.connect()
        # 阻塞，保持节点运行
  
    try:
        while True:
            time.sleep(0.1)
            # 获取当前关节状态
            dict1 = ros2_robot_interface.get_joint_state()
            if dict1 is None:
                continue
            # 生成新的命令字典（随机扰动 ±0.1）
            cmd_dict = copy.deepcopy(dict1)
            for joint_key, val in cmd_dict.items():
                if joint_key.endswith(".pos"):
                    cmd_dict[joint_key] = val + random.uniform(-0.1, 0.1)

            # 发送命令
            ros2_robot_interface.send_joint_commands(cmd_dict)

            
    except KeyboardInterrupt:
        ros2_robot_interface.disconnect()

if __name__ == "__main__":
    main()
