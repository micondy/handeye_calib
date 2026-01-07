from .ROS2RobotConfig import ROS2RobotConfig
from .ROS2RobotInterface import ROS2RobotInterface
import time
import logging
import copy
import random
logging.basicConfig(level=logging.INFO)

import cv2
import numpy as np
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
            "camera_rgbd":{
                "topic":"/camera/camera/rgbd",
                "type": "realsense2_camera_msgs/msg/RGBD",
            },
            "rgb_image":{
                "topic":"/camera/camera/color/image_raw",
                "type": "sensor_msgs/msg/Image",
            },
            "depth_image":{
                "topic":"/camera/camera/aligned_depth_to_color/image_raw",
                "type": "sensor_msgs/msg/Image",
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
            # # 获取当前关节状态
            # dict1 = ros2_robot_interface.get_joint_state()
            # #print(dict1)
            # if dict1 is None:
            #     continue
            # # 生成新的命令字典（随机扰动 ±0.1）
            # cmd_dict = copy.deepcopy(dict1)
            # for joint_key, val in cmd_dict.items():
            #     if joint_key.endswith(".pos"):
            #         cmd_dict[joint_key] = val + random.uniform(-0.1, 0.1)
            # #print(cmd_dict)
            # # 发送命令
            # ros2_robot_interface.send_joint_commands(cmd_dict)
                # dict_rgb = ros2_robot_interface.get_rgb_image()
                # if dict_rgb is None:
                #     continue

                # dict_depth = ros2_robot_interface.get_depth_image()
                # if dict_depth is None:
                #     continue
            rgbd = ros2_robot_interface.get_rgbd_data()
            if rgbd is None:
                continue
            rgb_image = rgbd["rgb_image"]["data"]
            bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

            depth_image = rgbd["depth_image"]["data"]
            depth_m = depth_image.astype(np.float32) / 1000.0
            # 限制最大显示深度（例如 3 米）
            max_depth = 3.0
            depth_m = np.clip(depth_m, 0.0, max_depth)

            # 归一化到 0–255
            depth_norm = (depth_m / max_depth * 255.0).astype(np.uint8)

            # 伪彩色
            depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)

            cv2.imshow("RGB", bgr)
            cv2.imshow("Depth", depth_color)
            cv2.waitKey(1)
            #time.sleep(0.5)

            
    except KeyboardInterrupt:
        ros2_robot_interface.disconnect()

if __name__ == "__main__":
    main()

