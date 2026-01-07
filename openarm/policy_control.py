#!/usr/bin/env python

import torch


from lerobot_ros2.config import Ros2RobotConfig
from lerobot_ros2.openarm import ROS2Robot
import time
import logging

logging.basicConfig(level=logging.INFO)

def main():
    # 配置机器人
    config = Ros2RobotConfig(
        robot_name="my_robot",
        joints={
            "left_forward_position_controller": ["openarm_left_joint1", "openarm_left_joint2", "openarm_left_joint3", "openarm_left_joint4", "openarm_left_joint5", "openarm_left_joint6", "openarm_left_joint7"],
            "right_forward_position_controller": ["openarm_right_joint1", "openarm_right_joint2", "openarm_right_joint3", "openarm_right_joint4", "openarm_right_joint5", "openarm_right_joint6", "openarm_right_joint7"]
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
        },
        cameras={
            "camera_rgb":{
                "height":480,
                "width":640,
                "channels":3,
            },
            "camera_depth":{
                "height":480,
                "width":640,
                "channels":1,
            },
        }
    )

    robot = ROS2Robot(config)
    robot.connect()

    # 示例：加载一个简单的策略（需要替换为实际路径）
    # policy = make_policy("act", config=policy_config, dataset_stats=dataset_stats)
    # policy.load_state_dict(torch.load("path/to/policy.pth"))
    # policy.eval()

    try:
        while True:
            time.sleep(0.1)
            obs = robot.get_observation()
            if not obs:
                continue

            # TODO: 使用策略生成动作
            # with torch.no_grad():
            #     action = policy.select_action(obs)
            # robot.send_action(action)
            #print(obs)
            # 暂时随机动作
            action = {}
            for key, val in obs.items():
                if key.endswith(".pos"):
                    # 尝试更大的动作
                    action[key] = val + 0.1  # 增加幅度
            print(f"Generated action: {action}")
            robot.send_action(action)
            time.sleep(0.5)

    except KeyboardInterrupt:
        robot.disconnect()

if __name__ == "__main__":
    main()


# Sending to left_forward_position_controller: array('d', [-0.05275272755016381, -0.07449683375295614, 1.5872106507972836, 0.06359731441214599, -0.10234454871442701, 0.04109025711451885, -0.11531471732661892])
# Sending to right_forward_position_controller: array('d', [0.06550469214923303, -0.023379110399023302, -0.24730525673304427, 0.011716639963378342, 0.025068284122987665, 0.043760585946440715, -0.14621423666742905])

# Sending to left_forward_position_controller: array('d', [-0.08400403327115089, -0.08069765341208804, 1.6188067973132727, 0.16427158084546906, -0.0997049011235141, 0.047811185374101714, -0.1636257180463231])
# Sending to right_forward_position_controller: array('d', [0.013453667076138723, -0.14297322635823245, -0.206605308925973, 0.10217709364963909, 0.12770849539448487, -0.07371432549380628, -0.017567344480956426])

