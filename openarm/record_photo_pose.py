"""
记录手眼标定数据：采集照片并同步保存各 link 位姿。

使用简介：
1. 启动 ROS2、机械臂状态发布和相机话题（确保可订阅到 `/camera/camera/rgbd`）。
2. 运行：`python -m openarm.record_photo_pose`
3. 在程序运行期间按空格键抓拍一次，保存： /home/scc/lerobot/lerobot_ros2/handeye_calib/calibration_data
    - 图像：`handeye_calib/calibration_data/<序号>.jpg`
    - 位姿：`handeye_calib/calibration_data/poses.json`
4. `Ctrl+C` 退出。
"""

from .Ros2RobotConfig import Ros2RobotConfig
from .Ros2RobotInterface import Ros2RobotInterface
import time
import os
import logging
import json
from pynput import keyboard
import copy
import random
logging.basicConfig(level=logging.INFO)

import cv2
import numpy as np

# 全局变量
ros2_robot_interface = None

link_names = [
    'openarm_body_link0',
    'openarm_left_hand',
    'openarm_left_hand_tcp',
    'openarm_left_left_finger',
    'openarm_left_link0',
    'openarm_left_link1',
    'openarm_left_link2',
    'openarm_left_link3',
    'openarm_left_link4',
    'openarm_left_link5',
    'openarm_left_link6',
    'openarm_left_link7',
    'openarm_left_right_finger',
    'openarm_right_hand',
    'openarm_right_hand_tcp',
    'openarm_right_left_finger',
    'openarm_right_link0',
    'openarm_right_link1',
    'openarm_right_link2',
    'openarm_right_link3',
    'openarm_right_link4',
    'openarm_right_link5',
    'openarm_right_link6',
    'openarm_right_link7',
    'openarm_right_right_finger',
    'world'
]

def on_press(key):
    if key == keyboard.Key.space:
        capture()

# 标定数据
count = 0
data_dir = "/home/scc/lerobot/lerobot_ros2/handeye_calib/calibration_data"
os.makedirs(data_dir, exist_ok=True)
poses = []
preview_scale = 0.75
MAX_POSE_STAMP_DIFF_SEC = 0.08

def _stamp_to_sec(stamp):
    if not isinstance(stamp, dict):
        return None
    sec = stamp.get("sec")
    nanosec = stamp.get("nanosec")
    if sec is None or nanosec is None:
        return None
    return float(sec) + float(nanosec) * 1e-9

def capture():
    global count, poses, ros2_robot_interface
    # 获取 RGB 图像
    rgbd = ros2_robot_interface.get_rgbd_data()
    if rgbd is None:
        print("未获取到RGBD数据，跳过本次采样")
        return

    rgb_image = rgbd["rgb_image"]["data"]
    image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

    # 与本次RGBD图像同时间戳的TF查询（统一采样入口）
    tf_stamp = rgbd["rgb_image"]["header"]["stamp"]
    tf_stamp_sec = _stamp_to_sec(tf_stamp)
  
    # 保存图片
    filename = f"{count}.jpg"
    filepath = os.path.join(data_dir, filename)
    cv2.imwrite(filepath, image)
    
    # 获取所有 link 的 pose
    current_poses = {}
    for link in link_names:
        pose = ros2_robot_interface.get_link_pose(
            link_name=link,
            reference_frame='world',
            stamp=tf_stamp,
            max_time_diff=0.08,
        )
        if pose is not None:
            pos = pose['position']
            quat = pose['orientation']
            pose_stamp = pose.get('pose_stamp')
            pose_stamp_sec = _stamp_to_sec(pose_stamp)
            stamp_diff_sec = None
            if tf_stamp_sec is not None and pose_stamp_sec is not None:
                stamp_diff_sec = abs(pose_stamp_sec - tf_stamp_sec)
                if stamp_diff_sec > MAX_POSE_STAMP_DIFF_SEC:
                    logging.warning(
                        f"{link} 的 pose_stamp 与 tf_stamp 差值过大: {stamp_diff_sec:.6f}s (> {MAX_POSE_STAMP_DIFF_SEC:.3f}s)"
                    )
            current_poses[link] = {
                "position": pos.tolist(),
                "orientation": quat.tolist(),
                "pose_stamp": pose_stamp,
                "stamp_diff_sec": stamp_diff_sec,
            }
        else:
            current_poses[link] = None
    
    # 保存到 poses
    poses.append({
        "image": filename,
        "tf_stamp": tf_stamp,
        "poses": current_poses
    })
    
    # 保存 json
    with open(os.path.join(data_dir, "poses.json"), "w") as f:
        json.dump(poses, f, indent=4)
    
    print(f"已保存图像 {filename} 和所有 link 的坐标到 poses.json，计数: {count}")
    count += 1

def main():
    global ros2_robot_interface
    enable_visualization = bool(os.environ.get("DISPLAY"))
    config = Ros2RobotConfig(
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
        },
    )
    ros2_robot_interface = Ros2RobotInterface(config)
    ros2_robot_interface.connect()
    
    # 启动键盘监听，按空格键捕获
    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    print("[record_photo_pose] 运行中：按空格保存图像+位姿，按 Ctrl+C 退出")
    
    try:

        while True:
            now = time.time()

            # 获取并显示图像
            rgbd = ros2_robot_interface.get_rgbd_data()
            if rgbd is not None:
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

                if enable_visualization:
                    try:
                        cv2.imshow("RGB", bgr)
                        cv2.imshow("Depth", depth_color)
                        cv2.waitKey(1)
                    except cv2.error as e:
                        logging.warning(f"OpenCV GUI 不可用，已自动关闭图像显示: {e}")
                        enable_visualization = False
            else:
                time.sleep(0.005)

            
    except KeyboardInterrupt:
        listener.stop()
        cv2.destroyAllWindows()
        ros2_robot_interface.disconnect()

if __name__ == "__main__":
    main()

