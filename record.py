#!/usr/bin/env python

from lerobot.configs.types import PipelineFeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import create_initial_features
from lerobot.processor import RobotObservation
from lerobot.scripts.lerobot_record import record_loop
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.visualization_utils import init_rerun
from lerobot.utils.utils import log_say
import cv2
import numpy as np

from lerobot_ros2.openarm.Ros2RobotConfig import ROS2RobotConfig
from lerobot_ros2.openarm.Ros2Robot import ROS2Robot
from lerobot.datasets.pipeline_features import create_initial_features
from lerobot.configs.types import PipelineFeatureType
from lerobot.datasets.pipeline_features import PipelineFeatureType
from lerobot.datasets.features import FeatureType
NUM_EPISODES = 2
FPS = 30
EPISODE_TIME_SEC = 60
RESET_TIME_SEC = 10
TASK_DESCRIPTION = "moveit_execution"
HF_REPO_ID = "<hf_username>/<dataset_repo_id>"

def main():
    # -----------------------
    # ROS2 Robot Config
    # -----------------------
    config = ROS2RobotConfig(
        robot_name="my_robot",
        joints={
            "left_arm": [
                "openarm_left_joint1",
                "openarm_left_joint2",
                "openarm_left_joint3",
                "openarm_left_joint4",
                "openarm_left_joint5",
                "openarm_left_joint6",
                "openarm_left_joint7",
            ],
            "right_arm": [
                "openarm_right_joint1",
                "openarm_right_joint2",
                "openarm_right_joint3",
                "openarm_right_joint4",
                "openarm_right_joint5",
                "openarm_right_joint6",
                "openarm_right_joint7",
            ],
        },
        topics_to_subscribe={
            "joint_states": {
                "topic": "/joint_states",
                "type": "sensor_msgs/msg/JointState",
            },
            "rgb_image": {
                "topic": "/camera/camera/color/image_raw",
                "type": "sensor_msgs/msg/Image",
            },
            "depth_image": {
                "topic": "/camera/camera/aligned_depth_to_color/image_raw",
                "type": "sensor_msgs/msg/Image",
            },
        },
        topics_to_publish={},  # ❗record-only，不发布控制
    )

    robot = ROS2Robot(config)
    robot.connect()
    # while True:
    #     obs1 = robot.get_observation()
    #     if obs1 is None:
    #         continue
    #     rgb_image = obs1["rgb_image"]
    #     bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

    #     depth_image = obs1["depth_image"]
    #     depth_m = depth_image.astype(np.float32) / 1000.0
    #     # 限制最大显示深度（例如 3 米）
    #     max_depth = 3.0
    #     depth_m = np.clip(depth_m, 0.0, max_depth)

    #     # 归一化到 0–255
    #     depth_norm = (depth_m / max_depth * 255.0).astype(np.uint8)

    #     # 伪彩色
    #     depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)

    #     cv2.imshow("RGB", bgr)
    #     cv2.imshow("Depth", depth_color)
    #     cv2.waitKey(1)
    # -----------------------
    # Dataset
    # -----------------------
    # 生成初始 feature schema
    # Flatten features dict: merge action and observation features into a single dict
    observation_features = {
        "rgb_image": {
            "pipeline_type": PipelineFeatureType.OBSERVATION,
            "feature_type": FeatureType.IMAGE,
        },
        "depth_image": {
            "pipeline_type": PipelineFeatureType.OBSERVATION,
            "feature_type": FeatureType.DEPTH,
        },
    }

    dataset = LeRobotDataset.create(
        repo_id=HF_REPO_ID,
        fps=FPS,
        robot_type=robot.name,
        features=observation_features,
        use_videos=True,
    )
  


    # -----------------------
    # UI / Keyboard / Viz
    # -----------------------
    listener, events = init_keyboard_listener()
    init_rerun(session_name="moveit_record")

    log_say("Start recording")

    episode_idx = 0
    while episode_idx < NUM_EPISODES and not events["stop_recording"]:
        log_say(f"Recording episode {episode_idx}")

        record_loop(
            robot=robot,
            events=events,
            fps=FPS,
            dataset=dataset,
            control_time_s=EPISODE_TIME_SEC,
            single_task=TASK_DESCRIPTION,
            display_data=True,
            robot_action_processor=None,       # ❗不记录 action
            teleop_action_processor=None,      # ❗不记录 action
        )

        if events["rerecord_episode"]:
            log_say("Re-record episode")
            events["rerecord_episode"] = False
            dataset.clear_episode_buffer()
            continue

        dataset.save_episode()
        episode_idx += 1

    log_say("Finish recording")
    listener.stop()
    robot.disconnect()

    dataset.finalize()
    dataset.push_to_hub()


if __name__ == "__main__":
    main()
