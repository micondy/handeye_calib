# !/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import combine_feature_dicts
from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import RobotAction, RobotObservation, RobotProcessorPipeline
from lerobot.processor.converters import (
    observation_to_transition,
    robot_action_observation_to_transition,
    transition_to_observation,
    transition_to_robot_action,
)
from lerobot.scripts.lerobot_record import record_loop
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun

from lerobot_ros2.openarm.Ros2RobotConfig import Ros2RobotConfig
from lerobot_ros2.openarm.ROS2Robot import ROS2Robot

NUM_EPISODES = 2
FPS = 30
EPISODE_TIME_SEC = 60
RESET_TIME_SEC = 30
TASK_DESCRIPTION = "My task description"
HF_REPO_ID = "<hf_username>/<dataset_repo_id>"


def main():
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
        }
    )

    robot = ROS2Robot(config)


    # Create the dataset (直接用关节状态)
    dataset = LeRobotDataset.create(
        repo_id=HF_REPO_ID,
        fps=FPS,
        features=robot.observation_features,
        robot_type=robot.name,
        use_videos=True,  # 如果你还想存视频
        image_writer_threads=4,
    )


    # Connect the robot and teleoperator
    robot.connect()


    # Initialize the keyboard listener and rerun visualization
    listener, events = init_keyboard_listener()
    init_rerun(session_name="recording_phone")

    if not robot.is_connected:
        raise ValueError("Robot is not connected!")

    print("Starting record loop...")
    episode_idx = 0
    while episode_idx < NUM_EPISODES and not events["stop_recording"]:
        log_say(f"Recording episode {episode_idx + 1} of {NUM_EPISODES}")

        # Main record loop
        record_loop(
            robot=robot,
            events=events,
            fps=FPS,
            dataset=dataset,
            control_time_s=EPISODE_TIME_SEC,
            single_task=TASK_DESCRIPTION,
            display_data=True,
        )

        # Reset the environment if not stopping or re-recording
        if not events["stop_recording"] and (episode_idx < NUM_EPISODES - 1 or events["rerecord_episode"]):
            log_say("Reset the environment")
            record_loop(
                robot=robot,
                events=events,
                fps=FPS,

                control_time_s=RESET_TIME_SEC,
                single_task=TASK_DESCRIPTION,
                display_data=True,
            )

        if events["rerecord_episode"]:
            log_say("Re-recording episode")
            events["rerecord_episode"] = False
            events["exit_early"] = False
            dataset.clear_episode_buffer()
            continue

        # Save episode
        dataset.save_episode()
        episode_idx += 1

    # Clean up
    log_say("Stop recording")
    robot.disconnect()

    listener.stop()

    dataset.finalize()
    dataset.push_to_hub()


if __name__ == "__main__":
    main()
