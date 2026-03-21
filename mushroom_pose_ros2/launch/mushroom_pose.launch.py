from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='mushroom_pose_ros2',
            executable='mushroom_pose_node',
            name='mushroom_pose_node',
            output='screen',
            parameters=[{
                'color_topic': '/camera/d435/color/image_raw',
                'depth_topic': '/camera/d435/aligned_depth_to_color/image_raw',
                'camera_info_topic': '/camera/d435/color/camera_info',
                'detection_request_topic': '/detection_request',
                'detected_json_topic': '/detected_objects_json',
                'base_frame': 'base_link',
                'camera_frame': 'd435_color_optical_frame',
                'target_class': 'harvestable',
                'depth_max_m': 1.5,
                'min_conf': 0.25,
                'publish_empty': True,
                'visualization_topic': '/mushroom/visualization',
                'enable_visualization_topic': True,
                'viz_rate_hz': 3.0,
                'viz_axis_length_m': 0.06,
            }]
        )
    ])
