#!/usr/bin/env python3
import argparse
import time
from typing import Optional, Tuple

import cv2
import numpy as np
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster

try:
    from .Ros2RobotConfig import Ros2RobotConfig
    from .Ros2RobotInterface import Ros2RobotInterface
except ImportError:
    from openarm.Ros2RobotConfig import Ros2RobotConfig
    from openarm.Ros2RobotInterface import Ros2RobotInterface

"""
ros2 launch realsense2_camera rs_launch.py \
enable_rgbd:=true \
enable_sync:=true \
align_depth.enable:=true \
enable_color:=true \
enable_depth:=true \
rgb_camera.color_profile:=1280x720x30


python -m openarm.click_point_to_base_tf --camera-frame camera_depth_optical_frame --base-frame world --camera-pose 0.32034904 0.05333502 0.73377816 -0.70317699 0.71050360 -0.01751946 -0.02048948


"""

class ClickPointToBaseTF:
    def __init__(self, camera_frame: str, base_frame: str, camera_pose: Optional[np.ndarray] = None):
        self.camera_frame = camera_frame
        self.base_frame = base_frame
        self.camera_pose = camera_pose

        self.interface: Optional[Ros2RobotInterface] = None
        self.tf_broadcaster: Optional[TransformBroadcaster] = None

        self.clicked_uv: Optional[Tuple[int, int]] = None
        self.clicked_base_xyz: Optional[np.ndarray] = None
        self.clicked_depth_m: Optional[float] = None
        self.T_base_cam_fixed: Optional[np.ndarray] = None

        if camera_pose is not None:
            camera_pose = np.asarray(camera_pose, dtype=np.float64).reshape(7)
            pos = camera_pose[:3]
            quat = camera_pose[3:]
            self.T_base_cam_fixed = self._pose_to_homogeneous(pos, quat)

    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.clicked_uv = (int(x), int(y))
            print(f'🖱️ 点击像素: (u={x}, v={y})')

    @staticmethod
    def _pixel_to_camera_xyz(u: int, v: int, depth_m: float, K: np.ndarray) -> np.ndarray:
        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]
        x = (u - cx) * depth_m / fx
        y = (v - cy) * depth_m / fy
        z = depth_m
        return np.array([x, y, z], dtype=np.float64)

    @staticmethod
    def _pose_to_homogeneous(position: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
        x, y, z, w = quat_xyzw
        R = np.array([
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ], dtype=np.float64)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = np.asarray(position, dtype=np.float64).reshape(3)
        return T

    def _publish_clicked_point_tf(self, child_frame: str = 'clicked_point'):
        if self.tf_broadcaster is None or self.interface is None or self.clicked_base_xyz is None:
            return

        node = self.interface._robot_node
        if node is None:
            return

        msg = TransformStamped()
        msg.header.stamp = node.get_clock().now().to_msg()
        msg.header.frame_id = self.base_frame
        msg.child_frame_id = child_frame

        msg.transform.translation.x = float(self.clicked_base_xyz[0])
        msg.transform.translation.y = float(self.clicked_base_xyz[1])
        msg.transform.translation.z = float(self.clicked_base_xyz[2])

        msg.transform.rotation.x = 0.0
        msg.transform.rotation.y = 0.0
        msg.transform.rotation.z = 0.0
        msg.transform.rotation.w = 1.0

        self.tf_broadcaster.sendTransform(msg)

    def _publish_camera_tf(self):
        if self.tf_broadcaster is None or self.interface is None or self.camera_pose is None:
            return

        node = self.interface._robot_node
        if node is None:
            return

        camera_pose = np.asarray(self.camera_pose, dtype=np.float64).reshape(7)
        pos = camera_pose[:3]
        quat = camera_pose[3:]

        msg = TransformStamped()
        msg.header.stamp = node.get_clock().now().to_msg()
        msg.header.frame_id = self.base_frame
        msg.child_frame_id = self.camera_frame
        msg.transform.translation.x = float(pos[0])
        msg.transform.translation.y = float(pos[1])
        msg.transform.translation.z = float(pos[2])
        msg.transform.rotation.x = float(quat[0])
        msg.transform.rotation.y = float(quat[1])
        msg.transform.rotation.z = float(quat[2])
        msg.transform.rotation.w = float(quat[3])
        self.tf_broadcaster.sendTransform(msg)

    def run(self):
        config = Ros2RobotConfig(
            robot_name='click_point_to_base_tf',
            joints={},
            topics_to_subscribe={
                'camera_rgbd': {
                    'topic': '/camera/camera/rgbd',
                    'type': 'realsense2_camera_msgs/msg/RGBD',
                },
            },
            topics_to_publish={},
        )

        self.interface = Ros2RobotInterface(config)
        self.interface.connect()
        if self.interface._robot_node is None:
            raise RuntimeError('ROS2 node 未初始化')

        self.tf_broadcaster = TransformBroadcaster(self.interface._robot_node)

        cv2.namedWindow('RGB Click')
        cv2.setMouseCallback('RGB Click', self._on_mouse)

        try:
            while True:
                self._publish_camera_tf()

                rgbd = self.interface.get_rgbd_data()
                if rgbd is None:
                    time.sleep(0.01)
                    continue

                rgb_image = rgbd['rgb_image']['data']
                depth_image = rgbd['depth_image']['data']
                K = np.array(rgbd['rgb_camera_info']['k'], dtype=np.float64).reshape(3, 3)

                bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

                if self.clicked_uv is not None:
                    u, v = self.clicked_uv
                    h, w = depth_image.shape[:2]
                    if 0 <= v < h and 0 <= u < w:
                        depth_m = float(depth_image[v, u]) / 1000.0
                        if depth_m > 0:
                            if self.T_base_cam_fixed is not None:
                                T_base_cam = self.T_base_cam_fixed
                            else:
                                stamp = rgbd['rgb_image']['header']['stamp']
                                pose = self.interface.get_link_pose(
                                    link_name=self.camera_frame,
                                    reference_frame=self.base_frame,
                                    stamp=stamp,
                                    max_time_diff=0.08,
                                )
                                if pose is None:
                                    print(
                                        f'⚠️ 未获取到 TF: {self.base_frame} <- {self.camera_frame}。'
                                        f'请确认两者在同一TF树；或直接传 --camera-pose x y z qx qy qz qw'
                                    )
                                    self.clicked_uv = None
                                    continue

                                T_base_cam = self._pose_to_homogeneous(
                                    pose['position'],
                                    pose['orientation'],
                                )

                            p_cam = self._pixel_to_camera_xyz(u, v, depth_m, K)
                            p_cam_h = np.array([p_cam[0], p_cam[1], p_cam[2], 1.0], dtype=np.float64)
                            p_base_h = T_base_cam @ p_cam_h
                            self.clicked_base_xyz = p_base_h[:3]
                            self.clicked_depth_m = depth_m

                            dist = float(np.linalg.norm(self.clicked_base_xyz))
                            print(
                                f'✅ {self.base_frame}坐标(来自{self.camera_frame}): x={self.clicked_base_xyz[0]:.4f}, '
                                f'y={self.clicked_base_xyz[1]:.4f}, z={self.clicked_base_xyz[2]:.4f}, '
                                f'|p|={dist:.4f} m, depth={depth_m:.4f} m'
                            )
                        else:
                            print('⚠️ 点击点深度为0，无法反投影')

                    self.clicked_uv = None

                if self.clicked_base_xyz is not None:
                    self._publish_clicked_point_tf(child_frame='clicked_point')
                    text = (
                        f'clicked_point @ {self.base_frame}: '
                        f'[{self.clicked_base_xyz[0]:.3f}, {self.clicked_base_xyz[1]:.3f}, {self.clicked_base_xyz[2]:.3f}] m'
                    )
                    cv2.putText(bgr, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

                cv2.putText(bgr, f'Left click to compute 3D point in {self.base_frame}', (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                cv2.imshow('RGB Click', bgr)
                key = cv2.waitKey(1)
                if key == 27:
                    break

        finally:
            cv2.destroyAllWindows()
            if self.interface is not None:
                self.interface.disconnect()


def main():
    parser = argparse.ArgumentParser(description='点击图像像素，通过TF(camera_link->base)转换并发布TF(clicked_point)')
    parser.add_argument(
        '--camera-frame',
        default='camera_link',
        help='相机坐标系名称（默认 camera_link）'
    )
    parser.add_argument(
        '--base-frame',
        default='openarm_body_link0',
        help='基座坐标系名称（默认 openarm_body_link0）'
    )
    parser.add_argument(
        '--camera-pose',
        type=float,
        nargs=7,
        default=None,
        metavar=('X', 'Y', 'Z', 'QX', 'QY', 'QZ', 'QW'),
        help='手动设置 camera_link 在 base_frame 下位姿: x y z qx qy qz qw'
    )
    args = parser.parse_args()

    app = ClickPointToBaseTF(args.camera_frame, args.base_frame, args.camera_pose)
    app.run()


if __name__ == '__main__':
    main()
