#!/usr/bin/env python3
import argparse
import json
import threading
import time
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from tf2_ros import TransformBroadcaster

try:
    from .Ros2RobotConfig import Ros2RobotConfig
    from .Ros2RobotInterface import Ros2RobotInterface
    from .tf_tools.tf_publisher import find_transform_by_child, load_transforms_from_json, publish_transform
except ImportError:
    from openarm.Ros2RobotConfig import Ros2RobotConfig
    from openarm.Ros2RobotInterface import Ros2RobotInterface
    from openarm.tf_tools.tf_publisher import find_transform_by_child, load_transforms_from_json, publish_transform

"""
ros2 launch realsense2_camera rs_launch.py \
enable_rgbd:=true \
enable_sync:=true \
align_depth.enable:=true \
enable_color:=true \
enable_depth:=true \
rgb_camera.color_profile:=1280x720x30

python -m openarm.click_point_to_base_tf --camera-frame camera_link --base-frame openarm_body_link0 --mushroom-topic /detected_objects_json

在 MoveIt2 程序里订阅这样即可：
from geometry_msgs.msg import PoseStamped
node.create_subscription(PoseStamped, '/click_point_target_pose', your_callback, 10)

启动时指定自定义 topic：
python -m openarm.click_point_to_base_tf --camera-frame camera_link --base-frame openarm_body_link0 --target-pose-topic /my_target_pose

# 末端工具 offset（夹爪/吸盘）示例：
# 1) 推荐使用 tool 坐标系（offset 会随姿态一起旋转）
python -m openarm.click_point_to_base_tf --camera-frame camera_link --base-frame openarm_body_link0 \
    --target-offset-xyz 0 0 0.10 --target-offset-frame tool

# 2) 若希望固定按 base 方向补偿（不随姿态变化）
python -m openarm.click_point_to_base_tf --camera-frame camera_link --base-frame openarm_body_link0 \
    --target-offset-xyz 0 0 0.10 --target-offset-frame base

# 3) 从 TF 文件读取相机位姿（无需实时 TF）
python -m openarm.click_point_to_base_tf --camera-frame camera_link --base-frame openarm_body_link0 \
    --tf-file openarm/tf_tools/tfs_example.json

# tf-file 最小格式示例（child_frame 必须与 --camera-frame 一致）：
# [
#   {
#     "parent_frame": "openarm_body_link0",
#     "child_frame": "camera_link",
#     "translation": [0.25, -0.02, 0.62],
#     "rotation": [0.0, 0.0, 0.0, 1.0]
#   }
# ]

# 正负号快速判断：
# - 若发现“越靠近目标越穿透”，说明补偿方向反了，把数值改为相反号。
# - tool 模式下通常沿工具 Z 轴补偿；base 模式下沿 base 的 XYZ 方向补偿。

Status 窗口（420×48 小条）：始终存在，显示当前视频状态和最后点位来源，V/ESC 在此接收
RGB Click 窗口：默认开启，按 V 关闭后销毁该窗口（鼠标点击功能暂停），再按 V 重新创建并恢复鼠标回调
视频关闭时后台仍正常处理 mushroom topic 并发布位姿，不影响主流程

无屏环境无法使用此代码


"""



class ClickPointToBaseTF:
    """将图像点击点/检测点转换到 base 坐标系，并发布目标 PoseStamped。

    数据来源：
    1) mushroom 话题（若有）
    2) 鼠标点击像素 + 深度反投影

    相机位姿来源优先级：
    1) --camera-pose（手动固定）
    2) --tf-file（文件固定）
    3) 实时 TF 查询
    """

    def __init__(
        self,
        camera_frame: str,
        base_frame: str,
        camera_pose: Optional[np.ndarray] = None,
        mushroom_topic: str = '/detected_objects_json',
        tf_file: Optional[str] = None,
        target_pose_topic: str = '/click_point_target_pose',
        target_offset_xyz: Optional[np.ndarray] = None,
        target_offset_frame: Optional[str] = None,
    ):
        """初始化参数。

        Args:
            camera_frame: 相机坐标系名（如 camera_link）。
            base_frame: 机械臂基座坐标系名。
            camera_pose: 可选固定外参 [x,y,z,qx,qy,qz,qw]（base->camera）。
            mushroom_topic: 检测结果话题（std_msgs/String）。
            tf_file: 可选 TF JSON 文件；用于提供固定相机外参。
            target_pose_topic: 发布目标位姿话题（PoseStamped）。
            target_offset_xyz: 末端 TCP 位置补偿（米）。
            target_offset_frame: 补偿参考系（tool/base/任意TF帧名），
        """
        self.camera_frame = camera_frame  # 相机 TF 名称（child_frame）
        self.base_frame = base_frame  # 基座 TF 名称（parent_frame）
        self.camera_pose = camera_pose  # 固定相机外参（可空），优先级高于 tf_file 中的相机位姿
        self.mushroom_topic = mushroom_topic 
        self.tf_file = tf_file # TF 文件路径（可空）用于加载固定相机外参，优先级低于直接传 camera_pose
        self.target_pose_topic = target_pose_topic
        self.target_offset_xyz = (
            np.asarray(target_offset_xyz, dtype=np.float64).reshape(3)
            if target_offset_xyz is not None
            else np.zeros(3, dtype=np.float64)
        )
        # offset 的语义：期望末端 TCP 相对“检测点”做的位置补偿（单位：米）。
        # 例如吸盘长度 0.10m，可配置为 (0, 0, 0.10) 或其反号（取决于工具轴方向）。
        self.target_offset_frame = str(target_offset_frame).strip() if target_offset_frame is not None else ''
        if self.target_offset_frame == '':
            raise ValueError('target_offset_frame 必填，请手动输入（tool/base 或任意 TF 帧名）。')
        self._has_target_offset = not np.allclose(self.target_offset_xyz, 0.0)  # 缓存：避免循环中重复判断
        self._offset_frame_warned = False

        self.interface: Optional[Ros2RobotInterface] = None
        self.tf_broadcaster: Optional[TransformBroadcaster] = None
        self._pose_publisher = None
        self._show_video: bool = True
        self._rgb_window_open: bool = False
        self._identity_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)

        self.clicked_uv: Optional[Tuple[int, int]] = None  # 最近一次点击像素
        self.clicked_base_xyz: Optional[np.ndarray] = None  # 最近一次目标点（base）
        self.clicked_base_quat: Optional[np.ndarray] = None  # 最近一次姿态（xyzw）；None 表示 identity
        self.clicked_depth_m: Optional[float] = None
        self.T_base_cam_fixed: Optional[np.ndarray] = None
        self.last_source: str = 'none'

        self._mushroom_lock = threading.Lock()
        self._pending_mushroom_xyz: Optional[np.ndarray] = None
        self._pending_mushroom_quat: Optional[np.ndarray] = None
        self._pending_mushroom_meta: str = ''
        self._offset_frame_rot_cache_quat: Optional[np.ndarray] = None
        self._offset_frame_rot_cache_ts: float = 0.0
        self._offset_frame_cache_ttl_sec: float = 0.05

        if camera_pose is not None:
            camera_pose = np.asarray(camera_pose, dtype=np.float64).reshape(7)
            pos = camera_pose[:3]
            quat = camera_pose[3:]
            self.T_base_cam_fixed = self._pose_to_homogeneous(pos, quat)

    @staticmethod
    def _normalize_quaternion(quat_xyzw: np.ndarray) -> np.ndarray:
        """四元数归一化，避免非单位四元数带来的旋转误差。"""
        quat = np.asarray(quat_xyzw, dtype=np.float64).reshape(4)
        norm = float(np.linalg.norm(quat))
        if norm < 1e-12:
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        return quat / norm

    def _open_rgb_window(self):
        """按需创建 RGB 窗口，避免重复创建。"""
        if self._rgb_window_open:
            return
        cv2.namedWindow('RGB Click')
        cv2.setMouseCallback('RGB Click', self._on_mouse)
        self._rgb_window_open = True

    def _close_rgb_window(self):
        """按需关闭 RGB 窗口，避免每帧 destroyWindow。"""
        if not self._rgb_window_open:
            return
        cv2.destroyWindow('RGB Click')
        self._rgb_window_open = False

    def _apply_target_offset(self, xyz: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
        """将 offset（TCP 补偿）旋转到 base 坐标后叠加到 xyz。

        - target_offset_frame == 'base'：直接在 base 下平移
        - target_offset_frame == 'tool'：使用当前目标姿态 quat_xyzw 旋转
        - 其他字符串：当作 TF 帧名，查询该帧在 base 下姿态后旋转
        """
        if not self._has_target_offset:
            return xyz

        # 快速路径：不依赖 TF 查询，开销最小
        offset_frame = self.target_offset_frame.lower()
        if offset_frame == 'base':
            return xyz + self.target_offset_xyz
        if offset_frame == 'tool':
            quat = self._normalize_quaternion(quat_xyzw)
            T = self._pose_to_homogeneous(np.zeros(3, dtype=np.float64), quat)
            offset_base = T[:3, :3] @ self.target_offset_xyz
            return xyz + offset_base

        offset_rot_quat = None
        if self.interface is not None and self.interface._robot_node is not None:
            now_ts = time.time()
            if (
                self._offset_frame_rot_cache_quat is not None
                and (now_ts - self._offset_frame_rot_cache_ts) < self._offset_frame_cache_ttl_sec
            ):
                offset_rot_quat = self._offset_frame_rot_cache_quat
            else:
                node = self.interface._robot_node
                pose = self.interface.get_link_pose(
                    link_name=self.target_offset_frame,
                    reference_frame=self.base_frame,
                    stamp=node.get_clock().now().to_msg(),
                    max_time_diff=0.2,
                )
                if pose is not None:
                    offset_rot_quat = pose.get('orientation')
                    if offset_rot_quat is not None:
                        self._offset_frame_rot_cache_quat = np.asarray(offset_rot_quat, dtype=np.float64).reshape(4)
                        self._offset_frame_rot_cache_ts = now_ts

        if offset_rot_quat is None:
            if not self._offset_frame_warned:
                print(
                    f'⚠️ 未获取到 offset frame TF: {self.base_frame} <- {self.target_offset_frame}，'
                    f'将回退为 tool 方式计算。'
                )
                self._offset_frame_warned = True
            quat = self._normalize_quaternion(quat_xyzw)
            T = self._pose_to_homogeneous(np.zeros(3, dtype=np.float64), quat)
            offset_base = T[:3, :3] @ self.target_offset_xyz
            return xyz + offset_base

        self._offset_frame_warned = False
        offset_rot_quat = self._normalize_quaternion(offset_rot_quat)
        T = self._pose_to_homogeneous(np.zeros(3, dtype=np.float64), offset_rot_quat)
        offset_base = T[:3, :3] @ self.target_offset_xyz
        return xyz + offset_base

    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.clicked_uv = (int(x), int(y))
            print(f'🖱️ 点击像素: (u={x}, v={y})')

    def _on_mushroom_msg(self, msg: String):
        try:
            obj = json.loads(msg.data)
        except Exception as exc:
            print(f'⚠️ mushroom JSON 解析失败: {exc}')
            return

        xyz, quat, meta = self._extract_pose_from_mushroom_payload(obj)
        if xyz is None:
            return

        with self._mushroom_lock:
            self._pending_mushroom_xyz = xyz
            self._pending_mushroom_quat = quat
            self._pending_mushroom_meta = meta

    def _extract_pose_from_mushroom_payload(
        self, payload: Dict[str, Any]
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
        """从 mushroom payload 中提取最优检测的位置和四元数姿态。
        返回 (xyz, quat_xyzw, meta)，无法解析时返回 (None, None, '')。
        """
        detections = payload.get('detections', []) if isinstance(payload, dict) else []
        if not isinstance(detections, list) or not detections:
            return None, None, ''

        best_det = None
        best_score = -1e9
        for det in detections:
            if not isinstance(det, dict):
                continue
            conf = det.get('confidence')
            if conf is None:
                conf = det.get('grasp_quality', 0.0)
            try:
                score = float(conf)
            except Exception:
                score = 0.0
            if score > best_score:
                best_score = score
                best_det = det

        if best_det is None:
            return None, None, ''

        # 优先从 grasp_pose 取位姿，再从顶层 position/orientation 取
        grasp_pose = best_det.get('grasp_pose', {})
        if isinstance(grasp_pose, dict) and grasp_pose:
            pos = grasp_pose.get('position')
            ori = grasp_pose.get('orientation')
        else:
            pos = best_det.get('position')
            ori = best_det.get('orientation')

        if not isinstance(pos, dict):
            return None, None, ''

        try:
            x = float(pos['x'])
            y = float(pos['y'])
            z = float(pos['z'])
        except Exception:
            return None, None, ''

        quat = None
        if isinstance(ori, dict):
            try:
                quat = np.array(
                    [float(ori['x']), float(ori['y']), float(ori['z']), float(ori['w'])],
                    dtype=np.float64,
                )
            except Exception:
                quat = None

        label = str(best_det.get('label', 'unknown'))
        conf_show = float(best_det.get('confidence', best_det.get('grasp_quality', 0.0)))
        has_ori = quat is not None
        meta = f'label={label}, conf={conf_show:.3f}, ori={"yes" if has_ori else "no"}'
        return np.array([x, y, z], dtype=np.float64), quat, meta

    def _consume_mushroom_point(self) -> Optional[Tuple[np.ndarray, Optional[np.ndarray], str]]:
        # 单次消费策略：每帧最多取一次，取出后清空，避免重复使用旧检测结果。
        with self._mushroom_lock:
            if self._pending_mushroom_xyz is None:
                return None
            xyz = self._pending_mushroom_xyz.copy()
            quat = self._pending_mushroom_quat.copy() if self._pending_mushroom_quat is not None else None
            meta = self._pending_mushroom_meta
            self._pending_mushroom_xyz = None
            self._pending_mushroom_quat = None
            self._pending_mushroom_meta = ''
        return xyz, quat, meta

    @staticmethod
    def _pixel_to_camera_xyz(u: int, v: int, depth_m: float, K: np.ndarray) -> np.ndarray:
        # 基于针孔模型反投影：
        # x = (u-cx)/fx * z, y = (v-cy)/fy * z, z = depth
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
        # 注意四元数顺序为 [x, y, z, w]。
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

    def _publish_target_pose(self):
        """将当前 clicked_base_xyz/quat 作为 PoseStamped 发布到 target_pose_topic。
        若 clicked_base_quat 有值（来自 mushroom）则使用真实姿态，否则用 identity 四元数。
        """
        if self._pose_publisher is None or self.interface is None or self.clicked_base_xyz is None:
            return
        node = self.interface._robot_node
        if node is None:
            return
        # 姿态统一归一化，避免上游数据模长漂移。
        quat_raw = self.clicked_base_quat if self.clicked_base_quat is not None else self._identity_quat
        quat = self._normalize_quaternion(quat_raw)
        # 这里应用 TCP offset：
        # - tool: 先按当前姿态旋转 offset，再叠加到 base 坐标
        # - base: 直接在 base 坐标下叠加
        target_xyz = self._apply_target_offset(self.clicked_base_xyz, quat)
        msg = PoseStamped()
        msg.header.frame_id = self.base_frame
        msg.header.stamp = node.get_clock().now().to_msg()
        msg.pose.position.x = float(target_xyz[0])
        msg.pose.position.y = float(target_xyz[1])
        msg.pose.position.z = float(target_xyz[2])
        msg.pose.orientation.x = float(quat[0])
        msg.pose.orientation.y = float(quat[1])
        msg.pose.orientation.z = float(quat[2])
        msg.pose.orientation.w = float(quat[3])
        self._pose_publisher.publish(msg)
        print(
            f'📤 已发布目标位姿 -> {self.target_pose_topic}: '
            f'xyz=[{target_xyz[0]:.4f}, {target_xyz[1]:.4f}, {target_xyz[2]:.4f}] '
            f'quat=[{quat[0]:.4f}, {quat[1]:.4f}, {quat[2]:.4f}, {quat[3]:.4f}]'
        )
        if self._has_target_offset:
            print(
                f'   ↳ offset(frame={self.target_offset_frame}) '
                f'[{self.target_offset_xyz[0]:.4f}, {self.target_offset_xyz[1]:.4f}, {self.target_offset_xyz[2]:.4f}] m'
            )

    def _publish_clicked_point_tf(self, child_frame: str = 'clicked_point'):
        if self.tf_broadcaster is None or self.interface is None or self.clicked_base_xyz is None:
            return

        node = self.interface._robot_node
        if node is None:
            return

        publish_transform(
            tf_broadcaster=self.tf_broadcaster,
            node=node,
            parent_frame=self.base_frame,
            child_frame=child_frame,
            translation_xyz=self.clicked_base_xyz,
            rotation_xyzw=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
        )

    def _publish_camera_tf(self):
        if self.tf_broadcaster is None or self.interface is None or self.camera_pose is None:
            return

        node = self.interface._robot_node
        if node is None:
            return

        camera_pose = np.asarray(self.camera_pose, dtype=np.float64).reshape(7)
        pos = camera_pose[:3]
        quat = camera_pose[3:]

        publish_transform(
            tf_broadcaster=self.tf_broadcaster,
            node=node,
            parent_frame=self.base_frame,
            child_frame=self.camera_frame,
            translation_xyz=pos,
            rotation_xyzw=quat,
        )

    def _try_load_camera_pose_from_file(self):
        """若提供 --tf-file 且未提供 --camera-pose，则尝试读取固定相机外参。"""
        if self.tf_file is None:
            return
        if self.camera_pose is not None:
            return

        transforms = load_transforms_from_json(self.tf_file)
        tf_item = find_transform_by_child(transforms, self.camera_frame)
        if tf_item is None:
            print(
                f'⚠️ TF 文件中未找到 child_frame={self.camera_frame}，将继续使用实时TF查询。'
            )
            return

        pos = tf_item['translation']
        quat = tf_item['rotation']
        self.camera_pose = np.concatenate([pos, quat], axis=0)
        self.T_base_cam_fixed = self._pose_to_homogeneous(pos, quat)
        print(f'📄 已从文件加载相机TF: {tf_item["parent_frame"]} -> {tf_item["child_frame"]}')

    def run(self):
        """主循环：取数据、解算目标、发布位姿和可视化。"""
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

        self._try_load_camera_pose_from_file()

        self.tf_broadcaster = TransformBroadcaster(self.interface._robot_node)
        self._pose_publisher = self.interface._robot_node.create_publisher(
            PoseStamped,
            self.target_pose_topic,
            10,
        )
        print(f'📢 已创建目标位姿发布者: {self.target_pose_topic}')
        self.interface._robot_node.create_subscription(
            String,
            self.mushroom_topic,
            self._on_mushroom_msg,
            10,
        )
        print(f'📡 已订阅 mushroom 点位话题: {self.mushroom_topic}')

        # 状态窗口：始终显示，用于接收按键（ESC/V）
        cv2.namedWindow('Status', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Status', 420, 48)
        # 视频窗口：可按 V 切换
        self._open_rgb_window()

        try:
            while True:
                # 1) 持续发布相机TF（若提供了固定相机位姿，则用于稳定对齐）
                self._publish_camera_tf()

                # 2) 优先消费 mushroom 检测结果；其自带姿态时直接用于目标发布
                mushroom_point = self._consume_mushroom_point()
                if mushroom_point is not None:
                    xyz, quat, meta = mushroom_point
                    self.clicked_base_xyz = xyz
                    self.clicked_base_quat = quat  # 使用 mushroom 提供的真实姿态
                    self.clicked_depth_m = None
                    self.last_source = 'mushroom_topic'
                    dist = float(np.linalg.norm(self.clicked_base_xyz))
                    ori_str = f'quat=[{quat[0]:.3f},{quat[1]:.3f},{quat[2]:.3f},{quat[3]:.3f}]' if quat is not None else 'ori=identity'
                    print(
                        f'🍄 采用 mushroom 结果 -> {self.base_frame}: '
                        f'x={self.clicked_base_xyz[0]:.4f}, y={self.clicked_base_xyz[1]:.4f}, '
                        f'z={self.clicked_base_xyz[2]:.4f}, |p|={dist:.4f} m, {ori_str} ({meta})'
                    )
                    self._publish_target_pose()

                rgbd = self.interface.get_rgbd_data()
                if rgbd is None:
                    time.sleep(0.01)
                    continue

                rgb_image = rgbd['rgb_image']['data']
                depth_image = rgbd['depth_image']['data']
                K = np.array(rgbd['rgb_camera_info']['k'], dtype=np.float64).reshape(3, 3)

                bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

                if self.clicked_uv is not None:
                    # 3) 鼠标点击路径：像素+深度 -> 相机坐标 -> base 坐标
                    u, v = self.clicked_uv
                    h, w = depth_image.shape[:2]
                    if 0 <= v < h and 0 <= u < w:
                        depth_m = float(depth_image[v, u]) / 1000.0
                        if depth_m > 0:
                            if self.T_base_cam_fixed is not None:
                                # 若用户提供了固定外参，避免实时 TF 查询抖动
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
                            self.clicked_base_quat = None  # 鼠标点击只有位置，姿态用 identity
                            self.clicked_depth_m = depth_m
                            self.last_source = 'mouse_click'

                            dist = float(np.linalg.norm(self.clicked_base_xyz))
                            print(
                                f'✅ {self.base_frame}坐标(来自{self.camera_frame}): x={self.clicked_base_xyz[0]:.4f}, '
                                f'y={self.clicked_base_xyz[1]:.4f}, z={self.clicked_base_xyz[2]:.4f}, '
                                f'|p|={dist:.4f} m, depth={depth_m:.4f} m'
                            )
                            self._publish_target_pose()
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
                    cv2.putText(bgr, f'source: {self.last_source}', (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 180, 80), 2)

                # 视频窗口
                if self._show_video:
                    self._open_rgb_window()
                    cv2.putText(bgr, f'Left click to compute 3D point in {self.base_frame}', (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    cv2.putText(bgr, '[V] hide video  [ESC] quit', (10, bgr.shape[0] - 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
                    cv2.imshow('RGB Click', bgr)
                else:
                    self._close_rgb_window()

                # 状态栏（始终显示）
                status_img = np.zeros((48, 420, 3), dtype=np.uint8)
                video_hint = '[V] show video' if not self._show_video else '[V] hide video'
                src_text = f'source: {self.last_source}' if self.clicked_base_xyz is not None else 'no point yet'
                cv2.putText(status_img, f'{video_hint}  [ESC] quit  |  {src_text}',
                            (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 220, 255), 1)
                cv2.imshow('Status', status_img)

                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    break
                elif key == ord('v') or key == ord('V'):
                    # 仅切换显示，不影响后台 topic 处理和位姿发布。
                    self._show_video = not self._show_video
                    state = '开启' if self._show_video else '关闭'
                    print(f'📺 视频显示已{state}（按 V 切换）')

        finally:
            cv2.destroyAllWindows()
            if self.interface is not None:
                self.interface.disconnect()
            print('👋 已退出')


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
    parser.add_argument(
        '--mushroom-topic',
        default='/detected_objects_json',
        help='订阅 mushroom_pose 发布结果的 topic（std_msgs/String, 默认 /detected_objects_json）'
    )
    parser.add_argument(
        '--tf-file',
        default=None,
        help='可选：TF JSON 文件路径。若提供且包含 camera_frame 的位姿，则优先使用文件中的相机位姿。'
    )
    parser.add_argument(
        '--target-pose-topic',
        default='/click_point_target_pose',
        help='发布目标位姿 (geometry_msgs/PoseStamped) 的 topic 名称（默认 /click_point_target_pose）'
    )
    parser.add_argument(
        '--target-offset-xyz',
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        metavar=('OX', 'OY', 'OZ'),
        help='目标位姿偏置（米）。用于补偿末端夹爪/吸盘长度，默认 0 0 0。'
    )
    parser.add_argument(
        '--target-offset-frame',
        type=str,
        required=True,
        help='offset 所在坐标系：tool/base 或任意 TF 帧名（必填）'
    )
    args = parser.parse_args()

    app = ClickPointToBaseTF(
        args.camera_frame,
        args.base_frame,
        args.camera_pose,
        args.mushroom_topic,
        args.tf_file,
        args.target_pose_topic,
        args.target_offset_xyz,
        args.target_offset_frame,
    )
    app.run()


if __name__ == '__main__':
    main()
