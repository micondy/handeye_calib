import json
import os
import threading
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import open3d as o3d
import rclpy
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener


class MushroomPoseNode(Node):
    def __init__(self) -> None:
        super().__init__('mushroom_pose_node')

        self.declare_parameter('color_topic', '/camera/d435/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/d435/aligned_depth_to_color/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/d435/color/camera_info')
        self.declare_parameter('detection_request_topic', '/detection_request')
        self.declare_parameter('detected_json_topic', '/detected_objects_json')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('camera_frame', 'd435_color_optical_frame')
        self.declare_parameter('target_class', 'harvestable')
        self.declare_parameter('model_path', '/home/winnnnnn/lododo-arm/src/YOLOv8-Magic-8.3.12/ultralytics-8.3.12/mushroom_sphere_fit/best.pt')
        self.declare_parameter('ultralytics_root', '/home/winnnnnn/lododo-arm/src/YOLOv8-Magic-8.3.12/ultralytics-8.3.12')
        self.declare_parameter('depth_max_m', 1.5)
        self.declare_parameter('min_conf', 0.25)
        self.declare_parameter('publish_empty', True)
        self.declare_parameter('max_instances', 8)
        self.declare_parameter('dbscan_eps', 0.012)
        self.declare_parameter('dbscan_min_points', 100)
        self.declare_parameter('sor_nb_neighbors', 25)
        self.declare_parameter('sor_std_ratio', 1.5)
        self.declare_parameter('world_up', [0.0, -1.0, 0.0])
        self.declare_parameter('visualization_topic', '/mushroom/visualization')
        self.declare_parameter('enable_visualization_topic', True)
        self.declare_parameter('viz_rate_hz', 3.0)
        self.declare_parameter('viz_axis_length_m', 0.06)

        self.color_topic = self.get_parameter('color_topic').value
        self.depth_topic = self.get_parameter('depth_topic').value
        self.camera_info_topic = self.get_parameter('camera_info_topic').value
        self.detection_request_topic = self.get_parameter('detection_request_topic').value
        self.detected_json_topic = self.get_parameter('detected_json_topic').value
        self.base_frame = self.get_parameter('base_frame').value
        self.camera_frame = self.get_parameter('camera_frame').value
        self.target_class = self.get_parameter('target_class').value
        self.model_path = self.get_parameter('model_path').value
        self.ultralytics_root = self.get_parameter('ultralytics_root').value
        self.depth_max_m = float(self.get_parameter('depth_max_m').value)
        self.min_conf = float(self.get_parameter('min_conf').value)
        self.publish_empty = bool(self.get_parameter('publish_empty').value)
        self.max_instances = int(self.get_parameter('max_instances').value)
        self.dbscan_eps = float(self.get_parameter('dbscan_eps').value)
        self.dbscan_min_points = int(self.get_parameter('dbscan_min_points').value)
        self.sor_nb_neighbors = int(self.get_parameter('sor_nb_neighbors').value)
        self.sor_std_ratio = float(self.get_parameter('sor_std_ratio').value)
        self.world_up = np.array(self.get_parameter('world_up').value, dtype=np.float64)
        self.visualization_topic = self.get_parameter('visualization_topic').value
        self.enable_visualization_topic = bool(self.get_parameter('enable_visualization_topic').value)
        self.viz_rate_hz = float(self.get_parameter('viz_rate_hz').value)
        self.viz_axis_length_m = float(self.get_parameter('viz_axis_length_m').value)

        self.bridge = CvBridge()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self._lock = threading.Lock()
        self._busy = False
        self._latest_rgb = None
        self._latest_depth = None
        self._latest_header = None
        self._last_viz_sec = 0.0
        self._infer_lock = threading.Lock()

        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        self._init_model()

        self.sub_info = self.create_subscription(
            CameraInfo, self.camera_info_topic, self._camera_info_cb, 10
        )

        self.rgb_sub = Subscriber(self, Image, self.color_topic)
        self.depth_sub = Subscriber(self, Image, self.depth_topic)
        self.sync = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub], queue_size=10, slop=0.1
        )
        self.sync.registerCallback(self._sync_cb)

        self.req_sub = self.create_subscription(
            String, self.detection_request_topic, self._request_cb, 10
        )
        self.pub = self.create_publisher(String, self.detected_json_topic, 10)
        self.viz_pub = self.create_publisher(Image, self.visualization_topic, 10)

        self.get_logger().info(
            f'mushroom_pose_node started. input=({self.color_topic}, {self.depth_topic}), '
            f'request={self.detection_request_topic}, output={self.detected_json_topic}'
        )

    def _init_model(self) -> None:
        if self.ultralytics_root and os.path.isdir(self.ultralytics_root):
            import sys
            if self.ultralytics_root not in sys.path:
                sys.path.append(self.ultralytics_root)

        from ultralytics import YOLO  # pylint: disable=import-outside-toplevel

        self.model = YOLO(self.model_path)
        names = getattr(self.model, 'names', {}) or {}
        self.name_to_id = {str(v).lower(): int(k) for k, v in names.items()}
        self.target_class_id = self.name_to_id.get(str(self.target_class).lower())
        if self.target_class_id is None:
            self.get_logger().warn(
                f"target_class '{self.target_class}' not in model names {list(self.name_to_id.keys())}, will keep all classes"
            )
        else:
            self.get_logger().info(f"target_class '{self.target_class}' -> class_id={self.target_class_id}")

    def _camera_info_cb(self, msg: CameraInfo) -> None:
        self.fx = float(msg.k[0])
        self.fy = float(msg.k[4])
        self.cx = float(msg.k[2])
        self.cy = float(msg.k[5])

    def _sync_cb(self, rgb_msg: Image, depth_msg: Image) -> None:
        try:
            rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
            depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        except Exception as exc:
            self.get_logger().error(f'cv_bridge convert failed: {exc}')
            return

        with self._lock:
            self._latest_rgb = rgb
            self._latest_depth = depth
            self._latest_header = rgb_msg.header

        self._maybe_publish_visualization(rgb, depth, rgb_msg.header)

    def _request_cb(self, msg: String) -> None:
        with self._lock:
            if self._busy:
                self.get_logger().warn('pipeline is busy, skip this request')
                return
            self._busy = True

        thread = threading.Thread(target=self._process_request, args=(msg.data,), daemon=True)
        thread.start()

    def _process_request(self, raw_request: str) -> None:
        try:
            site = self._parse_site(raw_request)
            frame = self._snapshot_frame()
            if frame is None:
                self.get_logger().warn('no synchronized frame yet')
                self._publish(site, [])
                return

            rgb_bgr, depth, header = frame
            if any(v is None for v in [self.fx, self.fy, self.cx, self.cy]):
                self.get_logger().warn('camera_info not received yet')
                self._publish(site, [])
                return

            detections = self._run_pipeline(rgb_bgr, depth, header)
            self._publish(site, detections)
        except Exception as exc:
            self.get_logger().error(f'pipeline exception: {exc}')
            self._publish(self._parse_site(raw_request), [])
        finally:
            with self._lock:
                self._busy = False

    def _snapshot_frame(self) -> Optional[Tuple[np.ndarray, np.ndarray, Any]]:
        with self._lock:
            if self._latest_rgb is None or self._latest_depth is None or self._latest_header is None:
                return None
            return self._latest_rgb.copy(), self._latest_depth.copy(), self._latest_header

    def _parse_site(self, raw: str) -> str:
        s = (raw or '').strip()
        if not s:
            return 'unknown'
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                return str(obj.get('site') or obj.get('request_id') or s)
        except Exception:
            pass
        return s

    def _run_pipeline(self, rgb_bgr: np.ndarray, depth_raw: np.ndarray, header: Any) -> List[Dict[str, Any]]:
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        depth_m = self._depth_to_m(depth_raw)
        result = self._predict(rgb_bgr)

        if result.masks is None or result.masks.data is None or result.boxes is None:
            return []

        masks = result.masks.data.cpu().numpy()
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        clss = result.boxes.cls.cpu().numpy().astype(int)

        order = np.argsort(-confs)
        detections = []

        for idx in order[: self.max_instances]:
            cls_id = int(clss[idx])
            label = str(result.names.get(cls_id, cls_id))
            conf = float(confs[idx])

            if self.target_class_id is not None and cls_id != self.target_class_id:
                continue

            mask = (masks[idx] > 0.5).astype(np.uint8) * 255
            mask_clean = self._refine_mask(mask, ksize=5)
            valid = self._depth_valid_mask(depth_m)
            mask_valid = cv2.bitwise_and(mask_clean, valid)

            points, colors = self._backproject(rgb, depth_m, mask_valid)
            if points.shape[0] < 300:
                continue

            pcd = self._points_to_pcd(points, colors)
            pcd = self._auto_crop(pcd)
            if len(pcd.points) < 300:
                continue

            pcd, _ = pcd.remove_statistical_outlier(
                nb_neighbors=self.sor_nb_neighbors,
                std_ratio=self.sor_std_ratio,
            )
            if len(pcd.points) < 300:
                continue

            pcd = self._keep_largest_cluster(pcd, self.dbscan_eps, self.dbscan_min_points)
            if len(pcd.points) < 300:
                continue

            pts = np.asarray(pcd.points)
            sphere = self._fit_sphere(pts)
            if sphere is None:
                continue

            center_cam = sphere['center']
            quat_cam = self._compute_grasp_quat_from_pca(pts)

            tf = self._lookup_tf(header)
            if tf is None:
                continue

            center_base = self._transform_point(center_cam, tf)
            quat_base = self._transform_quaternion(quat_cam, tf)

            x1, y1, x2, y2 = boxes[idx]
            u = float((x1 + x2) * 0.5)
            v = float((y1 + y2) * 0.5)

            det = {
                'class_id': cls_id,
                'label': label,
                'confidence': conf,
                'position': {
                    'x': float(center_base[0]),
                    'y': float(center_base[1]),
                    'z': float(center_base[2]),
                },
                'center_px': {'u': u, 'v': v},
                'bbox_px': {'w': float(x2 - x1), 'h': float(y2 - y1)},
                'measured': True,
                'grasp_quality': conf,
                'grasp_pose': {
                    'position': {
                        'x': float(center_base[0]),
                        'y': float(center_base[1]),
                        'z': float(center_base[2]),
                    },
                    'orientation': {
                        'x': float(quat_base[0]),
                        'y': float(quat_base[1]),
                        'z': float(quat_base[2]),
                        'w': float(quat_base[3]),
                    }
                }
            }
            detections.append(det)

        return detections

    def _predict(self, rgb_bgr: np.ndarray):
        with self._infer_lock:
            return self.model.predict(
                source=rgb_bgr,
                save=False,
                imgsz=640,
                conf=self.min_conf,
                iou=0.45,
                show=False,
                retina_masks=True,
                verbose=False,
            )[0]

    def _maybe_publish_visualization(self, rgb_bgr: np.ndarray, depth_raw: np.ndarray, header: Any) -> None:
        if not self.enable_visualization_topic:
            return
        if any(v is None for v in [self.fx, self.fy, self.cx, self.cy]):
            return
        now_sec = self.get_clock().now().nanoseconds / 1e9
        if self.viz_rate_hz <= 0:
            return
        if now_sec - self._last_viz_sec < (1.0 / self.viz_rate_hz):
            return
        if self._busy:
            return
        self._last_viz_sec = now_sec

        try:
            vis = self._build_visualization_frame(rgb_bgr, depth_raw)
            msg = self.bridge.cv2_to_imgmsg(vis, encoding='bgr8')
            msg.header = header
            self.viz_pub.publish(msg)
        except Exception as exc:
            self.get_logger().warn(f'visualization publish failed: {exc}')

    def _build_visualization_frame(self, rgb_bgr: np.ndarray, depth_raw: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        depth_m = self._depth_to_m(depth_raw)
        result = self._predict(rgb_bgr)
        vis = rgb_bgr.copy()

        if result.masks is None or result.masks.data is None or result.boxes is None:
            cv2.putText(vis, 'No target', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 180, 255), 2)
            return vis

        masks = result.masks.data.cpu().numpy()
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        clss = result.boxes.cls.cpu().numpy().astype(int)
        order = np.argsort(-confs)

        for idx in order[: self.max_instances]:
            cls_id = int(clss[idx])
            label = str(result.names.get(cls_id, cls_id))
            conf = float(confs[idx])
            if self.target_class_id is not None and cls_id != self.target_class_id:
                continue

            mask = (masks[idx] > 0.5).astype(np.uint8) * 255
            mask_clean = self._refine_mask(mask, ksize=5)
            valid = self._depth_valid_mask(depth_m)
            mask_valid = cv2.bitwise_and(mask_clean, valid)

            points, _ = self._backproject(rgb, depth_m, mask_valid)
            if points.shape[0] < 120:
                continue

            pcd = self._points_to_pcd(points, np.zeros((points.shape[0], 3), dtype=np.float32))
            pcd = self._auto_crop(pcd)
            if len(pcd.points) < 120:
                continue
            pcd, _ = pcd.remove_statistical_outlier(
                nb_neighbors=self.sor_nb_neighbors,
                std_ratio=self.sor_std_ratio,
            )
            if len(pcd.points) < 120:
                continue
            pcd = self._keep_largest_cluster(pcd, self.dbscan_eps, self.dbscan_min_points)
            if len(pcd.points) < 120:
                continue

            pts = np.asarray(pcd.points)
            center, axis = self._extract_main_axis(pts)
            axis_end = center + axis * self.viz_axis_length_m
            uv_center = self._project_cam_point_to_uv(center)
            uv_axis = self._project_cam_point_to_uv(axis_end)
            if uv_center is None or uv_axis is None:
                continue

            x1, y1, x2, y2 = boxes[idx].astype(int)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (60, 220, 60), 2)

            mask_overlay = np.zeros_like(vis, dtype=np.uint8)
            mask_overlay[:, :, 1] = (mask_valid > 0).astype(np.uint8) * 120
            vis = cv2.addWeighted(vis, 1.0, mask_overlay, 0.35, 0.0)

            cv2.circle(vis, uv_center, 5, (0, 0, 255), -1)
            cv2.arrowedLine(vis, uv_center, uv_axis, (255, 0, 0), 2, tipLength=0.2)
            text1 = f'{label} {conf:.2f}'
            text2 = f'C: ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})m'
            cv2.putText(vis, text1, (x1, max(20, y1 - 28)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 255, 20), 2)
            cv2.putText(vis, text2, (x1, max(40, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 220, 255), 2)

        return vis

    def _publish(self, site: str, detections: List[Dict[str, Any]]) -> None:
        if not detections and not self.publish_empty:
            return
        payload = {'site': site, 'detections': detections}
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.pub.publish(msg)
        self.get_logger().info(f'published {len(detections)} detections for site={site}')

    @staticmethod
    def _keep_largest_cc(mask_u8: np.ndarray) -> np.ndarray:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
        if num_labels <= 1:
            return mask_u8
        areas = stats[1:, cv2.CC_STAT_AREA]
        max_i = 1 + int(np.argmax(areas))
        out = np.zeros_like(mask_u8)
        out[labels == max_i] = 255
        return out

    def _refine_mask(self, mask_u8: np.ndarray, ksize: int = 5) -> np.ndarray:
        mask_u8 = (mask_u8 > 0).astype(np.uint8) * 255
        mask_u8 = self._keep_largest_cc(mask_u8)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, k, iterations=1)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, k, iterations=1)
        return mask_u8

    def _depth_to_m(self, depth: np.ndarray) -> np.ndarray:
        if depth.dtype == np.uint16:
            return depth.astype(np.float32) * 0.001
        return depth.astype(np.float32)

    def _depth_valid_mask(self, depth_m: np.ndarray) -> np.ndarray:
        valid = (depth_m > 0) & np.isfinite(depth_m) & (depth_m < self.depth_max_m)
        return (valid.astype(np.uint8) * 255)

    def _backproject(self, rgb: np.ndarray, depth_m: np.ndarray, mask_u8: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        ys, xs = np.where(mask_u8 > 0)
        if xs.size == 0:
            return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32)

        z = depth_m[ys, xs]
        valid = (z > 0) & np.isfinite(z) & (z < self.depth_max_m)
        xs = xs[valid]
        ys = ys[valid]
        z = z[valid]
        if xs.size == 0:
            return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32)

        x = (xs.astype(np.float32) - self.cx) * z / self.fx
        y = (ys.astype(np.float32) - self.cy) * z / self.fy
        points = np.stack([x, y, z], axis=1).astype(np.float32)
        colors = rgb[ys, xs].astype(np.float32) / 255.0
        return points, colors

    @staticmethod
    def _points_to_pcd(points: np.ndarray, colors: np.ndarray) -> o3d.geometry.PointCloud:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
        return pcd

    @staticmethod
    def _auto_crop(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        pts = np.asarray(pcd.points)
        lo = np.percentile(pts, 2.0, axis=0)
        hi = np.percentile(pts, 98.0, axis=0)
        span = np.maximum(hi - lo, 1e-6)
        margin = span * 0.08
        lo -= margin
        hi += margin

        keep = np.all((pts >= lo) & (pts <= hi), axis=1)
        idx = np.where(keep)[0]
        if idx.size == 0:
            return o3d.geometry.PointCloud()
        return pcd.select_by_index(idx)

    @staticmethod
    def _keep_largest_cluster(pcd: o3d.geometry.PointCloud, eps: float, min_points: int) -> o3d.geometry.PointCloud:
        labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
        if labels.size == 0 or labels.max() < 0:
            return o3d.geometry.PointCloud()
        valid = labels >= 0
        labs = labels[valid]
        counts = np.bincount(labs)
        best_label = int(np.argmax(counts))
        idx = np.where(labels == best_label)[0]
        return pcd.select_by_index(idx)

    @staticmethod
    def _sphere_from_4pts(p1, p2, p3, p4):
        p1, p2, p3, p4 = map(np.asarray, (p1, p2, p3, p4))
        a_mat = np.vstack([2 * (p2 - p1), 2 * (p3 - p1), 2 * (p4 - p1)])
        b_vec = np.array([
            np.dot(p2, p2) - np.dot(p1, p1),
            np.dot(p3, p3) - np.dot(p1, p1),
            np.dot(p4, p4) - np.dot(p1, p1),
        ], dtype=np.float64)
        if abs(np.linalg.det(a_mat)) < 1e-10:
            return None, None
        c = np.linalg.solve(a_mat, b_vec)
        r = float(np.linalg.norm(c - p1))
        if not np.isfinite(r) or r <= 1e-6:
            return None, None
        return c, r

    @staticmethod
    def _sphere_residuals(points, center, radius):
        d = np.linalg.norm(points - center[None, :], axis=1)
        return np.abs(d - radius)

    def _fit_sphere(self, points: np.ndarray) -> Optional[Dict[str, Any]]:
        n = points.shape[0]
        if n < 80:
            return None

        rng = np.random.default_rng(0)
        best = None
        best_mask = None
        best_cnt = -1

        for _ in range(600):
            idx = rng.choice(n, size=4, replace=False)
            center, radius = self._sphere_from_4pts(points[idx[0]], points[idx[1]], points[idx[2]], points[idx[3]])
            if center is None:
                continue
            res = self._sphere_residuals(points, center, radius)
            inliers = res < 0.006
            cnt = int(inliers.sum())
            if cnt > best_cnt:
                best_cnt = cnt
                best = (center, radius)
                best_mask = inliers

        if best is None or best_cnt < int(0.55 * n):
            return None

        pts_in = points[best_mask]
        x = pts_in.astype(np.float64)
        a_mat = np.column_stack([x, np.ones((x.shape[0], 1))])
        b_vec = -(x[:, 0] ** 2 + x[:, 1] ** 2 + x[:, 2] ** 2)
        params, *_ = np.linalg.lstsq(a_mat, b_vec, rcond=None)
        a, b, c, d = params
        center = np.array([-a / 2.0, -b / 2.0, -c / 2.0], dtype=np.float64)
        r2 = (a * a + b * b + c * c) / 4.0 - d
        if r2 <= 0 or not np.isfinite(r2):
            return None
        radius = float(np.sqrt(r2))
        return {'center': center, 'radius': radius}

    def _compute_grasp_quat_from_pca(self, points: np.ndarray) -> np.ndarray:
        _, axis = self._extract_main_axis(points)

        grasp_approach = -axis
        z_ee = grasp_approach / (np.linalg.norm(grasp_approach) + 1e-12)
        up = self.world_up / (np.linalg.norm(self.world_up) + 1e-12)
        x_ee = np.cross(up, z_ee)
        if np.linalg.norm(x_ee) < 1e-6:
            x_ee = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        x_ee /= (np.linalg.norm(x_ee) + 1e-12)
        y_ee = np.cross(z_ee, x_ee)
        y_ee /= (np.linalg.norm(y_ee) + 1e-12)
        r_mat = np.column_stack([x_ee, y_ee, z_ee])
        quat = R.from_matrix(r_mat).as_quat()  # xyzw
        return quat

    def _extract_main_axis(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        center = points.mean(axis=0)
        pts_c = points - center
        cov = np.cov(pts_c.T)
        _, eigvecs = np.linalg.eigh(cov)

        up = self.world_up / (np.linalg.norm(self.world_up) + 1e-12)
        e1 = eigvecs[:, -1] / (np.linalg.norm(eigvecs[:, -1]) + 1e-12)
        e2 = eigvecs[:, -2] / (np.linalg.norm(eigvecs[:, -2]) + 1e-12)
        s1 = abs(float(np.dot(e1, up)))
        s2 = abs(float(np.dot(e2, up)))
        axis = e1 if s1 >= s2 else e2
        if float(np.dot(axis, up)) < 0:
            axis = -axis
        return center, axis

    def _project_cam_point_to_uv(self, p_cam: np.ndarray) -> Optional[Tuple[int, int]]:
        z = float(p_cam[2])
        if z <= 1e-6:
            return None
        u = int((float(p_cam[0]) * self.fx / z) + self.cx)
        v = int((float(p_cam[1]) * self.fy / z) + self.cy)
        return (u, v)

    def _lookup_tf(self, header: Any):
        source = self.camera_frame or header.frame_id
        try:
            return self.tf_buffer.lookup_transform(
                self.base_frame,
                source,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.3),
            )
        except TransformException as exc:
            self.get_logger().warn(f'TF lookup failed {self.base_frame}<-{source}: {exc}')
            return None

    @staticmethod
    def _transform_point(point_xyz: np.ndarray, tf_msg) -> np.ndarray:
        t = tf_msg.transform.translation
        q = tf_msg.transform.rotation
        rot = R.from_quat([q.x, q.y, q.z, q.w])
        p = rot.apply(point_xyz) + np.array([t.x, t.y, t.z], dtype=np.float64)
        return p

    @staticmethod
    def _transform_quaternion(quat_xyzw: np.ndarray, tf_msg) -> np.ndarray:
        q = tf_msg.transform.rotation
        q_tf = R.from_quat([q.x, q.y, q.z, q.w])
        q_obj = R.from_quat(quat_xyzw)
        q_out = (q_tf * q_obj).as_quat()
        return q_out


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MushroomPoseNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
