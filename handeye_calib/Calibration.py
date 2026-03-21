#!/usr/bin/env python3
"""
纯 OpenCV 手眼标定脚本
无需 ROS2 / MoveIt2 / RViz2
支持从文件夹批量读取图片和位姿数据
"""

import cv2
import numpy as np
import json
import os
import glob
import yaml
import re

try:
    from scipy.spatial.transform import Rotation
except ImportError:
    Rotation = None


def _natural_sort_key(path_or_name):
    """自然排序键：'2.jpg' < '10.jpg'。"""
    name = os.path.basename(str(path_or_name))
    stem, _ = os.path.splitext(name)
    parts = re.split(r'(\d+)', stem)
    key = []
    for part in parts:
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part.lower())
    return key

class HandEyeCalibration:

    def __init__(self, board_size=(9, 6), square_size=0.024):
        # 1. 棋盘格参数
        self.board_size = board_size  # 棋盘格内角点数量（列, 行），如(9,6)
        self.square_size = square_size  # 每个格子的实际物理边长（单位：米）
        # 生成所有内角点的三维坐标（Z=0，平面上）
        self.objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)  # N×3，N为内角点总数
        # 填充X、Y坐标，排列顺序与OpenCV角点检测一致(0,0),(0,1),(0,2)...(9,5)
        self.objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
        # 按实际格子边长缩放，得到真实世界坐标（单位：米） ,乘上小格子边长
        self.objp *= square_size

        # 2. 相机内参
        self.camera_matrix = None
        self.dist_coeffs = None
        self.image_width = None
        self.image_height = None

        # 3. 样本存储
        self.R_gripper2base_list = []
        self.t_gripper2base_list = []
        self.R_target2cam_list = []
        self.t_target2cam_list = []
        self.sample_names = []
        self.pnp_reproj_errors = []
        self._last_reproj_error = None
        self.max_reproj_error_px = 3
        
        self.eye_in_hand = False

    def set_eye_mode(self, eye_in_hand: bool):
        """设置手眼模式。True=眼在手上, False=眼在手外。"""
        self.eye_in_hand = bool(eye_in_hand)
        mode = "Eye-in-Hand" if self.eye_in_hand else "Eye-to-Hand"
        print(f"🔧 手眼模式已设置: {mode}")

    def load_camera_intrinsics(self, json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)

        # 兼容嵌套结构：例如 {"rgb_camera_info": {...}} / {"camera_info": {...}}
        if isinstance(data, dict):
            if isinstance(data.get('rgb_camera_info'), dict):
                data = data['rgb_camera_info']
            elif isinstance(data.get('camera_info'), dict):
                data = data['camera_info']
            elif len(data) == 1:
                only_value = next(iter(data.values()))
                if isinstance(only_value, dict):
                    data = only_value

        def _get_first_existing(d, keys):
            for key in keys:
                if key in d:
                    return d[key]
            return None
        # 兼容多种常见相机参数格式：
        # 1) {'K': [...], 'D': [...], 'width': ..., 'height': ...}
        # 2) {'k': [...], 'd': [...], 'width': ..., 'height': ...}  (ROS2 常见导出)
        # 3) {'camera_matrix': {'data': [...]}, 'distortion_coefficients': {'data': [...]}, ...}
        K = _get_first_existing(data, ['K', 'k'])
        if K is None and isinstance(data.get('camera_matrix'), dict):
            K = data['camera_matrix'].get('data')

        D = _get_first_existing(data, ['D', 'd'])
        if D is None and isinstance(data.get('distortion_coefficients'), dict):
            D = data['distortion_coefficients'].get('data')

        if K is None:
            raise KeyError("相机内参文件缺少 K/k 或 camera_matrix.data 字段")
        if D is None:
            D = [0.0, 0.0, 0.0, 0.0, 0.0]

        self.camera_matrix = np.array([
            [K[0], K[1], K[2]],
            [K[3], K[4], K[5]],
            [K[6], K[7], K[8]]
        ], dtype=np.float64)
        self.dist_coeffs = np.array(D, dtype=np.float64)
        self.image_width = data.get('width', data.get('image_width'))
        self.image_height = data.get('height', data.get('image_height'))
        print(f"✅ 相机内参已加载: {json_path}")

    def detect_target(self, image, print_pose=False, sample_name=''):
        """
            return: success, rvec(3,), tvec(3,) 返回标定板相对相机的位姿
            可选：print_pose=True 时打印位姿。
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        if self.camera_matrix is None:
            raise RuntimeError("相机内参未设置：请先调用 load_camera_intrinsics 从文件加载内参")
        camera_matrix = self.camera_matrix
        ret, corners = cv2.findChessboardCorners(gray, self.board_size, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if not ret:
            self._last_reproj_error = None
            return False, None, None
        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        success, rvec, tvec = cv2.solvePnP(self.objp, corners_refined, camera_matrix, self.dist_coeffs)
        if not success:
            self._last_reproj_error = None
            return False, None, None

        # 记录当前样本PnP重投影误差（像素）
        projected, _ = cv2.projectPoints(self.objp, rvec, tvec, camera_matrix, self.dist_coeffs)
        reproj_error = np.mean(np.linalg.norm(projected.reshape(-1, 2) - corners_refined.reshape(-1, 2), axis=1))
        self._last_reproj_error = float(reproj_error)

        if print_pose:
            R_target2cam, _ = cv2.Rodrigues(rvec)
            sample_tag = f"[{sample_name}] " if sample_name else ""
            print(f"{sample_tag}target->camera rvec(rad): {rvec.flatten()}")
            print(f"{sample_tag}target->camera tvec(m): {tvec.flatten()}")
            print(f"{sample_tag}R_target2cam:\n{R_target2cam}")

        return True, rvec.flatten(), tvec.flatten()
    
    def add_sample(self, gripper_pose, image, name=""):
        """
        添加一个标定样本
        
        Args:
            gripper_pose: 机械臂末端位姿，可以是：
                - 4x4 齐次变换矩阵
                - dict: {'x', 'y', 'z', 'rx', 'ry', 'rz'} (位置 + 欧拉角，弧度)
                - dict: {'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'} (位置 + 四元数)
            image: 相机图像 (numpy array)
        
        Returns:
            success: bool
        """
        # 1. 检测标定板位姿
        success, rvec, tvec = self.detect_target(image, print_pose=True, sample_name=name)
        if not success:
            print("❌ 标定板检测失败！")
            return False

        if self._last_reproj_error is not None and self._last_reproj_error > self.max_reproj_error_px:
            print(f"❌ 丢弃样本 {name}: PnP重投影误差={self._last_reproj_error:.3f}px > {self.max_reproj_error_px:.3f}px")
            return False
        
        # 转换为旋转矩阵
        R_target2cam, _ = cv2.Rodrigues(rvec)
        
        # 2. 解析机械臂位姿
        # 眼在手外：输入视为 base->gripper，需要取逆为 gripper->base
        # 眼在手上：输入视为 gripper->base，不取逆
        R_in, t_in = self._parse_gripper_pose(gripper_pose)
        if self.eye_in_hand:
            R_gripper2base = np.asarray(R_in, dtype=np.float64)
            t_gripper2base = np.asarray(t_in, dtype=np.float64).reshape(3, 1)
            print(f"[DEBUG] gripper_pose(gripper->base) -> R:\n{R_gripper2base}\n[DEBUG] t: {np.asarray(t_gripper2base).reshape(3)}")
        else:
            R_gripper2base, t_gripper2base = self._invert_rt(R_in, t_in)
            print(f"[DEBUG] gripper_pose(base->gripper) -> R:\n{R_in}\n[DEBUG] t: {np.asarray(t_in).reshape(3)}")
            print(f"[DEBUG] converted gripper->base -> R:\n{R_gripper2base}\n[DEBUG] t: {np.asarray(t_gripper2base).reshape(3)}")
        print(f"[DEBUG] 输入 gripper_pose -> {gripper_pose}")
        
        # 3. 保存样本
        self.R_gripper2base_list.append(R_gripper2base)
        self.t_gripper2base_list.append(np.asarray(t_gripper2base, dtype=np.float64).reshape(3, 1))
        self.R_target2cam_list.append(R_target2cam)
        self.t_target2cam_list.append(tvec.reshape(3, 1))
        self.sample_names.append(name)
        if self._last_reproj_error is None:
            self.pnp_reproj_errors.append(np.nan)
            print(f"✅ 样本 {len(self.R_gripper2base_list)} 已添加: {name}")
        else:
            self.pnp_reproj_errors.append(self._last_reproj_error)
            print(f"✅ 样本 {len(self.R_gripper2base_list)} 已添加: {name} | PnP重投影误差={self._last_reproj_error:.3f}px")
        return True
    
    def _parse_gripper_pose(self, pose):
        """解析机械臂位姿为旋转矩阵和平移向量"""
        if isinstance(pose, np.ndarray) and pose.shape == (4, 4):
            # 4x4 齐次变换矩阵
            R = pose[:3, :3]
            t = pose[:3, 3]
        elif isinstance(pose, dict):
            t = np.array([pose['x'], pose['y'], pose['z']])
            if 'qw' in pose:
                # 四元数
                R = self._quaternion_to_rotation_matrix(
                    pose['qx'], pose['qy'], pose['qz'], pose['qw']
                )
            else:
                # 欧拉角 (rx, ry, rz) - 弧度
                R = self._euler_to_rotation_matrix(
                    pose['rx'], pose['ry'], pose['rz']
                )
        else:
            raise ValueError("不支持的位姿格式")
        
        return R, t
    
    def _quaternion_to_rotation_matrix(self, qx, qy, qz, qw):
        """四元数转旋转矩阵"""
        if Rotation is None:
            raise ImportError("缺少 scipy，请先安装: pip install scipy")
        quat_xyzw = np.array([qx, qy, qz, qw], dtype=np.float64)
        return Rotation.from_quat(quat_xyzw).as_matrix()
    
    def _euler_to_rotation_matrix(self, rx, ry, rz):
        """欧拉角(XYZ顺序)转旋转矩阵"""
        if Rotation is None:
            raise ImportError("缺少 scipy，请先安装: pip install scipy")
        return Rotation.from_euler('xyz', [rx, ry, rz], degrees=False).as_matrix()

    def _rotation_matrix_to_quaternion(self, R):
        """旋转矩阵转四元数，返回 [qx, qy, qz, qw]"""
        if Rotation is None:
            raise ImportError("缺少 scipy，请先安装: pip install scipy")
        R = np.asarray(R, dtype=np.float64)
        return Rotation.from_matrix(R).as_quat().astype(np.float64)

    def _to_homogeneous(self, R, t):
        """将 R(3x3), t(3,) 或 (3,1) 转为 4x4 齐次变换矩阵。"""
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = np.asarray(R, dtype=np.float64)
        T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
        return T

    def _from_homogeneous(self, T):
        """将 4x4 齐次变换矩阵拆分为 R(3x3), t(3,1)。"""
        T = np.asarray(T, dtype=np.float64)
        R = T[:3, :3]
        t = T[:3, 3].reshape(3, 1)
        return R, t

    def _invert_rt(self, R, t):
        """位姿求逆，输入/输出均为 (R, t)。"""
        R = np.asarray(R, dtype=np.float64)
        t = np.asarray(t, dtype=np.float64).reshape(3, 1)
        R_inv = R.T
        t_inv = -R_inv @ t
        return R_inv, t_inv

    def _compose_rt(self, R_ab, t_ab, R_bc, t_bc):
        """位姿复合：T_ac = T_ab * T_bc。"""
        R_ab = np.asarray(R_ab, dtype=np.float64)
        t_ab = np.asarray(t_ab, dtype=np.float64).reshape(3, 1)
        R_bc = np.asarray(R_bc, dtype=np.float64)
        t_bc = np.asarray(t_bc, dtype=np.float64).reshape(3, 1)
        R_ac = R_ab @ R_bc
        t_ac = R_ab @ t_bc + t_ab
        return R_ac, t_ac

    
    def solve(self, method='Tsai1989'):
        """
        求解手眼标定
        
        Args:
            method: 算法名称
                - 'Tsai1989'
                - 'Park1994' 
                - 'Horaud1995'
                - 'Andreff1999'
                - 'Daniilidis1998'
        Returns:
            R_cam2gripper: 旋转矩阵 (3x3)
            t_cam2gripper: 平移向量 (3x1)
        """
        # 选择算法
        methods = {
            'Tsai1989': cv2.CALIB_HAND_EYE_TSAI,
            'Park1994': cv2.CALIB_HAND_EYE_PARK,
            'Horaud1995': cv2.CALIB_HAND_EYE_HORAUD,
            'Andreff1999': cv2.CALIB_HAND_EYE_ANDREFF,
            'Daniilidis1998': cv2.CALIB_HAND_EYE_DANIILIDIS,
        }
        cv_method = methods.get(method, cv2.CALIB_HAND_EYE_TSAI)
        result_name = "cam2gripper" if self.eye_in_hand else "cam2base"
        R_result, t_result = cv2.calibrateHandEye(
            self.R_gripper2base_list,
            self.t_gripper2base_list,
            self.R_target2cam_list,
            self.t_target2cam_list,
            method=cv_method
        )
        print("\n========== 标定结果 ==========")
        print(f"使用算法: {method}")
        print(f"模式: {'Eye-in-Hand' if self.eye_in_hand else 'Eye-to-Hand'}")
        print(f"样本数量: {len(self.R_gripper2base_list)}")
        print(f"\n变换: {result_name}")
        print(f"\n旋转矩阵 R:\n{R_result}")
        print(f"\n平移向量 t (米):\n{t_result.flatten()}")
        rvec, _ = cv2.Rodrigues(R_result)
        qx, qy, qz, qw = self._rotation_matrix_to_quaternion(R_result)
        print(f"\n旋转向量 (弧度): {rvec.flatten()}")
        print(f"旋转向量 (角度): {np.degrees(rvec.flatten())}")
        print(f"四元数 [qx, qy, qz, qw]: [{qx:.8f}, {qy:.8f}, {qz:.8f}, {qw:.8f}]")

        if self.pnp_reproj_errors:
            valid_err = np.array([e for e in self.pnp_reproj_errors if np.isfinite(e)], dtype=np.float64)
            if valid_err.size > 0:
                print(f"PnP重投影误差统计: mean={valid_err.mean():.3f}px, median={np.median(valid_err):.3f}px, max={valid_err.max():.3f}px")

        return R_result, t_result

    def save_result(self, filename, R, t, method=None):
        """保存标定结果到 JSON 文件"""
        T = self._to_homogeneous(R, t)
        qx, qy, qz, qw = self._rotation_matrix_to_quaternion(R)

        result = {
            'method': method,
            'mode': 'Eye-in-Hand' if self.eye_in_hand else 'Eye-to-Hand',
            'transformation_matrix': T.tolist(),
            'rotation_matrix': np.asarray(R, dtype=np.float64).tolist(),
            'quaternion': {
                'x': float(qx),
                'y': float(qy),
                'z': float(qz),
                'w': float(qw),
            },
            'translation': np.asarray(t, dtype=np.float64).reshape(3).tolist(),
            'num_samples': len(self.R_gripper2base_list),
        }

        with open(filename, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"✅ 结果已保存到: {filename}")
    
    def clear_samples(self):
        """清空所有样本"""
        self.R_gripper2base_list.clear()
        self.t_gripper2base_list.clear()
        self.R_target2cam_list.clear()
        self.t_target2cam_list.clear()
        self.sample_names.clear()
        self.pnp_reproj_errors.clear()
        print("已清空所有样本")

    def load_from_folder(self, folder_path, pose_file='poses.json', robot_pose_key='openarm_left_hand_tcp'):
        """
        从文件夹批量加载图片和位姿数据
        
        文件夹结构示例:
        data_folder/
        ├── poses.json (或 poses.yaml)    # 位姿文件
        ├── 001.png (或 .jpg)             # 图片文件
        ├── 002.png
        ├── 003.png
        └── ...
        
        poses.json 格式示例:
        {
            "001": {"x": 0.3, "y": 0.1, "z": 0.4, "rx": 0.0, "ry": 3.14, "rz": 0.0},
            "002": {"x": 0.35, "y": 0.05, "z": 0.38, "rx": 0.1, "ry": 3.0, "rz": 0.2},
            ...
        }
        
        或者使用四元数:
        {
            "001": {"x": 0.3, "y": 0.1, "z": 0.4, "qx": 0, "qy": 0, "qz": 0, "qw": 1},
            ...
        }
        
        或者使用 4x4 矩阵:
        {
            "001": {"matrix": [[r11,r12,r13,tx], [r21,r22,r23,ty], [r31,r32,r33,tz], [0,0,0,1]]},
            ...
        }
        
        Args:
            folder_path: 数据文件夹路径
            pose_file: 位姿文件名 (支持 .json 或 .yaml)
            robot_pose_key: 当位姿文件为列表结构时，从 poses 字段中选取的机械臂链路名
        
        Returns:
            成功加载的样本数量
        """
        folder_path = os.path.abspath(folder_path)

        def _resolve_pose_path():
            direct_path = os.path.join(folder_path, pose_file)
            if os.path.exists(direct_path):
                return direct_path
            for ext in ['.json', '.yaml', '.yml']:
                candidate = os.path.join(folder_path, 'poses' + ext)
                if os.path.exists(candidate):
                    return candidate
            raise FileNotFoundError(f"找不到位姿文件: {direct_path}")

        def _load_pose_data(path):
            if path.endswith('.json'):
                with open(path, 'r') as f:
                    return json.load(f)
            if path.endswith(('.yaml', '.yml')):
                with open(path, 'r') as f:
                    return yaml.safe_load(f)
            raise ValueError(f"不支持的位姿文件格式: {path}")

        def _build_pose_map(data):
            if isinstance(data, dict):
                return data
            if isinstance(data, list):
                mapped = {}
                for item in data:
                    if not isinstance(item, dict):
                        continue
                    image_name = item.get('image')
                    if not image_name:
                        continue
                    image_key = os.path.splitext(os.path.basename(str(image_name)))[0]
                    all_poses = item.get('poses', {})
                    if isinstance(all_poses, dict) and robot_pose_key in all_poses:
                        mapped[image_key] = all_poses[robot_pose_key]
                print(f"🤖 列表格式位姿: 使用链路 {robot_pose_key}，可用 {len(mapped)} 条")
                return mapped
            raise ValueError("位姿文件格式无效：应为 dict 或 list")

        pose_path = _resolve_pose_path()
        print(f"📂 加载位姿文件: {pose_path}")
        poses_data = _load_pose_data(pose_path)
        pose_map = _build_pose_map(poses_data)

        image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
            image_files.extend(glob.glob(os.path.join(folder_path, ext)))
        image_files = sorted(image_files, key=_natural_sort_key)

        if not image_files:
            raise FileNotFoundError(f"文件夹中没有找到图片: {folder_path}")
        print(f"📷 找到 {len(image_files)} 张图片")

        success_count = 0
        failed_count = 0

        for image_path in image_files:
            image_name = os.path.splitext(os.path.basename(image_path))[0]

            if image_name not in pose_map:
                print(f"⚠️  跳过 {image_name}: 没有对应的位姿数据")
                failed_count += 1
                continue

            image = cv2.imread(image_path)
            if image is None:
                print(f"⚠️  跳过 {image_name}: 无法读取图片")
                failed_count += 1
                continue

            try:
                gripper_pose = self._parse_pose_from_data(pose_map[image_name])
            except Exception as e:
                print(f"⚠️  跳过 {image_name}: 位姿解析失败 - {e}")
                failed_count += 1
                continue

            if self.add_sample(gripper_pose, image, name=image_name):
                success_count += 1
                print(f"   ✅ {image_name} 加载成功")
            else:
                failed_count += 1
                print(f"   ❌ {image_name} 标定板检测失败")

        print(f"\n📊 加载完成: 成功 {success_count} 个, 失败 {failed_count} 个")
        return success_count
    
    def _parse_pose_from_data(self, pose_data):
        """从字典解析位姿数据"""
        if 'matrix' in pose_data:
            # 4x4 矩阵格式
            return np.array(pose_data['matrix'], dtype=np.float64)
        elif 'position' in pose_data and 'orientation' in pose_data:
            # 常见格式: {'position': [x,y,z], 'orientation': [qx,qy,qz,qw]}
            pos = pose_data['position']
            quat = pose_data['orientation']
            if len(pos) != 3 or len(quat) != 4:
                raise ValueError("position/orientation 长度应分别为 3 和 4")
            return {
                'x': float(pos[0]),
                'y': float(pos[1]),
                'z': float(pos[2]),
                'qx': float(quat[0]),
                'qy': float(quat[1]),
                'qz': float(quat[2]),
                'qw': float(quat[3]),
            }
        elif 'qw' in pose_data:
            # 四元数格式
            return {
                'x': float(pose_data['x']),
                'y': float(pose_data['y']),
                'z': float(pose_data['z']),
                'qx': float(pose_data['qx']),
                'qy': float(pose_data['qy']),
                'qz': float(pose_data['qz']),
                'qw': float(pose_data['qw']),
            }
        elif 'rx' in pose_data:
            # 欧拉角格式
            return {
                'x': float(pose_data['x']),
                'y': float(pose_data['y']),
                'z': float(pose_data['z']),
                'rx': float(pose_data['rx']),
                'ry': float(pose_data['ry']),
                'rz': float(pose_data['rz']),
            }
        else:
            raise ValueError(f"无法识别的位姿格式: {pose_data.keys()}")


def generate_chessboard(board_size=(9, 6), square_size_mm=24, output_file='chessboard.png'):
    """
    生成棋盘格标定板图像用于打印
    
    Args:
        board_size: 棋盘格数量 (列, 行)，例如 (9, 6) 表示 9列 x 6行
        square_size_mm: 格子边长（毫米）
        output_file: 输出文件名
    """
    # 计算图像尺寸（像素），假设 300 DPI 打印
    dpi = 300
    mm_to_inch = 1 / 25.4
    square_size_px = int(square_size_mm * mm_to_inch * dpi)
    
    cols, rows = board_size
    img_width = cols * square_size_px
    img_height = rows * square_size_px
    
    # 创建棋盘格图像
    img = np.zeros((img_height, img_width), dtype=np.uint8)
    
    for i in range(rows):
        for j in range(cols):
            if (i + j) % 2 == 0:
                y1 = i * square_size_px
                y2 = (i + 1) * square_size_px
                x1 = j * square_size_px
                x2 = (j + 1) * square_size_px
                img[y1:y2, x1:x2] = 255
    
    cv2.imwrite(output_file, img)
    print(f"✅ 棋盘格标定板已保存: {output_file}")
    print(f"   棋盘格尺寸: {cols} x {rows}")
    print(f"   格子边长: {square_size_mm} mm")
    print(f"   内角点数量: {cols-1} x {rows-1} = {(cols-1)*(rows-1)} 个")
    print(f"   请按 100% 比例打印，确保格子边长为 {square_size_mm} mm")




if __name__ == '__main__':
    
    #generate_chessboard(board_size=(11, 8), square_size_mm=10, output_file='calibration_chessboard.png')
    #1. 创建标定器
    calibrator = HandEyeCalibration(
        board_size=(11, 8),  # 内角点数量 (列, 行)，对应棋盘格的格子数 -1
        square_size=0.01  # 10mm
    )

    #2. True: Eye-in-Hand(眼在手上) / False: Eye-to-Hand(眼在手外)
    eye_in_hand_mode = True
    calibrator.set_eye_mode(eye_in_hand_mode)

    # 3. 仅从文件加载相机内参
    camera_info_path = 'calibration_data/camera_info.json'
    data_folder = 'calibration_data'
    if os.path.exists(camera_info_path):
        calibrator.load_camera_intrinsics(camera_info_path)
    else:
        raise FileNotFoundError(f"缺少相机内参文件: {camera_info_path}")

    # 4. 加载数据
    if not os.path.exists(data_folder):
        print("\n⚠️  数据文件夹不存在，创建示例结构...")
        print("\n请准备好数据后重新运行此脚本！")
        exit(0)
    # 4.1 最简人工检查：显示原图，按任意键下一张，ESC退出
    print("\n按任意键切换下一张图片，ESC退出角点检查...")
    image_files = sorted(glob.glob(os.path.join(data_folder, '*.jpg')), key=_natural_sort_key)
    for img_path in image_files:
        image = cv2.imread(img_path)
        if image is None:
            print(f"显示: {img_path} ❌ (无法读取图片)")
            continue
        display = image.copy()
        success, rvec, tvec = calibrator.detect_target(
            image,
            print_pose=True,
            sample_name=os.path.basename(img_path)
        )
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(
            gray,
            calibrator.board_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        if ret:
            corners_refined = cv2.cornerSubPix(
                gray,
                corners,
                (11, 11),
                (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            )
            cv2.drawChessboardCorners(display, calibrator.board_size, corners_refined, ret)
        cv2.imshow("Chessboard Check", display)
        print(f"显示: {img_path} {'✅' if success else '❌'}")
        key = cv2.waitKey(0)
        if key == 27:  # ESC
            cv2.destroyAllWindows()
            break
    cv2.destroyAllWindows()

    # 4. 正式加载数据
    num_samples = calibrator.load_from_folder(
        data_folder,
        robot_pose_key='openarm_left_hand_tcp'
    )
    if num_samples < 3:
        print(f"\n❌ 样本数不足！至少需要 3 个有效样本，当前只有 {num_samples} 个")
        exit(1)

    # 5. 多种方法对比求解
    methods = ['Tsai1989', 'Horaud1995', 'Andreff1999', 'Daniilidis1998']
    for method in methods:
        print(f"\n{'='*20} {method} {'='*20}")
        R, t = calibrator.solve(method=method)
        result_file = os.path.join(data_folder, f"handeye_result_{method}.json")
        calibrator.save_result(result_file, R, t, method=method)

   

