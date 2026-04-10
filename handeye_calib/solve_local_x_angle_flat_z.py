#!/usr/bin/env python3
"""
给定外参平移 t 和四元数 q（xyzw），仅允许绕“自身 X 轴”追加旋转，
求使一组相机点在世界坐标系下 z 值尽量相等的旋转角。

默认内置了用户提供的 7 组相机点（同一平面采样点）。

示例:
python3 solve_local_x_angle_flat_z.py \
  "0.030031; 0.0033526; 0.81756" \
  "-0.66796; 0.67902; -0.21022; 0.22038"
"""

from __future__ import annotations

import argparse
import json
import math
import re
from typing import Tuple

import numpy as np


DEFAULT_CAMERA_POINTS = np.array(
    [
        [0.0615, -0.2282, 0.9660],
        [0.0621, -0.1476, 0.9220],
        [0.0610, -0.0801, 0.8780],
        [0.0590, -0.0219, 0.8400],
        [0.0574, 0.0281, 0.8090],
        [0.0570, 0.0878, 0.7710],
        [0.0563, 0.1656, 0.7260],
    ],
    dtype=float,
)


def parse_vector(text: str, expected_len: int) -> np.ndarray:
    cleaned = text.strip().strip("[]()")
    cleaned = re.sub(r"[,;，]", " ", cleaned)
    parts = [p for p in cleaned.split() if p]
    if len(parts) != expected_len:
        raise ValueError(f"期望 {expected_len} 个数字，实际得到 {len(parts)} 个")
    return np.array([float(v) for v in parts], dtype=float)


def normalize_quaternion(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q)
    if n == 0:
        raise ValueError("四元数范数为 0，无法归一化")
    return q / n


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=float,
    )


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    return np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ],
        dtype=float,
    )


def quat_local_x(theta_rad: float) -> np.ndarray:
    half = 0.5 * theta_rad
    return np.array([math.sin(half), 0.0, 0.0, math.cos(half)], dtype=float)


def load_camera_points(json_path: str | None) -> np.ndarray:
    if not json_path:
        return DEFAULT_CAMERA_POINTS.copy()

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "camera_points" in data:
        pts = np.array(data["camera_points"], dtype=float)
    else:
        pts = np.array(data, dtype=float)

    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("点文件格式错误，必须是 Nx3")
    return pts


def evaluate_z_stats(
    t: np.ndarray,
    q_init: np.ndarray,
    theta_rad: float,
    camera_points: np.ndarray,
) -> Tuple[np.ndarray, float, float]:
    q_new = normalize_quaternion(quaternion_multiply(q_init, quat_local_x(theta_rad)))
    R = quaternion_to_rotation_matrix(q_new)
    world_points = (R @ camera_points.T).T + t
    z = world_points[:, 2]
    return z, float(np.std(z)), float(np.max(z) - np.min(z))


def find_best_theta(
    t: np.ndarray,
    q_init: np.ndarray,
    camera_points: np.ndarray,
    grid_size: int,
) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    if grid_size < 1001:
        raise ValueError("grid_size 建议 >= 1001")

    theta_grid = np.linspace(-math.pi, math.pi, grid_size)
    best_std = None
    best_theta = 0.0
    best_range = 0.0
    best_z = np.zeros((camera_points.shape[0],), dtype=float)
    best_q = q_init.copy()

    for theta in theta_grid:
        z, z_std, z_range = evaluate_z_stats(t, q_init, float(theta), camera_points)
        if best_std is None or z_std < best_std:
            best_std = z_std
            best_theta = float(theta)
            best_range = z_range
            best_z = z
            best_q = normalize_quaternion(
                quaternion_multiply(q_init, quat_local_x(best_theta))
            )

    assert best_std is not None
    return best_theta, best_std, best_range, best_z, best_q


def wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def main() -> None:
    parser = argparse.ArgumentParser(
        description="仅绕局部 X 旋转，求相机点变换后 z 最平的角度"
    )
    parser.add_argument("t_xyz", help="平移向量，格式: 'tx; ty; tz' 或 'tx ty tz'")
    parser.add_argument("q_xyzw", help="四元数，格式: 'qx; qy; qz; qw' 或空格分隔")
    parser.add_argument(
        "--camera-points-json",
        default=None,
        help=(
            "可选，相机点 JSON 文件（Nx3 或 {\"camera_points\": Nx3}）；"
            "不传则用内置 7 个点"
        ),
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=20001,
        help="搜索网格数量，默认 20001",
    )
    args = parser.parse_args()

    t = parse_vector(args.t_xyz, 3)
    q_init = normalize_quaternion(parse_vector(args.q_xyzw, 4))
    camera_points = load_camera_points(args.camera_points_json)

    theta, z_std, z_range, z_values, q_new = find_best_theta(
        t=t,
        q_init=q_init,
        camera_points=camera_points,
        grid_size=args.grid_size,
    )
    theta = wrap_to_pi(theta)

    print("输入平移 t [x y z]:", t)
    print("输入四元数 q [x y z w]:", q_init)
    print("样本点数量:", camera_points.shape[0])
    print()
    print("最优绕自身 X 轴角度:")
    print("theta(rad):", theta)
    print("theta(deg):", math.degrees(theta))
    print()
    print("旋转后四元数 q_new [x y z w]:", q_new)
    print("变换后 z 值:", z_values)
    print("z 标准差:", z_std)
    print("z 极差(max-min):", z_range)


if __name__ == "__main__":
    main()
