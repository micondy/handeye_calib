#!/usr/bin/env python3
"""
给定一个初始姿态四元数，仅绕“自身 y 轴”旋转，
使下面两个条件尽量满足：
1) 旋转后四元数的 x 轴 垂直于 世界坐标系 x 轴
2) 旋转后四元数的 z 轴 垂直于 世界坐标系 y 轴

四元数格式使用 [x, y, z, w]。
"""

from __future__ import annotations

import argparse
import math
from typing import Tuple

import numpy as np


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


def quat_local_y(theta: float) -> np.ndarray:
    half = 0.5 * theta
    return np.array([0.0, math.sin(half), 0.0, math.cos(half)], dtype=float)


def evaluate_constraints(R: np.ndarray) -> Tuple[float, float, float]:
    world_x = np.array([1.0, 0.0, 0.0], dtype=float)
    world_y = np.array([0.0, 1.0, 0.0], dtype=float)

    x_axis = R[:, 0]
    z_axis = R[:, 2]

    c1 = float(np.dot(x_axis, world_x))
    c2 = float(np.dot(z_axis, world_y))
    err = math.sqrt(c1 * c1 + c2 * c2)
    return c1, c2, err


def find_best_theta_for_local_y(
    q_init: np.ndarray,
) -> Tuple[float, float, float, float]:
    """
    返回 (theta, c1, c2, err)
    theta: 绕自身 y 轴旋转角（弧度）
    c1 = x_axis · world_x，理想值 0
    c2 = z_axis · world_y，理想值 0
    err = sqrt(c1^2 + c2^2)
    """
    R = quaternion_to_rotation_matrix(q_init)
    rx = R[:, 0]
    rz = R[:, 2]

    a = float(rx[0])
    b = float(rz[0])
    c = float(rx[1])
    d = float(rz[1])

    m = 0.5 * ((a * a + d * d) - (b * b + c * c))
    n = (-a * b + c * d)

    # E(theta) = c1^2 + c2^2 的解析最小点: 2*theta = atan2(n, m) + pi
    theta0 = 0.5 * (math.atan2(n, m) + math.pi)
    candidates = [theta0, theta0 + math.pi]

    best = None
    for theta in candidates:
        q_new = normalize_quaternion(
            quaternion_multiply(q_init, quat_local_y(theta))
        )
        c1, c2, err = evaluate_constraints(
            quaternion_to_rotation_matrix(q_new)
        )
        if best is None or err < best[3]:
            best = (theta, c1, c2, err)

    assert best is not None
    return best


def wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def main() -> None:
    parser = argparse.ArgumentParser(
        description="输入四元数 [x y z w]，求绕自身 y 轴旋转角使约束尽量满足"
    )
    parser.add_argument("x", type=float, help="四元数 x")
    parser.add_argument("y", type=float, help="四元数 y")
    parser.add_argument("z", type=float, help="四元数 z")
    parser.add_argument("w", type=float, help="四元数 w")
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-6,
        help="误差阈值，小于该值视为精确满足（默认 1e-6）",
    )
    args = parser.parse_args()

    q_init = normalize_quaternion(
        np.array([args.x, args.y, args.z, args.w], dtype=float)
    )
    theta, c1, c2, err = find_best_theta_for_local_y(q_init)
    theta = wrap_to_pi(theta)

    q_final = normalize_quaternion(
        quaternion_multiply(q_init, quat_local_y(theta))
    )

    print("初始四元数 [x y z w]:", q_init)
    print("绕自身 y 轴旋转角 theta (rad):", theta)
    print("绕自身 y 轴旋转角 theta (deg):", math.degrees(theta))
    print("最终四元数 [x y z w]:", q_final)
    print()
    print("约束检查（理想值都为 0）:")
    print("x_axis · world_x =", c1)
    print("z_axis · world_y =", c2)
    print("combined error =", err)

    if err <= args.tol:
        print("结果: 已满足条件（在容差范围内）")
    else:
        print("结果: 仅找到最优近似解（该初始姿态下可能无法同时精确满足两个条件）")


if __name__ == "__main__":
    main()
