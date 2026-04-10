#!/usr/bin/env python3
"""
输入初始四元数 [x, y, z, w]，绕自身 x 轴旋转一个角度，
计算旋转后的四元数。
"""

from __future__ import annotations

import argparse
import math
import re

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


def quat_local_x(theta_rad: float) -> np.ndarray:
    half = 0.5 * theta_rad
    return np.array(
        [math.sin(half), 0.0, 0.0, math.cos(half)],
        dtype=float,
    )


def parse_xyzw_text(text: str) -> np.ndarray:
    cleaned = text.strip().strip("[]()")
    cleaned = re.sub(r"[,;，]", " ", cleaned)
    parts = [p for p in cleaned.split() if p]
    if len(parts) != 4:
        raise ValueError("xyzw 文本必须包含 4 个数字")
    return np.array([float(p) for p in parts], dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "四元数绕自身 x 轴旋转并计算结果。"
            "支持输入: 'x y z w angle' 或 'x;y;z;w angle'"
        )
    )
    parser.add_argument(
        "values",
        nargs="+",
        help=(
            "输入值。可用: qx qy qz qw angle；"
            "或 'x;y;z;w' angle（注意分号格式需加引号）"
        ),
    )

    angle_group = parser.add_mutually_exclusive_group(required=False)
    angle_group.add_argument(
        "--angle-deg",
        type=float,
        help="绕自身 x 轴旋转角（度）",
    )
    angle_group.add_argument(
        "--angle-rad",
        type=float,
        help="绕自身 x 轴旋转角（弧度）",
    )
    parser.add_argument(
        "--unit",
        choices=["rad", "deg"],
        default="rad",
        help="紧凑输入中角度单位（默认 rad）",
    )

    args = parser.parse_args()

    q_raw: np.ndarray
    angle_input: float

    compact_mode = False
    if len(args.values) == 5:
        q_raw = np.array([float(v) for v in args.values[:4]], dtype=float)
        angle_input = float(args.values[4])
        compact_mode = True
    elif len(args.values) == 2:
        q_raw = parse_xyzw_text(args.values[0])
        angle_input = float(args.values[1])
        compact_mode = True
    elif len(args.values) == 4:
        q_raw = np.array([float(v) for v in args.values], dtype=float)
        angle_input = 0.0
    else:
        parser.error(
            "输入格式不正确。请使用: qx qy qz qw angle；"
            "或 'x;y;z;w' angle"
        )

    q_init = normalize_quaternion(q_raw)

    if compact_mode:
        if args.angle_rad is not None or args.angle_deg is not None:
            parser.error(
                "紧凑输入已包含角度，不要再传 --angle-rad/--angle-deg"
            )
        if args.unit == "rad":
            theta_rad = angle_input
        else:
            theta_rad = math.radians(angle_input)
    else:
        if args.angle_rad is None and args.angle_deg is None:
            parser.error(
                "当输入为 qx qy qz qw 时，需要传 --angle-rad 或 --angle-deg"
            )
        if args.angle_rad is not None:
            theta_rad = float(args.angle_rad)
        else:
            theta_rad = math.radians(float(args.angle_deg))

    # 绕自身轴旋转：右乘局部旋转四元数
    q_rot = quat_local_x(theta_rad)
    q_final = normalize_quaternion(quaternion_multiply(q_init, q_rot))

    print("初始四元数 [x y z w]:", q_init)
    print("旋转角 theta(rad):", theta_rad)
    print("旋转角 theta(deg):", math.degrees(theta_rad))
    print("旋转后四元数 [x y z w]:", q_final)


if __name__ == "__main__":
    main()
