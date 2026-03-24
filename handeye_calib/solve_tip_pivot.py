#!/usr/bin/env python3
"""基于 poses.json 的 pivot 求解脚本（调库版）。

功能：
- 固定一点、姿态变化时，最小二乘求解该点在 link 与 world 下坐标。

默认按 left_hand 作为参考 link，适合求 hand->tip 末端点偏移。

python3 solve_tip_pivot.py --poses calibration_data/poses.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy.spatial.transform import Rotation as R


def pose_to_hmat(position: List[float], quat_xyzw: List[float]) -> np.ndarray:
    t = np.asarray(position, dtype=float).reshape(3)
    rot = R.from_quat(np.asarray(quat_xyzw, dtype=float).reshape(4)).as_matrix()
    h = np.eye(4, dtype=float)
    h[:3, :3] = rot
    h[:3, 3] = t
    return h


def load_frames(path: Path) -> List[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("poses.json 顶层必须是 list")
    return data


def solve_pivot(frames: List[dict], pivot_link: str, max_stamp_diff_sec: Optional[float]) -> Dict:
    rots, trans, used = [], [], []

    for idx, frame in enumerate(frames):
        pose = frame.get("poses", {}).get(pivot_link)
        if pose is None:
            continue
        if max_stamp_diff_sec is not None and abs(float(pose.get("stamp_diff_sec", 0.0))) > max_stamp_diff_sec:
            continue

        h = pose_to_hmat(pose["position"], pose["orientation"])
        rots.append(h[:3, :3])
        trans.append(h[:3, 3])
        used.append(idx)

    n = len(rots)
    if n < 3:
        raise RuntimeError("pivot 求解至少需要 3 帧有效样本")

    # R_i * p_link - p_world = -t_i
    a = np.zeros((3 * n, 6), dtype=float)
    b = np.zeros((3 * n,), dtype=float)
    for i, (r_i, t_i) in enumerate(zip(rots, trans)):
        j = 3 * i
        a[j : j + 3, :3] = r_i
        a[j : j + 3, 3:] = -np.eye(3)
        b[j : j + 3] = -t_i

    x, residuals, rank, _ = np.linalg.lstsq(a, b, rcond=None)
    p_link = x[:3]
    p_world = x[3:]

    err_mm = []
    for r_i, t_i in zip(rots, trans):
        p_world_i = r_i @ p_link + t_i
        err_mm.append(np.linalg.norm((p_world_i - p_world) * 1000.0))
    err_mm = np.asarray(err_mm, dtype=float)

    return {
        "mode": "pivot",
        "pivot_link": pivot_link,
        "num_total_frames": len(frames),
        "num_used_frames": n,
        "used_frame_indices": used,
        "pivot_in_link_m": p_link.tolist(),
        "pivot_in_world_m": p_world.tolist(),
        "fit_rank": int(rank),
        "fit_residual_sum": float(residuals[0]) if residuals.size else 0.0,
        "pivot_error_mm": {
            "mean": float(err_mm.mean()),
            "std": float(err_mm.std()),
            "max": float(err_mm.max()),
        },
    }


def print_result(res: Dict) -> None:
    print(json.dumps(res, ensure_ascii=False, indent=2))
    pivot_link = res.get("pivot_link")
    pivot_in_link = res.get("pivot_in_link_m")
    if pivot_in_link is not None:
        print(
            f"\n建议：可将 {pivot_link} -> tip 的 xyz 偏移设为: "
            f"[{pivot_in_link[0]:.6f}, {pivot_in_link[1]:.6f}, {pivot_in_link[2]:.6f}] (m)"
        )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="poses.json pivot 求解脚本（调库版）")
    p.add_argument("--poses", type=Path, required=True, help="poses.json 路径")
    p.add_argument("--pivot-link", default="openarm_left_hand", help="pivot 模式 link")
    p.add_argument("--max-stamp-diff-sec", type=float, default=None, help="可选时间戳过滤")
    p.add_argument("--save-json", type=Path, default=None, help="可选输出结果 json")
    return p


def main() -> None:
    args = build_parser().parse_args()
    frames = load_frames(args.poses)

    res = solve_pivot(
        frames,
        pivot_link=args.pivot_link,
        max_stamp_diff_sec=args.max_stamp_diff_sec,
    )

    print_result(res)
    if args.save_json is not None:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        args.save_json.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
