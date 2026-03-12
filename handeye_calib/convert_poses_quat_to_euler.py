#!/usr/bin/env python3
import argparse
import json
import os
from typing import Any, Dict, List, Tuple

import transforms3d


def quat_xyzw_to_euler_xyz(qx: float, qy: float, qz: float, qw: float) -> Tuple[float, float, float]:
    q_wxyz = [qw, qx, qy, qz]
    rx, ry, rz = transforms3d.euler.quat2euler(q_wxyz, axes='sxyz')
    return float(rx), float(ry), float(rz)


def convert_pose_dict_quat_to_euler(pose: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    if not isinstance(pose, dict):
        return pose, False

    if 'position' in pose and 'orientation' in pose:
        pos = pose.get('position', None)
        quat = pose.get('orientation', None)
        if isinstance(pos, list) and len(pos) == 3 and isinstance(quat, list) and len(quat) == 4:
            x, y, z = pos
            qx, qy, qz, qw = quat
            rx, ry, rz = quat_xyzw_to_euler_xyz(qx, qy, qz, qw)
            converted = {
                'x': float(x),
                'y': float(y),
                'z': float(z),
                'rx': rx,
                'ry': ry,
                'rz': rz,
            }
            return converted, True

    if all(k in pose for k in ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']):
        rx, ry, rz = quat_xyzw_to_euler_xyz(
            float(pose['qx']), float(pose['qy']), float(pose['qz']), float(pose['qw'])
        )
        converted = {
            'x': float(pose['x']),
            'y': float(pose['y']),
            'z': float(pose['z']),
            'rx': rx,
            'ry': ry,
            'rz': rz,
        }
        return converted, True

    return pose, False


def convert_poses_data(data: Any) -> Tuple[Any, int]:
    converted_count = 0

    if isinstance(data, list):
        out: List[Any] = []
        for item in data:
            if isinstance(item, dict) and isinstance(item.get('poses', None), dict):
                new_item = dict(item)
                new_poses = {}
                for link_name, pose in item['poses'].items():
                    converted_pose, changed = convert_pose_dict_quat_to_euler(pose)
                    new_poses[link_name] = converted_pose
                    if changed:
                        converted_count += 1
                new_item['poses'] = new_poses
                out.append(new_item)
            else:
                converted_item, changed = convert_pose_dict_quat_to_euler(item)
                if changed:
                    converted_count += 1
                out.append(converted_item)
        return out, converted_count

    if isinstance(data, dict):
        out_dict: Dict[str, Any] = {}
        for k, v in data.items():
            converted_v, changed = convert_pose_dict_quat_to_euler(v) if isinstance(v, dict) else (v, False)
            out_dict[k] = converted_v
            if changed:
                converted_count += 1
        return out_dict, converted_count

    return data, converted_count


def main() -> None:
    parser = argparse.ArgumentParser(description='将 poses 中四元数位姿转换为欧拉角 rx,ry,rz（弧度）')
    parser.add_argument(
        '--input',
        default='calibration_data/poses.json',
        help='输入 poses 文件路径（json）'
    )
    parser.add_argument(
        '--output',
        default='calibration_data/poses_euler.json',
        help='输出文件路径（json）'
    )
    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    output_path = os.path.abspath(args.output)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f'输入文件不存在: {input_path}')

    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    converted_data, count = convert_poses_data(data)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, indent=2, ensure_ascii=False)

    print(f'输入文件: {input_path}')
    print(f'输出文件: {output_path}')
    print(f'成功转换位姿数量: {count}')


if __name__ == '__main__':
    main()
