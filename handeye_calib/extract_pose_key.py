#!/usr/bin/env python3
import argparse
import json
import os
from typing import Any, Dict, List


ALLOWED_KEYS = {"openarm_left_hand"}


def parse_keys(key_arg: str) -> List[str]:
    keys = [k.strip() for k in key_arg.split(',') if k.strip()]
    if not keys:
        raise ValueError("--keys 不能为空")
    invalid = [k for k in keys if k not in ALLOWED_KEYS]
    if invalid:
        raise ValueError(f"不支持的 key: {invalid}，仅支持: {sorted(ALLOWED_KEYS)}")
    return keys


def extract(data: Any, keys: List[str]) -> List[Dict[str, Any]]:
    if not isinstance(data, list):
        raise ValueError("输入文件必须是 list 结构（每项含 image/poses）")

    output = []
    missing_count = 0

    for item in data:
        if not isinstance(item, dict):
            continue
        image_name = item.get('image')
        poses = item.get('poses', {})
        if not isinstance(poses, dict):
            continue

        row = {'image': image_name, 'poses': {}}
        found_any = False
        for key in keys:
            pose = poses.get(key)
            if isinstance(pose, dict):
                row['poses'][key] = {
                    'x': float(pose['x']),
                    'y': float(pose['y']),
                    'z': float(pose['z']),
                    'rx': float(pose['rx']),
                    'ry': float(pose['ry']),
                    'rz': float(pose['rz']),
                }
                found_any = True
            else:
                missing_count += 1

        if found_any:
            output.append(row)

    print(f"输入样本数: {len(data)}")
    print(f"输出样本数: {len(output)}")
    print(f"缺失目标 key 次数: {missing_count}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="从 poses_euler.json 提取 openarm_left_hand/openarm_left_hand_tcp")
    parser.add_argument(
        '--input',
        default='calibration_data/poses_euler.json',
        help='输入文件路径'
    )
    parser.add_argument(
        '--output',
        default='calibration_data/poses_left_hand_only.json',
        help='输出文件路径'
    )
    parser.add_argument(
        '--keys',
        default='openarm_left_hand',
        help='提取的 key，多个用逗号分隔：openarm_left_hand,openarm_left_hand_tcp'
    )
    args = parser.parse_args()

    keys = parse_keys(args.keys)
    input_path = os.path.abspath(args.input)
    output_path = os.path.abspath(args.output)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    out = extract(data, keys)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"提取 key: {keys}")
    print(f"已保存: {output_path}")


if __name__ == '__main__':
    main()
