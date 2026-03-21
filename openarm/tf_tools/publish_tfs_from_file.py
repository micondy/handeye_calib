#!/usr/bin/env python3
import argparse
import time

import rclpy
from tf2_ros import TransformBroadcaster

try:
    from .tf_publisher import load_transforms_from_json, publish_transform
except ImportError:
    from openarm.tf_tools.tf_publisher import load_transforms_from_json, publish_transform


"""
在工程根目录运行：python3 -m openarm.tf_tools.publish_tfs_from_file --tf-file openarm/tf_tools/tfs_example.json
或先进入目录再运行：cd openarm/tf_tools && python3 -m openarm.tf_tools.publish_tfs_from_file --tf-file tfs_example.json
"""

def main():
    parser = argparse.ArgumentParser(description='从 JSON 文件读取多个 TF 并循环发布')
    parser.add_argument('--tf-file', required=True, help='TF 配置文件路径')
    parser.add_argument('--rate', type=float, default=30.0, help='发布频率 Hz（默认 30）')
    args = parser.parse_args()

    transforms = load_transforms_from_json(args.tf_file)
    if not transforms:
        raise RuntimeError('TF 配置为空，请检查 --tf-file')

    rclpy.init()
    node = rclpy.create_node('publish_tfs_from_file')
    tf_broadcaster = TransformBroadcaster(node)
    sleep_sec = 1.0 / max(args.rate, 1e-6)

    print(f'📄 已加载 {len(transforms)} 个 TF: {args.tf_file}')
    for item in transforms:
        print(f"  - {item['parent_frame']} -> {item['child_frame']}")

    try:
        while True:
            for item in transforms:
                publish_transform(
                    tf_broadcaster=tf_broadcaster,
                    node=node,
                    parent_frame=item['parent_frame'],
                    child_frame=item['child_frame'],
                    translation_xyz=item['translation'],
                    rotation_xyzw=item['rotation'],
                )
            time.sleep(sleep_sec)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
