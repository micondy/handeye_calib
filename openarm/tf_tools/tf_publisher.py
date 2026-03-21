#!/usr/bin/env python3
import json
from typing import Any, Dict, List, Optional

import numpy as np
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster


def publish_transform(
    tf_broadcaster: TransformBroadcaster,
    node,
    parent_frame: str,
    child_frame: str,
    translation_xyz: np.ndarray,
    rotation_xyzw: np.ndarray,
) -> None:
    msg = TransformStamped()
    msg.header.stamp = node.get_clock().now().to_msg()
    msg.header.frame_id = parent_frame
    msg.child_frame_id = child_frame
    msg.transform.translation.x = float(translation_xyz[0])
    msg.transform.translation.y = float(translation_xyz[1])
    msg.transform.translation.z = float(translation_xyz[2])
    msg.transform.rotation.x = float(rotation_xyzw[0])
    msg.transform.rotation.y = float(rotation_xyzw[1])
    msg.transform.rotation.z = float(rotation_xyzw[2])
    msg.transform.rotation.w = float(rotation_xyzw[3])
    tf_broadcaster.sendTransform(msg)


def load_transforms_from_json(file_path: str) -> List[Dict[str, Any]]:
    with open(file_path, 'r', encoding='utf-8') as f:
        payload = json.load(f)

    if isinstance(payload, dict):
        transforms = payload.get('transforms', [])
    elif isinstance(payload, list):
        transforms = payload
    else:
        raise ValueError('JSON 根节点必须是 list 或包含 transforms 字段的 dict')

    if not isinstance(transforms, list):
        raise ValueError('transforms 必须是 list')

    normalized: List[Dict[str, Any]] = []
    for idx, item in enumerate(transforms):
        if not isinstance(item, dict):
            raise ValueError(f'transforms[{idx}] 必须是对象')

        parent = item.get('parent_frame') or item.get('frame_id')
        child = item.get('child_frame') or item.get('child_frame_id')

        t = item.get('translation') or item.get('position')
        r = item.get('rotation') or item.get('orientation')

        if parent is None or child is None or t is None or r is None:
            raise ValueError(
                f'transforms[{idx}] 缺少字段，需要 parent_frame/frame_id、child_frame/child_frame_id、translation、rotation'
            )

        try:
            tx = float(t['x'])
            ty = float(t['y'])
            tz = float(t['z'])
            qx = float(r['x'])
            qy = float(r['y'])
            qz = float(r['z'])
            qw = float(r['w'])
        except Exception as exc:
            raise ValueError(f'transforms[{idx}] 数值字段解析失败: {exc}') from exc

        normalized.append(
            {
                'parent_frame': str(parent),
                'child_frame': str(child),
                'translation': np.array([tx, ty, tz], dtype=np.float64),
                'rotation': np.array([qx, qy, qz, qw], dtype=np.float64),
            }
        )

    return normalized


def find_transform_by_child(transforms: List[Dict[str, Any]], child_frame: str) -> Optional[Dict[str, Any]]:
    for item in transforms:
        if item.get('child_frame') == child_frame:
            return item
    return None
