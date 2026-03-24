# Hand-Eye Calibration (OpenCV)

纯 OpenCV 手眼标定脚本（不依赖 ROS2 / MoveIt2 / RViz2）。

当前主脚本：`Calibration.py`

辅助脚本：`solve_tip_pivot.py`

## 功能
- 文件读取相机内参
- 从文件批量读取图像与机械臂位姿进行手眼标定
- 支持 Eye-in-Hand / Eye-to-Hand 两种模式
- 多算法求解（Tsai / Horaud / Andreff / Daniilidis）
- 保存标定结果 JSON
- 基于 `poses.json` 的 pivot 求解（用于估计 hand -> tip 偏移）

## 环境准备
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 数据文件
- `calibration_data/camera_info.json`：相机内参
- `calibration_data/poses.json`：每帧机械臂位姿（含 link 姿态）
- `calibration_data/*.jpg`：标定图像

## 运行手眼标定
```bash
python3 Calibration.py
```

运行后会在数据目录下输出各算法结果文件，例如：
- `handeye_result_Tsai1989.json`
- `handeye_result_Horaud1995.json`
- `handeye_result_Andreff1999.json`
- `handeye_result_Daniilidis1998.json`

## 运行 pivot 求解（hand -> tip）
默认以 `openarm_left_hand` 作为 pivot link：

```bash
python3 solve_tip_pivot.py --poses calibration_data/poses.json
```

常用参数：
- `--pivot-link`：指定参考 link 名称
- `--max-stamp-diff-sec`：按时间戳差过滤样本
- `--save-json`：保存 pivot 求解结果

示例：
```bash
python3 solve_tip_pivot.py \
	--poses calibration_data/poses.json \
	--pivot-link openarm_left_hand \
	--max-stamp-diff-sec 0.05 \
	--save-json calibration_data/pivot_result.json
```
