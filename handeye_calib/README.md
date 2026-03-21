# Hand-Eye Calibration (OpenCV)

纯 OpenCV 手眼标定脚本（不依赖 ROS2 / MoveIt2 / RViz2）。

当前主脚本：`Calibration.py`

## 功能
- 文件读取相机内参
- 从文件批量读取图像与机械臂位姿进行手眼标定
- 支持 Eye-in-Hand / Eye-to-Hand 两种模式
- 多算法求解（Tsai / Horaud / Andreff / Daniilidis）
- 保存标定结果 JSON 
