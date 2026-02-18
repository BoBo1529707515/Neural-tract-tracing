# 神经元追踪工具 v1.0 说明文档

## 1. 项目简介
这是一个专为延时活细胞成像设计的神经元轴突追踪工具。它允许用户在视频序列中交互式地标记神经元，并自动跟踪其生长轨迹。工具采用了模块化设计，结合了人工交互与计算机视觉算法，旨在解决低信噪比、灰度显微图像中的微小结构追踪问题。

## 2. 核心功能
*   **交互式标记**：支持在任意帧进行标点，支持同一帧标记多个点（自动计算重心），支持平移、缩放查看细节。
*   **混合追踪算法**：结合了人工标点的强约束与 CSRT (Discriminative Correlation Filter with Channel and Spatial Reliability) 自动跟踪算法。
    *   **人工优先**：如果某帧有人工标点，强制对齐，消除累积误差。
    *   **自动填补**：在没有人工标点的帧，利用 CSRT 算法基于纹理特征自动预测位置。
*   **灰白图像适配**：内置 CLAHE (Contrast Limited Adaptive Histogram Equalization) 预处理，增强灰度图像对比度，提升特征识别率。
*   **结果导出**：自动生成带有轨迹叠加的 MP4 视频以及包含逐帧坐标的 CSV 数据文件。

## 3. 使用指南

### 3.1 环境依赖
请确保安装以下 Python 库：
```bash
pip install opencv-python opencv-contrib-python numpy
```
*注意：`opencv-contrib-python` 是必须的，因为 CSRT 跟踪器包含在 contrib 模块中。*

### 3.2 运行方法
在项目根目录下运行主入口脚本：
```bash
python neuron_trace_v1/main_05.py
```

### 3.3 操作快捷键
**交互选择界面 (01_选择神经元):**
*   **鼠标左键**: 添加标记点。
*   **鼠标右键**: 删除最近的一个标记点。
*   **鼠标中键拖拽**: 平移图像。
*   **鼠标滚轮**: 缩放图像。
*   **A / D**: 上一帧 / 下一帧。
*   **G**: 跳转到指定帧（输入数字后按回车）。
*   **Space (空格)**: 完成标记，开始自动追踪。
*   **Q / Esc**: 退出程序。

**追踪显示界面 (02_跟踪显示):**
*   **Q / Esc**: 提前终止追踪并保存结果。

## 4. 模块说明
*   `01_config.py`: 配置管理，定义视频路径、输出目录及参数。
*   `02_ui.py`: 交互式 UI 实现，处理鼠标键盘事件与坐标变换。
*   `03_tracker.py`: 核心追踪器封装，目前使用 CSRT 算法。
*   `04_pipeline.py`: 业务流程控制，串联读取、预处理、交互、追踪与导出。
*   `05_main.py`: 程序入口。

## 5. 常见问题
*   **Q: 为什么视频读取失败？**
    *   A: 请检查 `01_config.py` 中的 `video_path` 是否正确，路径中尽量不要包含中文（虽然已做处理，但部分系统仍敏感）。
*   **Q: 自动追踪漂移严重怎么办？**
    *   A: 可以在漂移开始的帧手动添加一个标点。算法会优先使用人工标点重置跟踪器，从而修正后续轨迹。
*   **Q: 报错 `AttributeError: module 'cv2' has no attribute 'TrackerCSRT_create'`?**
    *   A: 请卸载 `opencv-python` 并重新安装 `opencv-contrib-python`。

---
*Created by Trae AI Pair Programmer*
