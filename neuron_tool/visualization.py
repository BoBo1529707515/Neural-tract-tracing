"""
可视化模块
负责绘制标记、轨迹、界面元素等
"""

import cv2
import numpy as np
from .config import Config


class Visualizer:
    """
    可视化类

    功能:
        - 绘制标记点
        - 绘制轨迹
        - 绘制信息文字
        - 绘制选区预览
    """

    def __init__(self, data_manager):
        """
        初始化可视化器

        参数:
            data_manager: DataManager实例
        """
        self.data = data_manager

    def draw_marks_at_frame(self, canvas, frame_idx, current_neuron_id,
                            coord_transform_func):
        """
        绘制指定帧的标记点

        参数:
            canvas: 目标画布
            frame_idx: 帧索引
            current_neuron_id: 当前选中的神经元ID
            coord_transform_func: 坐标转换函数 (img_x, img_y) -> (screen_x, screen_y)
        """
        display_w, display_h = canvas.shape[1], canvas.shape[0]
        image_area_h = display_h - Config.PANEL_HEIGHT

        # 当前帧的标记
        marks = self.data.get_marks_at_frame(frame_idx)
        for mark in marks:
            color = tuple(mark["color"])
            sx, sy = coord_transform_func(mark["x"], mark["y"])

            # 检查是否在显示区域内
            if not (0 <= sx < display_w and 0 <= sy < image_area_h):
                continue

            nid = mark["neuron_id"]
            r = max(5, 8)  # 标记半径

            # 当前神经元的标记加高亮边框
            if nid == current_neuron_id:
                cv2.circle(canvas, (sx, sy), r + 5, Config.COLOR_MARK_HIGHLIGHT, 2)

            cv2.circle(canvas, (sx, sy), r, color, -1)
            cv2.putText(canvas, str(nid), (sx + r + 3, sy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    def draw_other_frame_marks(self, canvas, current_frame_idx, coord_transform_func):
        """
        绘制其他帧的标记（用十字表示）

        参数:
            canvas: 目标画布
            current_frame_idx: 当前帧索引
            coord_transform_func: 坐标转换函数
        """
        display_w, display_h = canvas.shape[1], canvas.shape[0]
        image_area_h = display_h - Config.PANEL_HEIGHT

        for nid, ndata in self.data.neuron_marks.items():
            color = tuple(ndata["color"])
            for mark in ndata["marks"]:
                if mark["frame"] != current_frame_idx:
                    sx, sy = coord_transform_func(mark["x"], mark["y"])
                    if 0 <= sx < display_w and 0 <= sy < image_area_h:
                        cv2.drawMarker(canvas, (sx, sy), color,
                                       cv2.MARKER_TILTED_CROSS, 8, 1)

    def draw_trajectories(self, canvas, coord_transform_func):
        """
        绘制所有轨迹

        参数:
            canvas: 目标画布
            coord_transform_func: 坐标转换函数 (img_y, img_x) -> (screen_x, screen_y)
                                  注意：轨迹坐标是(y, x)格式
        """
        display_w, display_h = canvas.shape[1], canvas.shape[0]
        image_area_h = display_h - Config.PANEL_HEIGHT

        for nid, traj in self.data.neuron_trajectories.items():
            if not traj:
                continue

            color = self.data.get_neuron_color(nid)

            # 绘制轨迹线
            for i in range(len(traj) - 1):
                # 轨迹点是(y, x)格式，转换时需要交换
                p1 = coord_transform_func(traj[i][1], traj[i][0])
                p2 = coord_transform_func(traj[i + 1][1], traj[i + 1][0])

                # 检查是否在显示区域内
                if not (0 <= p1[0] < display_w and 0 <= p1[1] < image_area_h):
                    continue

                # 检查距离，过大则跳过（避免跨区域连线）
                dist = np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
                if dist < 30:
                    cv2.line(canvas, p1, p2, color, 2, cv2.LINE_AA)

            # 绘制终点
            if traj:
                end = coord_transform_func(traj[-1][1], traj[-1][0])
                if 0 <= end[0] < display_w and 0 <= end[1] < image_area_h:
                    cv2.circle(canvas, end, 6, Config.COLOR_TRAJECTORY_END, -1)

    def draw_selection_preview(self, canvas, mouse_x, mouse_y, mark_radius,
                               zoom_level, neuron_color):
        """
        绘制选区预览圆圈

        参数:
            canvas: 目标画布
            mouse_x, mouse_y: 鼠标位置
            mark_radius: 标记半径
            zoom_level: 当前缩放级别
            neuron_color: 当前神经元颜色
        """
        preview_r = max(2, int(mark_radius * zoom_level))
        cv2.circle(canvas, (mouse_x, mouse_y), preview_r, neuron_color, 1)
        cv2.circle(canvas, (mouse_x, mouse_y), 2, Config.COLOR_MARK_HIGHLIGHT, -1)

    def draw_info_bar(self, canvas, frame_idx, total_frames, zoom_level, mode):
        """
        绘制信息栏

        参数:
            canvas: 目标画布
            frame_idx: 当前帧索引
            total_frames: 总帧数
            zoom_level: 缩放级别
            mode: 当前模式
        """
        info1 = f"Frame: {frame_idx + 1}/{total_frames}  |  Zoom: {zoom_level:.2f}x  |  Mode: {mode.upper()}"
        cv2.putText(canvas, info1, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    Config.COLOR_TEXT_INFO, 2, cv2.LINE_AA)

        stats = self.data.get_statistics()
        info2 = f"Marks: {stats['mark_count']}  |  Trajectories: {stats['trajectory_count']}"
        cv2.putText(canvas, info2, (650, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    Config.COLOR_TEXT_DIM, 2, cv2.LINE_AA)

    def draw_legend(self, canvas, current_neuron_id):
        """
        绘制神经元图例

        参数:
            canvas: 目标画布
            current_neuron_id: 当前选中的神经元ID
        """
        display_w = canvas.shape[1]
        lx = display_w - 230
        ly = 15

        cv2.putText(canvas, "Neurons (m=marks, t=traj):", (lx, ly + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, Config.COLOR_TEXT_DIM, 1, cv2.LINE_AA)

        all_nids = sorted(self.data.get_all_neuron_ids())
        for i, nid in enumerate(all_nids[:12]):
            color = self.data.get_neuron_color(nid)
            yp = ly + 38 + i * 20

            cv2.rectangle(canvas, (lx, yp), (lx + 14, yp + 14), color, -1)

            traj_len = len(self.data.get_trajectory(nid))
            mark_len = len(self.data.get_neuron_marks(nid))
            prefix = "► " if nid == current_neuron_id else ""

            cv2.putText(canvas, f"{prefix}N{nid}: {mark_len}m/{traj_len}t",
                        (lx + 20, yp + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        Config.COLOR_TEXT, 1, cv2.LINE_AA)

    def draw_tips(self, canvas):
        """
        绘制操作提示

        参数:
            canvas: 目标画布
        """
        image_area_h = canvas.shape[0] - Config.PANEL_HEIGHT
        tips = "L-Click: Mark | R-Click: Del | Wheel: Zoom | M-Drag: Pan"
        cv2.putText(canvas, tips, (10, image_area_h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, Config.COLOR_TEXT_DIM, 1, cv2.LINE_AA)

    def draw_frame_on_video(self, frame, frame_idx, total_frames):
        """
        在视频帧上绘制轨迹和信息（用于导出）

        参数:
            frame: 视频帧
            frame_idx: 帧索引
            total_frames: 总帧数

        返回:
            numpy.ndarray: 绘制后的帧
        """
        vis = frame.copy()

        # 绘制所有轨迹
        for nid, traj in self.data.neuron_trajectories.items():
            if not traj:
                continue

            color = self.data.get_neuron_color(nid)

            # 绘制线
            for i in range(len(traj) - 1):
                p1 = (int(traj[i][1]), int(traj[i][0]))
                p2 = (int(traj[i + 1][1]), int(traj[i + 1][0]))
                dist = np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
                if dist < 30:
                    cv2.line(vis, p1, p2, color, 2, cv2.LINE_AA)

            # 起点和终点
            if traj:
                start = (int(traj[0][1]), int(traj[0][0]))
                end = (int(traj[-1][1]), int(traj[-1][0]))
                cv2.circle(vis, start, 5, color, -1)
                cv2.circle(vis, end, 7, Config.COLOR_TRAJECTORY_END, -1)
                cv2.circle(vis, end, 9, Config.COLOR_MARK_HIGHLIGHT, 2)
                cv2.putText(vis, str(nid), (end[0] + 10, end[1] + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

        # 帧信息
        info = f"Frame: {frame_idx + 1}/{total_frames} | Neurons: {len(self.data.neuron_trajectories)}"
        cv2.putText(vis, info, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(vis, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

        # 统计
        total_pts = sum(len(t) for t in self.data.neuron_trajectories.values())
        cv2.putText(vis, f"Total Points: {total_pts}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2, cv2.LINE_AA)

        return vis
