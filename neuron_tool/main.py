"""
主程序模块
整合所有模块，提供完整的神经元标记与追踪功能
"""

import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog

# 导入各模块
from .config import Config
from .ui_components import Button, InputBox
from .video_handler import VideoHandler
from .image_processing import ImageProcessor
from .tracking_algorithm import NeuronTracker
from .data_manager import DataManager
from .visualization import Visualizer


class NeuronTool:
    """
    神经元标记与追踪主工具

    功能:
        - 视频浏览和缩放
        - 神经元标记
        - 自动追踪
        - 数据保存和加载
        - 视频导出
    """

    def __init__(self):
        """初始化工具"""
        # 初始化各模块
        self.video = VideoHandler()
        self.img_proc = ImageProcessor()
        self.data = DataManager()
        self.tracker = NeuronTracker(self.img_proc)
        self.vis = Visualizer(self.data)

        # 显示状态
        self.zoom_level = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.is_panning = False
        self.pan_start = (0, 0)

        # 标记状态
        self.mark_radius = Config.DEFAULT_MARK_RADIUS
        self.current_neuron_id = 0
        self.mode = 'mark'  # 'mark' 或 'track'

        # 输入框
        self.frame_input = None
        self.neuron_input = None
        self.active_input = None

        # UI组件
        self.buttons = {}

        # 鼠标状态
        self.mouse_x = 0
        self.mouse_y = 0
        self.mouse_in_image = False

        # 追踪状态
        self.is_tracking = False

        # 动作结果
        self.action_result = None

        # 隐藏的tkinter根窗口，用于文件对话框
        self.root = tk.Tk()
        self.root.withdraw()
        
        # 路径变量
        self.data_path = None
        self.output_video_path = None

    def _init_ui(self):
        """初始化UI组件"""
        y1 = self.image_area_height + 12
        y2 = self.image_area_height + 55
        y3 = self.image_area_height + 95
        bh = 36

        # ===== 第一行：帧导航 =====
        x = 15
        self.buttons['prev'] = Button(x, y1, 55, bh, '◀Prev');
        x += 60
        self.buttons['next'] = Button(x, y1, 55, bh, 'Next▶');
        x += 60
        self.buttons['prev10'] = Button(x, y1, 40, bh, '-10');
        x += 45
        self.buttons['next10'] = Button(x, y1, 40, bh, '+10');
        x += 55

        # 帧输入框
        self.frame_input = InputBox(x, y1 + 2, 90, 32, "Frame:")
        x += 105

        # 缩放按钮
        self.buttons['zin'] = Button(x, y1, 35, bh, 'Z+');
        x += 40
        self.buttons['zout'] = Button(x, y1, 35, bh, 'Z-');
        x += 40
        self.buttons['zfit'] = Button(x, y1, 35, bh, 'Fit');
        x += 45

        # 选区按钮
        self.buttons['r_down'] = Button(x, y1, 30, bh, 'R-');
        x += 35
        self.buttons['r_up'] = Button(x, y1, 30, bh, 'R+');
        x += 45

        # 神经元输入框
        self.neuron_input = InputBox(x, y1 + 2, 70, 32, "Neuron:")
        self.neuron_input.text = "0"
        x += 85

        # 神经元操作
        self.buttons['new_n'] = Button(x, y1, 50, bh, 'New');
        x += 55
        self.buttons['del_n'] = Button(x, y1, 50, bh, 'Del N');
        x += 55
        self.buttons['del_mark'] = Button(x, y1, 70, bh, 'Del Mark');
        x += 75

        # ===== 第二行：模式和操作 =====
        x = 15
        self.buttons['mode_mark'] = Button(x, y2, 70, bh, 'MARK', Config.COLOR_BUTTON);
        x += 75
        self.buttons['mode_track'] = Button(x, y2, 70, bh, 'TRACK', Config.COLOR_BUTTON);
        x += 85

        self.buttons['run_track'] = Button(x, y2, 100, bh, '▶ RUN', Config.COLOR_BUTTON_SUCCESS);
        x += 105
        self.buttons['stop'] = Button(x, y2, 50, bh, 'STOP', Config.COLOR_BUTTON_DANGER);
        x += 60

        self.buttons['undo'] = Button(x, y2, 50, bh, 'Undo');
        x += 55
        self.buttons['save'] = Button(x, y2, 50, bh, 'Save', Config.COLOR_BUTTON_SUCCESS);
        x += 55
        self.buttons['save_as'] = Button(x, y2, 65, bh, 'SaveAs', Config.COLOR_BUTTON_SUCCESS);
        x += 70
        self.buttons['load'] = Button(x, y2, 50, bh, 'Load');
        x += 60

        self.buttons['set_out'] = Button(x, y2, 70, bh, 'Set Out', Config.COLOR_BUTTON);
        x += 75

        self.buttons['clear_traj'] = Button(x, y2, 80, bh, 'Clear Traj', Config.COLOR_BUTTON_DANGER);
        x += 85
        self.buttons['clear_all'] = Button(x, y2, 80, bh, 'Clear ALL', Config.COLOR_BUTTON_DANGER);
        x += 95

        self.buttons['confirm'] = Button(x, y2, 80, bh, 'CONFIRM', Config.COLOR_BUTTON_SUCCESS);
        x += 85
        self.buttons['cancel'] = Button(x, y2, 60, bh, 'Cancel', Config.COLOR_BUTTON_DANGER)

        # ===== 第三行：快速神经元选择 =====
        for i in range(20):
            self.buttons[f'n{i}'] = Button(15 + i * 48, y3, 44, 28, f'N{i}')

    @property
    def image_area_height(self):
        """图像显示区域高度"""
        return Config.DISPLAY_HEIGHT - Config.PANEL_HEIGHT

    def screen_to_image(self, sx, sy):
        """屏幕坐标转图像坐标"""
        dw = int(self.video.frame_width * self.zoom_level)
        dh = int(self.video.frame_height * self.zoom_level)
        ox = (Config.DISPLAY_WIDTH - dw) // 2 + self.pan_x
        oy = (self.image_area_height - dh) // 2 + self.pan_y
        return (sx - ox) / self.zoom_level, (sy - oy) / self.zoom_level

    def image_to_screen(self, ix, iy):
        """图像坐标转屏幕坐标"""
        dw = int(self.video.frame_width * self.zoom_level)
        dh = int(self.video.frame_height * self.zoom_level)
        ox = (Config.DISPLAY_WIDTH - dw) // 2 + self.pan_x
        oy = (self.image_area_height - dh) // 2 + self.pan_y
        return int(ix * self.zoom_level + ox), int(iy * self.zoom_level + oy)

    def handle_button_click(self, name):
        """处理按钮点击"""
        if name == 'prev':
            self.video.read_frame(self.video.current_frame_idx - 1)
            self._update_frame_input()
        elif name == 'next':
            self.video.read_frame(self.video.current_frame_idx + 1)
            self._update_frame_input()
        elif name == 'prev10':
            self.video.read_frame(self.video.current_frame_idx - 10)
            self._update_frame_input()
        elif name == 'next10':
            self.video.read_frame(self.video.current_frame_idx + 10)
            self._update_frame_input()
        elif name == 'zin':
            self.zoom_level = min(Config.MAX_ZOOM, self.zoom_level * Config.ZOOM_STEP)
        elif name == 'zout':
            self.zoom_level = max(Config.MIN_ZOOM, self.zoom_level / Config.ZOOM_STEP)
        elif name == 'zfit':
            self._fit_zoom()
        elif name == 'r_down':
            self.mark_radius = max(Config.MIN_MARK_RADIUS, self.mark_radius - 1)
        elif name == 'r_up':
            self.mark_radius = min(Config.MAX_MARK_RADIUS, self.mark_radius + 1)
        elif name == 'new_n':
            self.current_neuron_id = self.data.get_new_neuron_id()
            self.neuron_input.text = str(self.current_neuron_id)
            print(f"  新建 N{self.current_neuron_id}")
        elif name == 'del_n':
            self.data.delete_neuron(self.current_neuron_id)
        elif name == 'del_mark':
            removed = self.data.remove_mark_at_frame(
                self.current_neuron_id, self.video.current_frame_idx)
            if removed:
                print(f"  删除 N{self.current_neuron_id} @ 帧{self.video.current_frame_idx + 1}")
        elif name == 'undo':
            removed = self.data.remove_last_mark(self.current_neuron_id)
            if removed:
                print(f"  撤销 N{self.current_neuron_id} 最后一个标记")
        elif name == 'save':
            return 'save'
        elif name == 'save_as':
            return 'save_as'
        elif name == 'load':
            return 'load'
        elif name == 'set_out':
            return 'set_out'
        elif name == 'mode_mark':
            self.mode = 'mark'
        elif name == 'mode_track':
            self.mode = 'track'
        elif name == 'run_track':
            return 'run_track'
        elif name == 'stop':
            self.is_tracking = False
        elif name == 'clear_traj':
            self.data.clear_all_trajectories()
        elif name == 'clear_all':
            self.data.clear_all_data()
        elif name == 'confirm':
            return 'confirm'
        elif name == 'cancel':
            return 'cancel'
        elif name.startswith('n') and name[1:].isdigit():
            self.current_neuron_id = int(name[1:])
            self.neuron_input.text = str(self.current_neuron_id)
        return None

    def _update_frame_input(self):
        """更新帧输入框显示"""
        self.frame_input.text = str(self.video.current_frame_idx + 1)

    def _fit_zoom(self):
        """适应窗口缩放"""
        sw = (Config.DISPLAY_WIDTH - 40) / self.video.frame_width
        sh = (self.image_area_height - 40) / self.video.frame_height
        self.zoom_level = min(sw, sh, 1.0)
        self.pan_x = 0
        self.pan_y = 0

    def mouse_callback(self, event, x, y, flags, param):
        """鼠标回调函数"""
        self.mouse_x, self.mouse_y = x, y
        in_panel = y >= self.image_area_height

        # 检查是否在图像区域内
        if not in_panel:
            ix, iy = self.screen_to_image(x, y)
            self.mouse_in_image = (0 <= ix < self.video.frame_width and
                                   0 <= iy < self.video.frame_height)
        else:
            self.mouse_in_image = False

        # 更新按钮hover状态
        for btn in self.buttons.values():
            btn.hover = btn.contains(x, y)

        # 更新快速选择按钮active状态
        for i in range(20):
            self.buttons[f'n{i}'].active = (i == self.current_neuron_id)

        self.buttons['mode_mark'].active = (self.mode == 'mark')
        self.buttons['mode_track'].active = (self.mode == 'track')

        if event == cv2.EVENT_LBUTTONDOWN:
            # 检查输入框点击
            if self.frame_input.contains(x, y):
                self.active_input = self.frame_input
                self.frame_input.active = True
                self.neuron_input.active = False
                return
            elif self.neuron_input.contains(x, y):
                self.active_input = self.neuron_input
                self.neuron_input.active = True
                self.frame_input.active = False
                return
            else:
                # 确认之前的输入
                self._confirm_input()

            # 检查按钮点击
            if in_panel:
                for name, btn in self.buttons.items():
                    if btn.contains(x, y):
                        result = self.handle_button_click(name)
                        if result:
                            self.action_result = result
                        return
            else:
                # 图像区域点击：添加标记
                if self.mode == 'mark':
                    ix, iy = self.screen_to_image(x, y)
                    if 0 <= ix < self.video.frame_width and 0 <= iy < self.video.frame_height:
                        mx, my = int(round(ix)), int(round(iy))
                        self.data.add_mark(self.current_neuron_id,
                                           self.video.current_frame_idx, mx, my)
                        print(f"  + N{self.current_neuron_id} @ 帧{self.video.current_frame_idx + 1}: ({mx}, {my})")

        elif event == cv2.EVENT_RBUTTONDOWN:
            # 右键删除标记
            if not in_panel and self.mode == 'mark':
                removed = self.data.remove_mark_at_frame(
                    self.current_neuron_id, self.video.current_frame_idx)
                if removed:
                    print(f"  - 删除 N{self.current_neuron_id} @ 帧{self.video.current_frame_idx + 1}")

        elif event == cv2.EVENT_MBUTTONDOWN:
            # 中键开始平移
            self.is_panning = True
            self.pan_start = (x, y)

        elif event == cv2.EVENT_MBUTTONUP:
            self.is_panning = False

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.is_panning:
                self.pan_x += x - self.pan_start[0]
                self.pan_y += y - self.pan_start[1]
                self.pan_start = (x, y)

        elif event == cv2.EVENT_MOUSEWHEEL:
            if not in_panel:
                if flags > 0:
                    self.zoom_level = min(Config.MAX_ZOOM, self.zoom_level * 1.2)
                else:
                    self.zoom_level = max(Config.MIN_ZOOM, self.zoom_level / 1.2)

    def _confirm_input(self):
        """确认输入框内容"""
        if self.frame_input.active:
            try:
                self.video.read_frame(int(self.frame_input.text) - 1)
            except:
                pass
            self.frame_input.active = False

        if self.neuron_input.active:
            try:
                self.current_neuron_id = max(0, min(
                    int(self.neuron_input.text), Config.MAX_NEURONS - 1))
                self.neuron_input.text = str(self.current_neuron_id)
            except:
                pass
            self.neuron_input.active = False

        self.active_input = None

    def draw(self):
        """绘制完整界面"""
        if self.video.current_frame is None:
            return None

        # 创建画布
        canvas = np.zeros((Config.DISPLAY_HEIGHT, Config.DISPLAY_WIDTH, 3), dtype=np.uint8)
        canvas[:] = Config.COLOR_BACKGROUND

        # ===== 绘制图像 =====
        frame = self.video.current_frame
        dw = int(self.video.frame_width * self.zoom_level)
        dh = int(self.video.frame_height * self.zoom_level)

        interp = cv2.INTER_LINEAR if self.zoom_level < 1 else cv2.INTER_NEAREST
        resized = cv2.resize(frame, (dw, dh), interpolation=interp)

        ox = (Config.DISPLAY_WIDTH - dw) // 2 + self.pan_x
        oy = (self.image_area_height - dh) // 2 + self.pan_y

        # 裁剪显示
        sx1 = max(0, -ox)
        sy1 = max(0, -oy)
        sx2 = min(dw, Config.DISPLAY_WIDTH - ox)
        sy2 = min(dh, self.image_area_height - oy)
        dx1 = max(0, ox)
        dy1 = max(0, oy)
        dx2 = min(Config.DISPLAY_WIDTH, ox + dw)
        dy2 = min(self.image_area_height, oy + dh)

        if sx2 > sx1 and sy2 > sy1:
            canvas[dy1:dy2, dx1:dx2] = resized[sy1:sy2, sx1:sx2]

        # ===== 绘制标记 =====
        self.vis.draw_marks_at_frame(
            canvas, self.video.current_frame_idx,
            self.current_neuron_id, self.image_to_screen)
        self.vis.draw_other_frame_marks(
            canvas, self.video.current_frame_idx, self.image_to_screen)

        # ===== 绘制轨迹（track模式）=====
        if self.mode == 'track':
            self.vis.draw_trajectories(canvas, self.image_to_screen)

        # ===== 选区预览 =====
        if self.mouse_in_image and self.mode == 'mark':
            color = self.data.get_neuron_color(self.current_neuron_id)
            self.vis.draw_selection_preview(
                canvas, self.mouse_x, self.mouse_y,
                self.mark_radius, self.zoom_level, color)

        # ===== 控制面板 =====
        cv2.rectangle(canvas, (0, self.image_area_height),
                      (Config.DISPLAY_WIDTH, Config.DISPLAY_HEIGHT),
                      Config.COLOR_PANEL, -1)
        cv2.line(canvas, (0, self.image_area_height),
                 (Config.DISPLAY_WIDTH, self.image_area_height), (80, 80, 80), 2)

        # 绘制按钮
        for btn in self.buttons.values():
            btn.draw(canvas)

        # 绘制输入框
        self.frame_input.draw(canvas)
        neuron_color = self.data.get_neuron_color(self.current_neuron_id)
        self.neuron_input.draw(canvas, highlight_color=neuron_color)

        # ===== 信息栏 =====
        self.vis.draw_info_bar(canvas, self.video.current_frame_idx,
                               self.video.total_frames, self.zoom_level, self.mode)
        self.vis.draw_legend(canvas, self.current_neuron_id)
        self.vis.draw_tips(canvas)

        return canvas

    def run_tracking(self, output_path):
        """
        执行实时追踪

        参数:
            output_path: 输出视频路径
        """
        if not self.data.neuron_marks:
            print("⚠ 没有标记数据")
            return

        print("\n开始实时追踪...")
        self.data.clear_all_trajectories()
        self.is_tracking = True

        # 找最早标记帧
        all_frames = []
        for nid in self.data.neuron_marks:
            for mark in self.data.get_neuron_marks(nid):
                all_frames.append(mark["frame"])
        start_frame = min(all_frames) if all_frames else 0

        # 开始写入视频
        self.video.start_writer(output_path)

        # 初始化帧
        self.video.read_frame(start_frame)
        skeleton = self.img_proc.preprocess(self.video.current_frame)

        # 初始化每个神经元
        for nid in self.data.neuron_marks:
            marks = sorted(self.data.get_neuron_marks(nid), key=lambda m: m["frame"])
            if not marks:
                continue

            mark = marks[0]
            pt = self.img_proc.find_nearest_skeleton_point(
                skeleton, self.video.current_frame, (mark["x"], mark["y"]))

            if pt:
                traj = self.tracker.trace_bidirectional(
                    skeleton, self.video.current_frame, pt)
                if len(traj) >= 5:
                    traj = sorted(traj, key=lambda p: p[1])
                    self.data.set_trajectory(nid, traj)
                    print(f"  N{nid}: 初始化 {len(traj)} 点")

        print(f"✓ 初始化 {len(self.data.neuron_trajectories)} 根神经元")

        # 追踪窗口
        win = 'Tracking Progress (Q to stop)'
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 1280, 720)

        # 追踪循环
        frame_delay = int(1000 / self.video.fps) if self.video.fps > 0 else 30
        for fidx in range(0, self.video.total_frames):
            if not self.is_tracking:
                print("追踪被停止")
                break

            self.video.read_frame(fidx)

            if fidx > start_frame:
                skeleton = self.img_proc.preprocess(self.video.current_frame)

                for nid, traj in list(self.data.neuron_trajectories.items()):
                    if not traj:
                        continue

                    y_center = np.mean([p[0] for p in traj])
                    new_pts = self.tracker.find_growth(
                        skeleton, self.video.current_frame, traj, y_center)

                    if new_pts:
                        traj.extend(new_pts)
                        self.data.set_trajectory(nid, sorted(traj, key=lambda p: p[1]))

            # 绘制并写入
            vis = self.vis.draw_frame_on_video(
                self.video.current_frame, fidx, self.video.total_frames, fidx >= start_frame)
            self.video.write_frame(vis)

            # 显示进度
            scale = min(1.0, 1280 / self.video.frame_width)
            display = cv2.resize(vis, None, fx=scale, fy=scale)
            cv2.imshow(win, display)

            if cv2.waitKey(frame_delay) & 0xFF in [ord('q'), 27]:
                self.is_tracking = False
                break

        # 清理
        self.video.stop_writer()
        self.is_tracking = False

        print(f"\n✓ 追踪完成! 视频: {output_path}")
        for nid, traj in self.data.neuron_trajectories.items():
            print(f"  N{nid}: {len(traj)} 点")

        last_frame = self.vis.draw_frame_on_video(
            self.video.current_frame, self.video.current_frame_idx, self.video.total_frames)
        scale = min(1.0, 1280 / self.video.frame_width)
        display = cv2.resize(last_frame, None, fx=scale, fy=scale)
        while True:
            cv2.imshow(win, display)
            if cv2.waitKey(30) & 0xFF in [ord('q'), 27]:
                break
        cv2.destroyWindow(win)

    def run(self, video_path, data_path=None):
        """
        运行主程序

        参数:
            video_path: 视频文件路径
            data_path: 数据文件路径（可选）
        """
        # 加载视频
        self.video.load(video_path)
        self.data.set_video_info(self.video.get_info())

        # 设置默认数据路径
        if data_path is None:
            self.data_path = os.path.splitext(video_path)[0] + "_data.json"
        else:
            self.data_path = data_path
            
        self.output_video_path = os.path.splitext(video_path)[0] + "_tracked.mp4"

        # 加载已有数据
        if os.path.exists(self.data_path):
            self.data.load(self.data_path)

        # 初始化UI
        self._init_ui()

        # 读取第一帧
        self.video.read_frame(0)
        self._update_frame_input()
        self._fit_zoom()

        self.action_result = None

        # 创建窗口
        win = 'Neuron Tool'
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, Config.DISPLAY_WIDTH, Config.DISPLAY_HEIGHT)
        cv2.setMouseCallback(win, self.mouse_callback)

        print("\n" + "=" * 70)
        print("神经元标记与追踪工具 v7.0")
        print("=" * 70)
        print("操作说明:")
        print("  左键: 添加标记    右键: 删除标记")
        print("  滚轮: 缩放        中键拖动: 平移")
        print("  A/D: 前后帧       W/S: ±10帧")
        print("  0-9: 切换神经元   X: 删除当前神经元")
        print("=" * 70 + "\n")

        # 主循环
        while True:
            disp = self.draw()
            if disp is not None:
                cv2.imshow(win, disp)

            # 处理动作
            if self.action_result == 'confirm':
                self.data.save(self.data_path)
                break
            elif self.action_result == 'cancel':
                break
            elif self.action_result == 'save':
                self.data.save(self.data_path)
                self.action_result = None
            elif self.action_result == 'save_as':
                path = filedialog.asksaveasfilename(
                    defaultextension=".json",
                    filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                    initialfile=os.path.basename(self.data_path),
                    initialdir=os.path.dirname(self.data_path)
                )
                if path:
                    self.data_path = path
                    self.data.save(self.data_path)
                self.action_result = None
            elif self.action_result == 'set_out':
                path = filedialog.asksaveasfilename(
                    defaultextension=".mp4",
                    filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")],
                    initialfile=os.path.basename(self.output_video_path),
                    initialdir=os.path.dirname(self.output_video_path)
                )
                if path:
                    self.output_video_path = path
                    print(f"输出路径设置为: {self.output_video_path}")
                self.action_result = None
            elif self.action_result == 'load':
                path = filedialog.askopenfilename(
                    defaultextension=".json",
                    filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                    initialdir=os.path.dirname(self.data_path)
                )
                if path:
                    self.data_path = path
                    self.data.load(self.data_path)
                self.action_result = None
            elif self.action_result == 'run_track':
                self.run_tracking(self.output_video_path)
                self.data.save(self.data_path)
                self.action_result = None

            # 处理键盘
            key = cv2.waitKey(30) & 0xFF

            if self.active_input:
                result = self.active_input.handle_key(key)
                if result == 'confirm':
                    self._confirm_input()
                elif result == 'cancel':
                    self.active_input.active = False
                    self.active_input = None
                continue

            if key == 27:  # Esc
                break
            elif key == ord('a'): # A: Prev Frame
                self.video.read_frame(self.video.current_frame_idx - 1)
                self._update_frame_input()
            elif key == ord('d'): # D: Next Frame
                self.video.read_frame(self.video.current_frame_idx + 1)
                self._update_frame_input()
            elif key == ord('w'): # W: Next 10 Frames
                self.video.read_frame(self.video.current_frame_idx + 10)
                self._update_frame_input()
            elif key == ord('s'): # S: Prev 10 Frames
                self.video.read_frame(self.video.current_frame_idx - 10)
                self._update_frame_input()
            
            # 方向键平移 (Up: 82, Down: 84, Left: 81, Right: 83)
            # 注意: 不同平台键值可能不同，这里兼容常见的Windows/Linux键值
            elif key == 81: # Left Arrow
                self.pan_x += 50
            elif key == 83: # Right Arrow
                self.pan_x -= 50
            elif key == 82: # Up Arrow
                self.pan_y += 50
            elif key == 84: # Down Arrow
                self.pan_y -= 50
                
            elif ord('0') <= key <= ord('9'):
                self.current_neuron_id = key - ord('0')
                self.neuron_input.text = str(self.current_neuron_id)
            elif key == ord('+') or key == ord('='):
                self.zoom_level = min(Config.MAX_ZOOM, self.zoom_level * Config.ZOOM_STEP)
            elif key == ord('-') or key == ord('_'):
                self.zoom_level = max(Config.MIN_ZOOM, self.zoom_level / Config.ZOOM_STEP)
            elif key == ord('x') or key == ord('X'):
                self.data.delete_neuron(self.current_neuron_id)
            elif key == 13:  # Enter
                self.data.save(self.data_path)
                break

        cv2.destroyAllWindows()
        self.video.release()
        self.root.destroy()  # 销毁tkinter窗口

        return self.data_path
