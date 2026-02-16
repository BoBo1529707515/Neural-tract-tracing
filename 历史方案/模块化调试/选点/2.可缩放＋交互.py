import cv2
import numpy as np
import json
import os
from datetime import datetime


def generate_colors(n):
    colors = []
    for i in range(n):
        hue = int(180 * i / n)
        hsv = np.uint8([[[hue, 255, 255]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
        colors.append((int(bgr[0]), int(bgr[1]), int(bgr[2])))
    return colors


class Button:
    """简单按钮类"""

    def __init__(self, x, y, w, h, text, color=(80, 80, 80), text_color=(255, 255, 255)):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.text = text
        self.color = color
        self.text_color = text_color
        self.hover = False
        self.active = False

    def contains(self, px, py):
        return self.x <= px < self.x + self.w and self.y <= py < self.y + self.h

    def draw(self, img):
        # 背景
        if self.active:
            bg_color = (100, 180, 100)
        elif self.hover:
            bg_color = (100, 100, 100)
        else:
            bg_color = self.color

        cv2.rectangle(img, (self.x, self.y), (self.x + self.w, self.y + self.h), bg_color, -1)
        cv2.rectangle(img, (self.x, self.y), (self.x + self.w, self.y + self.h), (150, 150, 150), 1)

        # 文字
        font_scale = 0.45
        thickness = 1
        (text_w, text_h), _ = cv2.getTextSize(self.text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        text_x = self.x + (self.w - text_w) // 2
        text_y = self.y + (self.h + text_h) // 2
        cv2.putText(img, self.text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, self.text_color, thickness)


class NeuronMarkerTool:
    """
    神经元标记工具 V2
    - 支持缩放和平移
    - 帧号输入
    - 按钮式界面
    """

    def __init__(self, max_neurons=15):
        self.max_neurons = max_neurons
        self.colors = generate_colors(max_neurons)

        # 标记数据
        self.neuron_marks = {}

        # 视频状态
        self.current_neuron_id = 0
        self.current_frame_idx = 0
        self.total_frames = 0
        self.video_path = ""
        self.cap = None
        self.current_frame = None
        self.video_info = {}

        # 缩放和平移状态
        self.zoom_level = 1.0
        self.min_zoom = 0.5
        self.max_zoom = 5.0
        self.pan_x = 0  # 平移偏移（图像坐标）
        self.pan_y = 0
        self.is_panning = False
        self.pan_start_x = 0
        self.pan_start_y = 0

        # 显示尺寸
        self.display_width = 1400
        self.display_height = 900
        self.control_panel_height = 100
        self.image_area_height = self.display_height - self.control_panel_height

        # 帧号输入模式
        self.input_mode = False
        self.input_text = ""

        # 按钮
        self.buttons = {}
        self._create_buttons()

    def _create_buttons(self):
        """创建按钮"""
        y = self.image_area_height + 10
        btn_h = 35
        gap = 10

        # 第一行：帧导航
        x = 10
        self.buttons['prev_frame'] = Button(x, y, 80, btn_h, '<< Prev')
        x += 90
        self.buttons['next_frame'] = Button(x, y, 80, btn_h, 'Next >>')
        x += 90
        self.buttons['prev_10'] = Button(x, y, 60, btn_h, '-10')
        x += 70
        self.buttons['next_10'] = Button(x, y, 60, btn_h, '+10')
        x += 70
        self.buttons['goto_frame'] = Button(x, y, 100, btn_h, 'Go To Frame')

        # 第一行：神经元选择
        x += 120
        self.buttons['prev_neuron'] = Button(x, y, 80, btn_h, '<< N')
        x += 90
        self.buttons['next_neuron'] = Button(x, y, 80, btn_h, 'N >>')
        x += 90
        self.buttons['new_neuron'] = Button(x, y, 80, btn_h, 'New N')
        x += 90
        self.buttons['del_neuron'] = Button(x, y, 80, btn_h, 'Del N')

        # 第一行：缩放
        x += 100
        self.buttons['zoom_in'] = Button(x, y, 60, btn_h, 'Zoom+')
        x += 70
        self.buttons['zoom_out'] = Button(x, y, 60, btn_h, 'Zoom-')
        x += 70
        self.buttons['zoom_reset'] = Button(x, y, 60, btn_h, 'Reset')

        # 第二行
        y += btn_h + 10
        x = 10
        self.buttons['undo'] = Button(x, y, 80, btn_h, 'Undo')
        x += 90
        self.buttons['save'] = Button(x, y, 80, btn_h, 'Save', color=(60, 100, 60))
        x += 90
        self.buttons['load'] = Button(x, y, 80, btn_h, 'Load')
        x += 100
        self.buttons['confirm'] = Button(x, y, 100, btn_h, 'CONFIRM', color=(60, 120, 60))
        x += 110
        self.buttons['cancel'] = Button(x, y, 80, btn_h, 'Cancel', color=(60, 60, 100))

        # 神经元ID快选按钮 (N0-N9)
        x += 120
        for i in range(10):
            self.buttons[f'n{i}'] = Button(x + i * 40, y, 35, btn_h, f'N{i}')

    def load_video(self, video_path):
        """加载视频"""
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))

        self.video_info = {
            "path": video_path,
            "width": self.frame_width,
            "height": self.frame_height,
            "fps": fps,
            "total_frames": self.total_frames
        }

        # 初始缩放以适应显示区域
        scale_w = self.display_width / self.frame_width
        scale_h = self.image_area_height / self.frame_height
        self.zoom_level = min(scale_w, scale_h, 1.0)

        print(f"✓ 加载视频: {self.frame_width}x{self.frame_height}, {fps}fps, {self.total_frames}帧")

        return True

    def read_frame(self, frame_idx):
        """读取指定帧"""
        if self.cap is None:
            return None

        frame_idx = max(0, min(frame_idx, self.total_frames - 1))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()

        if ret:
            self.current_frame_idx = frame_idx
            self.current_frame = frame
            return frame
        return None

    def screen_to_image(self, sx, sy):
        """屏幕坐标转图像坐标"""
        # 计算图像在显示区域的位置
        disp_w = int(self.frame_width * self.zoom_level)
        disp_h = int(self.frame_height * self.zoom_level)

        offset_x = (self.display_width - disp_w) // 2 + self.pan_x
        offset_y = (self.image_area_height - disp_h) // 2 + self.pan_y

        img_x = int((sx - offset_x) / self.zoom_level)
        img_y = int((sy - offset_y) / self.zoom_level)

        return img_x, img_y

    def image_to_screen(self, ix, iy):
        """图像坐标转屏幕坐标"""
        disp_w = int(self.frame_width * self.zoom_level)
        disp_h = int(self.frame_height * self.zoom_level)

        offset_x = (self.display_width - disp_w) // 2 + self.pan_x
        offset_y = (self.image_area_height - disp_h) // 2 + self.pan_y

        sx = int(ix * self.zoom_level + offset_x)
        sy = int(iy * self.zoom_level + offset_y)

        return sx, sy

    def add_mark(self, neuron_id, frame_idx, x, y):
        """添加标记点"""
        if neuron_id not in self.neuron_marks:
            color_idx = neuron_id % self.max_neurons
            self.neuron_marks[neuron_id] = {
                "color": list(self.colors[color_idx]),
                "marks": []
            }

        # 检查是否已存在相同帧的标记
        for mark in self.neuron_marks[neuron_id]["marks"]:
            if mark["frame"] == frame_idx:
                mark["x"] = x
                mark["y"] = y
                return

        self.neuron_marks[neuron_id]["marks"].append({
            "frame": frame_idx,
            "x": x,
            "y": y
        })

    def remove_last_mark(self, neuron_id):
        """删除神经元的最后一个标记"""
        if neuron_id in self.neuron_marks:
            if len(self.neuron_marks[neuron_id]["marks"]) > 0:
                return self.neuron_marks[neuron_id]["marks"].pop()
        return None

    def remove_mark_at_frame(self, neuron_id, frame_idx):
        """删除指定帧的标记"""
        if neuron_id in self.neuron_marks:
            marks = self.neuron_marks[neuron_id]["marks"]
            for i, mark in enumerate(marks):
                if mark["frame"] == frame_idx:
                    return marks.pop(i)
        return None

    def delete_neuron(self, neuron_id):
        """删除整个神经元"""
        if neuron_id in self.neuron_marks:
            del self.neuron_marks[neuron_id]
            return True
        return False

    def get_marks_at_frame(self, frame_idx):
        """获取指定帧的所有标记"""
        marks = []
        for neuron_id, data in self.neuron_marks.items():
            for mark in data["marks"]:
                if mark["frame"] == frame_idx:
                    marks.append({
                        "neuron_id": neuron_id,
                        "x": mark["x"],
                        "y": mark["y"],
                        "color": data["color"]
                    })
        return marks

    def save_marks(self, save_path):
        """保存标记"""
        data = {
            "version": "2.0",
            "created": datetime.now().isoformat(),
            "video_info": self.video_info,
            "neurons": self.neuron_marks
        }

        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"✓ 保存标记到: {save_path}")
        return True

    def load_marks(self, load_path):
        """加载标记"""
        if not os.path.exists(load_path):
            print(f"⚠ 文件不存在: {load_path}")
            return False

        with open(load_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.neuron_marks = {int(k): v for k, v in data.get("neurons", {}).items()}

        print(f"✓ 加载标记: {len(self.neuron_marks)} 个神经元")
        return True

    def handle_button_click(self, btn_name):
        """处理按钮点击"""
        if btn_name == 'prev_frame':
            self.read_frame(self.current_frame_idx - 1)
        elif btn_name == 'next_frame':
            self.read_frame(self.current_frame_idx + 1)
        elif btn_name == 'prev_10':
            self.read_frame(self.current_frame_idx - 10)
        elif btn_name == 'next_10':
            self.read_frame(self.current_frame_idx + 10)
        elif btn_name == 'goto_frame':
            self.input_mode = True
            self.input_text = ""
            print("  输入帧号后按Enter确认...")
        elif btn_name == 'prev_neuron':
            self.current_neuron_id = max(0, self.current_neuron_id - 1)
            print(f"  切换到 N{self.current_neuron_id}")
        elif btn_name == 'next_neuron':
            self.current_neuron_id = min(self.max_neurons - 1, self.current_neuron_id + 1)
            print(f"  切换到 N{self.current_neuron_id}")
        elif btn_name == 'new_neuron':
            existing_ids = set(self.neuron_marks.keys())
            new_id = 0
            while new_id in existing_ids:
                new_id += 1
            self.current_neuron_id = new_id
            print(f"  创建新神经元 N{self.current_neuron_id}")
        elif btn_name == 'del_neuron':
            if self.delete_neuron(self.current_neuron_id):
                print(f"  删除 N{self.current_neuron_id}")
        elif btn_name == 'zoom_in':
            self.zoom_level = min(self.max_zoom, self.zoom_level * 1.2)
        elif btn_name == 'zoom_out':
            self.zoom_level = max(self.min_zoom, self.zoom_level / 1.2)
        elif btn_name == 'zoom_reset':
            self.zoom_level = min(self.display_width / self.frame_width,
                                  self.image_area_height / self.frame_height, 1.0)
            self.pan_x = 0
            self.pan_y = 0
        elif btn_name == 'undo':
            removed = self.remove_last_mark(self.current_neuron_id)
            if removed:
                print(f"  撤销 N{self.current_neuron_id} @ 帧{removed['frame'] + 1}")
        elif btn_name == 'save':
            return 'save'
        elif btn_name == 'load':
            return 'load'
        elif btn_name == 'confirm':
            return 'confirm'
        elif btn_name == 'cancel':
            return 'cancel'
        elif btn_name.startswith('n') and btn_name[1:].isdigit():
            self.current_neuron_id = int(btn_name[1:])
            print(f"  切换到 N{self.current_neuron_id}")

        return None

    def mouse_callback(self, event, x, y, flags, param):
        """鼠标回调"""
        # 检查是否在控制面板区域
        in_control_panel = y >= self.image_area_height

        # 更新按钮hover状态
        for btn in self.buttons.values():
            btn.hover = btn.contains(x, y)

        # 更新当前神经元按钮active状态
        for i in range(10):
            self.buttons[f'n{i}'].active = (i == self.current_neuron_id)

        if event == cv2.EVENT_LBUTTONDOWN:
            if in_control_panel:
                # 检查按钮点击
                for btn_name, btn in self.buttons.items():
                    if btn.contains(x, y):
                        result = self.handle_button_click(btn_name)
                        if result:
                            self.action_result = result
                        return
            else:
                # 在图像区域添加标记
                img_x, img_y = self.screen_to_image(x, y)

                # 检查是否在图像范围内
                if 0 <= img_x < self.frame_width and 0 <= img_y < self.frame_height:
                    self.add_mark(self.current_neuron_id, self.current_frame_idx, img_x, img_y)
                    print(f"  + N{self.current_neuron_id} @ 帧{self.current_frame_idx + 1}: ({img_x}, {img_y})")

        elif event == cv2.EVENT_RBUTTONDOWN:
            if not in_control_panel:
                # 右键删除当前帧的标记
                removed = self.remove_mark_at_frame(self.current_neuron_id, self.current_frame_idx)
                if removed:
                    print(f"  - N{self.current_neuron_id} @ 帧{self.current_frame_idx + 1}: 已删除")

        elif event == cv2.EVENT_MBUTTONDOWN:
            # 中键开始平移
            self.is_panning = True
            self.pan_start_x = x
            self.pan_start_y = y

        elif event == cv2.EVENT_MBUTTONUP:
            self.is_panning = False

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.is_panning:
                dx = x - self.pan_start_x
                dy = y - self.pan_start_y
                self.pan_x += dx
                self.pan_y += dy
                self.pan_start_x = x
                self.pan_start_y = y

        elif event == cv2.EVENT_MOUSEWHEEL:
            if not in_control_panel:
                # 滚轮缩放
                if flags > 0:
                    self.zoom_level = min(self.max_zoom, self.zoom_level * 1.1)
                else:
                    self.zoom_level = max(self.min_zoom, self.zoom_level / 1.1)

    def draw_frame(self):
        """绘制完整界面"""
        if self.current_frame is None:
            return None

        # 创建画布
        canvas = np.zeros((self.display_height, self.display_width, 3), dtype=np.uint8)
        canvas[:] = (40, 40, 40)  # 深灰背景

        # === 绘制图像区域 ===
        disp_w = int(self.frame_width * self.zoom_level)
        disp_h = int(self.frame_height * self.zoom_level)

        # 缩放图像
        resized = cv2.resize(self.current_frame, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)

        # 计算图像位置
        offset_x = (self.display_width - disp_w) // 2 + self.pan_x
        offset_y = (self.image_area_height - disp_h) // 2 + self.pan_y

        # 计算可见区域
        src_x1 = max(0, -offset_x)
        src_y1 = max(0, -offset_y)
        src_x2 = min(disp_w, self.display_width - offset_x)
        src_y2 = min(disp_h, self.image_area_height - offset_y)

        dst_x1 = max(0, offset_x)
        dst_y1 = max(0, offset_y)
        dst_x2 = min(self.display_width, offset_x + disp_w)
        dst_y2 = min(self.image_area_height, offset_y + disp_h)

        if src_x2 > src_x1 and src_y2 > src_y1:
            canvas[dst_y1:dst_y2, dst_x1:dst_x2] = resized[src_y1:src_y2, src_x1:src_x2]

        # === 绘制标记点 ===
        # 当前帧的标记（大圆）
        marks_here = self.get_marks_at_frame(self.current_frame_idx)
        for mark in marks_here:
            color = tuple(mark["color"])
            sx, sy = self.image_to_screen(mark["x"], mark["y"])

            if 0 <= sx < self.display_width and 0 <= sy < self.image_area_height:
                nid = mark["neuron_id"]
                radius = int(8 * self.zoom_level)
                radius = max(4, min(radius, 20))

                if nid == self.current_neuron_id:
                    cv2.circle(canvas, (sx, sy), radius + 4, (255, 255, 255), 2)
                    cv2.circle(canvas, (sx, sy), radius, color, -1)
                else:
                    cv2.circle(canvas, (sx, sy), radius, color, -1)

                cv2.putText(canvas, str(nid), (sx + radius + 2, sy + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # 其他帧的标记（十字）
        for nid, ndata in self.neuron_marks.items():
            color = tuple(ndata["color"])
            for mark in ndata["marks"]:
                if mark["frame"] != self.current_frame_idx:
                    sx, sy = self.image_to_screen(mark["x"], mark["y"])
                    if 0 <= sx < self.display_width and 0 <= sy < self.image_area_height:
                        size = int(6 * self.zoom_level)
                        size = max(3, min(size, 15))
                        cv2.drawMarker(canvas, (sx, sy), color, cv2.MARKER_CROSS, size, 1)

        # === 控制面板背景 ===
        cv2.rectangle(canvas, (0, self.image_area_height),
                      (self.display_width, self.display_height), (30, 30, 30), -1)
        cv2.line(canvas, (0, self.image_area_height),
                 (self.display_width, self.image_area_height), (80, 80, 80), 1)

        # === 绘制按钮 ===
        for btn in self.buttons.values():
            btn.draw(canvas)

        # === 信息栏 ===
        # 帧信息
        info1 = f"Frame: {self.current_frame_idx + 1}/{self.total_frames}"
        cv2.putText(canvas, info1, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # 当前神经元
        ncolor = self.colors[self.current_neuron_id % len(self.colors)]
        info2 = f"Neuron: N{self.current_neuron_id}"
        cv2.putText(canvas, info2, (200, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, ncolor, 2)

        # 缩放
        info3 = f"Zoom: {self.zoom_level:.1f}x"
        cv2.putText(canvas, info3, (380, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        # 标记统计
        total_marks = sum(len(d["marks"]) for d in self.neuron_marks.values())
        info4 = f"Neurons: {len(self.neuron_marks)} | Marks: {total_marks}"
        cv2.putText(canvas, info4, (520, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        # 帧号输入模式
        if self.input_mode:
            input_text = f"Go to frame: {self.input_text}_"
            cv2.rectangle(canvas, (550, self.image_area_height + 15), (750, self.image_area_height + 45), (60, 60, 80),
                          -1)
            cv2.putText(canvas, input_text, (560, self.image_area_height + 37),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # 图例（右上角）
        legend_x = self.display_width - 200
        legend_y = 10
        cv2.putText(canvas, "Neurons:", (legend_x, legend_y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

        for i, (nid, ndata) in enumerate(list(self.neuron_marks.items())[:8]):
            color = tuple(ndata["color"])
            y_pos = legend_y + 30 + i * 18
            cv2.rectangle(canvas, (legend_x, y_pos), (legend_x + 12, y_pos + 12), color, -1)
            mark_count = len(ndata["marks"])
            cv2.putText(canvas, f"N{nid}: {mark_count}pts", (legend_x + 18, y_pos + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)

        # 操作提示
        tips = "LClick:Mark | RClick:Delete | MWheel:Zoom | MDrag:Pan"
        cv2.putText(canvas, tips, (10, self.image_area_height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

        return canvas

    def run(self, video_path, marks_path=None):
        """运行标记工具"""
        self.load_video(video_path)

        if marks_path and os.path.exists(marks_path):
            self.load_marks(marks_path)

        if marks_path is None:
            base = os.path.splitext(video_path)[0]
            marks_path = base + "_marks.json"

        self.read_frame(0)
        self.action_result = None

        window_name = 'Neuron Marker Tool V2'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self.display_width, self.display_height)
        cv2.setMouseCallback(window_name, self.mouse_callback)

        print("\n" + "=" * 70)
        print("神经元标记工具 V2")
        print("=" * 70)
        print("鼠标操作:")
        print("  左键点击图像 : 添加标记")
        print("  右键点击图像 : 删除当前帧标记")
        print("  滚轮 : 缩放")
        print("  中键拖动 : 平移")
        print("")
        print("快捷键:")
        print("  A/D, ←/→ : 上/下一帧")
        print("  W/S, ↑/↓ : 前/后10帧")
        print("  0-9 : 快速切换神经元")
        print("  G : 输入帧号跳转")
        print("  +/- : 缩放")
        print("  Ctrl+S : 保存")
        print("  Enter : 确认退出")
        print("  Esc : 取消退出")
        print("=" * 70 + "\n")

        while True:
            display = self.draw_frame()
            if display is not None:
                cv2.imshow(window_name, display)

            # 检查按钮触发的动作
            if self.action_result == 'confirm':
                self.save_marks(marks_path)
                print("✓ 确认并保存")
                break
            elif self.action_result == 'cancel':
                print("✗ 取消")
                break
            elif self.action_result == 'save':
                self.save_marks(marks_path)
                self.action_result = None
            elif self.action_result == 'load':
                self.load_marks(marks_path)
                self.action_result = None

            key = cv2.waitKey(30) & 0xFF

            # 帧号输入模式
            if self.input_mode:
                if key == 13 or key == 10:  # Enter
                    if self.input_text.isdigit():
                        frame_num = int(self.input_text) - 1  # 用户输入的是1-based
                        self.read_frame(frame_num)
                        print(f"  跳转到帧 {self.current_frame_idx + 1}")
                    self.input_mode = False
                    self.input_text = ""
                elif key == 27:  # Esc
                    self.input_mode = False
                    self.input_text = ""
                elif key == 8 or key == 127:  # Backspace
                    self.input_text = self.input_text[:-1]
                elif ord('0') <= key <= ord('9'):
                    self.input_text += chr(key)
                continue

            # 普通模式快捷键
            if key == ord('a') or key == 81:  # A / Left
                self.read_frame(self.current_frame_idx - 1)
            elif key == ord('d') or key == 83:  # D / Right
                self.read_frame(self.current_frame_idx + 1)
            elif key == ord('w') or key == 82:  # W / Up
                self.read_frame(self.current_frame_idx + 10)
            elif key == ord('s') or key == 84:  # S / Down
                self.read_frame(self.current_frame_idx - 10)
            elif key == ord('g') or key == ord('G'):
                self.input_mode = True
                self.input_text = ""
                print("  输入帧号...")
            elif ord('0') <= key <= ord('9'):
                self.current_neuron_id = key - ord('0')
                print(f"  切换到 N{self.current_neuron_id}")
            elif key == ord('+') or key == ord('='):
                self.zoom_level = min(self.max_zoom, self.zoom_level * 1.2)
            elif key == ord('-') or key == ord('_'):
                self.zoom_level = max(self.min_zoom, self.zoom_level / 1.2)
            elif key == 19:  # Ctrl+S
                self.save_marks(marks_path)
            elif key == 13 or key == 10:  # Enter
                self.save_marks(marks_path)
                print("✓ 确认并保存")
                break
            elif key == 27:  # Esc
                if len(self.neuron_marks) > 0:
                    print("有未保存的标记，是否保存? (Y/N)")
                    save_key = cv2.waitKey(0) & 0xFF
                    if save_key == ord('y') or save_key == ord('Y'):
                        self.save_marks(marks_path)
                break

        cv2.destroyAllWindows()
        if self.cap:
            self.cap.release()

        return marks_path


if __name__ == "__main__":
    VIDEO_PATH = r"/neuron_growth_50.mp4"
    MARKS_PATH = r"/neuron_marks.json"

    tool = NeuronMarkerTool(max_neurons=15)
    tool.run(VIDEO_PATH, MARKS_PATH)
