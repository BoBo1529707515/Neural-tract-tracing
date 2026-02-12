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
    """按钮类"""

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
        self.visible = True

    def contains(self, px, py):
        if not self.visible:
            return False
        return self.x <= px < self.x + self.w and self.y <= py < self.y + self.h

    def draw(self, img):
        if not self.visible:
            return

        if self.active:
            bg_color = (80, 160, 80)
        elif self.hover:
            bg_color = (100, 100, 100)
        else:
            bg_color = self.color

        cv2.rectangle(img, (self.x, self.y), (self.x + self.w, self.y + self.h), bg_color, -1)
        cv2.rectangle(img, (self.x, self.y), (self.x + self.w, self.y + self.h), (150, 150, 150), 1)

        font_scale = 0.5
        thickness = 2  # 加粗
        (text_w, text_h), _ = cv2.getTextSize(self.text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        text_x = self.x + (self.w - text_w) // 2
        text_y = self.y + (self.h + text_h) // 2
        cv2.putText(img, self.text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, self.text_color, thickness)


class DropdownMenu:
    """下拉菜单类"""

    def __init__(self, x, y, w, h, label, options, colors=None):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.label = label
        self.options = options  # [(value, display_text), ...]
        self.colors = colors  # 每个选项的颜色
        self.selected_idx = 0
        self.is_open = False
        self.hover_idx = -1
        self.max_visible = 10  # 最多显示10个选项
        self.scroll_offset = 0

    def get_selected_value(self):
        if 0 <= self.selected_idx < len(self.options):
            return self.options[self.selected_idx][0]
        return None

    def set_selected_value(self, value):
        for i, (v, _) in enumerate(self.options):
            if v == value:
                self.selected_idx = i
                return True
        return False

    def contains_header(self, px, py):
        return self.x <= px < self.x + self.w and self.y <= py < self.y + self.h

    def contains_dropdown(self, px, py):
        if not self.is_open:
            return False
        visible_count = min(len(self.options), self.max_visible)
        dropdown_h = visible_count * self.h
        return self.x <= px < self.x + self.w and self.y + self.h <= py < self.y + self.h + dropdown_h

    def get_clicked_option(self, px, py):
        if not self.is_open:
            return -1
        if not self.contains_dropdown(px, py):
            return -1

        rel_y = py - (self.y + self.h)
        idx = rel_y // self.h + self.scroll_offset
        if 0 <= idx < len(self.options):
            return idx
        return -1

    def draw(self, img):
        # 绘制标签
        cv2.putText(img, self.label, (self.x, self.y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        # 绘制主框
        bg_color = (70, 70, 70) if not self.is_open else (90, 90, 90)
        cv2.rectangle(img, (self.x, self.y), (self.x + self.w, self.y + self.h), bg_color, -1)
        cv2.rectangle(img, (self.x, self.y), (self.x + self.w, self.y + self.h), (150, 150, 150), 1)

        # 绘制当前选中项
        if 0 <= self.selected_idx < len(self.options):
            val, text = self.options[self.selected_idx]
            color = self.colors[self.selected_idx] if self.colors else (255, 255, 255)

            # 颜色方块
            cv2.rectangle(img, (self.x + 5, self.y + 5), (self.x + 20, self.y + self.h - 5), color, -1)

            # 文字
            cv2.putText(img, text, (self.x + 25, self.y + self.h - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # 下拉箭头
        arrow_x = self.x + self.w - 15
        arrow_y = self.y + self.h // 2
        if self.is_open:
            pts = np.array([[arrow_x - 5, arrow_y + 3], [arrow_x + 5, arrow_y + 3], [arrow_x, arrow_y - 4]])
        else:
            pts = np.array([[arrow_x - 5, arrow_y - 3], [arrow_x + 5, arrow_y - 3], [arrow_x, arrow_y + 4]])
        cv2.fillPoly(img, [pts], (200, 200, 200))

        # 绘制下拉列表
        if self.is_open:
            visible_count = min(len(self.options), self.max_visible)
            dropdown_h = visible_count * self.h

            # 背景
            cv2.rectangle(img, (self.x, self.y + self.h),
                          (self.x + self.w, self.y + self.h + dropdown_h), (50, 50, 50), -1)
            cv2.rectangle(img, (self.x, self.y + self.h),
                          (self.x + self.w, self.y + self.h + dropdown_h), (150, 150, 150), 1)

            # 选项
            for i in range(visible_count):
                opt_idx = i + self.scroll_offset
                if opt_idx >= len(self.options):
                    break

                val, text = self.options[opt_idx]
                opt_y = self.y + self.h + i * self.h

                # 高亮
                if opt_idx == self.hover_idx:
                    cv2.rectangle(img, (self.x + 1, opt_y + 1),
                                  (self.x + self.w - 1, opt_y + self.h - 1), (80, 80, 100), -1)
                elif opt_idx == self.selected_idx:
                    cv2.rectangle(img, (self.x + 1, opt_y + 1),
                                  (self.x + self.w - 1, opt_y + self.h - 1), (60, 100, 60), -1)

                # 颜色方块
                color = self.colors[opt_idx] if self.colors else (255, 255, 255)
                cv2.rectangle(img, (self.x + 5, opt_y + 5), (self.x + 20, opt_y + self.h - 5), color, -1)

                # 文字
                cv2.putText(img, text, (self.x + 25, opt_y + self.h - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)


class NeuronMarkerTool:
    """神经元标记工具 V3 - 下拉菜单 + 可调选区"""

    def __init__(self, max_neurons=20):
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

        # 缩放和平移
        self.zoom_level = 1.0
        self.min_zoom = 0.3
        self.max_zoom = 8.0
        self.pan_x = 0
        self.pan_y = 0
        self.is_panning = False
        self.pan_start_x = 0
        self.pan_start_y = 0

        # 选区半径（像素，图像坐标系）
        self.mark_radius = 1  # 默认1像素精度
        self.min_mark_radius = 1
        self.max_mark_radius = 20

        # 显示尺寸
        self.display_width = 1500
        self.display_height = 950
        self.control_panel_height = 130
        self.image_area_height = self.display_height - self.control_panel_height

        # 帧号输入
        self.input_mode = False
        self.input_text = ""
        self.input_target = ""  # 'frame' or 'neuron'

        # 鼠标位置（用于显示选区预览）
        self.mouse_x = 0
        self.mouse_y = 0
        self.mouse_in_image = False

        # 按钮和下拉菜单
        self.buttons = {}
        self.dropdown = None
        self._create_ui()

        self.action_result = None

    def _create_ui(self):
        """创建UI组件"""
        y1 = self.image_area_height + 15
        y2 = self.image_area_height + 60
        y3 = self.image_area_height + 100
        btn_h = 38

        # === 第一行：帧导航 ===
        x = 15
        self.buttons['prev_frame'] = Button(x, y1, 70, btn_h, '<< Prev')
        x += 80
        self.buttons['next_frame'] = Button(x, y1, 70, btn_h, 'Next >>')
        x += 80
        self.buttons['prev_10'] = Button(x, y1, 50, btn_h, '-10')
        x += 60
        self.buttons['next_10'] = Button(x, y1, 50, btn_h, '+10')
        x += 60
        self.buttons['goto_frame'] = Button(x, y1, 90, btn_h, 'GoTo Frame')

        # 缩放
        x += 110
        self.buttons['zoom_in'] = Button(x, y1, 55, btn_h, 'Z +')
        x += 65
        self.buttons['zoom_out'] = Button(x, y1, 55, btn_h, 'Z -')
        x += 65
        self.buttons['zoom_fit'] = Button(x, y1, 50, btn_h, 'Fit')

        # 选区半径
        x += 70
        self.buttons['radius_down'] = Button(x, y1, 40, btn_h, 'R-')
        x += 50
        self.buttons['radius_up'] = Button(x, y1, 40, btn_h, 'R+')

        # === 神经元下拉菜单 ===
        x += 70
        options = [(i, f"N{i}") for i in range(self.max_neurons)]
        self.dropdown = DropdownMenu(x, y1 + 5, 120, 30, "Select Neuron:", options, self.colors)

        # === 第二行：操作按钮 ===
        x = 15
        self.buttons['new_neuron'] = Button(x, y2, 80, btn_h, 'New N')
        x += 90
        self.buttons['del_neuron'] = Button(x, y2, 80, btn_h, 'Del N')
        x += 90
        self.buttons['undo'] = Button(x, y2, 70, btn_h, 'Undo')
        x += 90
        self.buttons['save'] = Button(x, y2, 70, btn_h, 'Save', color=(50, 90, 50))
        x += 80
        self.buttons['load'] = Button(x, y2, 70, btn_h, 'Load')

        x += 100
        self.buttons['confirm'] = Button(x, y2, 100, btn_h, 'CONFIRM', color=(50, 120, 50))
        x += 110
        self.buttons['cancel'] = Button(x, y2, 80, btn_h, 'Cancel', color=(90, 50, 50))

        # === 第三行：快速神经元选择 ===
        x = 15
        for i in range(15):
            self.buttons[f'n{i}'] = Button(x + i * 48, y3, 44, 30, f'N{i}')

    def load_video(self, video_path):
        """加载视频"""
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)

        self.video_info = {
            "path": video_path,
            "width": self.frame_width,
            "height": self.frame_height,
            "fps": fps,
            "total_frames": self.total_frames
        }

        # 初始缩放
        scale_w = (self.display_width - 20) / self.frame_width
        scale_h = (self.image_area_height - 20) / self.frame_height
        self.zoom_level = min(scale_w, scale_h, 1.0)

        print(f"✓ 加载视频: {self.frame_width}x{self.frame_height}, {fps:.1f}fps, {self.total_frames}帧")
        return True

    def read_frame(self, frame_idx):
        """读取帧"""
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
        disp_w = int(self.frame_width * self.zoom_level)
        disp_h = int(self.frame_height * self.zoom_level)

        offset_x = (self.display_width - disp_w) // 2 + self.pan_x
        offset_y = (self.image_area_height - disp_h) // 2 + self.pan_y

        img_x = (sx - offset_x) / self.zoom_level
        img_y = (sy - offset_y) / self.zoom_level

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
        """添加标记"""
        if neuron_id not in self.neuron_marks:
            color_idx = neuron_id % self.max_neurons
            self.neuron_marks[neuron_id] = {
                "color": list(self.colors[color_idx]),
                "marks": []
            }

        # 更新已有或添加新
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
        if neuron_id in self.neuron_marks:
            if len(self.neuron_marks[neuron_id]["marks"]) > 0:
                return self.neuron_marks[neuron_id]["marks"].pop()
        return None

    def remove_mark_at_frame(self, neuron_id, frame_idx):
        if neuron_id in self.neuron_marks:
            marks = self.neuron_marks[neuron_id]["marks"]
            for i, mark in enumerate(marks):
                if mark["frame"] == frame_idx:
                    return marks.pop(i)
        return None

    def delete_neuron(self, neuron_id):
        if neuron_id in self.neuron_marks:
            del self.neuron_marks[neuron_id]
            return True
        return False

    def get_marks_at_frame(self, frame_idx):
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
        data = {
            "version": "3.0",
            "created": datetime.now().isoformat(),
            "video_info": self.video_info,
            "neurons": self.neuron_marks
        }

        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"✓ 保存: {save_path}")
        return True

    def load_marks(self, load_path):
        if not os.path.exists(load_path):
            print(f"⚠ 文件不存在: {load_path}")
            return False

        with open(load_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.neuron_marks = {int(k): v for k, v in data.get("neurons", {}).items()}
        print(f"✓ 加载: {len(self.neuron_marks)} 个神经元")
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
            self.input_target = 'frame'
        elif btn_name == 'zoom_in':
            self.zoom_level = min(self.max_zoom, self.zoom_level * 1.3)
        elif btn_name == 'zoom_out':
            self.zoom_level = max(self.min_zoom, self.zoom_level / 1.3)
        elif btn_name == 'zoom_fit':
            scale_w = (self.display_width - 20) / self.frame_width
            scale_h = (self.image_area_height - 20) / self.frame_height
            self.zoom_level = min(scale_w, scale_h, 1.0)
            self.pan_x = 0
            self.pan_y = 0
        elif btn_name == 'radius_down':
            self.mark_radius = max(self.min_mark_radius, self.mark_radius - 1)
            print(f"  选区半径: {self.mark_radius}px")
        elif btn_name == 'radius_up':
            self.mark_radius = min(self.max_mark_radius, self.mark_radius + 1)
            print(f"  选区半径: {self.mark_radius}px")
        elif btn_name == 'new_neuron':
            existing_ids = set(self.neuron_marks.keys())
            new_id = 0
            while new_id in existing_ids:
                new_id += 1
            self.current_neuron_id = new_id
            self.dropdown.set_selected_value(new_id)
            print(f"  新建 N{self.current_neuron_id}")
        elif btn_name == 'del_neuron':
            if self.delete_neuron(self.current_neuron_id):
                print(f"  删除 N{self.current_neuron_id}")
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
            nid = int(btn_name[1:])
            self.current_neuron_id = nid
            self.dropdown.set_selected_value(nid)
            print(f"  切换到 N{self.current_neuron_id}")

        return None

    def mouse_callback(self, event, x, y, flags, param):
        """鼠标回调"""
        self.mouse_x = x
        self.mouse_y = y

        in_control_panel = y >= self.image_area_height
        in_image_area = y < self.image_area_height

        # 检查是否在图像范围内
        if in_image_area:
            img_x, img_y = self.screen_to_image(x, y)
            self.mouse_in_image = (0 <= img_x < self.frame_width and 0 <= img_y < self.frame_height)
        else:
            self.mouse_in_image = False

        # 更新按钮hover
        for btn in self.buttons.values():
            btn.hover = btn.contains(x, y)

        # 更新当前神经元按钮active
        for i in range(15):
            self.buttons[f'n{i}'].active = (i == self.current_neuron_id)

        # 下拉菜单hover
        if self.dropdown.is_open:
            self.dropdown.hover_idx = self.dropdown.get_clicked_option(x, y)

        if event == cv2.EVENT_LBUTTONDOWN:
            # 下拉菜单
            if self.dropdown.is_open:
                clicked_idx = self.dropdown.get_clicked_option(x, y)
                if clicked_idx >= 0:
                    self.dropdown.selected_idx = clicked_idx
                    self.current_neuron_id = self.dropdown.get_selected_value()
                    print(f"  切换到 N{self.current_neuron_id}")
                self.dropdown.is_open = False
                return

            if self.dropdown.contains_header(x, y):
                self.dropdown.is_open = not self.dropdown.is_open
                return

            if in_control_panel:
                # 按钮
                for btn_name, btn in self.buttons.items():
                    if btn.contains(x, y):
                        result = self.handle_button_click(btn_name)
                        if result:
                            self.action_result = result
                        return
            else:
                # 图像区域添加标记
                img_x, img_y = self.screen_to_image(x, y)

                if 0 <= img_x < self.frame_width and 0 <= img_y < self.frame_height:
                    # 使用选区半径计算中心点
                    mark_x = int(round(img_x))
                    mark_y = int(round(img_y))

                    self.add_mark(self.current_neuron_id, self.current_frame_idx, mark_x, mark_y)
                    print(f"  + N{self.current_neuron_id} @ 帧{self.current_frame_idx + 1}: ({mark_x}, {mark_y})")

        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.dropdown.is_open:
                self.dropdown.is_open = False
                return

            if not in_control_panel:
                removed = self.remove_mark_at_frame(self.current_neuron_id, self.current_frame_idx)
                if removed:
                    print(f"  - N{self.current_neuron_id} @ 帧{self.current_frame_idx + 1}")

        elif event == cv2.EVENT_MBUTTONDOWN:
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
                # 以鼠标位置为中心缩放
                img_x, img_y = self.screen_to_image(x, y)

                if flags > 0:
                    new_zoom = min(self.max_zoom, self.zoom_level * 1.15)
                else:
                    new_zoom = max(self.min_zoom, self.zoom_level / 1.15)

                # 调整平移以保持鼠标位置不变
                if 0 <= img_x < self.frame_width and 0 <= img_y < self.frame_height:
                    self.pan_x = int(self.pan_x + (x - self.display_width / 2) * (1 - new_zoom / self.zoom_level))
                    self.pan_y = int(self.pan_y + (y - self.image_area_height / 2) * (1 - new_zoom / self.zoom_level))

                self.zoom_level = new_zoom

    def draw_frame(self):
        """绘制界面"""
        if self.current_frame is None:
            return None

        canvas = np.zeros((self.display_height, self.display_width, 3), dtype=np.uint8)
        canvas[:] = (35, 35, 35)

        # === 图像区域 ===
        disp_w = int(self.frame_width * self.zoom_level)
        disp_h = int(self.frame_height * self.zoom_level)

        resized = cv2.resize(self.current_frame, (disp_w, disp_h),
                             interpolation=cv2.INTER_LINEAR if self.zoom_level < 1 else cv2.INTER_NEAREST)

        offset_x = (self.display_width - disp_w) // 2 + self.pan_x
        offset_y = (self.image_area_height - disp_h) // 2 + self.pan_y

        # 裁剪显示
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

        # === 绘制标记 ===
        marks_here = self.get_marks_at_frame(self.current_frame_idx)
        for mark in marks_here:
            color = tuple(mark["color"])
            sx, sy = self.image_to_screen(mark["x"], mark["y"])

            if 0 <= sx < self.display_width and 0 <= sy < self.image_area_height:
                nid = mark["neuron_id"]

                # 根据缩放调整显示大小
                radius = max(4, int(6 * self.zoom_level))
                radius = min(radius, 25)

                if nid == self.current_neuron_id:
                    cv2.circle(canvas, (sx, sy), radius + 5, (255, 255, 255), 2)
                    cv2.circle(canvas, (sx, sy), radius, color, -1)
                else:
                    cv2.circle(canvas, (sx, sy), radius, color, -1)
                    cv2.circle(canvas, (sx, sy), radius, (180, 180, 180), 1)

                # 编号（加粗）
                cv2.putText(canvas, str(nid), (sx + radius + 3, sy + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
                cv2.putText(canvas, str(nid), (sx + radius + 3, sy + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 其他帧标记
        for nid, ndata in self.neuron_marks.items():
            color = tuple(ndata["color"])
            for mark in ndata["marks"]:
                if mark["frame"] != self.current_frame_idx:
                    sx, sy = self.image_to_screen(mark["x"], mark["y"])
                    if 0 <= sx < self.display_width and 0 <= sy < self.image_area_height:
                        size = max(4, int(5 * self.zoom_level))
                        cv2.drawMarker(canvas, (sx, sy), color, cv2.MARKER_TILTED_CROSS, size, 1)

        # === 选区预览（鼠标位置） ===
        if self.mouse_in_image and self.mouse_y < self.image_area_height:
            # 显示选区圆圈
            preview_radius = max(2, int(self.mark_radius * self.zoom_level))
            ncolor = self.colors[self.current_neuron_id % len(self.colors)]
            cv2.circle(canvas, (self.mouse_x, self.mouse_y), preview_radius, ncolor, 1)
            cv2.circle(canvas, (self.mouse_x, self.mouse_y), 2, (255, 255, 255), -1)

        # === 控制面板 ===
        cv2.rectangle(canvas, (0, self.image_area_height),
                      (self.display_width, self.display_height), (25, 25, 25), -1)
        cv2.line(canvas, (0, self.image_area_height),
                 (self.display_width, self.image_area_height), (80, 80, 80), 2)

        # 按钮
        for btn in self.buttons.values():
            btn.draw(canvas)

        # 下拉菜单
        self.dropdown.draw(canvas)

        # === 信息栏（加粗） ===
        # 帧信息
        info1 = f"Frame: {self.current_frame_idx + 1} / {self.total_frames}"
        cv2.putText(canvas, info1, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4)
        cv2.putText(canvas, info1, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 230, 230), 2)

        # 当前神经元
        ncolor = self.colors[self.current_neuron_id % len(self.colors)]
        info2 = f"Neuron: N{self.current_neuron_id}"
        cv2.putText(canvas, info2, (252, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4)
        cv2.putText(canvas, info2, (250, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, ncolor, 2)

        # 缩放
        info3 = f"Zoom: {self.zoom_level:.2f}x"
        cv2.putText(canvas, info3, (450, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 2)

        # 选区半径
        info4 = f"Radius: {self.mark_radius}px"
        cv2.putText(canvas, info4, (600, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 220, 180), 2)

        # 统计
        total_marks = sum(len(d["marks"]) for d in self.neuron_marks.values())
        info5 = f"Neurons: {len(self.neuron_marks)} | Marks: {total_marks}"
        cv2.putText(canvas, info5, (760, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 2)

        # 输入模式
        if self.input_mode:
            prompt = f"Enter frame number: {self.input_text}_"
            cv2.rectangle(canvas, (500, 40), (750, 70), (50, 50, 80), -1)
            cv2.rectangle(canvas, (500, 40), (750, 70), (150, 150, 200), 1)
            cv2.putText(canvas, prompt, (510, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 100), 2)

        # === 图例（右上角） ===
        legend_x = self.display_width - 220
        legend_y = 10
        cv2.putText(canvas, "Marked Neurons:", (legend_x, legend_y + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 2)

        for i, (nid, ndata) in enumerate(list(self.neuron_marks.items())[:10]):
            color = tuple(ndata["color"])
            y_pos = legend_y + 38 + i * 22
            cv2.rectangle(canvas, (legend_x, y_pos), (legend_x + 16, y_pos + 16), color, -1)
            mark_count = len(ndata["marks"])
            highlight = " ◄" if nid == self.current_neuron_id else ""
            cv2.putText(canvas, f"N{nid}: {mark_count} pts{highlight}", (legend_x + 22, y_pos + 13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        # === 操作提示 ===
        tips = "L-Click: Mark | R-Click: Delete | Wheel: Zoom | M-Drag: Pan | R+/R-: Adjust radius"
        cv2.putText(canvas, tips, (10, self.image_area_height - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 100, 100), 1)

        return canvas

    def run(self, video_path, marks_path=None):
        """运行"""
        self.load_video(video_path)

        if marks_path and os.path.exists(marks_path):
            self.load_marks(marks_path)

        if marks_path is None:
            base = os.path.splitext(video_path)[0]
            marks_path = base + "_marks.json"

        self.read_frame(0)
        self.action_result = None

        window_name = 'Neuron Marker V3'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self.display_width, self.display_height)
        cv2.setMouseCallback(window_name, self.mouse_callback)

        print("\n" + "=" * 70)
        print("神经元标记工具 V3")
        print("=" * 70)
        print("鼠标操作:")
        print("  左键点击图像 : 添加标记（在选区圆圈中心）")
        print("  右键点击图像 : 删除当前帧标记")
        print("  滚轮 : 缩放（以鼠标为中心）")
        print("  中键拖动 : 平移")
        print("")
        print("快捷键:")
        print("  A/D, ←/→ : 上/下一帧")
        print("  W/S, ↑/↓ : 前/后10帧")
        print("  0-9 : 快速切换神经元 N0-N9")
        print("  G : 输入帧号跳转")
        print("  +/- : 缩放")
        print("  [/] : 减小/增大选区半径")
        print("  Ctrl+S : 保存")
        print("  Enter : 确认退出")
        print("  Esc : 取消退出")
        print("=" * 70 + "\n")

        while True:
            display = self.draw_frame()
            if display is not None:
                cv2.imshow(window_name, display)

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

            # 输入模式
            if self.input_mode:
                if key == 13 or key == 10:  # Enter
                    if self.input_text.isdigit():
                        frame_num = int(self.input_text) - 1
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

            # 关闭下拉菜单
            if key == 27 and self.dropdown.is_open:
                self.dropdown.is_open = False
                continue

            # 快捷键
            if key == ord('a') or key == 81:
                self.read_frame(self.current_frame_idx - 1)
            elif key == ord('d') or key == 83:
                self.read_frame(self.current_frame_idx + 1)
            elif key == ord('w') or key == 82:
                self.read_frame(self.current_frame_idx + 10)
            elif key == ord('s') or key == 84:
                self.read_frame(self.current_frame_idx - 10)
            elif key == ord('g') or key == ord('G'):
                self.input_mode = True
                self.input_text = ""
            elif ord('0') <= key <= ord('9'):
                nid = key - ord('0')
                self.current_neuron_id = nid
                self.dropdown.set_selected_value(nid)
                print(f"  切换到 N{self.current_neuron_id}")
            elif key == ord('+') or key == ord('='):
                self.zoom_level = min(self.max_zoom, self.zoom_level * 1.3)
            elif key == ord('-') or key == ord('_'):
                self.zoom_level = max(self.min_zoom, self.zoom_level / 1.3)
            elif key == ord('['):
                self.mark_radius = max(self.min_mark_radius, self.mark_radius - 1)
                print(f"  选区半径: {self.mark_radius}px")
            elif key == ord(']'):
                self.mark_radius = min(self.max_mark_radius, self.mark_radius + 1)
                print(f"  选区半径: {self.mark_radius}px")
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
    VIDEO_PATH = r"F:\工作文件\RA\python\项目汇总\神经图像\cde7198975117a224bb23963f96cbd3d.mp4"
    MARKS_PATH = r"F:\工作文件\RA\python\项目汇总\神经图像\neuron_marks.json"

    tool = NeuronMarkerTool(max_neurons=20)
    tool.run(VIDEO_PATH, MARKS_PATH)
