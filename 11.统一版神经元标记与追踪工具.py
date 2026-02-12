import cv2
import numpy as np
import json
import os
from datetime import datetime
from skimage.morphology import skeletonize, remove_small_objects

try:
    import torch
    import torch.nn.functional as F

    HAS_CUDA = torch.cuda.is_available()
    if HAS_CUDA:
        DEVICE = torch.device('cuda')
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    else:
        DEVICE = torch.device('cpu')
except ImportError:
    HAS_CUDA = False
    DEVICE = None


def generate_colors(n):
    colors = []
    for i in range(n):
        hue = int(180 * i / n)
        hsv = np.uint8([[[hue, 255, 255]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
        colors.append((int(bgr[0]), int(bgr[1]), int(bgr[2])))
    return colors


class Button:
    def __init__(self, x, y, w, h, text, color=(70, 70, 70), text_color=(255, 255, 255)):
        self.x, self.y, self.w, self.h = x, y, w, h
        self.text = text
        self.color = color
        self.text_color = text_color
        self.hover = False
        self.active = False

    def contains(self, px, py):
        return self.x <= px < self.x + self.w and self.y <= py < self.y + self.h

    def draw(self, img):
        bg = (70, 140, 70) if self.active else ((100, 100, 100) if self.hover else self.color)
        cv2.rectangle(img, (self.x, self.y), (self.x + self.w, self.y + self.h), bg, -1)
        cv2.rectangle(img, (self.x, self.y), (self.x + self.w, self.y + self.h), (140, 140, 140), 1)

        (tw, th), _ = cv2.getTextSize(self.text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        tx = self.x + (self.w - tw) // 2
        ty = self.y + (self.h + th) // 2
        cv2.putText(img, self.text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, self.text_color, 2, cv2.LINE_AA)


class InputBox:
    """输入框"""

    def __init__(self, x, y, w, h, label, default=""):
        self.x, self.y, self.w, self.h = x, y, w, h
        self.label = label
        self.text = default
        self.active = False
        self.cursor_visible = True
        self.cursor_timer = 0

    def contains(self, px, py):
        return self.x <= px < self.x + self.w and self.y <= py < self.y + self.h

    def draw(self, img, highlight_color=None):
        # 标签
        cv2.putText(img, self.label, (self.x, self.y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
                    cv2.LINE_AA)

        # 背景
        bg = (60, 60, 90) if self.active else (50, 50, 50)
        cv2.rectangle(img, (self.x, self.y), (self.x + self.w, self.y + self.h), bg, -1)
        border = (100, 150, 255) if self.active else (140, 140, 140)
        cv2.rectangle(img, (self.x, self.y), (self.x + self.w, self.y + self.h), border, 2)

        # 颜色块（如果有）
        if highlight_color:
            cv2.rectangle(img, (self.x + 5, self.y + 5), (self.x + 25, self.y + self.h - 5), highlight_color, -1)
            text_x = self.x + 32
        else:
            text_x = self.x + 10

        # 文字
        display_text = self.text
        if self.active:
            self.cursor_timer = (self.cursor_timer + 1) % 30
            if self.cursor_timer < 20:
                display_text += "_"

        cv2.putText(img, display_text, (text_x, self.y + self.h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 100) if self.active else (220, 220, 220), 2, cv2.LINE_AA)


class NeuronTool:
    """神经元标记与追踪工具 - 实时追踪版"""

    def __init__(self, max_neurons=50):
        self.max_neurons = max_neurons
        self.colors = generate_colors(max_neurons)

        # 数据
        self.neuron_marks = {}
        self.neuron_trajectories = {}

        # 视频
        self.video_path = ""
        self.cap = None
        self.current_frame = None
        self.current_frame_idx = 0
        self.total_frames = 0
        self.frame_width = 0
        self.frame_height = 0
        self.video_info = {}

        # 缩放平移
        self.zoom_level = 1.0
        self.min_zoom, self.max_zoom = 0.2, 10.0
        self.pan_x, self.pan_y = 0, 0
        self.is_panning = False
        self.pan_start = (0, 0)

        # 选区
        self.mark_radius = 1

        # 显示
        self.display_width = 1600
        self.display_height = 1000
        self.panel_height = 120
        self.image_area_height = self.display_height - self.panel_height

        # 当前神经元
        self.current_neuron_id = 0

        # 输入框
        self.frame_input = None
        self.neuron_input = None
        self.active_input = None  # 当前激活的输入框

        # UI
        self.buttons = {}
        self.mode = 'mark'
        self._init_ui()

        # 鼠标
        self.mouse_x, self.mouse_y = 0, 0
        self.mouse_in_image = False

        # 追踪参数
        self.green_threshold = 40
        self.max_gap = 15
        self.y_tolerance = 35
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

        # 追踪状态
        self.is_tracking = False
        self.video_writer = None

        self.action_result = None

    def _init_ui(self):
        y1 = self.image_area_height + 15
        y2 = self.image_area_height + 65
        bh = 38

        # 行1：帧导航
        x = 15
        self.buttons['prev'] = Button(x, y1, 60, bh, '◀ Prev');
        x += 70
        self.buttons['next'] = Button(x, y1, 60, bh, 'Next ▶');
        x += 70
        self.buttons['prev10'] = Button(x, y1, 45, bh, '-10');
        x += 55
        self.buttons['next10'] = Button(x, y1, 45, bh, '+10');
        x += 65

        # 帧输入框
        self.frame_input = InputBox(x, y1 + 3, 100, 32, "Frame:")
        x += 120

        # 缩放
        self.buttons['zin'] = Button(x, y1, 40, bh, 'Z+');
        x += 50
        self.buttons['zout'] = Button(x, y1, 40, bh, 'Z-');
        x += 50
        self.buttons['zfit'] = Button(x, y1, 40, bh, 'Fit');
        x += 55

        # 选区
        self.buttons['r_down'] = Button(x, y1, 35, bh, 'R-');
        x += 45
        self.buttons['r_up'] = Button(x, y1, 35, bh, 'R+');
        x += 55

        # 神经元输入框
        self.neuron_input = InputBox(x, y1 + 3, 80, 32, "Neuron:")
        self.neuron_input.text = "0"
        x += 100

        # 神经元操作
        self.buttons['new_n'] = Button(x, y1, 60, bh, 'New');
        x += 70
        self.buttons['del_n'] = Button(x, y1, 50, bh, 'Del');
        x += 60

        # 行2：模式和操作
        x = 15
        self.buttons['mode_mark'] = Button(x, y2, 80, bh, 'MARK', (60, 60, 90));
        x += 90
        self.buttons['mode_track'] = Button(x, y2, 80, bh, 'TRACK', (60, 60, 90));
        x += 100

        self.buttons['run_track'] = Button(x, y2, 120, bh, '▶ RUN TRACK', (40, 100, 40));
        x += 130
        self.buttons['stop'] = Button(x, y2, 60, bh, 'STOP', (100, 40, 40));
        x += 80

        self.buttons['undo'] = Button(x, y2, 55, bh, 'Undo');
        x += 65
        self.buttons['save'] = Button(x, y2, 55, bh, 'Save', (50, 80, 50));
        x += 65
        self.buttons['load'] = Button(x, y2, 55, bh, 'Load');
        x += 75

        self.buttons['confirm'] = Button(x, y2, 90, bh, 'CONFIRM', (50, 110, 50));
        x += 100
        self.buttons['cancel'] = Button(x, y2, 70, bh, 'Cancel', (90, 50, 50))

        # 快速神经元按钮
        x += 90
        for i in range(10):
            self.buttons[f'n{i}'] = Button(x + i * 45, y2, 40, 32, f'N{i}')

    def load_video(self, path):
        self.video_path = path
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise ValueError(f"无法打开: {path}")

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        self.video_info = {
            "path": path, "width": self.frame_width, "height": self.frame_height,
            "fps": self.fps, "total_frames": self.total_frames
        }

        sw = (self.display_width - 40) / self.frame_width
        sh = (self.image_area_height - 40) / self.frame_height
        self.zoom_level = min(sw, sh, 1.0)

        print(f"✓ 视频: {self.frame_width}x{self.frame_height}, {self.fps:.1f}fps, {self.total_frames}帧")
        return True

    def read_frame(self, idx):
        if not self.cap:
            return None
        idx = max(0, min(idx, self.total_frames - 1))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        if ret:
            self.current_frame_idx = idx
            self.current_frame = frame
            self.frame_input.text = str(idx + 1)
            return frame
        return None

    def screen_to_image(self, sx, sy):
        dw = int(self.frame_width * self.zoom_level)
        dh = int(self.frame_height * self.zoom_level)
        ox = (self.display_width - dw) // 2 + self.pan_x
        oy = (self.image_area_height - dh) // 2 + self.pan_y
        return (sx - ox) / self.zoom_level, (sy - oy) / self.zoom_level

    def image_to_screen(self, ix, iy):
        dw = int(self.frame_width * self.zoom_level)
        dh = int(self.frame_height * self.zoom_level)
        ox = (self.display_width - dw) // 2 + self.pan_x
        oy = (self.image_area_height - dh) // 2 + self.pan_y
        return int(ix * self.zoom_level + ox), int(iy * self.zoom_level + oy)

    def add_mark(self, nid, fidx, x, y):
        if nid not in self.neuron_marks:
            self.neuron_marks[nid] = {"color": list(self.colors[nid % len(self.colors)]), "marks": []}
        for m in self.neuron_marks[nid]["marks"]:
            if m["frame"] == fidx:
                m["x"], m["y"] = x, y
                return
        self.neuron_marks[nid]["marks"].append({"frame": fidx, "x": x, "y": y})

    def remove_mark_at_frame(self, nid, fidx):
        if nid in self.neuron_marks:
            marks = self.neuron_marks[nid]["marks"]
            for i, m in enumerate(marks):
                if m["frame"] == fidx:
                    return marks.pop(i)
        return None

    def remove_last_mark(self, nid):
        if nid in self.neuron_marks and self.neuron_marks[nid]["marks"]:
            return self.neuron_marks[nid]["marks"].pop()
        return None

    def delete_neuron(self, nid):
        deleted = False
        if nid in self.neuron_marks:
            del self.neuron_marks[nid]
            deleted = True
        if nid in self.neuron_trajectories:
            del self.neuron_trajectories[nid]
            deleted = True
        return deleted

    def get_marks_at_frame(self, fidx):
        result = []
        for nid, data in self.neuron_marks.items():
            for m in data["marks"]:
                if m["frame"] == fidx:
                    result.append({"neuron_id": nid, "x": m["x"], "y": m["y"], "color": data["color"]})
        return result

    def save_data(self, path):
        data = {
            "version": "5.0",
            "created": datetime.now().isoformat(),
            "video_info": self.video_info,
            "neurons": self.neuron_marks,
            "trajectories": {k: [list(p) for p in v] for k, v in self.neuron_trajectories.items()}
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"✓ 保存数据: {path}")

    def load_data(self, path):
        if not os.path.exists(path):
            return False
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.neuron_marks = {int(k): v for k, v in data.get("neurons", {}).items()}
        trajs = data.get("trajectories", {})
        self.neuron_trajectories = {int(k): [tuple(p) for p in v] for k, v in trajs.items()}
        print(f"✓ 加载: {len(self.neuron_marks)}标记, {len(self.neuron_trajectories)}轨迹")
        return True

    # ========== 追踪 ==========

    def is_green(self, frame, y, x):
        h, w = frame.shape[:2]
        if not (0 <= y < h and 0 <= x < w):
            return False
        b, g, r = int(frame[y, x, 0]), int(frame[y, x, 1]), int(frame[y, x, 2])
        if g < 50 or g - r < self.green_threshold or g - b < self.green_threshold:
            return False
        if r > 150 and b > 150 and g > 150 and abs(r - g) < 50 and abs(b - g) < 50:
            return False
        return True

    def preprocess(self, frame):
        b, g, r = cv2.split(frame)
        b, g, r = b.astype(np.int16), g.astype(np.int16), r.astype(np.int16)
        mask = (g > 50) & ((g - r) > self.green_threshold) & ((g - b) > self.green_threshold)
        mask &= ~((r > 150) & (b > 150) & (g > 150) & (np.abs(r - g) < 50) & (np.abs(b - g) < 50))
        green_mask = mask.astype(np.uint8) * 255

        enhanced = self.clahe.apply(frame[:, :, 1])
        denoised = cv2.medianBlur(enhanced, 5)
        binary = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -3)
        binary = cv2.bitwise_and(binary, green_mask)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        cleaned = remove_small_objects(closed > 0, min_size=50, connectivity=1)
        skeleton = skeletonize(cleaned)

        return skeleton

    def find_nearest_skeleton(self, skeleton, frame, point, radius=30):
        h, w = skeleton.shape
        py, px = int(point[1]), int(point[0])
        best, best_d = None, float('inf')
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                ny, nx = py + dy, px + dx
                if 0 <= ny < h and 0 <= nx < w and skeleton[ny, nx]:
                    if self.is_green(frame, ny, nx):
                        d = dy * dy + dx * dx
                        if d < best_d:
                            best_d, best = d, (ny, nx)
        return best

    def trace_direction(self, skeleton, frame, start, y_center, go_left=True):
        if start is None:
            return []
        h, w = skeleton.shape
        visited = set()
        traj = [start]
        visited.add(start)
        current = start

        while True:
            next_pt = None
            best_score = float('-inf')

            for sr in [1, 2, self.max_gap]:
                for dy in range(-sr, sr + 1):
                    for dx in range(-sr, sr + 1):
                        if dy == 0 and dx == 0:
                            continue
                        ny, nx = current[0] + dy, current[1] + dx
                        if 0 <= ny < h and 0 <= nx < w:
                            if skeleton[ny, nx] and (ny, nx) not in visited:
                                if self.is_green(frame, ny, nx):
                                    if abs(ny - y_center) > self.y_tolerance * 2:
                                        continue
                                    dist = np.sqrt(dy * dy + dx * dx)
                                    score = (-dx if go_left else dx) * 10 - abs(dy) * 3 - dist * 2
                                    if score > best_score:
                                        best_score, next_pt = score, (ny, nx)
                if next_pt:
                    break

            if not next_pt:
                break
            visited.add(next_pt)
            traj.append(next_pt)
            current = next_pt

        return traj

    def trace_neuron(self, skeleton, frame, start_point):
        if start_point is None:
            return []
        y_center = start_point[0]
        left_traj = self.trace_direction(skeleton, frame, start_point, y_center, go_left=True)
        right_traj = self.trace_direction(skeleton, frame, start_point, y_center, go_left=False)
        full = left_traj[::-1] + right_traj[1:] if len(right_traj) > 1 else left_traj[::-1]
        return full

    def _find_growth(self, skeleton, frame, tip, existing, y_center):
        h, w = skeleton.shape
        visited = set(existing)
        new_pts = []
        queue = [tip]

        while queue:
            cur = queue.pop(0)
            cy, cx = cur
            for dy in range(-3, 4):
                for dx in range(-3, 4):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        if (ny, nx) not in visited:
                            if skeleton[ny, nx] and self.is_green(frame, ny, nx):
                                if abs(ny - y_center) < self.y_tolerance * 2:
                                    visited.add((ny, nx))
                                    new_pts.append((ny, nx))
                                    queue.append((ny, nx))
        return new_pts

    def draw_trajectories_on_frame(self, frame):
        """在帧上绘制所有轨迹"""
        vis = frame.copy()

        for nid, traj in self.neuron_trajectories.items():
            if not traj:
                continue
            color = tuple(self.colors[nid % len(self.colors)])

            for i in range(len(traj) - 1):
                p1 = (int(traj[i][1]), int(traj[i][0]))
                p2 = (int(traj[i + 1][1]), int(traj[i + 1][0]))
                d = np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
                if d < 30:
                    cv2.line(vis, p1, p2, color, 2, cv2.LINE_AA)

            if traj:
                start = (int(traj[0][1]), int(traj[0][0]))
                end = (int(traj[-1][1]), int(traj[-1][0]))
                cv2.circle(vis, start, 5, color, -1)
                cv2.circle(vis, end, 7, (0, 0, 255), -1)
                cv2.circle(vis, end, 9, (255, 255, 255), 2)
                cv2.putText(vis, str(nid), (end[0] + 10, end[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
                            cv2.LINE_AA)

        return vis

    def run_tracking_realtime(self, output_path):
        """实时追踪并保存视频"""
        if not self.neuron_marks:
            print("⚠ 没有标记数据")
            return

        print("\n开始实时追踪...")
        self.neuron_trajectories = {}
        self.is_tracking = True

        # 找最早帧
        all_frames = []
        for ndata in self.neuron_marks.values():
            for m in ndata["marks"]:
                all_frames.append(m["frame"])
        start_frame = min(all_frames) if all_frames else 0

        # 准备视频写入
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(output_path, fourcc, int(self.fps), (self.frame_width, self.frame_height))

        # 初始化帧
        self.read_frame(start_frame)
        skeleton = self.preprocess(self.current_frame)

        for nid, ndata in self.neuron_marks.items():
            marks = sorted(ndata["marks"], key=lambda m: m["frame"])
            if not marks:
                continue
            mark = marks[0]
            pt = self.find_nearest_skeleton(skeleton, self.current_frame, (mark["x"], mark["y"]))
            if pt:
                traj = self.trace_neuron(skeleton, self.current_frame, pt)
                if len(traj) >= 5:
                    traj = sorted(traj, key=lambda p: p[1])
                    self.neuron_trajectories[nid] = traj
                    print(f"  N{nid}: 初始化 {len(traj)} 点")

        print(f"✓ 初始化 {len(self.neuron_trajectories)} 根神经元")

        # 创建显示窗口
        win = 'Tracking Progress'
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 1280, 720)

        # 追踪循环
        for fidx in range(start_frame, self.total_frames):
            if not self.is_tracking:
                print("追踪被停止")
                break

            self.read_frame(fidx)

            if fidx > start_frame:
                skeleton = self.preprocess(self.current_frame)

                for nid, traj in self.neuron_trajectories.items():
                    if not traj:
                        continue
                    tip = traj[-1]
                    y_center = np.mean([p[0] for p in traj])
                    new_pts = self._find_growth(skeleton, self.current_frame, tip, traj, y_center)
                    if new_pts:
                        traj.extend(new_pts)
                        self.neuron_trajectories[nid] = sorted(traj, key=lambda p: p[1])

            # 绘制并写入
            vis = self.draw_trajectories_on_frame(self.current_frame)

            # 添加信息
            info = f"Frame: {fidx + 1}/{self.total_frames} | Neurons: {len(self.neuron_trajectories)}"
            cv2.putText(vis, info, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(vis, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

            # 统计
            total_pts = sum(len(t) for t in self.neuron_trajectories.values())
            stats = f"Total Points: {total_pts}"
            cv2.putText(vis, stats, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2, cv2.LINE_AA)

            self.video_writer.write(vis)

            # 显示（缩小以加快速度）
            scale = min(1.0, 1280 / self.frame_width)
            display = cv2.resize(vis, None, fx=scale, fy=scale)
            cv2.imshow(win, display)

            # 检查按键
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                self.is_tracking = False
                break

        # 完成
        self.video_writer.release()
        self.video_writer = None
        cv2.destroyWindow(win)
        self.is_tracking = False

        print(f"\n✓ 追踪完成! 视频保存: {output_path}")
        for nid, traj in self.neuron_trajectories.items():
            print(f"  N{nid}: {len(traj)} 点")

    # ========== UI ==========

    def handle_click(self, name):
        if name == 'prev':
            self.read_frame(self.current_frame_idx - 1)
        elif name == 'next':
            self.read_frame(self.current_frame_idx + 1)
        elif name == 'prev10':
            self.read_frame(self.current_frame_idx - 10)
        elif name == 'next10':
            self.read_frame(self.current_frame_idx + 10)
        elif name == 'zin':
            self.zoom_level = min(self.max_zoom, self.zoom_level * 1.4)
        elif name == 'zout':
            self.zoom_level = max(self.min_zoom, self.zoom_level / 1.4)
        elif name == 'zfit':
            sw = (self.display_width - 40) / self.frame_width
            sh = (self.image_area_height - 40) / self.frame_height
            self.zoom_level = min(sw, sh, 1.0)
            self.pan_x = self.pan_y = 0
        elif name == 'r_down':
            self.mark_radius = max(1, self.mark_radius - 1)
        elif name == 'r_up':
            self.mark_radius = min(20, self.mark_radius + 1)
        elif name == 'new_n':
            ids = set(self.neuron_marks.keys())
            nid = 0
            while nid in ids:
                nid += 1
            self.current_neuron_id = nid
            self.neuron_input.text = str(nid)
        elif name == 'del_n':
            self.delete_neuron(self.current_neuron_id)
        elif name == 'undo':
            self.remove_last_mark(self.current_neuron_id)
        elif name == 'save':
            return 'save'
        elif name == 'load':
            return 'load'
        elif name == 'mode_mark':
            self.mode = 'mark'
        elif name == 'mode_track':
            self.mode = 'track'
        elif name == 'run_track':
            return 'run_track'
        elif name == 'stop':
            self.is_tracking = False
        elif name == 'confirm':
            return 'confirm'
        elif name == 'cancel':
            return 'cancel'
        elif name.startswith('n') and name[1:].isdigit():
            self.current_neuron_id = int(name[1:])
            self.neuron_input.text = str(self.current_neuron_id)
        return None

    def mouse_cb(self, event, x, y, flags, param):
        self.mouse_x, self.mouse_y = x, y
        in_panel = y >= self.image_area_height

        if not in_panel:
            ix, iy = self.screen_to_image(x, y)
            self.mouse_in_image = 0 <= ix < self.frame_width and 0 <= iy < self.frame_height
        else:
            self.mouse_in_image = False

        # hover
        for btn in self.buttons.values():
            btn.hover = btn.contains(x, y)

        for i in range(10):
            self.buttons[f'n{i}'].active = (i == self.current_neuron_id)

        self.buttons['mode_mark'].active = (self.mode == 'mark')
        self.buttons['mode_track'].active = (self.mode == 'track')

        if event == cv2.EVENT_LBUTTONDOWN:
            # 输入框
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
                # 处理输入确认
                if self.frame_input.active:
                    try:
                        self.read_frame(int(self.frame_input.text) - 1)
                    except:
                        pass
                    self.frame_input.active = False
                if self.neuron_input.active:
                    try:
                        self.current_neuron_id = int(self.neuron_input.text)
                    except:
                        pass
                    self.neuron_input.active = False
                self.active_input = None

            if in_panel:
                for name, btn in self.buttons.items():
                    if btn.contains(x, y):
                        result = self.handle_click(name)
                        if result:
                            self.action_result = result
                        return
            else:
                if self.mode == 'mark':
                    ix, iy = self.screen_to_image(x, y)
                    if 0 <= ix < self.frame_width and 0 <= iy < self.frame_height:
                        mx, my = int(round(ix)), int(round(iy))
                        self.add_mark(self.current_neuron_id, self.current_frame_idx, mx, my)
                        print(f"  + N{self.current_neuron_id} @ 帧{self.current_frame_idx + 1}: ({mx}, {my})")

        elif event == cv2.EVENT_RBUTTONDOWN:
            if not in_panel and self.mode == 'mark':
                self.remove_mark_at_frame(self.current_neuron_id, self.current_frame_idx)

        elif event == cv2.EVENT_MBUTTONDOWN:
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
                self.zoom_level = min(self.max_zoom, self.zoom_level * 1.2) if flags > 0 else max(self.min_zoom,
                                                                                                  self.zoom_level / 1.2)

    def draw(self):
        if self.current_frame is None:
            return None

        canvas = np.zeros((self.display_height, self.display_width, 3), dtype=np.uint8)
        canvas[:] = (30, 30, 30)

        # 图像
        dw = int(self.frame_width * self.zoom_level)
        dh = int(self.frame_height * self.zoom_level)
        interp = cv2.INTER_LINEAR if self.zoom_level < 1 else cv2.INTER_NEAREST
        resized = cv2.resize(self.current_frame, (dw, dh), interpolation=interp)

        ox = (self.display_width - dw) // 2 + self.pan_x
        oy = (self.image_area_height - dh) // 2 + self.pan_y

        sx1, sy1 = max(0, -ox), max(0, -oy)
        sx2, sy2 = min(dw, self.display_width - ox), min(dh, self.image_area_height - oy)
        dx1, dy1 = max(0, ox), max(0, oy)
        dx2, dy2 = min(self.display_width, ox + dw), min(self.image_area_height, oy + dh)

        if sx2 > sx1 and sy2 > sy1:
            canvas[dy1:dy2, dx1:dx2] = resized[sy1:sy2, sx1:sx2]

        # 标记点
        for m in self.get_marks_at_frame(self.current_frame_idx):
            color = tuple(m["color"])
            sx, sy = self.image_to_screen(m["x"], m["y"])
            if 0 <= sx < self.display_width and 0 <= sy < self.image_area_height:
                nid = m["neuron_id"]
                r = max(5, int(8 * self.zoom_level))
                if nid == self.current_neuron_id:
                    cv2.circle(canvas, (sx, sy), r + 5, (255, 255, 255), 2)
                cv2.circle(canvas, (sx, sy), r, color, -1)
                cv2.putText(canvas, str(nid), (sx + r + 3, sy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
                            cv2.LINE_AA)

        # 其他帧标记
        for nid, ndata in self.neuron_marks.items():
            color = tuple(ndata["color"])
            for m in ndata["marks"]:
                if m["frame"] != self.current_frame_idx:
                    sx, sy = self.image_to_screen(m["x"], m["y"])
                    if 0 <= sx < self.display_width and 0 <= sy < self.image_area_height:
                        cv2.drawMarker(canvas, (sx, sy), color, cv2.MARKER_TILTED_CROSS, 8, 1)

        # 轨迹（track模式）
        if self.mode == 'track':
            for nid, traj in self.neuron_trajectories.items():
                if not traj:
                    continue
                color = tuple(self.colors[nid % len(self.colors)])
                for i in range(len(traj) - 1):
                    p1 = self.image_to_screen(traj[i][1], traj[i][0])
                    p2 = self.image_to_screen(traj[i + 1][1], traj[i + 1][0])
                    if 0 <= p1[0] < self.display_width and 0 <= p1[1] < self.image_area_height:
                        cv2.line(canvas, p1, p2, color, 2, cv2.LINE_AA)
                if traj:
                    end = self.image_to_screen(traj[-1][1], traj[-1][0])
                    cv2.circle(canvas, end, 6, (0, 0, 255), -1)

        # 选区预览
        if self.mouse_in_image and self.mode == 'mark':
            pr = max(2, int(self.mark_radius * self.zoom_level))
            nc = self.colors[self.current_neuron_id % len(self.colors)]
            cv2.circle(canvas, (self.mouse_x, self.mouse_y), pr, nc, 1)
            cv2.circle(canvas, (self.mouse_x, self.mouse_y), 2, (255, 255, 255), -1)

        # 控制面板
        cv2.rectangle(canvas, (0, self.image_area_height), (self.display_width, self.display_height), (25, 25, 25), -1)
        cv2.line(canvas, (0, self.image_area_height), (self.display_width, self.image_area_height), (80, 80, 80), 2)

        # 按钮
        for btn in self.buttons.values():
            btn.draw(canvas)

        # 输入框
        self.frame_input.draw(canvas)
        nc = tuple(self.colors[self.current_neuron_id % len(self.colors)])
        self.neuron_input.draw(canvas, highlight_color=nc)

        # 信息栏
        info1 = f"Frame: {self.current_frame_idx + 1}/{self.total_frames}  |  Zoom: {self.zoom_level:.2f}x  |  Mode: {self.mode.upper()}"
        cv2.putText(canvas, info1, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 220), 2, cv2.LINE_AA)

        total_marks = sum(len(d["marks"]) for d in self.neuron_marks.values())
        total_trajs = len(self.neuron_trajectories)
        info2 = f"Marks: {total_marks}  |  Trajectories: {total_trajs}  |  Radius: {self.mark_radius}px"
        cv2.putText(canvas, info2, (700, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 2, cv2.LINE_AA)

        # 图例
        lx = self.display_width - 220
        ly = 15
        cv2.putText(canvas, "Neurons:", (lx, ly + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 2, cv2.LINE_AA)

        for i, (nid, ndata) in enumerate(list(self.neuron_marks.items())[:12]):
            color = tuple(ndata["color"])
            yp = ly + 40 + i * 22
            cv2.rectangle(canvas, (lx, yp), (lx + 16, yp + 16), color, -1)
            traj_len = len(self.neuron_trajectories.get(nid, []))
            mark_len = len(ndata["marks"])
            prefix = "► " if nid == self.current_neuron_id else ""
            cv2.putText(canvas, f"{prefix}N{nid}: {mark_len}m/{traj_len}t", (lx + 22, yp + 13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)

        # 提示
        tips = "L-Click: Mark | R-Click: Del | Wheel: Zoom | M-Drag: Pan | Enter: confirm input"
        cv2.putText(canvas, tips, (10, self.image_area_height - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 100, 100), 1,
                    cv2.LINE_AA)

        return canvas

    def run(self, video_path, data_path=None):
        self.load_video(video_path)

        if data_path is None:
            data_path = os.path.splitext(video_path)[0] + "_data.json"

        output_video_path = os.path.splitext(video_path)[0] + "_tracked.mp4"

        if os.path.exists(data_path):
            self.load_data(data_path)

        self.read_frame(0)
        self.action_result = None

        win = 'Neuron Tool'
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, self.display_width, self.display_height)
        cv2.setMouseCallback(win, self.mouse_cb)

        print("\n" + "=" * 70)
        print("神经元标记与追踪工具")
        print("=" * 70)
        print("标记模式: 左键标记, 右键删除")
        print("追踪模式: 点击 RUN TRACK 实时追踪并自动保存视频")
        print("输入框: 点击后输入数字，按Enter确认")
        print("快捷键: A/D=前后帧, W/S=±10帧, 0-9=切换神经元")
        print("=" * 70 + "\n")

        while True:
            disp = self.draw()
            if disp is not None:
                cv2.imshow(win, disp)

            # 动作
            if self.action_result == 'confirm':
                self.save_data(data_path)
                break
            elif self.action_result == 'cancel':
                break
            elif self.action_result == 'save':
                self.save_data(data_path)
                self.action_result = None
            elif self.action_result == 'load':
                self.load_data(data_path)
                self.action_result = None
            elif self.action_result == 'run_track':
                self.run_tracking_realtime(output_video_path)
                self.save_data(data_path)  # 自动保存数据
                self.action_result = None

            key = cv2.waitKey(30) & 0xFF

            # 输入模式
            if self.active_input:
                if key == 13:  # Enter
                    if self.active_input == self.frame_input:
                        try:
                            self.read_frame(int(self.frame_input.text) - 1)
                        except:
                            pass
                    elif self.active_input == self.neuron_input:
                        try:
                            self.current_neuron_id = max(0, min(int(self.neuron_input.text), self.max_neurons - 1))
                            self.neuron_input.text = str(self.current_neuron_id)
                        except:
                            pass
                    self.active_input.active = False
                    self.active_input = None
                elif key == 27:  # Esc
                    self.active_input.active = False
                    self.active_input = None
                elif key == 8 or key == 127:  # Backspace
                    self.active_input.text = self.active_input.text[:-1]
                elif ord('0') <= key <= ord('9'):
                    self.active_input.text += chr(key)
                continue

            # 普通快捷键
            if key == 27:
                break
            elif key == ord('a') or key == 81:
                self.read_frame(self.current_frame_idx - 1)
            elif key == ord('d') or key == 83:
                self.read_frame(self.current_frame_idx + 1)
            elif key == ord('w') or key == 82:
                self.read_frame(self.current_frame_idx + 10)
            elif key == ord('s') or key == 84:
                self.read_frame(self.current_frame_idx - 10)
            elif ord('0') <= key <= ord('9'):
                self.current_neuron_id = key - ord('0')
                self.neuron_input.text = str(self.current_neuron_id)
            elif key == ord('+') or key == ord('='):
                self.zoom_level = min(self.max_zoom, self.zoom_level * 1.4)
            elif key == ord('-') or key == ord('_'):
                self.zoom_level = max(self.min_zoom, self.zoom_level / 1.4)
            elif key == 13:
                self.save_data(data_path)
                break

        cv2.destroyAllWindows()
        if self.cap:
            self.cap.release()

        return data_path


if __name__ == "__main__":
    VIDEO_PATH = r"F:\工作文件\RA\python\项目汇总\神经图像\neuron_growth_50.mp4"
    DATA_PATH = r"F:\工作文件\RA\python\项目汇总\神经图像\neuron_data.json"

    tool = NeuronTool(max_neurons=50)
    tool.run(VIDEO_PATH, DATA_PATH)
