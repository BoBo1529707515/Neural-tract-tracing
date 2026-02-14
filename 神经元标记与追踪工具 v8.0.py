"""
神经元标记与追踪工具 v8.0
完整单文件版本

功能:
    - 视频浏览、缩放、平移
    - 同一帧可标记多个点（同一神经元）
    - 多点约束追踪（A*路径连接）
    - 数据保存/加载 (JSON)
    - 视频导出

操作:
    - 左键: 添加标记
    - 右键: 删除最近的标记
    - 滚轮: 缩放
    - 中键拖动: 平移
    - A/D: 前后帧
    - W/S: ±10帧
    - 0-9: 切换神经元
    - X: 删除当前神经元
"""

import cv2
import numpy as np
import json
import os
import heapq
from datetime import datetime
from skimage.morphology import skeletonize, remove_small_objects

try:
    import tkinter as tk
    from tkinter import filedialog

    HAS_TK = True
except ImportError:
    HAS_TK = False
    print("⚠ tkinter不可用，文件对话框功能受限")


# ============================================================================
#                              配置参数
# ============================================================================

class Config:
    """全局配置"""

    # 神经元参数
    MAX_NEURONS = 50

    # 显示参数
    DISPLAY_WIDTH = 1600
    DISPLAY_HEIGHT = 1000
    PANEL_HEIGHT = 130

    # 缩放参数
    MIN_ZOOM = 0.2
    MAX_ZOOM = 10.0
    ZOOM_STEP = 1.4

    # 标记参数
    MIN_MARK_RADIUS = 1
    MAX_MARK_RADIUS = 20
    DEFAULT_MARK_RADIUS = 1

    # 图像处理参数
    GREEN_THRESHOLD = 40
    CLAHE_CLIP_LIMIT = 3.0
    CLAHE_GRID_SIZE = (8, 8)
    MEDIAN_KERNEL = 5
    ADAPTIVE_BLOCK_SIZE = 15
    ADAPTIVE_C = -3
    MORPH_KERNEL_SIZE = 3
    MORPH_ITERATIONS = 2
    MIN_OBJECT_SIZE = 50

    # 追踪参数
    MAX_GAP = 15
    Y_TOLERANCE = 35
    DIRECTION_WEIGHT = 15
    DIRECTION_HISTORY = 8
    ASTAR_MAX_ITERATIONS = 5000

    # 颜色 (BGR)
    COLOR_BACKGROUND = (30, 30, 30)
    COLOR_PANEL = (25, 25, 25)
    COLOR_BUTTON = (65, 65, 65)
    COLOR_BUTTON_HOVER = (85, 85, 85)
    COLOR_BUTTON_ACTIVE = (70, 130, 70)
    COLOR_BUTTON_SUCCESS = (60, 110, 60)
    COLOR_BUTTON_DANGER = (60, 60, 110)
    COLOR_TEXT = (255, 255, 255)
    COLOR_TEXT_INFO = (255, 220, 0)
    COLOR_TEXT_DIM = (150, 150, 150)
    COLOR_BORDER = (100, 100, 100)


def generate_colors(n):
    """生成n种不同颜色"""
    colors = []
    for i in range(n):
        hue = int(180 * i / n)
        hsv = np.uint8([[[hue, 255, 255]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
        colors.append((int(bgr[0]), int(bgr[1]), int(bgr[2])))
    return colors


# ============================================================================
#                              UI组件
# ============================================================================

class Button:
    """按钮组件"""

    def __init__(self, x, y, w, h, text, color=None):
        self.x, self.y, self.w, self.h = x, y, w, h
        self.text = text
        self.color = color if color else Config.COLOR_BUTTON
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
            bg = Config.COLOR_BUTTON_ACTIVE
        elif self.hover:
            bg = Config.COLOR_BUTTON_HOVER
        else:
            bg = self.color

        cv2.rectangle(img, (self.x, self.y), (self.x + self.w, self.y + self.h), bg, -1)
        cv2.rectangle(img, (self.x, self.y), (self.x + self.w, self.y + self.h), Config.COLOR_BORDER, 1)

        (tw, th), _ = cv2.getTextSize(self.text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        tx = self.x + (self.w - tw) // 2
        ty = self.y + (self.h + th) // 2
        cv2.putText(img, self.text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, Config.COLOR_TEXT, 2, cv2.LINE_AA)


class InputBox:
    """输入框组件"""

    def __init__(self, x, y, w, h, label, default=""):
        self.x, self.y, self.w, self.h = x, y, w, h
        self.label = label
        self.text = default
        self.active = False
        self.cursor_timer = 0

    def contains(self, px, py):
        return self.x <= px < self.x + self.w and self.y <= py < self.y + self.h

    def draw(self, img, highlight_color=None):
        # 标签
        cv2.putText(img, self.label, (self.x, self.y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1,
                    cv2.LINE_AA)

        # 背景
        bg = (55, 55, 75) if self.active else (45, 45, 45)
        cv2.rectangle(img, (self.x, self.y), (self.x + self.w, self.y + self.h), bg, -1)
        border = (150, 180, 255) if self.active else Config.COLOR_BORDER
        cv2.rectangle(img, (self.x, self.y), (self.x + self.w, self.y + self.h), border, 1 if not self.active else 2)

        # 颜色块
        text_x = self.x + 6
        if highlight_color:
            cv2.rectangle(img, (self.x + 4, self.y + 4), (self.x + 20, self.y + self.h - 4), highlight_color, -1)
            text_x = self.x + 26

        # 文字
        display = self.text
        if self.active:
            self.cursor_timer = (self.cursor_timer + 1) % 30
            if self.cursor_timer < 20:
                display += "|"

        color = (255, 255, 100) if self.active else (220, 220, 220)
        cv2.putText(img, display, (text_x, self.y + self.h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    def handle_key(self, key):
        if key == 13 or key == 10:  # Enter
            return 'confirm'
        elif key == 27:  # Esc
            return 'cancel'
        elif key == 8 or key == 127:  # Backspace
            self.text = self.text[:-1]
        elif ord('0') <= key <= ord('9'):
            self.text += chr(key)
        return None


# ============================================================================
#                              视频处理
# ============================================================================

class VideoHandler:
    """视频处理类"""

    def __init__(self):
        self.cap = None
        self.writer = None
        self.video_path = ""
        self.total_frames = 0
        self.frame_width = 0
        self.frame_height = 0
        self.fps = 30
        self.current_frame = None
        self.current_frame_idx = 0

    def load(self, path):
        """加载视频"""
        self.video_path = path
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise ValueError(f"无法打开视频: {path}")

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30

        print(f"✓ 视频: {self.frame_width}x{self.frame_height}, {self.fps:.1f}fps, {self.total_frames}帧")
        return self.get_info()

    def get_info(self):
        return {
            "path": self.video_path,
            "width": self.frame_width,
            "height": self.frame_height,
            "fps": self.fps,
            "total_frames": self.total_frames
        }

    def read_frame(self, idx):
        """读取指定帧"""
        if not self.cap:
            return None
        idx = max(0, min(idx, self.total_frames - 1))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        if ret:
            self.current_frame_idx = idx
            self.current_frame = frame
            return frame
        return None

    def start_writer(self, path):
        """开始写入视频"""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(path, fourcc, int(self.fps), (self.frame_width, self.frame_height))

    def write_frame(self, frame):
        if self.writer:
            self.writer.write(frame)

    def stop_writer(self):
        if self.writer:
            self.writer.release()
            self.writer = None

    def release(self):
        if self.cap:
            self.cap.release()
        self.stop_writer()


# ============================================================================
#                              图像处理
# ============================================================================

class ImageProcessor:
    """图像处理类"""

    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=Config.CLAHE_CLIP_LIMIT, tileGridSize=Config.CLAHE_GRID_SIZE)
        self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                      (Config.MORPH_KERNEL_SIZE, Config.MORPH_KERNEL_SIZE))

    def is_green_pixel(self, frame, y, x):
        """检测是否为绿色像素"""
        h, w = frame.shape[:2]
        if not (0 <= y < h and 0 <= x < w):
            return False
        b, g, r = int(frame[y, x, 0]), int(frame[y, x, 1]), int(frame[y, x, 2])
        if g < 50:
            return False
        if g - r < Config.GREEN_THRESHOLD or g - b < Config.GREEN_THRESHOLD:
            return False
        if r > 150 and b > 150 and g > 150 and abs(r - g) < 50 and abs(b - g) < 50:
            return False
        return True

    def preprocess(self, frame):
        """预处理：提取骨架"""
        # 绿色掩码
        b, g, r = cv2.split(frame)
        b, g, r = b.astype(np.int16), g.astype(np.int16), r.astype(np.int16)
        mask = (g > 50) & ((g - r) > Config.GREEN_THRESHOLD) & ((g - b) > Config.GREEN_THRESHOLD)
        mask &= ~((r > 150) & (b > 150) & (g > 150) & (np.abs(r - g) < 50) & (np.abs(b - g) < 50))
        green_mask = mask.astype(np.uint8) * 255

        # 增强和二值化
        enhanced = self.clahe.apply(frame[:, :, 1])
        denoised = cv2.medianBlur(enhanced, Config.MEDIAN_KERNEL)
        binary = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                       Config.ADAPTIVE_BLOCK_SIZE, Config.ADAPTIVE_C)
        binary = cv2.bitwise_and(binary, green_mask)

        # 形态学和骨架化
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, self.morph_kernel, iterations=Config.MORPH_ITERATIONS)
        cleaned = remove_small_objects(closed > 0, min_size=Config.MIN_OBJECT_SIZE, connectivity=1)
        skeleton = skeletonize(cleaned)

        return skeleton

    def find_nearest_skeleton_point(self, skeleton, frame, point, radius=30):
        """找最近的骨架点"""
        h, w = skeleton.shape
        px, py = int(point[0]), int(point[1])
        best, best_d = None, float('inf')
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                ny, nx = py + dy, px + dx
                if 0 <= ny < h and 0 <= nx < w and skeleton[ny, nx]:
                    if self.is_green_pixel(frame, ny, nx):
                        d = dy * dy + dx * dx
                        if d < best_d:
                            best_d, best = d, (ny, nx)
        return best


# ============================================================================
#                              追踪算法
# ============================================================================

class NeuronTracker:
    """神经元追踪器（支持多点约束）"""

    def __init__(self, img_proc):
        self.img_proc = img_proc

    def get_direction(self, traj, n=None):
        """计算轨迹方向"""
        if n is None:
            n = Config.DIRECTION_HISTORY
        if len(traj) < 2:
            return (0, 1)
        recent = traj[-min(n, len(traj)):]
        dy = recent[-1][0] - recent[0][0]
        dx = recent[-1][1] - recent[0][1]
        length = np.sqrt(dy * dy + dx * dx)
        if length < 0.001:
            return (0, 1)
        return (dy / length, dx / length)

    def trace_with_markers(self, skeleton, frame, markers):
        """
        使用多个标记点约束的追踪

        参数:
            skeleton: 骨架图像
            frame: BGR图像
            markers: 标记列表 [{"x": x, "y": y}, ...]

        返回:
            轨迹点列表 [(y, x), ...]
        """
        if not markers:
            return []

        # 转换为骨架点
        skeleton_points = []
        for m in markers:
            pt = self.img_proc.find_nearest_skeleton_point(skeleton, frame, (m["x"], m["y"]))
            if pt:
                skeleton_points.append(pt)

        if not skeleton_points:
            return []

        # 去重（距离<5的合并）
        unique = []
        for pt in skeleton_points:
            is_dup = False
            for upt in unique:
                if abs(pt[0] - upt[0]) < 5 and abs(pt[1] - upt[1]) < 5:
                    is_dup = True
                    break
            if not is_dup:
                unique.append(pt)

        # 按X排序
        unique = sorted(unique, key=lambda p: p[1])

        # 单点：普通双向追踪
        if len(unique) == 1:
            return self.trace_bidirectional(skeleton, frame, unique[0])

        # 多点：A*连接
        full_traj = []
        for i in range(len(unique) - 1):
            segment = self._find_path_astar(skeleton, frame, unique[i], unique[i + 1])
            if segment:
                if full_traj and segment and segment[0] == full_traj[-1]:
                    segment = segment[1:]
                full_traj.extend(segment)

        # 左端延伸
        if unique:
            left_ext = self._trace_direction(skeleton, frame, unique[0], unique[0][0], go_left=True,
                                             avoid=set(full_traj))
            full_traj = left_ext[::-1] + full_traj

        # 右端延伸
        if unique:
            right_ext = self._trace_direction(skeleton, frame, unique[-1], unique[-1][0], go_left=False,
                                              avoid=set(full_traj))
            full_traj = full_traj + right_ext

        # 去重并排序
        seen = set()
        result = []
        for p in full_traj:
            if p not in seen:
                seen.add(p)
                result.append(p)
        return sorted(result, key=lambda p: p[1])

    def _find_path_astar(self, skeleton, frame, start, end):
        """A*找两点间路径"""
        h, w = skeleton.shape

        def heuristic(p):
            return np.sqrt((p[0] - end[0]) ** 2 + (p[1] - end[1]) ** 2)

        counter = 0
        open_set = [(heuristic(start), counter, start, [start])]
        heapq.heapify(open_set)
        visited = {start}

        for _ in range(Config.ASTAR_MAX_ITERATIONS):
            if not open_set:
                break

            f, _, current, path = heapq.heappop(open_set)

            if current == end or heuristic(current) < 3:
                return path + [end] if current != end else path

            cy, cx = current
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = cy + dy, cx + dx
                    if not (0 <= ny < h and 0 <= nx < w):
                        continue
                    if (ny, nx) in visited:
                        continue
                    if not skeleton[ny, nx]:
                        continue
                    if not self.img_proc.is_green_pixel(frame, ny, nx):
                        continue

                    visited.add((ny, nx))
                    g = len(path)
                    h_val = heuristic((ny, nx))
                    counter += 1
                    heapq.heappush(open_set, (g + h_val, counter, (ny, nx), path + [(ny, nx)]))

        # A*失败，尝试直连
        return self._direct_connect(start, end)

    def _direct_connect(self, start, end):
        """直线连接（A*失败时的备选）"""
        path = [start]
        steps = int(np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)) + 1
        for i in range(1, steps + 1):
            t = i / steps
            y = int(start[0] + t * (end[0] - start[0]))
            x = int(start[1] + t * (end[1] - start[1]))
            if (y, x) != path[-1]:
                path.append((y, x))
        if path[-1] != end:
            path.append(end)
        return path

    def _trace_direction(self, skeleton, frame, start, y_center, go_left=True, avoid=None):
        """单向追踪"""
        if start is None:
            return []

        h, w = skeleton.shape
        visited = set(avoid) if avoid else set()
        traj = [start]
        visited.add(start)
        current = start

        while True:
            current_dir = self.get_direction(traj)
            candidates = []

            for sr in [1, 2, Config.MAX_GAP]:
                for dy in range(-sr, sr + 1):
                    for dx in range(-sr, sr + 1):
                        if dy == 0 and dx == 0:
                            continue
                        ny, nx = current[0] + dy, current[1] + dx
                        if not (0 <= ny < h and 0 <= nx < w):
                            continue
                        if (ny, nx) in visited:
                            continue
                        if not skeleton[ny, nx]:
                            continue
                        if not self.img_proc.is_green_pixel(frame, ny, nx):
                            continue
                        if abs(ny - y_center) > Config.Y_TOLERANCE * 2:
                            continue

                        dist = np.sqrt(dy * dy + dx * dx)
                        base = (-dx if go_left else dx) * 10 - abs(dy) * 3 - dist * 2

                        if dist > 0:
                            cand_dir = (dy / dist, dx / dist)
                            dir_score = current_dir[0] * cand_dir[0] + current_dir[1] * cand_dir[1]
                            base += dir_score * Config.DIRECTION_WEIGHT

                        candidates.append(((ny, nx), base))

                if candidates:
                    break

            if not candidates:
                break

            candidates.sort(key=lambda x: x[1], reverse=True)
            best = candidates[0][0]
            visited.add(best)
            traj.append(best)
            current = best

        return traj

    def trace_bidirectional(self, skeleton, frame, start):
        """双向追踪"""
        if start is None:
            return []
        y_center = start[0]
        left = self._trace_direction(skeleton, frame, start, y_center, go_left=True)
        right = self._trace_direction(skeleton, frame, start, y_center, go_left=False)
        full = left[::-1] + right[1:] if len(right) > 1 else left[::-1]
        return full

    def find_growth(self, skeleton, frame, traj, y_center):
        """找生长点"""
        if not traj:
            return []
        return self._trace_direction(skeleton, frame, traj[-1], y_center, go_left=False, avoid=set(traj))


# ============================================================================
#                              数据管理
# ============================================================================

class DataManager:
    """数据管理类（支持同帧多点）"""

    def __init__(self):
        self.neuron_marks = {}
        self.neuron_trajectories = {}
        self.video_info = {}
        self.colors = generate_colors(Config.MAX_NEURONS)

    def set_video_info(self, info):
        self.video_info = info

    def add_mark(self, nid, frame_idx, x, y):
        """添加标记（同帧可多点，相近点更新）"""
        if nid not in self.neuron_marks:
            self.neuron_marks[nid] = {"color": list(self.colors[nid % len(self.colors)]), "marks": []}

        marks = self.neuron_marks[nid]["marks"]

        # 检查同帧是否有相近的点
        for m in marks:
            if m["frame"] == frame_idx:
                dist = np.sqrt((m["x"] - x) ** 2 + (m["y"] - y) ** 2)
                if dist < 5:
                    m["x"], m["y"] = x, y
                    return "updated"

        marks.append({"frame": frame_idx, "x": x, "y": y})
        return "added"

    def remove_mark_near(self, nid, frame_idx, x, y, radius=20):
        """删除附近的标记"""
        if nid not in self.neuron_marks:
            return None

        marks = self.neuron_marks[nid]["marks"]
        best_idx, best_dist = None, float('inf')

        for i, m in enumerate(marks):
            if m["frame"] == frame_idx:
                dist = np.sqrt((m["x"] - x) ** 2 + (m["y"] - y) ** 2)
                if dist < radius and dist < best_dist:
                    best_dist = dist
                    best_idx = i

        if best_idx is not None:
            return marks.pop(best_idx)
        return None

    def remove_last_mark(self, nid):
        """删除最后一个标记"""
        if nid in self.neuron_marks and self.neuron_marks[nid]["marks"]:
            return self.neuron_marks[nid]["marks"].pop()
        return None

    def get_marks_at_frame(self, frame_idx, nid=None):
        """获取指定帧的标记"""
        result = []
        neurons = {nid: self.neuron_marks[nid]} if nid is not None and nid in self.neuron_marks else self.neuron_marks
        for n, data in neurons.items():
            for m in data["marks"]:
                if m["frame"] == frame_idx:
                    result.append({"neuron_id": n, "x": m["x"], "y": m["y"], "color": data["color"]})
        return result

    def get_neuron_marks(self, nid):
        """获取神经元的所有标记"""
        if nid in self.neuron_marks:
            return self.neuron_marks[nid]["marks"]
        return []

    def get_marks_count_at_frame(self, nid, frame_idx):
        """获取指定帧的标记数量"""
        return sum(1 for m in self.get_neuron_marks(nid) if m["frame"] == frame_idx)

    def delete_neuron(self, nid):
        """删除神经元"""
        deleted = False
        if nid in self.neuron_marks:
            del self.neuron_marks[nid]
            deleted = True
        if nid in self.neuron_trajectories:
            del self.neuron_trajectories[nid]
            deleted = True
        if deleted:
            print(f"  ✓ 删除 N{nid}")
        return deleted

    def get_new_neuron_id(self):
        """获取新神经元ID"""
        ids = set(self.neuron_marks.keys()) | set(self.neuron_trajectories.keys())
        nid = 0
        while nid in ids:
            nid += 1
        return nid

    def get_neuron_color(self, nid):
        """获取神经元颜色"""
        if nid in self.neuron_marks:
            return tuple(self.neuron_marks[nid]["color"])
        return tuple(self.colors[nid % len(self.colors)])

    def set_trajectory(self, nid, traj):
        self.neuron_trajectories[nid] = traj

    def get_trajectory(self, nid):
        return self.neuron_trajectories.get(nid, [])

    def clear_all_trajectories(self):
        count = len(self.neuron_trajectories)
        self.neuron_trajectories = {}
        print(f"  ✓ 清空 {count} 条轨迹")

    def clear_all_data(self):
        self.neuron_marks = {}
        self.neuron_trajectories = {}
        print("  ✓ 清空所有数据")

    def get_all_neuron_ids(self):
        return sorted(set(self.neuron_marks.keys()) | set(self.neuron_trajectories.keys()))

    def save(self, path):
        """保存数据"""
        data = {
            "version": "8.0",
            "created": datetime.now().isoformat(),
            "video_info": self.video_info,
            "neurons": self.neuron_marks,
            "trajectories": {str(k): [list(p) for p in v] for k, v in self.neuron_trajectories.items()}
        }
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"✓ 保存: {path}")

    def load(self, path):
        """加载数据"""
        if not os.path.exists(path):
            return False
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.neuron_marks = {int(k): v for k, v in data.get("neurons", {}).items()}
        self.neuron_trajectories = {int(k): [tuple(p) for p in v] for k, v in data.get("trajectories", {}).items()}
        self.video_info = data.get("video_info", {})
        total = sum(len(d["marks"]) for d in self.neuron_marks.values())
        print(f"✓ 加载: {len(self.neuron_marks)}神经元, {total}标记, {len(self.neuron_trajectories)}轨迹")
        return True


# ============================================================================
#                              可视化
# ============================================================================

class Visualizer:
    """可视化类"""

    def __init__(self, data):
        self.data = data

    def draw_marks_at_frame(self, canvas, frame_idx, current_nid, coord_func):
        """绘制当前帧标记"""
        for m in self.data.get_marks_at_frame(frame_idx):
            color = tuple(m["color"])
            sx, sy = coord_func(m["x"], m["y"])
            if 0 <= sx < canvas.shape[1] and 0 <= sy < Config.DISPLAY_HEIGHT - Config.PANEL_HEIGHT:
                nid = m["neuron_id"]
                r = 6
                if nid == current_nid:
                    cv2.circle(canvas, (sx, sy), r + 4, (255, 255, 255), 2)
                cv2.circle(canvas, (sx, sy), r, color, -1)
                cv2.putText(canvas, str(nid), (sx + r + 2, sy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2,
                            cv2.LINE_AA)

    def draw_other_frame_marks(self, canvas, frame_idx, coord_func):
        """绘制其他帧标记（十字）"""
        for nid, ndata in self.data.neuron_marks.items():
            color = tuple(ndata["color"])
            for m in ndata["marks"]:
                if m["frame"] != frame_idx:
                    sx, sy = coord_func(m["x"], m["y"])
                    if 0 <= sx < canvas.shape[1] and 0 <= sy < Config.DISPLAY_HEIGHT - Config.PANEL_HEIGHT:
                        cv2.drawMarker(canvas, (sx, sy), color, cv2.MARKER_TILTED_CROSS, 8, 1)

    def draw_trajectories(self, canvas, coord_func):
        """绘制轨迹"""
        for nid, traj in self.data.neuron_trajectories.items():
            if not traj:
                continue
            color = self.data.get_neuron_color(nid)
            for i in range(len(traj) - 1):
                p1 = coord_func(traj[i][1], traj[i][0])
                p2 = coord_func(traj[i + 1][1], traj[i + 1][0])
                d = np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
                if d < 50:
                    cv2.line(canvas, p1, p2, color, 2, cv2.LINE_AA)
            if traj:
                end = coord_func(traj[-1][1], traj[-1][0])
                cv2.circle(canvas, end, 5, (0, 0, 255), -1)

    def draw_selection_preview(self, canvas, mx, my, radius, zoom, color):
        """绘制选区预览"""
        r = max(2, int(radius * zoom))
        cv2.circle(canvas, (mx, my), r, color, 1)
        cv2.circle(canvas, (mx, my), 2, (255, 255, 255), -1)

    def draw_info_bar(self, canvas, frame_idx, total, zoom, mode):
        """绘制信息栏"""
        info = f"Frame: {frame_idx + 1}/{total}  |  Zoom: {zoom:.2f}x  |  Mode: {mode.upper()}"
        cv2.putText(canvas, info, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, Config.COLOR_TEXT_INFO, 2, cv2.LINE_AA)

        stats = self.data
        total_marks = sum(len(d["marks"]) for d in stats.neuron_marks.values())
        total_trajs = len(stats.neuron_trajectories)
        info2 = f"Marks: {total_marks}  |  Trajs: {total_trajs}"
        cv2.putText(canvas, info2, (600, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, Config.COLOR_TEXT_DIM, 1, cv2.LINE_AA)

    def draw_legend(self, canvas, current_nid):
        """绘制图例"""
        lx = Config.DISPLAY_WIDTH - 200
        ly = 15
        cv2.putText(canvas, "Neurons:", (lx, ly + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, Config.COLOR_TEXT_DIM, 1,
                    cv2.LINE_AA)

        for i, nid in enumerate(self.data.get_all_neuron_ids()[:10]):
            color = self.data.get_neuron_color(nid)
            yp = ly + 35 + i * 18
            cv2.rectangle(canvas, (lx, yp), (lx + 12, yp + 12), color, -1)
            t_len = len(self.data.get_trajectory(nid))
            m_len = len(self.data.get_neuron_marks(nid))
            prefix = ">" if nid == current_nid else " "
            cv2.putText(canvas, f"{prefix}N{nid}: {m_len}m/{t_len}t", (lx + 18, yp + 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.38, (180, 180, 180), 1, cv2.LINE_AA)

    def draw_tips(self, canvas):
        """绘制提示"""
        tips = "L-Click:Mark | R-Click:Del | Wheel:Zoom | Mid-Drag:Pan | A/D:Frame | W/S:+/-10"
        y = Config.DISPLAY_HEIGHT - Config.PANEL_HEIGHT - 8
        cv2.putText(canvas, tips, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 80, 80), 1, cv2.LINE_AA)

    def draw_frame_on_video(self, frame, idx, total):
        """绘制输出视频帧"""
        vis = frame.copy()
        for nid, traj in self.data.neuron_trajectories.items():
            if not traj:
                continue
            color = self.data.get_neuron_color(nid)
            for i in range(len(traj) - 1):
                p1 = (int(traj[i][1]), int(traj[i][0]))
                p2 = (int(traj[i + 1][1]), int(traj[i + 1][0]))
                d = np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
                if d < 30:
                    cv2.line(vis, p1, p2, color, 2, cv2.LINE_AA)
            if traj:
                start = (int(traj[0][1]), int(traj[0][0]))
                end = (int(traj[-1][1]), int(traj[-1][0]))
                cv2.circle(vis, start, 4, color, -1)
                cv2.circle(vis, end, 6, (0, 0, 255), -1)
                cv2.putText(vis, str(nid), (end[0] + 8, end[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2,
                            cv2.LINE_AA)

        info = f"Frame: {idx + 1}/{total} | Neurons: {len(self.data.neuron_trajectories)}"
        cv2.putText(vis, info, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, info, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        return vis


# ============================================================================
#                              主程序
# ============================================================================

class NeuronTool:
    """神经元标记与追踪主工具"""

    def __init__(self):
        self.video = VideoHandler()
        self.img_proc = ImageProcessor()
        self.data = DataManager()
        self.tracker = NeuronTracker(self.img_proc)
        self.vis = Visualizer(self.data)

        self.zoom_level = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.is_panning = False
        self.pan_start = (0, 0)

        self.mark_radius = Config.DEFAULT_MARK_RADIUS
        self.current_neuron_id = 0
        self.mode = 'mark'

        self.frame_input = None
        self.neuron_input = None
        self.active_input = None

        self.buttons = {}

        self.mouse_x = 0
        self.mouse_y = 0
        self.mouse_in_image = False

        self.is_tracking = False
        self.action_result = None

        self.data_path = None
        self.output_video_path = None

        if HAS_TK:
            self.root = tk.Tk()
            self.root.withdraw()
        else:
            self.root = None

    @property
    def image_area_height(self):
        return Config.DISPLAY_HEIGHT - Config.PANEL_HEIGHT

    def _init_ui(self):
        """初始化UI"""
        y1 = self.image_area_height + 12
        y2 = self.image_area_height + 55
        y3 = self.image_area_height + 95
        bh = 36

        # 行1
        x = 15
        self.buttons['prev'] = Button(x, y1, 55, bh, '<Prev');
        x += 60
        self.buttons['next'] = Button(x, y1, 55, bh, 'Next>');
        x += 60
        self.buttons['prev10'] = Button(x, y1, 40, bh, '-10');
        x += 45
        self.buttons['next10'] = Button(x, y1, 40, bh, '+10');
        x += 55

        self.frame_input = InputBox(x, y1 + 2, 85, 32, "Frame:")
        x += 100

        self.buttons['zin'] = Button(x, y1, 35, bh, 'Z+');
        x += 40
        self.buttons['zout'] = Button(x, y1, 35, bh, 'Z-');
        x += 40
        self.buttons['zfit'] = Button(x, y1, 35, bh, 'Fit');
        x += 45

        self.buttons['r_down'] = Button(x, y1, 30, bh, 'R-');
        x += 35
        self.buttons['r_up'] = Button(x, y1, 30, bh, 'R+');
        x += 45

        self.neuron_input = InputBox(x, y1 + 2, 65, 32, "Neuron:")
        self.neuron_input.text = "0"
        x += 80

        self.buttons['new_n'] = Button(x, y1, 45, bh, 'New');
        x += 50
        self.buttons['del_n'] = Button(x, y1, 50, bh, 'Del N');
        x += 55
        self.buttons['del_mark'] = Button(x, y1, 65, bh, 'Del Mk');
        x += 70

        # 行2
        x = 15
        self.buttons['mode_mark'] = Button(x, y2, 65, bh, 'MARK');
        x += 70
        self.buttons['mode_track'] = Button(x, y2, 65, bh, 'TRACK');
        x += 80

        self.buttons['run_track'] = Button(x, y2, 90, bh, '> RUN', Config.COLOR_BUTTON_SUCCESS);
        x += 95
        self.buttons['stop'] = Button(x, y2, 50, bh, 'STOP', Config.COLOR_BUTTON_DANGER);
        x += 55

        self.buttons['undo'] = Button(x, y2, 45, bh, 'Undo');
        x += 50
        self.buttons['save'] = Button(x, y2, 45, bh, 'Save', Config.COLOR_BUTTON_SUCCESS);
        x += 50
        self.buttons['save_as'] = Button(x, y2, 60, bh, 'SaveAs');
        x += 65
        self.buttons['load'] = Button(x, y2, 45, bh, 'Load');
        x += 50
        self.buttons['set_out'] = Button(x, y2, 60, bh, 'SetOut');
        x += 65

        self.buttons['clear_traj'] = Button(x, y2, 70, bh, 'ClrTraj', Config.COLOR_BUTTON_DANGER);
        x += 75
        self.buttons['clear_all'] = Button(x, y2, 70, bh, 'ClrALL', Config.COLOR_BUTTON_DANGER);
        x += 80

        self.buttons['confirm'] = Button(x, y2, 75, bh, 'CONFIRM', Config.COLOR_BUTTON_SUCCESS);
        x += 80
        self.buttons['cancel'] = Button(x, y2, 55, bh, 'Cancel', Config.COLOR_BUTTON_DANGER)

        # 行3：快速选择
        for i in range(20):
            self.buttons[f'n{i}'] = Button(15 + i * 48, y3, 44, 28, f'N{i}')

    def screen_to_image(self, sx, sy):
        dw = int(self.video.frame_width * self.zoom_level)
        dh = int(self.video.frame_height * self.zoom_level)
        ox = (Config.DISPLAY_WIDTH - dw) // 2 + self.pan_x
        oy = (self.image_area_height - dh) // 2 + self.pan_y
        return (sx - ox) / self.zoom_level, (sy - oy) / self.zoom_level

    def image_to_screen(self, ix, iy):
        dw = int(self.video.frame_width * self.zoom_level)
        dh = int(self.video.frame_height * self.zoom_level)
        ox = (Config.DISPLAY_WIDTH - dw) // 2 + self.pan_x
        oy = (self.image_area_height - dh) // 2 + self.pan_y
        return int(ix * self.zoom_level + ox), int(iy * self.zoom_level + oy)

    def _fit_zoom(self):
        sw = (Config.DISPLAY_WIDTH - 40) / self.video.frame_width
        sh = (self.image_area_height - 40) / self.video.frame_height
        self.zoom_level = min(sw, sh, 1.0)
        self.pan_x = self.pan_y = 0

    def _update_frame_input(self):
        self.frame_input.text = str(self.video.current_frame_idx + 1)

    def handle_button_click(self, name):
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
            ix, iy = self.screen_to_image(self.mouse_x, self.mouse_y)
            removed = self.data.remove_mark_near(self.current_neuron_id, self.video.current_frame_idx, int(ix), int(iy))
            if removed:
                print(f"  删除 N{self.current_neuron_id} @ 帧{self.video.current_frame_idx + 1}")
        elif name == 'undo':
            removed = self.data.remove_last_mark(self.current_neuron_id)
            if removed:
                print(f"  撤销 N{self.current_neuron_id} 最后标记")
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

    def _confirm_input(self):
        if self.frame_input.active:
            try:
                self.video.read_frame(int(self.frame_input.text) - 1)
            except:
                pass
            self.frame_input.active = False

        if self.neuron_input.active:
            try:
                self.current_neuron_id = max(0, min(int(self.neuron_input.text), Config.MAX_NEURONS - 1))
                self.neuron_input.text = str(self.current_neuron_id)
            except:
                pass
            self.neuron_input.active = False

        self.active_input = None

    def mouse_callback(self, event, x, y, flags, param):
        self.mouse_x, self.mouse_y = x, y
        in_panel = y >= self.image_area_height

        if not in_panel:
            ix, iy = self.screen_to_image(x, y)
            self.mouse_in_image = 0 <= ix < self.video.frame_width and 0 <= iy < self.video.frame_height
        else:
            self.mouse_in_image = False

        for btn in self.buttons.values():
            btn.hover = btn.contains(x, y)

        for i in range(20):
            self.buttons[f'n{i}'].active = (i == self.current_neuron_id)

        self.buttons['mode_mark'].active = (self.mode == 'mark')
        self.buttons['mode_track'].active = (self.mode == 'track')

        if event == cv2.EVENT_LBUTTONDOWN:
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
                self._confirm_input()

            if in_panel:
                for name, btn in self.buttons.items():
                    if btn.contains(x, y):
                        result = self.handle_button_click(name)
                        if result:
                            self.action_result = result
                        return
            else:
                if self.mode == 'mark':
                    ix, iy = self.screen_to_image(x, y)
                    if 0 <= ix < self.video.frame_width and 0 <= iy < self.video.frame_height:
                        mx, my = int(round(ix)), int(round(iy))
                        result = self.data.add_mark(self.current_neuron_id, self.video.current_frame_idx, mx, my)
                        count = self.data.get_marks_count_at_frame(self.current_neuron_id, self.video.current_frame_idx)
                        print(
                            f"  + N{self.current_neuron_id} @ F{self.video.current_frame_idx + 1}: ({mx},{my}) [帧内{count}点]")

        elif event == cv2.EVENT_RBUTTONDOWN:
            if not in_panel and self.mode == 'mark':
                ix, iy = self.screen_to_image(x, y)
                removed = self.data.remove_mark_near(self.current_neuron_id, self.video.current_frame_idx, int(ix),
                                                     int(iy))
                if removed:
                    print(f"  - 删除 N{self.current_neuron_id} @ F{self.video.current_frame_idx + 1}")

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
                if flags > 0:
                    self.zoom_level = min(Config.MAX_ZOOM, self.zoom_level * 1.2)
                else:
                    self.zoom_level = max(Config.MIN_ZOOM, self.zoom_level / 1.2)

    def draw(self):
        if self.video.current_frame is None:
            return None

        canvas = np.zeros((Config.DISPLAY_HEIGHT, Config.DISPLAY_WIDTH, 3), dtype=np.uint8)
        canvas[:] = Config.COLOR_BACKGROUND

        # 图像
        frame = self.video.current_frame
        dw = int(self.video.frame_width * self.zoom_level)
        dh = int(self.video.frame_height * self.zoom_level)
        interp = cv2.INTER_LINEAR if self.zoom_level < 1 else cv2.INTER_NEAREST
        resized = cv2.resize(frame, (dw, dh), interpolation=interp)

        ox = (Config.DISPLAY_WIDTH - dw) // 2 + self.pan_x
        oy = (self.image_area_height - dh) // 2 + self.pan_y

        sx1, sy1 = max(0, -ox), max(0, -oy)
        sx2, sy2 = min(dw, Config.DISPLAY_WIDTH - ox), min(dh, self.image_area_height - oy)
        dx1, dy1 = max(0, ox), max(0, oy)
        dx2, dy2 = min(Config.DISPLAY_WIDTH, ox + dw), min(self.image_area_height, oy + dh)

        if sx2 > sx1 and sy2 > sy1:
            canvas[dy1:dy2, dx1:dx2] = resized[sy1:sy2, sx1:sx2]

        # 标记
        self.vis.draw_marks_at_frame(canvas, self.video.current_frame_idx, self.current_neuron_id, self.image_to_screen)
        self.vis.draw_other_frame_marks(canvas, self.video.current_frame_idx, self.image_to_screen)

        # 轨迹
        if self.mode == 'track':
            self.vis.draw_trajectories(canvas, self.image_to_screen)

        # 选区预览
        if self.mouse_in_image and self.mode == 'mark':
            color = self.data.get_neuron_color(self.current_neuron_id)
            self.vis.draw_selection_preview(canvas, self.mouse_x, self.mouse_y, self.mark_radius, self.zoom_level,
                                            color)

        # 面板
        cv2.rectangle(canvas, (0, self.image_area_height), (Config.DISPLAY_WIDTH, Config.DISPLAY_HEIGHT),
                      Config.COLOR_PANEL, -1)
        cv2.line(canvas, (0, self.image_area_height), (Config.DISPLAY_WIDTH, self.image_area_height), (60, 60, 60), 2)

        for btn in self.buttons.values():
            btn.draw(canvas)

        self.frame_input.draw(canvas)
        self.neuron_input.draw(canvas, self.data.get_neuron_color(self.current_neuron_id))

        self.vis.draw_info_bar(canvas, self.video.current_frame_idx, self.video.total_frames, self.zoom_level,
                               self.mode)
        self.vis.draw_legend(canvas, self.current_neuron_id)
        self.vis.draw_tips(canvas)

        return canvas

    def run_tracking(self, output_path):
        """执行追踪"""
        if not self.data.neuron_marks:
            print("⚠ 没有标记数据")
            return

        if not output_path:
            print("⚠ 未设置输出路径")
            return

        print("\n开始追踪...")
        self.data.clear_all_trajectories()
        self.is_tracking = True

        # 找最早标记帧
        all_frames = []
        for nid in self.data.neuron_marks:
            for m in self.data.get_neuron_marks(nid):
                all_frames.append(m["frame"])
        start_frame = min(all_frames) if all_frames else 0

        self.video.start_writer(output_path)

        # 初始化
        self.video.read_frame(start_frame)
        skeleton = self.img_proc.preprocess(self.video.current_frame)

        for nid in self.data.neuron_marks:
            marks = self.data.get_neuron_marks(nid)
            if not marks:
                continue

            # 使用多点约束追踪
            traj = self.tracker.trace_with_markers(skeleton, self.video.current_frame, marks)
            if len(traj) >= 5:
                traj = sorted(traj, key=lambda p: p[1])
                self.data.set_trajectory(nid, traj)
                print(f"  N{nid}: 初始化 {len(traj)} 点 (从{len(marks)}个标记)")

        print(f"✓ 初始化 {len(self.data.neuron_trajectories)} 根神经元")

        win = 'Tracking (Q to stop)'
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 1280, 720)

        delay = max(1, int(1000 / self.video.fps)) if self.video.fps > 0 else 30

        for fidx in range(start_frame, self.video.total_frames):
            if not self.is_tracking:
                print("追踪停止")
                break

            self.video.read_frame(fidx)

            if fidx > start_frame:
                skeleton = self.img_proc.preprocess(self.video.current_frame)

                for nid, traj in list(self.data.neuron_trajectories.items()):
                    if not traj:
                        continue
                    y_center = np.mean([p[0] for p in traj])
                    new_pts = self.tracker.find_growth(skeleton, self.video.current_frame, traj, y_center)
                    if new_pts:
                        traj.extend(new_pts)
                        self.data.set_trajectory(nid, sorted(traj, key=lambda p: p[1]))

            vis = self.vis.draw_frame_on_video(self.video.current_frame, fidx, self.video.total_frames)
            self.video.write_frame(vis)

            scale = min(1.0, 1280 / self.video.frame_width)
            display = cv2.resize(vis, None, fx=scale, fy=scale)
            cv2.imshow(win, display)

            if cv2.waitKey(delay) & 0xFF in [ord('q'), 27]:
                self.is_tracking = False
                break

        self.video.stop_writer()
        self.is_tracking = False
        cv2.destroyWindow(win)

        print(f"\n✓ 追踪完成! 视频: {output_path}")
        for nid, traj in self.data.neuron_trajectories.items():
            print(f"  N{nid}: {len(traj)} 点")

    def run(self, video_path, data_path=None):
        """运行主程序"""
        self.video.load(video_path)
        self.data.set_video_info(self.video.get_info())

        if data_path is None:
            self.data_path = os.path.splitext(video_path)[0] + "_data.json"
        else:
            self.data_path = data_path

        self.output_video_path = os.path.splitext(video_path)[0] + "_tracked.mp4"

        if os.path.exists(self.data_path):
            self.data.load(self.data_path)

        self._init_ui()
        self.video.read_frame(0)
        self._update_frame_input()
        self._fit_zoom()

        self.action_result = None

        win = 'Neuron Tool v8'
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, Config.DISPLAY_WIDTH, Config.DISPLAY_HEIGHT)
        cv2.setMouseCallback(win, self.mouse_callback)

        print("\n" + "=" * 60)
        print("神经元标记与追踪工具 v8.0")
        print("=" * 60)
        print("操作: 左键=标记 | 右键=删除 | 滚轮=缩放 | 中键=平移")
        print("快捷: A/D=帧 | W/S=±10帧 | 0-9=神经元 | X=删除神经元")
        print("=" * 60 + "\n")

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
                if HAS_TK:
                    path = filedialog.asksaveasfilename(
                        defaultextension=".json",
                        filetypes=[("JSON", "*.json")],
                        initialfile=os.path.basename(self.data_path)
                    )
                    if path:
                        self.data_path = path
                        self.data.save(self.data_path)
                else:
                    self.data.save(self.data_path)
                self.action_result = None
            elif self.action_result == 'load':
                if HAS_TK:
                    path = filedialog.askopenfilename(
                        filetypes=[("JSON", "*.json")]
                    )
                    if path:
                        self.data_path = path
                        self.data.load(self.data_path)
                self.action_result = None
            elif self.action_result == 'set_out':
                if HAS_TK:
                    path = filedialog.asksaveasfilename(
                        defaultextension=".mp4",
                        filetypes=[("MP4", "*.mp4")],
                        initialfile=os.path.basename(self.output_video_path)
                    )
                    if path:
                        self.output_video_path = path
                        print(f"输出路径: {self.output_video_path}")
                self.action_result = None
            elif self.action_result == 'run_track':
                self.run_tracking(self.output_video_path)
                self.data.save(self.data_path)
                self.action_result = None

            # 键盘
            key = cv2.waitKey(30) & 0xFF

            if self.active_input:
                result = self.active_input.handle_key(key)
                if result == 'confirm':
                    self._confirm_input()
                elif result == 'cancel':
                    self.active_input.active = False
                    self.active_input = None
                continue

            if key == 27:
                break
            elif key == ord('a'):
                self.video.read_frame(self.video.current_frame_idx - 1)
                self._update_frame_input()
            elif key == ord('d'):
                self.video.read_frame(self.video.current_frame_idx + 1)
                self._update_frame_input()
            elif key == ord('w'):
                self.video.read_frame(self.video.current_frame_idx + 10)
                self._update_frame_input()
            elif key == ord('s'):
                self.video.read_frame(self.video.current_frame_idx - 10)
                self._update_frame_input()
            elif key == 81:  # Left
                self.pan_x += 50
            elif key == 83:  # Right
                self.pan_x -= 50
            elif key == 82:  # Up
                self.pan_y += 50
            elif key == 84:  # Down
                self.pan_y -= 50
            elif ord('0') <= key <= ord('9'):
                self.current_neuron_id = key - ord('0')
                self.neuron_input.text = str(self.current_neuron_id)
            elif key == ord('+') or key == ord('='):
                self.zoom_level = min(Config.MAX_ZOOM, self.zoom_level * Config.ZOOM_STEP)
            elif key == ord('-'):
                self.zoom_level = max(Config.MIN_ZOOM, self.zoom_level / Config.ZOOM_STEP)
            elif key == ord('x') or key == ord('X'):
                self.data.delete_neuron(self.current_neuron_id)
            elif key == 13:
                self.data.save(self.data_path)
                break

        cv2.destroyAllWindows()
        self.video.release()
        if self.root:
            self.root.destroy()

        return self.data_path


# ============================================================================
#                              入口
# ============================================================================

if __name__ == "__main__":
    # 修改为你的视频路径
    VIDEO_PATH = r"F:\工作文件\RA\python\项目汇总\神经图像\neuron_growth_50.mp4"

    # 可选：指定数据文件路径（默认与视频同名.json）
    DATA_PATH = None

    tool = NeuronTool()
    tool.run(VIDEO_PATH, DATA_PATH)
