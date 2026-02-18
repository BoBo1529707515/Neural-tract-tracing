import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
from math import sqrt, cos, radians
from collections import defaultdict

DEFAULT_INPUT_DIR = r"F:\工作文件\RA\python\项目汇总\神经图像\05_二次空间去噪\frames_final"
DEFAULT_OUTPUT_DIR = r"F:\工作文件\RA\python\项目汇总\神经图像\07_生长追踪"


class NeuronTracker:
    """神经元追踪 - 多帧叠加版（三边界终止）"""

    def __init__(self):
        self.frames = []
        self.height = 0
        self.width = 0

        self.brightness_threshold = 30
        self.search_radius = 30
        self.linearity_weight = 2.0
        self.left_margin = 3
        self.edge_margin = 3
        self.max_turn_angle = 60
        self.max_search_radius = 150
        self.search_radius_step = 20

        self.weight_decay = 0.8
        self.min_overlap_ratio = 0.3
        self.signal_search_radius = 15
        self.max_backtrack_check = 50

        self.tracking_results = []

    def load_frames(self, input_dir, progress_callback=None):
        self.frames = []
        frame_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.png')])
        if not frame_files:
            return False

        sample = os.path.join(input_dir, frame_files[0])
        with open(sample, 'rb') as f:
            img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
        self.height, self.width = img.shape

        for i, fn in enumerate(frame_files):
            with open(os.path.join(input_dir, fn), 'rb') as f:
                img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
            self.frames.append(img)
            if progress_callback and i % 10 == 0:
                progress_callback(i / len(frame_files))

        if progress_callback:
            progress_callback(1.0)
        return True

    def is_bright(self, frame_idx, x, y):
        if frame_idx < 0 or frame_idx >= len(self.frames):
            return False
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return False
        return self.frames[frame_idx][int(y), int(x)] > self.brightness_threshold

    def has_signal_near(self, frame_idx, x, y, radius=None):
        if radius is None:
            radius = self.signal_search_radius
        if frame_idx < 0 or frame_idx >= len(self.frames):
            return False
        frame = self.frames[frame_idx]
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                nx, ny = int(x + dx), int(y + dy)
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    if frame[ny, nx] > self.brightness_threshold:
                        return True
        return False

    def find_signal_range(self, start_frame, x, y):
        earliest = start_frame
        for f in range(start_frame - 1, max(-1, start_frame - self.max_backtrack_check), -1):
            if self.has_signal_near(f, x, y):
                earliest = f
            else:
                break
        return earliest, start_frame

    def check_boundary_reached(self, x, y):
        """
        检查是否到达有效边界（左/上/下都算成功）
        返回: (是否到达, 边界类型)
        """
        if x <= self.left_margin:
            return True, 'left'
        if y <= self.edge_margin:
            return True, 'top'
        if y >= self.height - self.edge_margin:
            return True, 'bottom'
        return False, None

    def is_smooth_transition(self, path, new_point):
        if len(path) < 2:
            return True
        n = min(5, len(path))
        recent = path[-n:]
        dx_prev, dy_prev = 0, 0
        for i in range(1, len(recent)):
            dx_prev += recent[i][0] - recent[i - 1][0]
            dy_prev += recent[i][1] - recent[i - 1][1]
        prev_len = sqrt(dx_prev ** 2 + dy_prev ** 2)
        if prev_len == 0:
            return True
        last = path[-1]
        dx_new = new_point[0] - last[0]
        dy_new = new_point[1] - last[1]
        new_len = sqrt(dx_new ** 2 + dy_new ** 2)
        if new_len == 0:
            return True
        cos_angle = (dx_prev * dx_new + dy_prev * dy_new) / (prev_len * new_len)
        min_cos = cos(radians(self.max_turn_angle))
        return cos_angle >= min_cos

    def compute_linearity(self, path, new_point):
        if len(path) < 2:
            return 1.0
        n = min(5, len(path))
        recent = path[-n:]
        dx_sum, dy_sum = 0, 0
        for i in range(1, len(recent)):
            dx_sum += recent[i][0] - recent[i - 1][0]
            dy_sum += recent[i][1] - recent[i - 1][1]
        dir_len = sqrt(dx_sum ** 2 + dy_sum ** 2)
        if dir_len == 0:
            return 1.0
        last = path[-1]
        new_dx = new_point[0] - last[0]
        new_dy = new_point[1] - last[1]
        new_len = sqrt(new_dx ** 2 + new_dy ** 2)
        if new_len == 0:
            return 1.0
        cos_angle = (dx_sum * new_dx + dy_sum * new_dy) / (dir_len * new_len)
        return (cos_angle + 1) / 2

    def find_leftward_candidates(self, frame_idx, cx, cy, path, direction, visited, radius=None):
        if radius is None:
            radius = self.search_radius
        frame = self.frames[frame_idx]
        candidates = []
        search_range = int(radius)
        for dy in range(-search_range, search_range + 1):
            for dx in range(-search_range, 1):
                if dx == 0 and dy == 0:
                    continue
                dist = sqrt(dx ** 2 + dy ** 2)
                if dist > radius:
                    continue
                nx, ny = cx + dx, cy + dy
                if (nx, ny) in visited:
                    continue
                if nx < 0 or nx >= self.width or ny < 0 or ny >= self.height:
                    continue
                brightness = frame[ny, nx]
                if brightness < self.brightness_threshold:
                    continue
                if not self.is_smooth_transition(path, (nx, ny)):
                    continue
                linearity = self.compute_linearity(path, (nx, ny))
                score = brightness * (linearity ** self.linearity_weight) / (dist + 0.5)
                left_bonus = abs(dx) / (dist + 0.1)
                score *= (1 + left_bonus)
                if direction:
                    dir_x, dir_y = direction
                    dir_len = sqrt(dir_x ** 2 + dir_y ** 2)
                    if dir_len > 0:
                        cos_a = (dx * dir_x + dy * dir_y) / (dist * dir_len)
                        score *= (1 + (cos_a + 1) / 2)
                candidates.append((nx, ny, score, dx, dy))
        return candidates

    def trace_single_frame(self, frame_idx, start_x, start_y, initial_direction=None):
        """在单帧内追踪，三边界都算成功终止"""
        path = [(int(start_x), int(start_y))]
        visited = set()
        visited.add((int(start_x), int(start_y)))
        cx, cy = int(start_x), int(start_y)
        direction = initial_direction if initial_direction else (-1, 0)
        boundary_reached = None

        for step in range(3000):
            # 检查三个边界
            reached, boundary_type = self.check_boundary_reached(cx, cy)
            if reached:
                boundary_reached = boundary_type
                break

            candidates = self.find_leftward_candidates(frame_idx, cx, cy, path, direction, visited)

            if not candidates:
                for radius in range(self.search_radius, self.max_search_radius + 1, self.search_radius_step):
                    candidates = self.find_leftward_candidates(frame_idx, cx, cy, path, direction, visited, radius)
                    if candidates:
                        break

            if not candidates:
                break

            candidates.sort(key=lambda x: -x[2])
            chosen = candidates[0]
            nx, ny = chosen[0], chosen[1]
            dx, dy = chosen[3], chosen[4]

            if dx > 0:
                left_only = [c for c in candidates if c[3] <= 0]
                if left_only:
                    chosen = left_only[0]
                    nx, ny = chosen[0], chosen[1]
                    dx, dy = chosen[3], chosen[4]
                else:
                    break

            if (nx, ny) in visited:
                found = False
                for c in candidates:
                    if (c[0], c[1]) not in visited and c[3] <= 0:
                        nx, ny = c[0], c[1]
                        dx, dy = c[3], c[4]
                        found = True
                        break
                if not found:
                    break

            path.append((nx, ny))
            visited.add((nx, ny))

            dist = sqrt(dx ** 2 + dy ** 2)
            if dist > 0:
                new_dir = (dx / dist, dy / dist)
                if new_dir[0] <= 0:
                    alpha = 0.3
                    direction = (
                        direction[0] * (1 - alpha) + new_dir[0] * alpha,
                        direction[1] * (1 - alpha) + new_dir[1] * alpha
                    )
                    dlen = sqrt(direction[0] ** 2 + direction[1] ** 2)
                    if dlen > 0:
                        direction = (direction[0] / dlen, direction[1] / dlen)

            cx, cy = nx, ny

        return path, boundary_reached

    def compute_frame_weight(self, frame_idx, earliest_frame, latest_frame):
        if earliest_frame == latest_frame:
            return 1.0
        position = (frame_idx - earliest_frame) / (latest_frame - earliest_frame)
        weight = self.weight_decay ** position
        return weight

    def compute_weighted_path(self, paths_by_frame, earliest_frame, latest_frame):
        if not paths_by_frame:
            return []

        weights = {}
        total_weight = 0
        for f in paths_by_frame:
            w = self.compute_frame_weight(f, earliest_frame, latest_frame)
            weights[f] = w
            total_weight += w

        for f in weights:
            weights[f] /= total_weight

        print(f"  帧权重: {[(f, round(w, 3)) for f, w in sorted(weights.items())]}")

        all_x = []
        for path in paths_by_frame.values():
            for pt in path:
                all_x.append(pt[0])

        if not all_x:
            return []

        x_to_weighted_y = defaultdict(list)

        for frame_idx, path in paths_by_frame.items():
            weight = weights[frame_idx]
            path_x_to_y = {}
            for pt in path:
                x, y = pt
                if x not in path_x_to_y:
                    path_x_to_y[x] = []
                path_x_to_y[x].append(y)

            for x, ys in path_x_to_y.items():
                avg_y = sum(ys) / len(ys)
                x_to_weighted_y[x].append((avg_y, weight))

        final_path = []
        for x in sorted(x_to_weighted_y.keys()):
            y_weights = x_to_weighted_y[x]
            participation = sum(w for _, w in y_weights)
            if participation >= self.min_overlap_ratio:
                weighted_y = sum(y * w for y, w in y_weights) / participation
                final_path.append((x, int(round(weighted_y))))

        if len(final_path) > 3:
            final_path = self.smooth_path(final_path)

        return final_path

    def smooth_path(self, path, window=3):
        if len(path) <= window:
            return path
        smoothed = [path[0]]
        for i in range(1, len(path) - 1):
            start = max(0, i - window // 2)
            end = min(len(path), i + window // 2 + 1)
            avg_x = sum(p[0] for p in path[start:end]) / (end - start)
            avg_y = sum(p[1] for p in path[start:end]) / (end - start)
            smoothed.append((int(round(avg_x)), int(round(avg_y))))
        smoothed.append(path[-1])
        return smoothed

    def trace_with_multi_frame_overlay(self, start_frame_idx, start_x, start_y):
        print(f"\n{'=' * 60}")
        print(f"多帧叠加追踪（三边界终止）")
        print(f"选点: ({start_x}, {start_y}) @ Frame {start_frame_idx}")
        print(f"{'=' * 60}")

        earliest_frame, latest_frame = self.find_signal_range(start_frame_idx, start_x, start_y)
        print(f"\n信号帧范围: {earliest_frame} ~ {latest_frame}")

        paths_by_frame = {}
        boundaries_reached = {}

        print(f"\n独立追踪每帧...")

        for frame_idx in range(earliest_frame, latest_frame + 1):
            if self.has_signal_near(frame_idx, start_x, start_y):
                best_pt = None
                best_dist = float('inf')
                for dy in range(-self.signal_search_radius, self.signal_search_radius + 1):
                    for dx in range(-self.signal_search_radius, self.signal_search_radius + 1):
                        nx, ny = int(start_x + dx), int(start_y + dy)
                        if self.is_bright(frame_idx, nx, ny):
                            dist = dx ** 2 + dy ** 2
                            if dist < best_dist:
                                best_dist = dist
                                best_pt = (nx, ny)

                if best_pt:
                    path, boundary = self.trace_single_frame(frame_idx, best_pt[0], best_pt[1])
                    if len(path) > 5:
                        paths_by_frame[frame_idx] = path
                        boundaries_reached[frame_idx] = boundary
                        left_x = min(p[0] for p in path)
                        boundary_str = f"→{boundary}" if boundary else ""
                        print(f"  帧 {frame_idx}: {len(path)}点, 左x={left_x} {boundary_str}")

        if not paths_by_frame:
            print("  没有找到有效路径！")
            return None

        # 统计边界到达情况
        boundary_counts = {'left': 0, 'top': 0, 'bottom': 0, None: 0}
        for b in boundaries_reached.values():
            boundary_counts[b] = boundary_counts.get(b, 0) + 1

        print(
            f"\n边界统计: 左={boundary_counts['left']}, 上={boundary_counts['top']}, 下={boundary_counts['bottom']}, 未达={boundary_counts[None]}")

        print(f"\n计算加权叠加...")
        final_path = self.compute_weighted_path(paths_by_frame, earliest_frame, latest_frame)

        if not final_path:
            print("  叠加后路径为空！")
            return None

        if len(final_path) >= 2 and final_path[0][0] > final_path[-1][0]:
            final_path = final_path[::-1]

        # 判断最终路径的边界状态
        end_pt = final_path[0]  # 最左端点
        final_boundary_reached, final_boundary_type = self.check_boundary_reached(end_pt[0], end_pt[1])

        left_x = final_path[0][0]
        print(f"  最终路径: {len(final_path)}点, 端点=({end_pt[0]},{end_pt[1]})")
        if final_boundary_reached:
            print(f"  ✓ 到达{final_boundary_type}边界")

        all_paths = {}
        for frame_idx in range(earliest_frame, latest_frame + 1):
            if frame_idx in paths_by_frame:
                all_paths[frame_idx] = paths_by_frame[frame_idx]
            else:
                all_paths[frame_idx] = final_path.copy()

        for frame_idx in range(0, earliest_frame):
            all_paths[frame_idx] = final_path.copy()

        print(f"\n完成: 覆盖 {len(all_paths)} 帧")

        return all_paths, paths_by_frame, final_path, (earliest_frame, latest_frame), (final_boundary_reached,
                                                                                       final_boundary_type)


class TrackerGUI:
    """GUI"""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("神经元追踪 - 多帧叠加版（三边界）")
        self.root.geometry("1500x950")

        self.tracker = NeuronTracker()
        self.current_frame_idx = 0
        self.zoom_level = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.dragging = False
        self.drag_start = None
        self.preview_paths = {}
        self.preview_individual_paths = {}
        self.preview_final_path = []
        self.preview_signal_range = None
        self.preview_boundary_info = None
        self.click_point = None

        self.input_dir = DEFAULT_INPUT_DIR
        self.output_dir = DEFAULT_OUTPUT_DIR
        self.colors = [(255, 100, 100), (100, 255, 100), (100, 100, 255), (255, 255, 100), (255, 100, 255),
                       (100, 255, 255)]

        self.setup_ui()
        self.bind_events()

    def setup_ui(self):
        self.main_paned = ttk.PanedWindow(self.root, orient="horizontal")
        self.main_paned.pack(fill="both", expand=True, padx=5, pady=5)

        left = ttk.Frame(self.main_paned)
        self.main_paned.add(left, weight=3)
        left.columnconfigure(0, weight=1)
        left.rowconfigure(0, weight=1)
        self.canvas = tk.Canvas(left, bg='black')
        self.canvas.grid(row=0, column=0, sticky="nsew")

        right = ttk.Frame(self.main_paned, width=350)
        self.main_paned.add(right, weight=0)

        self.ctrl_canvas = tk.Canvas(right, width=330)
        scrollbar = ttk.Scrollbar(right, orient="vertical", command=self.ctrl_canvas.yview)
        self.ctrl_frame = ttk.Frame(self.ctrl_canvas)
        self.ctrl_canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.ctrl_canvas.pack(side="left", fill="both", expand=True)
        self.ctrl_window = self.ctrl_canvas.create_window((0, 0), window=self.ctrl_frame, anchor="nw")
        self.ctrl_frame.bind("<Configure>",
                             lambda e: self.ctrl_canvas.configure(scrollregion=self.ctrl_canvas.bbox("all")))
        self.ctrl_canvas.bind("<Configure>", lambda e: self.ctrl_canvas.itemconfig(self.ctrl_window, width=e.width))
        self.ctrl_canvas.bind("<Enter>", lambda e: self.ctrl_canvas.bind_all("<MouseWheel>",
                                                                             lambda ev: self.ctrl_canvas.yview_scroll(
                                                                                 int(-1 * (ev.delta / 120)), "units")))
        self.ctrl_canvas.bind("<Leave>", lambda e: self.ctrl_canvas.unbind_all("<MouseWheel>"))

        self.setup_controls()

    def setup_controls(self):
        p = self.ctrl_frame
        pad = 3

        # 文件
        f1 = ttk.LabelFrame(p, text="📁 文件", padding=5)
        f1.pack(fill="x", pady=pad, padx=3)
        ttk.Label(f1, text="输入:").grid(row=0, column=0, sticky="w")
        self.input_entry = ttk.Entry(f1, width=22)
        self.input_entry.insert(0, self.input_dir)
        self.input_entry.grid(row=0, column=1, sticky="ew")
        ttk.Button(f1, text="...", command=self.browse_input, width=3).grid(row=0, column=2)
        ttk.Label(f1, text="输出:").grid(row=1, column=0, sticky="w", pady=2)
        self.output_entry = ttk.Entry(f1, width=22)
        self.output_entry.insert(0, self.output_dir)
        self.output_entry.grid(row=1, column=1, sticky="ew")
        ttk.Button(f1, text="...", command=self.browse_output, width=3).grid(row=1, column=2)
        ttk.Button(f1, text="🔄 加载帧", command=self.load_frames).grid(row=2, column=0, columnspan=3, pady=5,
                                                                       sticky="ew")
        f1.columnconfigure(1, weight=1)

        # 导航
        f2 = ttk.LabelFrame(p, text="🎬 帧导航", padding=5)
        f2.pack(fill="x", pady=pad, padx=3)
        row1 = ttk.Frame(f2)
        row1.pack(fill="x")
        ttk.Label(row1, text="帧:").pack(side="left")
        self.frame_var = tk.StringVar(value="0")
        ttk.Entry(row1, textvariable=self.frame_var, width=6).pack(side="left", padx=2)
        self.frame_label = ttk.Label(row1, text="/ 0")
        self.frame_label.pack(side="left")
        ttk.Button(row1, text="Go", command=self.goto_frame, width=4).pack(side="left", padx=3)
        row2 = ttk.Frame(f2)
        row2.pack(fill="x", pady=3)
        ttk.Button(row2, text="⏮-10", command=lambda: self.change_frame(-10), width=5).pack(side="left", padx=1)
        ttk.Button(row2, text="◀-1", command=lambda: self.change_frame(-1), width=5).pack(side="left", padx=1)
        ttk.Button(row2, text="+1▶", command=lambda: self.change_frame(1), width=5).pack(side="left", padx=1)
        ttk.Button(row2, text="+10⏭", command=lambda: self.change_frame(10), width=5).pack(side="left", padx=1)
        self.frame_slider = ttk.Scale(f2, from_=0, to=100, orient="horizontal", command=self.on_slider)
        self.frame_slider.pack(fill="x", pady=2)

        # 缩放
        f3 = ttk.LabelFrame(p, text="🔍 缩放", padding=5)
        f3.pack(fill="x", pady=pad, padx=3)
        row = ttk.Frame(f3)
        row.pack(fill="x")
        ttk.Button(row, text="−", command=lambda: self.zoom(-0.25), width=3).pack(side="left")
        self.zoom_var = tk.StringVar(value="100%")
        ttk.Label(row, textvariable=self.zoom_var, width=6).pack(side="left", padx=3)
        ttk.Button(row, text="+", command=lambda: self.zoom(0.25), width=3).pack(side="left")
        ttk.Button(row, text="适应", command=self.fit_zoom, width=5).pack(side="left", padx=5)

        # 参数
        f4 = ttk.LabelFrame(p, text="⚙️ 参数", padding=5)
        f4.pack(fill="x", pady=pad, padx=3)
        g = ttk.Frame(f4)
        g.pack(fill="x")

        ttk.Label(g, text="亮度阈值:").grid(row=0, column=0, sticky="w")
        self.thresh_var = tk.StringVar(value="30")
        ttk.Entry(g, textvariable=self.thresh_var, width=5).grid(row=0, column=1)
        ttk.Label(g, text="搜索半径:").grid(row=0, column=2, sticky="w", padx=(8, 0))
        self.radius_var = tk.StringVar(value="30")
        ttk.Entry(g, textvariable=self.radius_var, width=5).grid(row=0, column=3)

        ttk.Label(g, text="权重衰减:").grid(row=1, column=0, sticky="w", pady=2)
        self.decay_var = tk.StringVar(value="0.8")
        ttk.Entry(g, textvariable=self.decay_var, width=5).grid(row=1, column=1)
        ttk.Label(g, text="最小重叠:").grid(row=1, column=2, sticky="w", padx=(8, 0))
        self.overlap_var = tk.StringVar(value="0.3")
        ttk.Entry(g, textvariable=self.overlap_var, width=5).grid(row=1, column=3)

        ttk.Label(g, text="最大转角:").grid(row=2, column=0, sticky="w", pady=2)
        self.angle_var = tk.StringVar(value="60")
        ttk.Entry(g, textvariable=self.angle_var, width=5).grid(row=2, column=1)
        ttk.Label(g, text="回退检查:").grid(row=2, column=2, sticky="w", padx=(8, 0))
        self.backcheck_var = tk.StringVar(value="50")
        ttk.Entry(g, textvariable=self.backcheck_var, width=5).grid(row=2, column=3)

        ttk.Button(f4, text="应用参数", command=self.apply_params).pack(pady=3)

        # 边界说明
        f5 = ttk.LabelFrame(p, text="📐 终止边界", padding=5)
        f5.pack(fill="x", pady=pad, padx=3)
        ttk.Label(f5, text="✓ 左边界 (x ≤ 3)\n✓ 上边界 (y ≤ 3)\n✓ 下边界 (y ≥ H-3)\n三边界任一都算成功！",
                  font=("", 9), justify="left").pack(anchor="w")

        # 显示选项
        f6 = ttk.LabelFrame(p, text="👁️ 显示", padding=5)
        f6.pack(fill="x", pady=pad, padx=3)
        self.show_individual_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(f6, text="显示各帧独立轨迹", variable=self.show_individual_var,
                        command=self.update_display).pack(anchor="w")

        # 操作
        f7 = ttk.LabelFrame(p, text="📍 操作", padding=5)
        f7.pack(fill="x", pady=pad, padx=3)
        ttk.Label(f7, text="左键选点 | 右键取消 | Enter确认").pack(anchor="w")
        self.preview_info = tk.StringVar(value="未选择")
        ttk.Label(f7, textvariable=self.preview_info, foreground="blue", wraplength=300).pack(anchor="w", pady=3)
        btn_row = ttk.Frame(f7)
        btn_row.pack(fill="x", pady=3)
        ttk.Button(btn_row, text="✓ 确认", command=self.confirm, width=10).pack(side="left", padx=2)
        ttk.Button(btn_row, text="✗ 取消", command=self.cancel, width=10).pack(side="left", padx=2)

        # 结果
        f8 = ttk.LabelFrame(p, text="📊 结果", padding=5)
        f8.pack(fill="x", pady=pad, padx=3)
        self.result_list = tk.Listbox(f8, height=4, font=("Consolas", 9))
        self.result_list.pack(fill="x")
        btn_row2 = ttk.Frame(f8)
        btn_row2.pack(fill="x", pady=3)
        ttk.Button(btn_row2, text="删除", command=self.delete_result, width=8).pack(side="left", padx=2)
        ttk.Button(btn_row2, text="清空", command=self.clear_results, width=8).pack(side="left", padx=2)

        # 导出
        f9 = ttk.LabelFrame(p, text="💾 导出", padding=5)
        f9.pack(fill="x", pady=pad, padx=3)
        btn_row3 = ttk.Frame(f9)
        btn_row3.pack(fill="x")
        ttk.Button(btn_row3, text="导出视频", command=self.export_video, width=10).pack(side="left", padx=2)
        ttk.Button(btn_row3, text="截图", command=self.screenshot, width=8).pack(side="left", padx=2)

        # 状态
        f10 = ttk.LabelFrame(p, text="📌 状态", padding=5)
        f10.pack(fill="x", pady=pad, padx=3)
        self.status = tk.StringVar(value="请加载帧")
        ttk.Label(f10, textvariable=self.status, wraplength=300, foreground="blue").pack(fill="x")
        self.progress = ttk.Progressbar(f10, mode='determinate')
        self.progress.pack(fill="x", pady=3)

    def bind_events(self):
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<Button-2>", self.on_mid_press)
        self.canvas.bind("<Button-3>", lambda e: self.cancel())
        self.canvas.bind("<B2-Motion>", self.on_mid_drag)
        self.canvas.bind("<ButtonRelease-2>", lambda e: setattr(self, 'dragging', False))
        self.canvas.bind("<MouseWheel>", self.on_scroll)
        self.canvas.bind("<Configure>", lambda e: self.update_display())
        self.root.bind("<Left>", lambda e: self.change_frame(-1))
        self.root.bind("<Right>", lambda e: self.change_frame(1))
        self.root.bind("<Up>", lambda e: self.change_frame(10))
        self.root.bind("<Down>", lambda e: self.change_frame(-10))
        self.root.bind("<Return>", lambda e: self.confirm())
        self.root.bind("<Escape>", lambda e: self.cancel())

    def browse_input(self):
        p = filedialog.askdirectory()
        if p:
            self.input_entry.delete(0, tk.END)
            self.input_entry.insert(0, p)

    def browse_output(self):
        p = filedialog.askdirectory()
        if p:
            self.output_entry.delete(0, tk.END)
            self.output_entry.insert(0, p)

    def load_frames(self):
        self.input_dir = self.input_entry.get()
        if not os.path.exists(self.input_dir):
            messagebox.showerror("错误", "目录不存在")
            return
        self.status.set("加载中...")
        self.progress['value'] = 0
        self.root.update()
        if self.tracker.load_frames(self.input_dir,
                                    lambda p: (setattr(self.progress, 'value', p * 100), self.root.update())):
            n = len(self.tracker.frames)
            self.frame_slider.configure(to=n - 1)
            self.frame_label.configure(text=f"/ {n - 1}")
            self.current_frame_idx = 0
            self.root.after(100, self.fit_zoom)
            self.status.set(f"已加载 {n} 帧")
        else:
            messagebox.showerror("错误", "未找到PNG")
        self.progress['value'] = 0

    def change_frame(self, d):
        if not self.tracker.frames: return
        self.current_frame_idx = max(0, min(len(self.tracker.frames) - 1, self.current_frame_idx + d))
        self.frame_var.set(str(self.current_frame_idx))
        self.frame_slider.set(self.current_frame_idx)
        self.update_display()

    def goto_frame(self):
        try:
            idx = int(self.frame_var.get())
            self.current_frame_idx = max(0, min(len(self.tracker.frames) - 1, idx))
            self.frame_var.set(str(self.current_frame_idx))
            self.frame_slider.set(self.current_frame_idx)
            self.update_display()
        except:
            pass

    def on_slider(self, v):
        if self.tracker.frames:
            self.current_frame_idx = int(float(v))
            self.frame_var.set(str(self.current_frame_idx))
            self.update_display()

    def zoom(self, d):
        old = self.zoom_level
        self.zoom_level = max(0.1, min(10, self.zoom_level + d))
        self.zoom_var.set(f"{int(self.zoom_level * 100)}%")
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        r = self.zoom_level / old
        self.pan_x = cw / 2 - (cw / 2 - self.pan_x) * r
        self.pan_y = ch / 2 - (ch / 2 - self.pan_y) * r
        self.update_display()

    def fit_zoom(self):
        if not self.tracker.frames: return
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        if cw < 10: cw, ch = 800, 600
        self.zoom_level = min(cw / self.tracker.width, ch / self.tracker.height) * 0.95
        self.zoom_var.set(f"{int(self.zoom_level * 100)}%")
        self.pan_x = (cw - self.tracker.width * self.zoom_level) / 2
        self.pan_y = (ch - self.tracker.height * self.zoom_level) / 2
        self.update_display()

    def on_scroll(self, e):
        self.zoom(0.2 if e.delta > 0 else -0.2)

    def on_mid_press(self, e):
        self.dragging = True
        self.drag_start = (e.x, e.y)

    def on_mid_drag(self, e):
        if self.dragging:
            self.pan_x += e.x - self.drag_start[0]
            self.pan_y += e.y - self.drag_start[1]
            self.drag_start = (e.x, e.y)
            self.update_display()

    def canvas_to_image(self, cx, cy):
        return int((cx - self.pan_x) / self.zoom_level), int((cy - self.pan_y) / self.zoom_level)

    def on_click(self, e):
        if not self.tracker.frames: return
        ix, iy = self.canvas_to_image(e.x, e.y)
        if not (0 <= ix < self.tracker.width and 0 <= iy < self.tracker.height): return

        self.apply_params()
        self.click_point = (ix, iy, self.current_frame_idx)
        self.status.set(f"多帧叠加追踪中... ({ix}, {iy})")
        self.root.update()

        result = self.tracker.trace_with_multi_frame_overlay(self.current_frame_idx, ix, iy)

        if result:
            all_paths, individual_paths, final_path, signal_range, boundary_info = result
            self.preview_paths = all_paths
            self.preview_individual_paths = individual_paths
            self.preview_final_path = final_path
            self.preview_signal_range = signal_range
            self.preview_boundary_info = boundary_info

            leftmost = final_path[0][0] if final_path else -1
            earliest, latest = signal_range
            n_frames = len(individual_paths)

            boundary_reached, boundary_type = boundary_info
            boundary_str = f"✓到达{boundary_type}边界" if boundary_reached else "未到达边界"

            self.preview_info.set(
                f"信号帧: {earliest}~{latest} ({n_frames}帧)\n"
                f"叠加路径: {len(final_path)}点\n"
                f"{boundary_str}"
            )
            self.status.set("预览就绪 | Enter确认")
        else:
            self.preview_info.set("追踪失败")
            self.status.set("未找到有效轨迹")

        self.update_display()

    def confirm(self):
        if not self.preview_paths: return
        self.tracker.tracking_results.append((
            self.preview_paths.copy(),
            self.preview_final_path.copy(),
            self.click_point,
            self.preview_signal_range,
            self.preview_boundary_info
        ))
        self.preview_paths = {}
        self.preview_individual_paths = {}
        self.preview_final_path = []
        self.preview_signal_range = None
        self.preview_boundary_info = None
        self.click_point = None
        self.preview_info.set("已确认")
        self.update_result_list()
        self.update_display()
        self.status.set(f"已确认！共 {len(self.tracker.tracking_results)} 条")

    def cancel(self):
        self.preview_paths = {}
        self.preview_individual_paths = {}
        self.preview_final_path = []
        self.preview_signal_range = None
        self.preview_boundary_info = None
        self.click_point = None
        self.preview_info.set("已取消")
        self.update_display()

    def apply_params(self):
        try:
            self.tracker.brightness_threshold = int(self.thresh_var.get())
            self.tracker.search_radius = int(self.radius_var.get())
            self.tracker.weight_decay = float(self.decay_var.get())
            self.tracker.min_overlap_ratio = float(self.overlap_var.get())
            self.tracker.max_turn_angle = int(self.angle_var.get())
            self.tracker.max_backtrack_check = int(self.backcheck_var.get())
        except:
            pass

    def update_result_list(self):
        self.result_list.delete(0, tk.END)
        for i, item in enumerate(self.tracker.tracking_results):
            paths, final_path, _, signal_range, boundary_info = item
            e, l = signal_range if signal_range else (0, 0)
            reached, btype = boundary_info if boundary_info else (False, None)
            bstr = f"→{btype}" if reached else ""
            self.result_list.insert(tk.END, f"#{i + 1}: 帧{e}~{l} {bstr}")

    def delete_result(self):
        sel = self.result_list.curselection()
        if sel:
            del self.tracker.tracking_results[sel[0]]
            self.update_result_list()
            self.update_display()

    def clear_results(self):
        self.tracker.tracking_results.clear()
        self.update_result_list()
        self.update_display()

    def export_video(self):
        if not self.tracker.tracking_results:
            messagebox.showwarning("警告", "无结果")
            return
        self.output_dir = self.output_entry.get()
        os.makedirs(self.output_dir, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        path = os.path.join(self.output_dir, "tracking_3boundary.mp4")
        out = cv2.VideoWriter(path, fourcc, 10, (self.tracker.width, self.tracker.height), True)
        for i in range(len(self.tracker.frames)):
            out.write(self.render(i, True))
            self.progress['value'] = (i + 1) / len(self.tracker.frames) * 100
            self.root.update()
        out.release()
        self.progress['value'] = 0
        self.status.set(f"已保存: {path}")
        messagebox.showinfo("完成", "导出完成!")

    def screenshot(self):
        if not self.tracker.frames: return
        self.output_dir = self.output_entry.get()
        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir, f"frame_{self.current_frame_idx:04d}.png")
        cv2.imwrite(path, self.render(self.current_frame_idx, True))
        self.status.set(f"已保存: {path}")

    def render(self, idx, for_export=False):
        frame = self.tracker.frames[idx]
        vis = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        # 绘制三条边界线
        cv2.line(vis, (self.tracker.left_margin, 0),
                 (self.tracker.left_margin, self.tracker.height), (0, 100, 0), 1)
        cv2.line(vis, (0, self.tracker.edge_margin),
                 (self.tracker.width, self.tracker.edge_margin), (0, 100, 0), 1)
        cv2.line(vis, (0, self.tracker.height - self.tracker.edge_margin),
                 (self.tracker.width, self.tracker.height - self.tracker.edge_margin), (0, 100, 0), 1)

        for i, item in enumerate(self.tracker.tracking_results):
            paths, final_path, _, _, _ = item
            color = self.colors[i % len(self.colors)]
            if idx in paths:
                path = paths[idx]
                for j in range(1, len(path)):
                    cv2.line(vis, path[j - 1], path[j], color, 2)
                if path:
                    cv2.circle(vis, path[0], 5, (0, 255, 0), -1)
                    cv2.circle(vis, path[-1], 5, color, -1)

        if not for_export:
            if self.show_individual_var.get() and self.preview_individual_paths:
                if idx in self.preview_individual_paths:
                    path = self.preview_individual_paths[idx]
                    for j in range(1, len(path)):
                        cv2.line(vis, path[j - 1], path[j], (150, 150, 255), 1)

            if self.preview_paths and idx in self.preview_paths:
                path = self.preview_paths[idx]
                for j in range(1, len(path)):
                    cv2.line(vis, path[j - 1], path[j], (0, 255, 255), 2)
                if path:
                    cv2.circle(vis, path[0], 6, (0, 255, 0), -1)
                    cv2.circle(vis, path[-1], 6, (0, 255, 255), -1)

            if self.click_point and idx == self.click_point[2]:
                cv2.circle(vis, (self.click_point[0], self.click_point[1]), 10, (0, 0, 255), 2)

        if for_export:
            cv2.putText(vis, f"Frame {idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        return vis

    def update_display(self):
        if not self.tracker.frames: return
        vis = cv2.cvtColor(self.render(self.current_frame_idx), cv2.COLOR_BGR2RGB)
        nw, nh = int(self.tracker.width * self.zoom_level), int(self.tracker.height * self.zoom_level)
        if nw < 1 or nh < 1: return
        vis_scaled = cv2.resize(vis, (nw, nh))
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        if cw < 10: return
        canvas_img = np.zeros((ch, cw, 3), dtype=np.uint8)
        x0, y0 = int(self.pan_x), int(self.pan_y)
        sx1, sy1 = max(0, -x0), max(0, -y0)
        sx2, sy2 = min(nw, cw - x0), min(nh, ch - y0)
        dx1, dy1 = max(0, x0), max(0, y0)
        if sx2 > sx1 and sy2 > sy1:
            canvas_img[dy1:dy1 + (sy2 - sy1), dx1:dx1 + (sx2 - sx1)] = vis_scaled[sy1:sy2, sx1:sx2]
        self.photo = ImageTk.PhotoImage(Image.fromarray(canvas_img))
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    print("启动神经元追踪 - 多帧叠加版（三边界）")
    TrackerGUI().run()
