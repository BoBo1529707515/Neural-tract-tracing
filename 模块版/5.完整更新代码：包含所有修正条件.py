import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
from math import sqrt, cos, radians

# ╔══════════════════════════════════════════════════════════════╗
# ║  神经元追踪 Part1 · 完整版（所有8条规则）                        ║
# ╚══════════════════════════════════════════════════════════════╝

DEFAULT_INPUT_DIR = r"F:\工作文件\RA\python\项目汇总\神经图像\05_二次空间去噪\frames_final"
DEFAULT_OUTPUT_DIR = r"F:\工作文件\RA\python\项目汇总\神经图像\07_生长追踪"


class NeuronTracker:
    """神经元追踪核心类"""

    def __init__(self):
        self.frames = []
        self.height = 0
        self.width = 0

        # 基本参数
        self.brightness_threshold = 30
        self.search_radius = 30
        self.linearity_weight = 2.0

        # 分叉处理
        self.branch_score_ratio = 0.85
        self.max_backtrack_frames = 10

        # 边界
        self.left_margin = 3
        self.edge_margin = 3

        # 平滑约束
        self.max_turn_angle = 60

        # 自适应搜索
        self.max_search_radius = 150
        self.search_radius_step = 20

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

    def is_smooth_transition(self, path, new_point):
        """规则7: 检查平滑性"""
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
        """规则4: 计算线性度"""
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
        """规则1: 只找向左的候选点 (dx <= 0)"""
        if radius is None:
            radius = self.search_radius

        frame = self.frames[frame_idx]
        candidates = []
        search_range = int(radius)

        for dy in range(-search_range, search_range + 1):
            for dx in range(-search_range, 1):  # 规则1: dx <= 0
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

                # 规则7: 平滑性检查
                if not self.is_smooth_transition(path, (nx, ny)):
                    continue

                # 规则4: 线性度评分
                linearity = self.compute_linearity(path, (nx, ny))
                score = brightness * (linearity ** self.linearity_weight) / (dist + 0.5)

                # 偏好更向左的点
                left_bonus = abs(dx) / (dist + 0.1)
                score *= (1 + left_bonus)

                # 方向一致性加成
                if direction:
                    dir_x, dir_y = direction
                    dir_len = sqrt(dir_x ** 2 + dir_y ** 2)
                    if dir_len > 0:
                        cos_a = (dx * dir_x + dy * dir_y) / (dist * dir_len)
                        score *= (1 + (cos_a + 1) / 2)

                candidates.append((nx, ny, score, dx, dy))

        return candidates

    def detect_branch(self, candidates):
        """规则3/5: 检测分叉"""
        if len(candidates) < 2:
            return False, candidates

        sorted_cands = sorted(candidates, key=lambda x: -x[2])
        best_score = sorted_cands[0][2]

        high_score = [c for c in sorted_cands if c[2] >= best_score * self.branch_score_ratio]

        if len(high_score) >= 2:
            dirs = []
            for c in high_score[:3]:
                dx, dy = c[3], c[4]
                dist = sqrt(dx ** 2 + dy ** 2)
                if dist > 0:
                    dirs.append((dx / dist, dy / dist))

            if len(dirs) >= 2:
                cos_angle = dirs[0][0] * dirs[1][0] + dirs[0][1] * dirs[1][1]
                if cos_angle < 0.5:
                    return True, sorted_cands

        return False, sorted_cands

    def get_direction_no_branch(self, frame_idx, position, max_steps=30):
        """规则3: 获取无分叉方向"""
        px, py = position

        start_pt = None
        min_dist = float('inf')

        for dy in range(-15, 16):
            for dx in range(-15, 16):
                nx, ny = px + dx, py + dy
                if self.is_bright(frame_idx, nx, ny):
                    dist = dx ** 2 + dy ** 2
                    if dist < min_dist:
                        min_dist = dist
                        start_pt = (nx, ny)

        if start_pt is None:
            return None

        path = [start_pt]
        cx, cy = start_pt
        visited = {start_pt}
        direction = (-1, 0)

        for _ in range(max_steps):
            candidates = self.find_leftward_candidates(frame_idx, cx, cy, path, direction, visited)

            if not candidates:
                break

            is_branch, sorted_cands = self.detect_branch(candidates)

            if is_branch:
                break

            best = sorted_cands[0]
            nx, ny = best[0], best[1]
            dx, dy = best[3], best[4]

            path.append((nx, ny))
            visited.add((nx, ny))

            dist = sqrt(dx ** 2 + dy ** 2)
            if dist > 0:
                direction = (dx / dist, dy / dist)

            cx, cy = nx, ny

            if cx <= self.left_margin:
                break

        if len(path) >= 3:
            dx = path[-1][0] - path[0][0]
            dy = path[-1][1] - path[0][1]
            length = sqrt(dx ** 2 + dy ** 2)
            if length > 0 and dx < 0:
                return (dx / length, dy / length)

        return (-1, 0)

    def resolve_branch(self, current_frame, position, candidates):
        """规则3: 分叉时回退到更早帧"""
        print(f"    分叉 @ 帧{current_frame}")

        ref_direction = None

        for back_frame in range(current_frame - 1, max(-1, current_frame - self.max_backtrack_frames - 1), -1):
            if back_frame < 0:
                break

            dir_earlier = self.get_direction_no_branch(back_frame, position)

            if dir_earlier is not None and dir_earlier[0] < 0:
                ref_direction = dir_earlier
                print(f"    回退到帧{back_frame}找到方向: {ref_direction}")
                break

        if ref_direction is None:
            ref_direction = (-1, 0)

        best = None
        best_score = -999

        for c in candidates:
            dx, dy = c[3], c[4]
            if dx > 0:
                continue
            dist = sqrt(dx ** 2 + dy ** 2)
            if dist > 0:
                cos_a = (dx * ref_direction[0] + dy * ref_direction[1]) / dist
                if cos_a > best_score:
                    best_score = cos_a
                    best = c

        return best if best else (candidates[0] if candidates else None)

    def search_with_expanding_radius(self, frame_idx, cx, cy, path, direction, visited):
        """规则8: 自适应扩大搜索半径"""
        for radius in range(self.search_radius, self.max_search_radius + 1, self.search_radius_step):
            candidates = self.find_leftward_candidates(frame_idx, cx, cy, path, direction, visited, radius)

            if candidates:
                print(f"    扩大半径到{radius}找到候选")
                return candidates, radius

        return [], self.max_search_radius

    def trace_leftward(self, frame_idx, start_x, start_y, initial_direction, all_paths):
        """在单帧内追踪，包含所有规则"""
        path = [(int(start_x), int(start_y))]
        visited = set()
        visited.add((int(start_x), int(start_y)))

        cx, cy = int(start_x), int(start_y)
        direction = initial_direction if initial_direction else (-1, 0)

        for step in range(5000):
            # 规则2: 检查左边界
            if cx <= self.left_margin:
                print(f"    到达左边界 x={cx}")
                break

            # 检查上下边界
            if cy <= self.edge_margin or cy >= self.height - self.edge_margin:
                print(f"    到达上下边界 y={cy}")
                break

            # 寻找候选
            candidates = self.find_leftward_candidates(frame_idx, cx, cy, path, direction, visited)

            # 规则8: 找不到时扩大搜索
            if not candidates and cx > self.left_margin:
                candidates, used_radius = self.search_with_expanding_radius(
                    frame_idx, cx, cy, path, direction, visited
                )

            if not candidates:
                break

            # 规则3/5: 检测分叉
            is_branch, sorted_cands = self.detect_branch(candidates)

            if is_branch:
                chosen = self.resolve_branch(frame_idx, (cx, cy), sorted_cands)
            else:
                chosen = sorted_cands[0] if sorted_cands else None

            if chosen is None:
                break

            nx, ny = chosen[0], chosen[1]
            dx, dy = chosen[3], chosen[4]

            # 规则1: 再次确认不向右
            if dx > 0:
                left_only = [c for c in sorted_cands if c[3] <= 0]
                if left_only:
                    chosen = left_only[0]
                    nx, ny = chosen[0], chosen[1]
                    dx, dy = chosen[3], chosen[4]
                else:
                    break

            if (nx, ny) in visited:
                found = False
                for c in sorted_cands:
                    if (c[0], c[1]) not in visited and c[3] <= 0:
                        nx, ny = c[0], c[1]
                        dx, dy = c[3], c[4]
                        found = True
                        break
                if not found:
                    break

            path.append((nx, ny))
            visited.add((nx, ny))

            # 更新方向（平滑）
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

        return path

    def find_start_in_frame(self, frame_idx, reference_path, direction):
        """找起点"""
        for pt in reference_path:
            if self.is_bright(frame_idx, pt[0], pt[1]):
                return pt

        if reference_path:
            start = reference_path[0]
            for step in range(1, self.max_search_radius):
                cx = int(start[0] + direction[0] * step)
                cy = int(start[1] + direction[1] * step)

                if cx > start[0]:
                    continue

                for dy in range(-8, 9):
                    for dx in range(-8, 1):
                        if self.is_bright(frame_idx, cx + dx, cy + dy):
                            return (cx + dx, cy + dy)

        return None

    def trace_backward_complete(self, start_frame_idx, start_x, start_y):
        """主追踪函数"""
        print(f"\n{'=' * 60}")
        print(f"开始追踪（完整版8条规则）")
        print(f"起点: ({start_x}, {start_y}) @ Frame {start_frame_idx}")
        print(f"{'=' * 60}")

        all_paths = {}

        # 当前帧追踪
        print(f"\n帧 {start_frame_idx}...")
        current_path = self.trace_leftward(start_frame_idx, start_x, start_y, (-1, 0), all_paths)

        if len(current_path) >= 2 and current_path[0][0] > current_path[-1][0]:
            current_path = current_path[::-1]

        all_paths[start_frame_idx] = current_path

        left_x = current_path[0][0] if current_path else start_x
        print(f"  路径: {len(current_path)}点, 最左x={left_x}")

        # 向前回溯
        prev_path = current_path
        direction = (-1, 0)

        if len(prev_path) >= 2:
            dx = prev_path[0][0] - prev_path[-1][0]
            dy = prev_path[0][1] - prev_path[-1][1]
            length = sqrt(dx ** 2 + dy ** 2)
            if length > 0 and dx < 0:
                direction = (dx / length, dy / length)

        consecutive_gaps = 0

        for frame_idx in range(start_frame_idx - 1, -1, -1):
            start_pt = self.find_start_in_frame(frame_idx, prev_path, direction)

            if start_pt:
                new_path = self.trace_leftward(frame_idx, start_pt[0], start_pt[1], direction, all_paths)

                if len(new_path) >= 2 and new_path[0][0] > new_path[-1][0]:
                    new_path = new_path[::-1]

                all_paths[frame_idx] = new_path
                prev_path = new_path

                if len(new_path) >= 2:
                    dx = new_path[0][0] - new_path[-1][0]
                    dy = new_path[0][1] - new_path[-1][1]
                    length = sqrt(dx ** 2 + dy ** 2)
                    if length > 0 and dx < 0:
                        direction = (dx / length, dy / length)

                consecutive_gaps = 0

                if frame_idx % 20 == 0 or frame_idx < 5:
                    left_x = new_path[0][0] if new_path else -1
                    print(f"  帧 {frame_idx}: {len(new_path)}点, 左x={left_x}")
            else:
                consecutive_gaps += 1
                all_paths[frame_idx] = prev_path.copy()

                if frame_idx % 20 == 0:
                    print(f"  帧 {frame_idx}: 断裂 ({consecutive_gaps})")

                if consecutive_gaps > 30:
                    print(f"  停止回溯")
                    for f in range(frame_idx - 1, -1, -1):
                        all_paths[f] = prev_path.copy()
                    break

        for f in range(start_frame_idx + 1):
            if f not in all_paths:
                for nearby in range(f + 1, start_frame_idx + 1):
                    if nearby in all_paths:
                        all_paths[f] = all_paths[nearby].copy()
                        break

        leftmost = min(p[0][0] for p in all_paths.values() if p)
        print(f"\n完成: {len(all_paths)}帧, 最左x={leftmost}")

        return all_paths


class TrackerGUI:
    """GUI"""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("神经元追踪 - 完整版（8条规则）")
        self.root.geometry("1400x900")

        self.tracker = NeuronTracker()
        self.current_frame_idx = 0
        self.zoom_level = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.dragging = False
        self.drag_start = None
        self.preview_paths = {}
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

        right = ttk.Frame(self.main_paned, width=320)
        self.main_paned.add(right, weight=0)

        self.ctrl_canvas = tk.Canvas(right, width=300)
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
        self.input_entry = ttk.Entry(f1, width=20)
        self.input_entry.insert(0, self.input_dir)
        self.input_entry.grid(row=0, column=1, sticky="ew")
        ttk.Button(f1, text="...", command=self.browse_input, width=3).grid(row=0, column=2)
        ttk.Label(f1, text="输出:").grid(row=1, column=0, sticky="w", pady=2)
        self.output_entry = ttk.Entry(f1, width=20)
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
        ttk.Label(g, text="线性权重:").grid(row=1, column=0, sticky="w", pady=2)
        self.linear_var = tk.StringVar(value="2.0")
        ttk.Entry(g, textvariable=self.linear_var, width=5).grid(row=1, column=1)
        ttk.Label(g, text="最大转角:").grid(row=1, column=2, sticky="w", padx=(8, 0))
        self.angle_var = tk.StringVar(value="60")
        ttk.Entry(g, textvariable=self.angle_var, width=5).grid(row=1, column=3)
        ttk.Label(g, text="最大半径:").grid(row=2, column=0, sticky="w", pady=2)
        self.max_radius_var = tk.StringVar(value="150")
        ttk.Entry(g, textvariable=self.max_radius_var, width=5).grid(row=2, column=1)
        ttk.Label(g, text="回退帧数:").grid(row=2, column=2, sticky="w", padx=(8, 0))
        self.backtrack_var = tk.StringVar(value="10")
        ttk.Entry(g, textvariable=self.backtrack_var, width=5).grid(row=2, column=3)
        ttk.Button(f4, text="应用参数", command=self.apply_params).pack(pady=3)

        # 规则说明
        f5 = ttk.LabelFrame(p, text="📐 8条规则", padding=5)
        f5.pack(fill="x", pady=pad, padx=3)
        rules = """1. 严格向左 (dx≤0)
2. 必须到达左边界
3. 分叉时回退到更早帧
4. 线性度优先评分
5. 单一路径不分叉
6. 断裂时可跳跃搜索
7. 平滑约束(限制转角)
8. 自适应扩大搜索半径"""
        ttk.Label(f5, text=rules, font=("", 8), justify="left").pack(anchor="w")

        # 操作
        f6 = ttk.LabelFrame(p, text="📍 操作", padding=5)
        f6.pack(fill="x", pady=pad, padx=3)
        ttk.Label(f6, text="左键选点 | 右键取消 | Enter确认").pack(anchor="w")
        self.preview_info = tk.StringVar(value="未选择")
        ttk.Label(f6, textvariable=self.preview_info, foreground="blue").pack(anchor="w", pady=3)
        btn_row = ttk.Frame(f6)
        btn_row.pack(fill="x", pady=3)
        ttk.Button(btn_row, text="✓ 确认", command=self.confirm, width=10).pack(side="left", padx=2)
        ttk.Button(btn_row, text="✗ 取消", command=self.cancel, width=10).pack(side="left", padx=2)

        # 结果
        f7 = ttk.LabelFrame(p, text="📊 结果", padding=5)
        f7.pack(fill="x", pady=pad, padx=3)
        self.result_list = tk.Listbox(f7, height=4, font=("Consolas", 9))
        self.result_list.pack(fill="x")
        btn_row2 = ttk.Frame(f7)
        btn_row2.pack(fill="x", pady=3)
        ttk.Button(btn_row2, text="删除", command=self.delete_result, width=8).pack(side="left", padx=2)
        ttk.Button(btn_row2, text="清空", command=self.clear_results, width=8).pack(side="left", padx=2)

        # 导出
        f8 = ttk.LabelFrame(p, text="💾 导出", padding=5)
        f8.pack(fill="x", pady=pad, padx=3)
        btn_row3 = ttk.Frame(f8)
        btn_row3.pack(fill="x")
        ttk.Button(btn_row3, text="导出视频", command=self.export_video, width=10).pack(side="left", padx=2)
        ttk.Button(btn_row3, text="截图", command=self.screenshot, width=8).pack(side="left", padx=2)

        # 状态
        f9 = ttk.LabelFrame(p, text="📌 状态", padding=5)
        f9.pack(fill="x", pady=pad, padx=3)
        self.status = tk.StringVar(value="请加载帧")
        ttk.Label(f9, textvariable=self.status, wraplength=280, foreground="blue").pack(fill="x")
        self.progress = ttk.Progressbar(f9, mode='determinate')
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
        self.dragging = True; self.drag_start = (e.x, e.y)

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
        self.status.set(f"追踪中... ({ix}, {iy})")
        self.root.update()
        paths = self.tracker.trace_backward_complete(self.current_frame_idx, ix, iy)
        self.preview_paths = paths
        total_pts = sum(len(p) for p in paths.values())
        leftmost = min(p[0][0] for p in paths.values() if p)
        reached = "✓到达左边界" if leftmost <= self.tracker.left_margin else f"未到达(x={leftmost})"
        self.preview_info.set(f"F0-F{self.current_frame_idx}, 左x={leftmost} {reached}")
        self.status.set("预览就绪 | Enter确认")
        self.update_display()

    def confirm(self):
        if not self.preview_paths: return
        tips = {f: p[-1] for f, p in self.preview_paths.items() if p}
        self.tracker.tracking_results.append((self.preview_paths.copy(), tips, self.click_point))
        self.preview_paths = {}
        self.click_point = None
        self.preview_info.set("已确认")
        self.update_result_list()
        self.update_display()
        self.status.set(f"已确认！共 {len(self.tracker.tracking_results)} 条")

    def cancel(self):
        self.preview_paths = {}
        self.click_point = None
        self.preview_info.set("已取消")
        self.update_display()

    def apply_params(self):
        try:
            self.tracker.brightness_threshold = int(self.thresh_var.get())
            self.tracker.search_radius = int(self.radius_var.get())
            self.tracker.linearity_weight = float(self.linear_var.get())
            self.tracker.max_turn_angle = int(self.angle_var.get())
            self.tracker.max_search_radius = int(self.max_radius_var.get())
            self.tracker.max_backtrack_frames = int(self.backtrack_var.get())
        except:
            pass

    def update_result_list(self):
        self.result_list.delete(0, tk.END)
        for i, (paths, _, _) in enumerate(self.tracker.tracking_results):
            leftmost = min(p[0][0] for p in paths.values() if p)
            self.result_list.insert(tk.END, f"#{i + 1}: 左x={leftmost}")

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
        path = os.path.join(self.output_dir, "tracking_complete.mp4")
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
        cv2.line(vis, (self.tracker.left_margin, 0), (self.tracker.left_margin, self.tracker.height), (0, 100, 0), 1)
        for i, (paths, _, _) in enumerate(self.tracker.tracking_results):
            color = self.colors[i % len(self.colors)]
            if idx in paths:
                path = paths[idx]
                for j in range(1, len(path)):
                    cv2.line(vis, path[j - 1], path[j], color, 2)
                if path:
                    cv2.circle(vis, path[0], 5, (0, 255, 0), -1)
                    cv2.circle(vis, path[-1], 5, color, -1)
        if not for_export and self.preview_paths and idx in self.preview_paths:
            path = self.preview_paths[idx]
            for j in range(1, len(path)):
                cv2.line(vis, path[j - 1], path[j], (0, 255, 255), 2)
            if path:
                cv2.circle(vis, path[0], 6, (0, 255, 0), -1)
                cv2.circle(vis, path[-1], 6, (0, 255, 255), -1)
        if not for_export and self.click_point and idx == self.click_point[2]:
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
    print("启动神经元追踪 - 完整版（8条规则）")
    TrackerGUI().run()
