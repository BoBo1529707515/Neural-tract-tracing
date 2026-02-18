import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk

# ╔══════════════════════════════════════════════════════════════╗
# ║  神经元追踪 Part1 · 向前回溯（改进版）                          ║
# ║  - 一直追到最左端                                              ║
# ║  - 不分叉，单路径                                              ║
# ║  - 依赖前帧，找最近亮点继续                                     ║
# ╚══════════════════════════════════════════════════════════════╝

DEFAULT_INPUT_DIR = r"F:\工作文件\RA\python\项目汇总\神经图像\05_二次空间去噪\frames_final"
DEFAULT_OUTPUT_DIR = r"F:\工作文件\RA\python\项目汇总\神经图像\07_生长追踪"


class NeuronTracker:
    """神经元追踪 - 向前回溯（改进版）"""

    def __init__(self):
        self.frames = []
        self.height = 0
        self.width = 0

        # 参数
        self.brightness_threshold = 30
        self.search_radius = 25  # 断裂搜索半径
        self.linearity_weight = 2.0
        self.max_gap_distance = 50  # 最大跳跃距离

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

    def get_brightness(self, frame_idx, x, y):
        if frame_idx < 0 or frame_idx >= len(self.frames):
            return 0
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return 0
        return self.frames[frame_idx][int(y), int(x)]

    def compute_linearity(self, path, new_point):
        """计算线性度"""
        if len(path) < 2:
            return 1.0

        n = min(5, len(path))
        recent = path[-n:]

        dx_sum, dy_sum = 0, 0
        for i in range(1, len(recent)):
            dx_sum += recent[i][0] - recent[i - 1][0]
            dy_sum += recent[i][1] - recent[i - 1][1]

        dir_len = np.sqrt(dx_sum ** 2 + dy_sum ** 2)
        if dir_len == 0:
            return 1.0

        last = path[-1]
        new_dx = new_point[0] - last[0]
        new_dy = new_point[1] - last[1]
        new_len = np.sqrt(new_dx ** 2 + new_dy ** 2)

        if new_len == 0:
            return 1.0

        cos_angle = (dx_sum * new_dx + dy_sum * new_dy) / (dir_len * new_len)
        return (cos_angle + 1) / 2

    def trace_in_frame(self, frame_idx, start_x, start_y, direction=None, max_steps=500):
        """在单帧内追踪线性路径"""
        frame = self.frames[frame_idx]
        path = [(int(start_x), int(start_y))]
        visited = set()
        visited.add((int(start_x), int(start_y)))

        cx, cy = int(start_x), int(start_y)

        for _ in range(max_steps):
            candidates = []

            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    if dx == 0 and dy == 0:
                        continue

                    nx, ny = cx + dx, cy + dy

                    if (nx, ny) in visited:
                        continue
                    if nx < 0 or nx >= self.width or ny < 0 or ny >= self.height:
                        continue

                    brightness = frame[ny, nx]
                    if brightness < self.brightness_threshold:
                        continue

                    dist = np.sqrt(dx ** 2 + dy ** 2)
                    linearity = self.compute_linearity(path, (nx, ny))
                    score = brightness * (linearity ** self.linearity_weight) / (dist + 0.5)

                    if direction is not None and len(path) < 5:
                        dir_x, dir_y = direction
                        dir_len = np.sqrt(dir_x ** 2 + dir_y ** 2)
                        if dir_len > 0:
                            cos_a = (dx * dir_x + dy * dir_y) / (dist * dir_len)
                            score *= (1 + (cos_a + 1) / 2)

                    candidates.append((nx, ny, score, dx, dy))

            if not candidates:
                break

            candidates.sort(key=lambda x: -x[2])
            nx, ny, _, dx, dy = candidates[0]

            path.append((nx, ny))
            visited.add((nx, ny))
            direction = (dx, dy)
            cx, cy = nx, ny

        return path

    def find_nearest_bright_on_trajectory(self, frame_idx, trajectory, direction):
        """
        在前一帧中，沿着轨迹方向找最近的亮点

        trajectory: 上一帧的路径点
        direction: 追踪方向 (dx, dy)

        返回: 最近的亮点坐标，或None
        """
        # 策略1：检查轨迹上的点在当前帧是否亮
        for pt in trajectory:
            if self.is_bright(frame_idx, pt[0], pt[1]):
                return pt

        # 策略2：轨迹上没有亮点，沿方向搜索
        if len(trajectory) > 0:
            # 从轨迹末端沿方向搜索
            last_pt = trajectory[-1]
            dir_x, dir_y = direction
            dir_len = np.sqrt(dir_x ** 2 + dir_y ** 2)

            if dir_len > 0:
                dir_x, dir_y = dir_x / dir_len, dir_y / dir_len

                # 沿方向逐步搜索
                for step in range(1, self.max_gap_distance):
                    cx = int(last_pt[0] + dir_x * step)
                    cy = int(last_pt[1] + dir_y * step)

                    # 在该点周围小范围搜索
                    for dy in range(-3, 4):
                        for dx in range(-3, 4):
                            nx, ny = cx + dx, cy + dy
                            if self.is_bright(frame_idx, nx, ny):
                                return (nx, ny)

        return None

    def check_path_in_frame(self, path, frame_idx):
        """
        检查路径上哪些点在指定帧中仍然亮

        返回:
            last_bright_idx: 最后一个亮点的索引（从头开始数）
            bright_points: 所有亮点列表
        """
        bright_points = []
        last_bright_idx = -1

        for i, pt in enumerate(path):
            if self.is_bright(frame_idx, pt[0], pt[1]):
                bright_points.append((i, pt))
                last_bright_idx = i

        return last_bright_idx, bright_points

    def trace_backward_improved(self, start_frame_idx, start_x, start_y):
        """
        ========== 改进版向前回溯算法 ==========

        核心逻辑：
        1. 在当前帧追踪完整路径（双向到端点）
        2. 向前回溯时：
           - 检查当前路径在前一帧的哪些点仍亮
           - 从最后一个亮点（最"老"的部分）继续向左追踪
           - 如果完全断裂，沿方向跳跃搜索
        3. 一直追到第0帧或完全找不到
        """

        print(f"\n{'=' * 60}")
        print(f"开始向前回溯（改进版）")
        print(f"起点: ({start_x}, {start_y}) @ Frame {start_frame_idx}")
        print(f"{'=' * 60}")

        # ===== Step 1: 在当前帧追踪完整路径 =====

        # 双向追踪
        path_a = self.trace_in_frame(start_frame_idx, start_x, start_y)

        if len(path_a) >= 2:
            rev_dir = (path_a[0][0] - path_a[1][0], path_a[0][1] - path_a[1][1])
        else:
            rev_dir = None

        path_b = self.trace_in_frame(start_frame_idx, start_x, start_y, rev_dir)

        # 合并路径
        path_b_rev = path_b[::-1]
        if len(path_b_rev) > 1:
            current_path = path_b_rev[:-1] + path_a
        else:
            current_path = path_a

        # 去重
        seen = set()
        current_path = [p for p in current_path if not (p in seen or seen.add(p))]

        if not current_path:
            current_path = [(int(start_x), int(start_y))]

        print(f"当前帧完整路径: {len(current_path)} 点")
        print(f"  头部(左): {current_path[0]}")
        print(f"  尾部(右): {current_path[-1]}")

        # 存储结果
        paths = {start_frame_idx: current_path.copy()}

        # ===== Step 2: 向前回溯 =====

        # 追踪方向：从尾部指向头部（向左）
        if len(current_path) >= 2:
            # 用路径前几个点计算方向
            head_dir_x = current_path[0][0] - current_path[min(5, len(current_path) - 1)][0]
            head_dir_y = current_path[0][1] - current_path[min(5, len(current_path) - 1)][1]
            head_dir_len = np.sqrt(head_dir_x ** 2 + head_dir_y ** 2)
            if head_dir_len > 0:
                backward_dir = (head_dir_x / head_dir_len, head_dir_y / head_dir_len)
            else:
                backward_dir = (-1, 0)
        else:
            backward_dir = (-1, 0)

        print(f"\n回溯方向: {backward_dir}")

        prev_path = current_path
        consecutive_gaps = 0

        for frame_idx in range(start_frame_idx - 1, -1, -1):

            # 检查前一帧路径在当前帧的状态
            # 我们要找：路径的哪个部分在这一帧仍然存在

            # 从路径头部（左端）开始检查
            first_bright_pt = None
            first_bright_idx = -1

            for i, pt in enumerate(prev_path):
                if self.is_bright(frame_idx, pt[0], pt[1]):
                    first_bright_pt = pt
                    first_bright_idx = i
                    break

            if first_bright_pt is not None:
                # 找到了亮点，从这里继续向左追踪
                new_path = self.trace_in_frame(frame_idx, first_bright_pt[0], first_bright_pt[1], backward_dir)

                paths[frame_idx] = new_path
                prev_path = new_path

                # 更新方向
                if len(new_path) >= 2:
                    head_dir_x = new_path[-1][0] - new_path[0][0]
                    head_dir_y = new_path[-1][1] - new_path[0][1]
                    head_dir_len = np.sqrt(head_dir_x ** 2 + head_dir_y ** 2)
                    if head_dir_len > 0:
                        backward_dir = (head_dir_x / head_dir_len, head_dir_y / head_dir_len)

                consecutive_gaps = 0

                if frame_idx % 20 == 0 or frame_idx < 5:
                    print(f"  Frame {frame_idx}: 从索引{first_bright_idx}继续, {len(new_path)}点")

            else:
                # 整条路径都不亮了！尝试沿方向跳跃搜索
                jump_pt = self.find_nearest_bright_on_trajectory(frame_idx, prev_path, backward_dir)

                if jump_pt is not None:
                    # 找到了跳跃点
                    new_path = self.trace_in_frame(frame_idx, jump_pt[0], jump_pt[1], backward_dir)

                    paths[frame_idx] = new_path
                    prev_path = new_path

                    if len(new_path) >= 2:
                        head_dir_x = new_path[-1][0] - new_path[0][0]
                        head_dir_y = new_path[-1][1] - new_path[0][1]
                        head_dir_len = np.sqrt(head_dir_x ** 2 + head_dir_y ** 2)
                        if head_dir_len > 0:
                            backward_dir = (head_dir_x / head_dir_len, head_dir_y / head_dir_len)

                    consecutive_gaps = 0
                    print(f"  Frame {frame_idx}: 跳跃到 {jump_pt}, {len(new_path)}点")

                else:
                    # 完全找不到
                    consecutive_gaps += 1
                    paths[frame_idx] = prev_path.copy()

                    if frame_idx % 20 == 0:
                        print(f"  Frame {frame_idx}: 断裂 (连续{consecutive_gaps}帧)")

                    # 如果连续断裂太多，尝试更大范围搜索
                    if consecutive_gaps > 10:
                        # 最后尝试：在更大范围内搜索任何亮点
                        found = False
                        if len(prev_path) > 0:
                            center = prev_path[0]  # 从路径头部搜索
                            for r in range(self.search_radius, self.max_gap_distance, 5):
                                for dy in range(-r, r + 1, 3):
                                    for dx in range(-r, r + 1, 3):
                                        nx, ny = center[0] + dx, center[1] + dy
                                        if self.is_bright(frame_idx, nx, ny):
                                            new_path = self.trace_in_frame(frame_idx, nx, ny, backward_dir)
                                            if len(new_path) > 3:
                                                paths[frame_idx] = new_path
                                                prev_path = new_path
                                                consecutive_gaps = 0
                                                found = True
                                                print(f"  Frame {frame_idx}: 大范围搜索找到 ({nx},{ny})")
                                                break
                                    if found:
                                        break
                                if found:
                                    break

                        if not found and consecutive_gaps > 20:
                            print(f"  连续断裂超过20帧，停止回溯")
                            # 剩余帧复制
                            for f in range(frame_idx - 1, -1, -1):
                                paths[f] = prev_path.copy()
                            break

        print(f"\n回溯完成: 覆盖 {len(paths)} 帧")

        # 确保所有帧都有路径
        for f in range(start_frame_idx + 1):
            if f not in paths:
                # 找最近的有路径的帧
                for nearby in range(f + 1, start_frame_idx + 1):
                    if nearby in paths:
                        paths[f] = paths[nearby].copy()
                        break

        return paths


class TrackerGUI:
    """GUI界面"""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("神经元追踪 Part1 - 向前回溯（改进版）")
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

        self.colors = [(255, 100, 100), (100, 255, 100), (100, 100, 255),
                       (255, 255, 100), (255, 100, 255), (100, 255, 255)]

        self.setup_ui()
        self.bind_events()

    def setup_ui(self):
        self.main_paned = ttk.PanedWindow(self.root, orient="horizontal")
        self.main_paned.pack(fill="both", expand=True, padx=5, pady=5)

        # 左侧图像
        left = ttk.Frame(self.main_paned)
        self.main_paned.add(left, weight=3)
        left.columnconfigure(0, weight=1)
        left.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(left, bg='black')
        self.canvas.grid(row=0, column=0, sticky="nsew")

        # 右侧控制面板
        right = ttk.Frame(self.main_paned, width=300)
        self.main_paned.add(right, weight=0)

        self.ctrl_canvas = tk.Canvas(right, width=280)
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

        self.setup_controls(self.ctrl_frame)

    def setup_controls(self, parent):
        pad = 3

        # 文件
        f1 = ttk.LabelFrame(parent, text="📁 文件", padding=5)
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

        # 帧导航
        f2 = ttk.LabelFrame(parent, text="🎬 帧导航", padding=5)
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
        f3 = ttk.LabelFrame(parent, text="🔍 缩放", padding=5)
        f3.pack(fill="x", pady=pad, padx=3)

        row = ttk.Frame(f3)
        row.pack(fill="x")
        ttk.Button(row, text="−", command=lambda: self.zoom(-0.25), width=3).pack(side="left")
        self.zoom_var = tk.StringVar(value="100%")
        ttk.Label(row, textvariable=self.zoom_var, width=6).pack(side="left", padx=3)
        ttk.Button(row, text="+", command=lambda: self.zoom(0.25), width=3).pack(side="left")
        ttk.Button(row, text="适应", command=self.fit_zoom, width=5).pack(side="left", padx=5)

        # 参数
        f4 = ttk.LabelFrame(parent, text="⚙️ 参数", padding=5)
        f4.pack(fill="x", pady=pad, padx=3)

        g = ttk.Frame(f4)
        g.pack(fill="x")

        ttk.Label(g, text="亮度阈值:").grid(row=0, column=0, sticky="w")
        self.thresh_var = tk.StringVar(value="30")
        ttk.Entry(g, textvariable=self.thresh_var, width=5).grid(row=0, column=1)

        ttk.Label(g, text="搜索半径:").grid(row=0, column=2, sticky="w", padx=(8, 0))
        self.radius_var = tk.StringVar(value="25")
        ttk.Entry(g, textvariable=self.radius_var, width=5).grid(row=0, column=3)

        ttk.Label(g, text="线性权重:").grid(row=1, column=0, sticky="w", pady=2)
        self.linear_var = tk.StringVar(value="2.0")
        ttk.Entry(g, textvariable=self.linear_var, width=5).grid(row=1, column=1)

        ttk.Label(g, text="最大跳跃:").grid(row=1, column=2, sticky="w", padx=(8, 0))
        self.gap_var = tk.StringVar(value="50")
        ttk.Entry(g, textvariable=self.gap_var, width=5).grid(row=1, column=3)

        ttk.Button(f4, text="应用参数", command=self.apply_params).pack(pady=3)

        # 算法说明
        f5 = ttk.LabelFrame(parent, text="📐 改进算法", padding=5)
        f5.pack(fill="x", pady=pad, padx=3)

        algo_text = """向前回溯（改进版）:

1. 在当前帧画出完整轨迹
2. 回溯到前一帧时:
   • 检查轨迹哪些点仍然亮
   • 从第一个亮点继续向左追踪
3. 如果完全断裂:
   • 沿方向跳跃搜索最近亮点
   • 允许跨越空白区域
4. 不分叉，保持单一路径
5. 一直追到第0帧"""
        ttk.Label(f5, text=algo_text, font=("", 8), justify="left").pack(anchor="w")

        # 操作
        f6 = ttk.LabelFrame(parent, text="📍 点击追踪", padding=5)
        f6.pack(fill="x", pady=pad, padx=3)

        ttk.Label(f6, text="左键: 选点预览\n右键/Esc: 取消\nEnter: 确认", font=("", 9)).pack(anchor="w")

        self.preview_info = tk.StringVar(value="未选择")
        ttk.Label(f6, textvariable=self.preview_info, foreground="blue").pack(anchor="w", pady=3)

        btn_row = ttk.Frame(f6)
        btn_row.pack(fill="x", pady=3)
        ttk.Button(btn_row, text="✓ 确认", command=self.confirm, width=10).pack(side="left", padx=2)
        ttk.Button(btn_row, text="✗ 取消", command=self.cancel, width=10).pack(side="left", padx=2)

        # 结果
        f7 = ttk.LabelFrame(parent, text="📊 追踪结果", padding=5)
        f7.pack(fill="x", pady=pad, padx=3)

        self.result_list = tk.Listbox(f7, height=4, font=("Consolas", 9))
        self.result_list.pack(fill="x")

        btn_row2 = ttk.Frame(f7)
        btn_row2.pack(fill="x", pady=3)
        ttk.Button(btn_row2, text="删除", command=self.delete_result, width=8).pack(side="left", padx=2)
        ttk.Button(btn_row2, text="清空", command=self.clear_results, width=8).pack(side="left", padx=2)

        # 导出
        f8 = ttk.LabelFrame(parent, text="💾 导出", padding=5)
        f8.pack(fill="x", pady=pad, padx=3)

        btn_row3 = ttk.Frame(f8)
        btn_row3.pack(fill="x")
        ttk.Button(btn_row3, text="导出视频", command=self.export_video, width=10).pack(side="left", padx=2)
        ttk.Button(btn_row3, text="截图", command=self.screenshot, width=8).pack(side="left", padx=2)

        # 状态
        f9 = ttk.LabelFrame(parent, text="📌 状态", padding=5)
        f9.pack(fill="x", pady=pad, padx=3)

        self.status = tk.StringVar(value="请加载帧")
        ttk.Label(f9, textvariable=self.status, wraplength=260, foreground="blue").pack(fill="x")

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

        if self.tracker.load_frames(self.input_dir, lambda p: (
                setattr(self.progress, 'value', p * 100), self.root.update())):
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
        if not self.tracker.frames:
            return
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
        if not self.tracker.frames:
            return
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        if cw < 10:
            cw, ch = 800, 600

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
        if not self.tracker.frames:
            return

        ix, iy = self.canvas_to_image(e.x, e.y)
        if not (0 <= ix < self.tracker.width and 0 <= iy < self.tracker.height):
            return

        self.apply_params()
        self.click_point = (ix, iy, self.current_frame_idx)
        self.status.set(f"追踪中... ({ix}, {iy})")
        self.root.update()

        # 执行改进版回溯
        paths = self.tracker.trace_backward_improved(self.current_frame_idx, ix, iy)
        self.preview_paths = paths

        # 统计
        total_pts = sum(len(p) for p in paths.values())
        f_min, f_max = min(paths.keys()), max(paths.keys())

        self.preview_info.set(f"F{f_min}→F{f_max}, {total_pts}点")
        self.status.set("预览就绪 | Enter确认 | Esc取消")
        self.update_display()

    def confirm(self):
        if not self.preview_paths:
            return

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
            self.tracker.max_gap_distance = int(self.gap_var.get())
        except:
            pass

    def update_result_list(self):
        self.result_list.delete(0, tk.END)
        for i, (paths, _, click) in enumerate(self.tracker.tracking_results):
            f_min, f_max = min(paths.keys()), max(paths.keys())
            pts = sum(len(p) for p in paths.values())
            self.result_list.insert(tk.END, f"#{i + 1}: F{f_min}→F{f_max}, {pts}pts")

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
            messagebox.showwarning("警告", "无追踪结果")
            return

        self.output_dir = self.output_entry.get()
        os.makedirs(self.output_dir, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        path = os.path.join(self.output_dir, "tracking_backward.mp4")
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
        if not self.tracker.frames:
            return
        self.output_dir = self.output_entry.get()
        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir, f"frame_{self.current_frame_idx:04d}.png")
        cv2.imwrite(path, self.render(self.current_frame_idx, True))
        self.status.set(f"已保存: {path}")

    def render(self, idx, for_export=False):
        frame = self.tracker.frames[idx]
        vis = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        # 已确认的结果
        for i, (paths, _, _) in enumerate(self.tracker.tracking_results):
            color = self.colors[i % len(self.colors)]
            if idx in paths:
                path = paths[idx]
                for j in range(1, len(path)):
                    cv2.line(vis, path[j - 1], path[j], color, 2)
                if path:
                    cv2.circle(vis, path[0], 5, (0, 255, 0), -1)  # 头部绿色
                    cv2.circle(vis, path[-1], 5, color, -1)  # 尾部

        # 预览
        if not for_export and self.preview_paths and idx in self.preview_paths:
            path = self.preview_paths[idx]
            for j in range(1, len(path)):
                cv2.line(vis, path[j - 1], path[j], (0, 255, 255), 2)
            if path:
                cv2.circle(vis, path[0], 6, (0, 255, 0), -1)  # 头
                cv2.circle(vis, path[-1], 6, (0, 255, 255), -1)  # 尾

        # 点击点
        if not for_export and self.click_point and idx == self.click_point[2]:
            cv2.circle(vis, (self.click_point[0], self.click_point[1]), 10, (0, 0, 255), 2)

        if for_export:
            cv2.putText(vis, f"Frame {idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        return vis

    def update_display(self):
        if not self.tracker.frames:
            return

        vis = cv2.cvtColor(self.render(self.current_frame_idx), cv2.COLOR_BGR2RGB)

        nw, nh = int(self.tracker.width * self.zoom_level), int(self.tracker.height * self.zoom_level)
        if nw < 1 or nh < 1:
            return

        vis_scaled = cv2.resize(vis, (nw, nh))

        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        if cw < 10:
            return

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
    print("启动神经元追踪 Part1 - 向前回溯（改进版）")
    TrackerGUI().run()
