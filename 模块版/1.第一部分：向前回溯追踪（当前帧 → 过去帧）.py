import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk

# ╔══════════════════════════════════════════════════════════════╗
# ║  神经元追踪 Part1 · 选点后向前回溯绘制路径                       ║
# ╚══════════════════════════════════════════════════════════════╝

DEFAULT_INPUT_DIR = r"F:\工作文件\RA\python\项目汇总\神经图像\05_二次空间去噪\frames_final"
DEFAULT_OUTPUT_DIR = r"F:\工作文件\RA\python\项目汇总\神经图像\07_生长追踪"


class NeuronTracker:
    """神经元追踪 - Part1: 向前回溯"""

    def __init__(self):
        self.frames = []
        self.height = 0
        self.width = 0

        # 参数
        self.brightness_threshold = 30  # 亮度阈值
        self.search_radius = 20  # 断裂时搜索半径
        self.linearity_weight = 2.0  # 线性度权重
        self.gap_tolerance = 5  # 断裂容忍帧数

        self.tracking_results = []

    def load_frames(self, input_dir, progress_callback=None):
        """加载帧"""
        self.frames = []
        frame_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.png')])

        if not frame_files:
            return False

        # 获取尺寸
        sample_path = os.path.join(input_dir, frame_files[0])
        with open(sample_path, 'rb') as f:
            img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
        self.height, self.width = img.shape

        # 加载所有帧
        for i, filename in enumerate(frame_files):
            path = os.path.join(input_dir, filename)
            with open(path, 'rb') as f:
                img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
            self.frames.append(img)

            if progress_callback and i % 10 == 0:
                progress_callback(i / len(frame_files))

        if progress_callback:
            progress_callback(1.0)

        return True

    def is_bright(self, frame_idx, x, y):
        """检查点是否足够亮"""
        if frame_idx < 0 or frame_idx >= len(self.frames):
            return False
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return False
        return self.frames[frame_idx][int(y), int(x)] > self.brightness_threshold

    def get_brightness(self, frame_idx, x, y):
        """获取亮度值"""
        if frame_idx < 0 or frame_idx >= len(self.frames):
            return 0
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return 0
        return self.frames[frame_idx][int(y), int(x)]

    def compute_linearity(self, path, new_point):
        """
        计算加入新点后的线性度（方向一致性）
        返回 0~1，1表示完全一致
        """
        if len(path) < 2:
            return 1.0

        # 取最近几个点计算平均方向
        n = min(5, len(path))
        recent = path[-n:]

        dx_sum, dy_sum = 0, 0
        for i in range(1, len(recent)):
            dx_sum += recent[i][0] - recent[i - 1][0]
            dy_sum += recent[i][1] - recent[i - 1][1]

        dir_len = np.sqrt(dx_sum ** 2 + dy_sum ** 2)
        if dir_len == 0:
            return 1.0

        # 新点方向
        last = path[-1]
        new_dx = new_point[0] - last[0]
        new_dy = new_point[1] - last[1]
        new_len = np.sqrt(new_dx ** 2 + new_dy ** 2)

        if new_len == 0:
            return 1.0

        # cos夹角
        cos_angle = (dx_sum * new_dx + dy_sum * new_dy) / (dir_len * new_len)

        # 归一化到 0~1
        return (cos_angle + 1) / 2

    def trace_in_frame(self, frame_idx, start_x, start_y, direction=None, max_steps=500):
        """
        在单帧内从起点追踪线性路径

        direction: 初始方向 (dx, dy)，None则自动探索
        返回: 路径点列表 [(x,y), ...]
        """
        frame = self.frames[frame_idx]
        path = [(start_x, start_y)]
        visited = set()
        visited.add((start_x, start_y))

        cx, cy = start_x, start_y

        for _ in range(max_steps):
            candidates = []

            # 搜索3x3邻域
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

                    # 评分 = 亮度 × 线性度^权重 / 距离
                    score = brightness * (linearity ** self.linearity_weight) / (dist + 0.5)

                    # 初始方向加成
                    if direction is not None and len(path) < 5:
                        dir_x, dir_y = direction
                        dir_len = np.sqrt(dir_x ** 2 + dir_y ** 2)
                        if dir_len > 0:
                            cos_a = (dx * dir_x + dy * dir_y) / (dist * dir_len)
                            score *= (1 + (cos_a + 1) / 2)

                    candidates.append((nx, ny, score, dx, dy))

            if not candidates:
                break

            # 选最优
            candidates.sort(key=lambda x: -x[2])
            nx, ny, _, dx, dy = candidates[0]

            path.append((nx, ny))
            visited.add((nx, ny))
            direction = (dx, dy)
            cx, cy = nx, ny

        return path

    def get_path_direction(self, path, from_start=False):
        """
        获取路径方向

        from_start=False: 从末端看的前进方向
        from_start=True: 从起点看的前进方向
        """
        if len(path) < 2:
            return (0, 0)

        n = min(5, len(path))

        if from_start:
            segment = path[:n]
        else:
            segment = path[-n:]

        dx_sum, dy_sum = 0, 0
        for i in range(1, len(segment)):
            dx_sum += segment[i][0] - segment[i - 1][0]
            dy_sum += segment[i][1] - segment[i - 1][1]

        length = np.sqrt(dx_sum ** 2 + dy_sum ** 2)
        if length == 0:
            return (0, 0)

        return (dx_sum / length, dy_sum / length)

    def search_semicircle(self, frame_idx, center, direction, side='left'):
        """
        在半圆范围内搜索最佳亮点

        side='left': 向前回溯时，搜索左半圆
        side='right': 向后追踪时，搜索右半圆

        返回: (x, y) 或 None
        """
        cx, cy = center
        dir_x, dir_y = direction

        # 归一化
        dir_len = np.sqrt(dir_x ** 2 + dir_y ** 2)
        if dir_len == 0:
            return None
        dir_x, dir_y = dir_x / dir_len, dir_y / dir_len

        # 垂直方向（用于判断左右）
        if side == 'left':
            perp_x, perp_y = dir_y, -dir_x  # 逆时针90度
        else:
            perp_x, perp_y = -dir_y, dir_x  # 顺时针90度

        frame = self.frames[frame_idx]
        candidates = []

        for dy in range(-self.search_radius, self.search_radius + 1):
            for dx in range(-self.search_radius, self.search_radius + 1):
                if dx == 0 and dy == 0:
                    continue

                nx, ny = int(cx + dx), int(cy + dy)

                if nx < 0 or nx >= self.width or ny < 0 or ny >= self.height:
                    continue

                dist = np.sqrt(dx ** 2 + dy ** 2)
                if dist > self.search_radius:
                    continue

                brightness = frame[ny, nx]
                if brightness < self.brightness_threshold:
                    continue

                # 判断是否在半圆内
                pt_dir_x, pt_dir_y = dx / dist, dy / dist

                # 与前进方向的点积（前方还是后方）
                forward_dot = pt_dir_x * dir_x + pt_dir_y * dir_y
                # 与垂直方向的点积（左还是右）
                side_dot = pt_dir_x * perp_x + pt_dir_y * perp_y

                # 半圆条件：大致在前方 + 在指定侧
                if forward_dot < -0.2:  # 不能太靠后
                    continue
                if side_dot < -0.2:  # 必须在指定侧
                    continue

                # 评分：亮度 × 前向程度 / 距离
                score = brightness * (forward_dot + 1) / (dist + 1)
                candidates.append((nx, ny, score))

        if not candidates:
            return None

        candidates.sort(key=lambda x: -x[2])
        return (candidates[0][0], candidates[0][1])

    def trace_backward(self, start_frame_idx, start_x, start_y):
        """
        ========== 核心算法：向前回溯 ==========

        1. 在当前帧追踪完整路径（双向到端点）
        2. 从当前帧向第0帧回溯
           - 如果路径起点在前一帧仍亮 → 继续追踪
           - 如果不亮 → 在左半圆搜索新起点

        返回: paths = {frame_idx: [(x,y), ...], ...}
        """

        print(f"\n=== 开始向前回溯 ===")
        print(f"起点: ({start_x}, {start_y}) @ Frame {start_frame_idx}")

        # ===== Step 1: 在当前帧追踪完整路径 =====

        # 先向一个方向追踪
        path_forward = self.trace_in_frame(start_frame_idx, start_x, start_y)
        print(f"正向追踪: {len(path_forward)} 点")

        # 反向追踪
        if len(path_forward) >= 2:
            reverse_dir = (path_forward[0][0] - path_forward[1][0],
                           path_forward[0][1] - path_forward[1][1])
        else:
            reverse_dir = None

        path_backward = self.trace_in_frame(start_frame_idx, start_x, start_y, reverse_dir)
        print(f"反向追踪: {len(path_backward)} 点")

        # 合并：反向路径翻转 + 正向路径（去除重复起点）
        path_backward_rev = path_backward[::-1]
        if len(path_backward_rev) > 1:
            current_path = path_backward_rev[:-1] + path_forward
        else:
            current_path = path_forward

        # 去重
        seen = set()
        current_path = [p for p in current_path if not (p in seen or seen.add(p))]

        print(f"当前帧完整路径: {len(current_path)} 点")

        if not current_path:
            current_path = [(start_x, start_y)]

        # 存储结果
        paths = {start_frame_idx: current_path.copy()}

        # ===== Step 2: 向前回溯到第0帧 =====

        # 获取起点（路径的"头部"，用于回溯）
        origin = current_path[0]

        # 获取回溯方向（从起点向外看的方向，即正向方向的反向）
        forward_dir = self.get_path_direction(current_path, from_start=True)
        back_dir = (-forward_dir[0], -forward_dir[1])

        print(f"\n回溯起点: {origin}")
        print(f"回溯方向: {back_dir}")

        gap_count = 0

        for frame_idx in range(start_frame_idx - 1, -1, -1):

            # 检查起点在前一帧是否仍亮
            if self.is_bright(frame_idx, origin[0], origin[1]):
                # 亮！从这里继续追踪
                new_path = self.trace_in_frame(frame_idx, origin[0], origin[1], back_dir)
                paths[frame_idx] = new_path

                # 更新起点和方向
                if len(new_path) > 0:
                    origin = new_path[-1]  # 新路径的末端成为下一次的起点
                    back_dir = self.get_path_direction(new_path, from_start=False)

                gap_count = 0

                if frame_idx % 20 == 0:
                    print(f"  Frame {frame_idx}: 连续, {len(new_path)} 点")

            else:
                # 不亮！在左半圆搜索
                new_start = self.search_semicircle(frame_idx, origin, back_dir, side='left')

                if new_start:
                    # 找到了
                    new_path = self.trace_in_frame(frame_idx, new_start[0], new_start[1], back_dir)
                    paths[frame_idx] = new_path

                    if len(new_path) > 0:
                        origin = new_path[-1]
                        back_dir = self.get_path_direction(new_path, from_start=False)

                    gap_count = 0
                    print(f"  Frame {frame_idx}: 左半圆找到 {new_start}, {len(new_path)} 点")

                else:
                    # 找不到
                    gap_count += 1
                    # 复制上一帧的路径
                    paths[frame_idx] = paths.get(frame_idx + 1, current_path).copy()

                    if frame_idx % 20 == 0:
                        print(f"  Frame {frame_idx}: 断裂 (gap={gap_count})")

                    if gap_count > self.gap_tolerance:
                        print(f"  断裂超过容忍度，停止回溯")
                        # 剩余帧复制当前路径
                        for f in range(frame_idx - 1, -1, -1):
                            paths[f] = paths[frame_idx].copy()
                        break

        print(f"\n回溯完成: {len(paths)} 帧")

        return paths


class TrackerGUI:
    """GUI界面"""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("神经元追踪 Part1 - 向前回溯")
        self.root.geometry("1400x900")

        self.tracker = NeuronTracker()

        self.current_frame_idx = 0
        self.zoom_level = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.dragging = False
        self.drag_start = None

        self.preview_paths = {}

        self.input_dir = DEFAULT_INPUT_DIR
        self.output_dir = DEFAULT_OUTPUT_DIR

        self.colors = [(255, 100, 100), (100, 255, 100), (100, 100, 255),
                       (255, 255, 100), (255, 100, 255), (100, 255, 255)]

        self.setup_ui()
        self.bind_events()

    def setup_ui(self):
        # 主布局
        self.main_paned = ttk.PanedWindow(self.root, orient="horizontal")
        self.main_paned.pack(fill="both", expand=True, padx=5, pady=5)

        # 左侧：图像
        left = ttk.Frame(self.main_paned)
        self.main_paned.add(left, weight=3)
        left.columnconfigure(0, weight=1)
        left.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(left, bg='black')
        self.canvas.grid(row=0, column=0, sticky="nsew")

        # 右侧：控制面板（带滚动）
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

        # 鼠标滚轮滚动控制面板
        self.ctrl_canvas.bind("<Enter>", lambda e: self.ctrl_canvas.bind_all("<MouseWheel>",
                                                                             lambda ev: self.ctrl_canvas.yview_scroll(
                                                                                 int(-1 * (ev.delta / 120)), "units")))
        self.ctrl_canvas.bind("<Leave>", lambda e: self.ctrl_canvas.unbind_all("<MouseWheel>"))

        self.setup_controls(self.ctrl_frame)

    def setup_controls(self, parent):
        pad = 3

        # === 文件 ===
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

        # === 帧导航 ===
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

        # === 缩放 ===
        f3 = ttk.LabelFrame(parent, text="🔍 缩放", padding=5)
        f3.pack(fill="x", pady=pad, padx=3)

        row = ttk.Frame(f3)
        row.pack(fill="x")
        ttk.Button(row, text="−", command=lambda: self.zoom(-0.25), width=3).pack(side="left")
        self.zoom_var = tk.StringVar(value="100%")
        ttk.Label(row, textvariable=self.zoom_var, width=6).pack(side="left", padx=3)
        ttk.Button(row, text="+", command=lambda: self.zoom(0.25), width=3).pack(side="left")
        ttk.Button(row, text="适应", command=self.fit_zoom, width=5).pack(side="left", padx=5)

        ttk.Label(f3, text="滚轮缩放 | 中键拖动", font=("", 8), foreground="gray").pack()

        # === 参数 ===
        f4 = ttk.LabelFrame(parent, text="⚙️ 追踪参数", padding=5)
        f4.pack(fill="x", pady=pad, padx=3)

        g = ttk.Frame(f4)
        g.pack(fill="x")

        ttk.Label(g, text="亮度阈值:").grid(row=0, column=0, sticky="w")
        self.thresh_var = tk.StringVar(value="30")
        ttk.Entry(g, textvariable=self.thresh_var, width=5).grid(row=0, column=1)

        ttk.Label(g, text="搜索半径:").grid(row=0, column=2, sticky="w", padx=(8, 0))
        self.radius_var = tk.StringVar(value="20")
        ttk.Entry(g, textvariable=self.radius_var, width=5).grid(row=0, column=3)

        ttk.Label(g, text="线性权重:").grid(row=1, column=0, sticky="w", pady=2)
        self.linear_var = tk.StringVar(value="2.0")
        ttk.Entry(g, textvariable=self.linear_var, width=5).grid(row=1, column=1)

        ttk.Label(g, text="断裂容忍:").grid(row=1, column=2, sticky="w", padx=(8, 0))
        self.gap_var = tk.StringVar(value="5")
        ttk.Entry(g, textvariable=self.gap_var, width=5).grid(row=1, column=3)

        ttk.Button(f4, text="应用参数", command=self.apply_params).pack(pady=3)

        # === 算法说明 ===
        f5 = ttk.LabelFrame(parent, text="📐 Part1: 向前回溯算法", padding=5)
        f5.pack(fill="x", pady=pad, padx=3)

        algo_text = """1. 点击选定神经元上的一点
2. 在当前帧追踪出完整线性路径
3. 从当前帧向第0帧回溯：
   • 起点仍亮 → 继续追踪
   • 起点不亮 → 左半圆搜索

左半圆 = 回溯方向的左侧180°范围"""
        ttk.Label(f5, text=algo_text, font=("", 8), justify="left").pack(anchor="w")

        # === 操作 ===
        f6 = ttk.LabelFrame(parent, text="📍 点击追踪", padding=5)
        f6.pack(fill="x", pady=pad, padx=3)

        ttk.Label(f6, text="左键: 选点并预览\n右键/Esc: 取消\nEnter: 确认", font=("", 9)).pack(anchor="w")

        self.preview_info = tk.StringVar(value="未选择")
        ttk.Label(f6, textvariable=self.preview_info, foreground="blue").pack(anchor="w", pady=3)

        btn_row = ttk.Frame(f6)
        btn_row.pack(fill="x", pady=3)
        ttk.Button(btn_row, text="✓ 确认", command=self.confirm, width=10).pack(side="left", padx=2)
        ttk.Button(btn_row, text="✗ 取消", command=self.cancel, width=10).pack(side="left", padx=2)

        # === 结果 ===
        f7 = ttk.LabelFrame(parent, text="📊 追踪结果", padding=5)
        f7.pack(fill="x", pady=pad, padx=3)

        self.result_list = tk.Listbox(f7, height=4, font=("Consolas", 9))
        self.result_list.pack(fill="x")

        btn_row2 = ttk.Frame(f7)
        btn_row2.pack(fill="x", pady=3)
        ttk.Button(btn_row2, text="删除", command=self.delete_result, width=8).pack(side="left", padx=2)
        ttk.Button(btn_row2, text="清空", command=self.clear_results, width=8).pack(side="left", padx=2)

        # === 导出 ===
        f8 = ttk.LabelFrame(parent, text="💾 导出", padding=5)
        f8.pack(fill="x", pady=pad, padx=3)

        btn_row3 = ttk.Frame(f8)
        btn_row3.pack(fill="x")
        ttk.Button(btn_row3, text="导出视频", command=self.export_video, width=10).pack(side="left", padx=2)
        ttk.Button(btn_row3, text="截图", command=self.screenshot, width=8).pack(side="left", padx=2)

        # === 状态 ===
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

    # === 文件 ===
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
            self.status.set(f"已加载 {n} 帧 ({self.tracker.width}×{self.tracker.height})")
        else:
            messagebox.showerror("错误", "未找到PNG")

        self.progress['value'] = 0

    # === 导航 ===
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

    # === 缩放 ===
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

    # === 点击追踪 ===
    def canvas_to_image(self, cx, cy):
        return int((cx - self.pan_x) / self.zoom_level), int((cy - self.pan_y) / self.zoom_level)

    def on_click(self, e):
        if not self.tracker.frames:
            return

        ix, iy = self.canvas_to_image(e.x, e.y)
        if not (0 <= ix < self.tracker.width and 0 <= iy < self.tracker.height):
            return

        self.apply_params()
        self.status.set(f"追踪中... ({ix}, {iy})")
        self.root.update()

        # 执行向前回溯
        paths = self.tracker.trace_backward(self.current_frame_idx, ix, iy)
        self.preview_paths = paths

        # 统计
        total_pts = sum(len(p) for p in paths.values())
        f_min, f_max = min(paths.keys()), max(paths.keys())

        self.preview_info.set(f"F{f_min}→F{f_max}, {total_pts}点")
        self.status.set("预览就绪，Enter确认/Esc取消")
        self.update_display()

    def confirm(self):
        if not self.preview_paths:
            return

        tips = {f: p[-1] for f, p in self.preview_paths.items() if p}
        self.tracker.tracking_results.append((self.preview_paths.copy(), tips))

        self.preview_paths = {}
        self.preview_info.set("已确认")
        self.update_result_list()
        self.update_display()
        self.status.set(f"已确认！共 {len(self.tracker.tracking_results)} 条")

    def cancel(self):
        self.preview_paths = {}
        self.preview_info.set("已取消")
        self.update_display()

    def apply_params(self):
        try:
            self.tracker.brightness_threshold = int(self.thresh_var.get())
            self.tracker.search_radius = int(self.radius_var.get())
            self.tracker.linearity_weight = float(self.linear_var.get())
            self.tracker.gap_tolerance = int(self.gap_var.get())
        except:
            pass

    def update_result_list(self):
        self.result_list.delete(0, tk.END)
        for i, (paths, _) in enumerate(self.tracker.tracking_results):
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

    # === 导出 ===
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

    # === 渲染 ===
    def render(self, idx, for_export=False):
        frame = self.tracker.frames[idx]
        vis = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        # 已确认的
        for i, (paths, _) in enumerate(self.tracker.tracking_results):
            color = self.colors[i % len(self.colors)]
            if idx in paths:
                path = paths[idx]
                for j in range(1, len(path)):
                    cv2.line(vis, path[j - 1], path[j], color, 2)
                if path:
                    cv2.circle(vis, path[0], 5, (0, 255, 0), -1)  # 起点绿
                    cv2.circle(vis, path[-1], 5, color, -1)  # 末端

        # 预览（黄色）
        if not for_export and self.preview_paths and idx in self.preview_paths:
            path = self.preview_paths[idx]
            for j in range(1, len(path)):
                cv2.line(vis, path[j - 1], path[j], (0, 255, 255), 2)
            if path:
                cv2.circle(vis, path[0], 6, (0, 255, 0), -1)
                cv2.circle(vis, path[-1], 6, (0, 255, 255), -1)

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
    print("启动神经元追踪 Part1 - 向前回溯")
    TrackerGUI().run()
