import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import time

from colors import NEURON_COLORS
from config import DEFAULT_INPUT_DIR, DEFAULT_OUTPUT_DIR
from tracker import NeuronTracker


class TrackerGUI:
    """GUI"""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("神经元追踪 - 平滑约束优化版")
        self.root.geometry("1550x980")

        self.tracker = NeuronTracker()
        self.current_frame_idx = 0
        self.zoom_level = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.dragging = False
        self.drag_start = None
        self._pan_updating = False

        self.current_neuron_id = 1
        self.preview_result = None
        self.preview_neuron_id = None
        self.preview_backup_result = None
        self.preview_backup_neuron_id = None
        self.preview_is_edit_mode = False
        self.first_marker_by_neuron = {}

        self.input_dir = DEFAULT_INPUT_DIR
        self.output_dir = DEFAULT_OUTPUT_DIR

        # 自动重算：当某个神经元已经计算过后，继续加点/删点会自动触发重算预览
        self.auto_recompute_enabled = True
        self.auto_recompute_delay_ms = 250  # 防抖：多次快速点击只重算一次
        self._auto_recompute_job = None

        # 速度导出参数
        self.fps_var = tk.StringVar(value="10")
        self.pixel_um_var = tk.StringVar(value="1.0")

        # ── 撤销栈 ────────────────────────────────────────────────────────
        # 每条记录: {'action': 'add'|'remove', 'neuron_id': int,
        #            'frame_idx': int, 'point': (x, y)}
        self.marker_undo_stack = []

        self.setup_styles()
        self.setup_ui()
        self.bind_events()

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')

        bg_color = '#F5F7FA'
        fg_color = '#2D3436'
        accent_color = '#74B9FF'
        keyword_color = '#2C3E50'
        accent_hover = '#0984E3'
        button_bg = '#FFFFFF'
        button_fg = '#2D3436'
        entry_bg = '#FFFFFF'
        entry_fg = '#2D3436'
        select_bg = '#74B9FF'
        border_color = '#DFE6E9'

        self.root.configure(bg=bg_color)

        style.configure('.', background=bg_color, foreground=fg_color,
                        fieldbackground=entry_bg, font=('Segoe UI', 10))
        style.configure('TFrame', background=bg_color)
        style.configure('TLabel', background=bg_color, foreground=fg_color)

        style.configure('TButton',
                        background=button_bg, foreground=button_fg,
                        borderwidth=1, bordercolor=border_color,
                        focuscolor=select_bg, relief='flat', padding=(10, 6))
        style.map('TButton',
                  background=[('active', '#F0F3F5'), ('pressed', '#E2E6EA')],
                  foreground=[('active', accent_hover)],
                  bordercolor=[('active', accent_color)])

        style.configure('Accent.TButton',
                        background=accent_color, foreground='white',
                        borderwidth=0, relief='flat', padding=(10, 6))
        style.map('Accent.TButton',
                  background=[('active', accent_hover), ('pressed', '#0062CC')])

        style.configure('TLabelframe',
                        background=bg_color, foreground=fg_color,
                        bordercolor=border_color, borderwidth=1, relief='solid')
        style.configure('TLabelframe.Label',
                        background=bg_color, foreground=keyword_color,
                        font=('Segoe UI', 11, 'bold'))

        style.configure('TEntry',
                        fieldbackground=entry_bg, foreground=entry_fg,
                        insertcolor='black', borderwidth=1, relief='solid',
                        bordercolor=border_color, padding=5)

        style.configure('TPanedwindow', background=bg_color)
        style.configure('Horizontal.TScale', background=bg_color,
                        troughcolor=border_color, sliderthickness=15)
        style.configure('Vertical.TScale', background=bg_color,
                        troughcolor=border_color, sliderthickness=15)
        style.configure('Horizontal.TProgressbar', background=accent_color,
                        troughcolor=border_color, borderwidth=0)

        self.root.option_add('*Listbox.Background', entry_bg)
        self.root.option_add('*Listbox.Foreground', entry_fg)
        self.root.option_add('*Listbox.selectBackground', select_bg)
        self.root.option_add('*Listbox.selectForeground', 'white')
        self.root.option_add('*Listbox.relief', 'flat')
        self.root.option_add('*Listbox.font', ('Segoe UI', 10))

        style.configure("Vertical.TScrollbar",
                        background='#B2BEC3', troughcolor=bg_color,
                        arrowcolor='#2D3436', relief="flat", borderwidth=0)

    def get_neuron_color(self, neuron_id):
        return NEURON_COLORS[(neuron_id - 1) % len(NEURON_COLORS)]

    def setup_ui(self):
        self.main_paned = ttk.PanedWindow(self.root, orient="horizontal")
        self.main_paned.pack(fill="both", expand=True, padx=5, pady=5)

        left = ttk.Frame(self.main_paned)
        self.main_paned.add(left, weight=3)
        left.columnconfigure(0, weight=1)
        left.columnconfigure(1, weight=0)
        left.rowconfigure(0, weight=1)
        left.rowconfigure(1, weight=0)
        self.canvas = tk.Canvas(left, bg='black', highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.v_pan_scale = ttk.Scale(left, orient="vertical", command=self.on_pan_y_slider)
        self.v_pan_scale.grid(row=0, column=1, sticky="ns")
        self.h_pan_scale = ttk.Scale(left, orient="horizontal", command=self.on_pan_x_slider)
        self.h_pan_scale.grid(row=1, column=0, sticky="ew")

        right = ttk.Frame(self.main_paned, width=420)
        self.main_paned.add(right, weight=0)

        self.ctrl_canvas = tk.Canvas(right, width=400, bg='#F5F7FA', highlightthickness=0)
        scrollbar = ttk.Scrollbar(right, orient="vertical", command=self.ctrl_canvas.yview)
        self.ctrl_frame = ttk.Frame(self.ctrl_canvas)
        self.ctrl_canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.ctrl_canvas.pack(side="left", fill="both", expand=True)
        self.ctrl_window = self.ctrl_canvas.create_window((0, 0), window=self.ctrl_frame, anchor="nw")
        self.ctrl_frame.bind("<Configure>",
                             lambda e: self.ctrl_canvas.configure(
                                 scrollregion=self.ctrl_canvas.bbox("all")))
        self.ctrl_canvas.bind("<Configure>",
                              lambda e: self.ctrl_canvas.itemconfig(
                                  self.ctrl_window, width=e.width))
        self.ctrl_canvas.bind("<MouseWheel>", self.on_ctrl_canvas_scroll)

        self.setup_controls()

    def setup_controls(self):
        p = self.ctrl_frame
        pad = 5

        # ── 文件 ──────────────────────────────────────────────────────────
        f1 = ttk.LabelFrame(p, text="📁 文件", padding=10)
        f1.pack(fill="x", pady=pad, padx=5)
        ttk.Label(f1, text="输入:").grid(row=0, column=0, sticky="w")
        self.input_entry = ttk.Entry(f1, width=28)
        self.input_entry.insert(0, self.input_dir)
        self.input_entry.grid(row=0, column=1, sticky="ew", padx=5)
        ttk.Button(f1, text="...", command=self.browse_input, width=3).grid(row=0, column=2)
        ttk.Label(f1, text="输出:").grid(row=1, column=0, sticky="w", pady=5)
        self.output_entry = ttk.Entry(f1, width=28)
        self.output_entry.insert(0, self.output_dir)
        self.output_entry.grid(row=1, column=1, sticky="ew", padx=5)
        ttk.Button(f1, text="...", command=self.browse_output, width=3).grid(row=1, column=2)
        ttk.Button(f1, text="🔄 加载帧", command=self.load_frames,
                   style='Accent.TButton').grid(row=2, column=0, columnspan=3, pady=10, sticky="ew")
        f1.columnconfigure(1, weight=1)

        # ── 帧导航 ────────────────────────────────────────────────────────
        f2 = ttk.LabelFrame(p, text="🎬 帧导航", padding=10)
        f2.pack(fill="x", pady=pad, padx=5)
        row1 = ttk.Frame(f2)
        row1.pack(fill="x")
        ttk.Label(row1, text="帧:").pack(side="left")
        self.frame_var = tk.StringVar(value="0")
        ttk.Entry(row1, textvariable=self.frame_var, width=6).pack(side="left", padx=5)
        self.frame_label = ttk.Label(row1, text="/ 0")
        self.frame_label.pack(side="left")
        ttk.Button(row1, text="Go", command=self.goto_frame, width=4,
                   style='Accent.TButton').pack(side="left", padx=10)

        row2 = ttk.Frame(f2)
        row2.pack(fill="x", pady=5)
        ttk.Button(row2, text="⏮", command=lambda: self.change_frame(-10), width=4).pack(side="left", padx=2)
        ttk.Button(row2, text="◀", command=lambda: self.change_frame(-1), width=4).pack(side="left", padx=2)
        ttk.Button(row2, text="▶", command=lambda: self.change_frame(1), width=4).pack(side="left", padx=2)
        ttk.Button(row2, text="⏭", command=lambda: self.change_frame(10), width=4).pack(side="left", padx=2)

        self.frame_slider = ttk.Scale(f2, from_=0, to=100, orient="horizontal", command=self.on_slider)
        self.frame_slider.pack(fill="x", pady=5)

        # ── 缩放 ──────────────────────────────────────────────────────────
        f3 = ttk.LabelFrame(p, text="🔍 缩放", padding=10)
        f3.pack(fill="x", pady=pad, padx=5)
        row = ttk.Frame(f3)
        row.pack(fill="x")
        ttk.Button(row, text="−", command=lambda: self.zoom(-0.25), width=4).pack(side="left")
        self.zoom_var = tk.StringVar(value="100%")
        ttk.Label(row, textvariable=self.zoom_var, width=8, anchor="center").pack(side="left", padx=5)
        ttk.Button(row, text="+", command=lambda: self.zoom(0.25), width=4).pack(side="left")
        ttk.Button(row, text="适应", command=self.fit_zoom, width=6).pack(side="right")

        # ── 神经元选择 ────────────────────────────────────────────────────
        f4 = ttk.LabelFrame(p, text="🧠 神经元选择", padding=10)
        f4.pack(fill="x", pady=pad, padx=5)
        row_n = ttk.Frame(f4)
        row_n.pack(fill="x")
        ttk.Label(row_n, text="ID:").pack(side="left")
        self.neuron_var = tk.StringVar(value="1")
        self.neuron_spinbox = ttk.Spinbox(row_n, from_=1, to=99, width=5,
                                          textvariable=self.neuron_var,
                                          command=self.on_neuron_change)
        self.neuron_spinbox.pack(side="left", padx=5)
        self.neuron_spinbox.bind('<Return>', lambda e: self.on_neuron_change())
        self.neuron_color_label = tk.Label(row_n, text="      ", bg='red', width=6, relief="flat")
        self.neuron_color_label.pack(side="left", padx=10)
        self.update_neuron_color_display()

        # ── 标记必经点 ────────────────────────────────────────────────────
        f5 = ttk.LabelFrame(p, text="📍 标记必经点", padding=10)
        f5.pack(fill="x", pady=pad, padx=5)
        ttk.Label(f5, text="左键：添加 | 右键：删除末点 | Ctrl+Z：撤销",
                  font=("Segoe UI", 9), foreground="#808080").pack(anchor="w")

        self.marker_info = tk.StringVar(value="当前神经元无标记")
        ttk.Label(f5, textvariable=self.marker_info, foreground="#4A90E2",
                  wraplength=360).pack(anchor="w", pady=5)

        btn_row = ttk.Frame(f5)
        btn_row.pack(fill="x", pady=5)
        ttk.Button(btn_row, text="↩ 撤销",
                   command=self.undo_last_marker).pack(side="left", fill="x", expand=True, padx=(0, 2))
        ttk.Button(btn_row, text="清除当前",
                   command=self.clear_current_markers).pack(side="left", fill="x", expand=True, padx=2)
        ttk.Button(btn_row, text="清除全部",
                   command=self.clear_all_markers).pack(side="left", fill="x", expand=True, padx=(2, 0))

        # 一键清除：当前神经元的标记 + 结果（更符合高频工作流）
        ttk.Button(
            f5,
            text="🧹 一键清除（当前神经元）",
            command=self.clear_current_everything
        ).pack(fill="x", pady=(0, 5))

        # ── 计算轨迹 ──────────────────────────────────────────────────────
        f6 = ttk.LabelFrame(p, text="🔬 计算轨迹", padding=10)
        f6.pack(fill="x", pady=pad, padx=5)

        btn_row2 = ttk.Frame(f6)
        btn_row2.pack(fill="x", pady=5)
        ttk.Button(btn_row2, text="▶ 计算当前", command=self.compute_current_neuron,
                   style='Accent.TButton').pack(side="left", fill="x", expand=True, padx=(0, 2))
        ttk.Button(btn_row2, text="▶▶ 计算全部", command=self.compute_all_neurons,
                   style='Accent.TButton').pack(side="left", fill="x", expand=True, padx=(2, 0))

        self.compute_info = tk.StringVar(value="")
        ttk.Label(f6, textvariable=self.compute_info, foreground="#50C878",
                  wraplength=360).pack(anchor="w", pady=5)

        btn_row3 = ttk.Frame(f6)
        btn_row3.pack(fill="x", pady=5)
        ttk.Button(btn_row3, text="✓ 确认", command=self.confirm_result,
                   style='Accent.TButton').pack(side="left", fill="x", expand=True, padx=(0, 2))
        ttk.Button(btn_row3, text="✗ 取消",
                   command=self.cancel_preview).pack(side="left", fill="x", expand=True, padx=(2, 0))

        # ── 参数 ──────────────────────────────────────────────────────────
        f7 = ttk.LabelFrame(p, text="⚙️ 参数", padding=10)
        f7.pack(fill="x", pady=pad, padx=5)
        g = ttk.Frame(f7)
        g.pack(fill="x")

        ttk.Label(g, text="亮度:").grid(row=0, column=0, sticky="w")
        self.thresh_var = tk.StringVar(value="30")
        ttk.Entry(g, textvariable=self.thresh_var, width=5).grid(row=0, column=1, padx=5)
        ttk.Label(g, text="半径:").grid(row=0, column=2, sticky="w", padx=(10, 0))
        self.radius_var = tk.StringVar(value="30")
        ttk.Entry(g, textvariable=self.radius_var, width=5).grid(row=0, column=3, padx=5)

        ttk.Label(g, text="转角:").grid(row=1, column=0, sticky="w", pady=5)
        self.angle_var = tk.StringVar(value="60")
        ttk.Entry(g, textvariable=self.angle_var, width=5).grid(row=1, column=1, padx=5)
        ttk.Label(g, text="°").grid(row=1, column=1, sticky="e")
        ttk.Label(g, text="步距:").grid(row=1, column=2, sticky="w", padx=(10, 0))
        self.step_var = tk.StringVar(value="15")
        ttk.Entry(g, textvariable=self.step_var, width=5).grid(row=1, column=3, padx=5)

        ttk.Button(f7, text="应用参数", command=self.apply_params).pack(pady=10, fill="x")

        # ── 平滑约束说明 ──────────────────────────────────────────────────
        f_smooth = ttk.LabelFrame(p, text="🔄 平滑约束", padding=10)
        f_smooth.pack(fill="x", pady=pad, padx=5)
        ttk.Label(f_smooth, text="✓ 转角限制  ✓ 步距限制  ✓ 方向连续",
                  font=("Segoe UI", 9), justify="left", foreground="#50C878").pack(anchor="w")

        # ── 追踪结果 ──────────────────────────────────────────────────────
        f8 = ttk.LabelFrame(p, text="📊 追踪结果", padding=10)
        f8.pack(fill="x", pady=pad, padx=5)

        self.result_list = tk.Listbox(f8, height=5, font=("Consolas", 9),
                                      borderwidth=0, highlightthickness=0)
        self.result_list.pack(fill="x", pady=5)
        self.result_list.bind('<<ListboxSelect>>', self.on_result_select)

        btn_row4 = ttk.Frame(f8)
        btn_row4.pack(fill="x", pady=5)
        ttk.Button(btn_row4, text="✏️ 编辑重算",
                   command=self.edit_selected_result).pack(side="left", fill="x", expand=True, padx=(0, 2))
        ttk.Button(btn_row4, text="删除选中",
                   command=self.delete_selected_result).pack(side="left", fill="x", expand=True, padx=2)
        ttk.Button(btn_row4, text="清空全部",
                   command=self.clear_all_results).pack(side="left", fill="x", expand=True, padx=(2, 0))

        # ── 导出 ──────────────────────────────────────────────────────────
        f9 = ttk.LabelFrame(p, text="💾 导出", padding=10)
        f9.pack(fill="x", pady=pad, padx=5)

        param_row = ttk.Frame(f9)
        param_row.pack(fill="x", pady=(0, 6))
        ttk.Label(param_row, text="FPS:").pack(side="left")
        ttk.Entry(param_row, textvariable=self.fps_var, width=5).pack(side="left", padx=(2, 10))
        ttk.Label(param_row, text="像素/μm:").pack(side="left")
        ttk.Entry(param_row, textvariable=self.pixel_um_var, width=5).pack(side="left", padx=2)

        btn_row5 = ttk.Frame(f9)
        btn_row5.pack(fill="x")
        ttk.Button(btn_row5, text="导出视频", command=self.export_video,
                   style='Accent.TButton').pack(side="left", fill="x", expand=True, padx=(0, 2))
        ttk.Button(btn_row5, text="截图",
                   command=self.screenshot).pack(side="left", fill="x", expand=True, padx=2)
        ttk.Button(btn_row5, text="CSV路径",
                   command=self.export_data).pack(side="left", fill="x", expand=True, padx=(2, 0))

        ttk.Button(f9, text="📈 导出速度CSV",
                   command=self.export_speed_data,
                   style='Accent.TButton').pack(fill="x", pady=(6, 0))

        # ── 状态 ──────────────────────────────────────────────────────────
        f10 = ttk.LabelFrame(p, text="📌 状态", padding=10)
        f10.pack(fill="x", pady=pad, padx=5)
        self.status = tk.StringVar(value="请加载帧")
        ttk.Label(f10, textvariable=self.status, wraplength=360,
                  foreground="#4A90E2").pack(fill="x")
        self.progress = ttk.Progressbar(f10, mode='determinate',
                                        style='Horizontal.TProgressbar')
        self.progress.pack(fill="x", pady=5)

    def bind_events(self):
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<Button-2>", self.on_mid_press)
        self.canvas.bind("<Button-3>", self.on_right_click)
        self.canvas.bind("<B2-Motion>", self.on_mid_drag)
        self.canvas.bind("<ButtonRelease-2>", lambda e: setattr(self, 'dragging', False))
        self.canvas.bind("<MouseWheel>", self.on_scroll)
        self.canvas.bind("<Configure>", lambda e: self.update_display())

        self.root.bind("<Left>", lambda e: self.change_frame(-1))
        self.root.bind("<Right>", lambda e: self.change_frame(1))
        self.root.bind("<Up>", lambda e: self.change_frame(10))
        self.root.bind("<Down>", lambda e: self.change_frame(-10))
        self.root.bind("<Return>", lambda e: self.confirm_result())
        self.root.bind("<Escape>", lambda e: self.cancel_preview())

        # Ctrl+Z 撤销标记
        self.root.bind("<Control-z>", lambda e: self.undo_last_marker())

        for i in range(1, 10):
            self.root.bind(f"<Key-{i}>", lambda e, n=i: self.quick_select_neuron(n))

    def quick_select_neuron(self, n):
        self.neuron_var.set(str(n))
        self.on_neuron_change()

    def update_neuron_color_display(self):
        color = self.get_neuron_color(self.current_neuron_id)
        hex_color = f'#{color[2]:02x}{color[1]:02x}{color[0]:02x}'
        self.neuron_color_label.configure(bg=hex_color)

    def on_neuron_change(self):
        try:
            n = int(self.neuron_var.get())
            n = max(1, min(99, n))
            self.current_neuron_id = n
            self.neuron_var.set(str(n))
            self.update_neuron_color_display()
            self.update_marker_info()
            self.update_display()
        except ValueError:
            pass

    def update_marker_info(self):
        markers = self.tracker.get_neuron_markers(self.current_neuron_id)
        waypoints = self.tracker.get_all_waypoints(self.current_neuron_id)
        undo_depth = len(self.marker_undo_stack)
        if not markers:
            self.marker_info.set(f"N{self.current_neuron_id}: 无标记  (撤销栈:{undo_depth})")
        else:
            frames = sorted(markers.keys())
            self.marker_info.set(
                f"N{self.current_neuron_id}: {len(waypoints)}个必经点 @ 帧{frames}"
                f"  (撤销栈:{undo_depth})"
            )

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

        # 加载新帧时清空撤销栈
        self.marker_undo_stack.clear()

        start_time = time.time()
        if self.tracker.load_frames(
                self.input_dir,
                lambda p: (setattr(self.progress, 'value', p * 100), self.root.update())):
            n = len(self.tracker.frames)
            load_time = time.time() - start_time
            self.frame_slider.configure(to=n - 1)
            self.frame_label.configure(text=f"/ {n - 1}")
            self.current_frame_idx = 0
            self.root.after(100, self.fit_zoom)
            self.status.set(f"已加载 {n} 帧 ({load_time:.2f}秒)")
        else:
            messagebox.showerror("错误", "未找到PNG")
        self.progress['value'] = 0

    def change_frame(self, d):
        if not self.tracker.frames:
            return
        self.current_frame_idx = max(0, min(len(self.tracker.frames) - 1,
                                            self.current_frame_idx + d))
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
        except ValueError:
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

    def on_ctrl_canvas_scroll(self, e):
        self.ctrl_canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")

    def on_mid_press(self, e):
        self.dragging = True
        self.drag_start = (e.x, e.y)

    def on_mid_drag(self, e):
        if self.dragging:
            self.pan_x += e.x - self.drag_start[0]
            self.pan_y += e.y - self.drag_start[1]
            self.drag_start = (e.x, e.y)
            self.update_display()

    def on_pan_x_slider(self, v):
        if self._pan_updating:
            return
        self.pan_x = float(v)
        self.update_display()

    def on_pan_y_slider(self, v):
        if self._pan_updating:
            return
        self.pan_y = float(v)
        self.update_display()

    def update_pan_limits(self, nw, nh, cw, ch):
        if nw <= 0 or nh <= 0 or cw <= 0 or ch <= 0:
            return
        if nw <= cw:
            min_x = max_x = (cw - nw) / 2
        else:
            min_x = cw - nw
            max_x = 0
        if nh <= ch:
            min_y = max_y = (ch - nh) / 2
        else:
            min_y = ch - nh
            max_y = 0

        self.pan_x = min(max(self.pan_x, min_x), max_x)
        self.pan_y = min(max(self.pan_y, min_y), max_y)

        self._pan_updating = True
        self.h_pan_scale.configure(from_=min_x, to=max_x)
        self.v_pan_scale.configure(from_=min_y, to=max_y)
        self.h_pan_scale.set(self.pan_x)
        self.v_pan_scale.set(self.pan_y)
        self._pan_updating = False

    def canvas_to_image(self, cx, cy):
        return (int((cx - self.pan_x) / self.zoom_level),
                int((cy - self.pan_y) / self.zoom_level))

    # ------------------------------------------------------------------ #
    #  标记操作（含撤销栈）                                                 #
    # ------------------------------------------------------------------ #

    def on_click(self, e):
        """左键：添加标记，记录到撤销栈。"""
        if not self.tracker.frames:
            return
        ix, iy = self.canvas_to_image(e.x, e.y)
        if not (0 <= ix < self.tracker.width and 0 <= iy < self.tracker.height):
            return

        pt = (ix, iy)
        self.tracker.add_marker(self.current_neuron_id, self.current_frame_idx, ix, iy)
        # GUI 侧也维护一个“最靠左的起点标记”，用于渲染/导出的一致性
        if self.current_neuron_id not in self.first_marker_by_neuron:
            self.first_marker_by_neuron[self.current_neuron_id] = pt
        else:
            cur = self.first_marker_by_neuron[self.current_neuron_id]
            if pt[0] < cur[0]:
                self.first_marker_by_neuron[self.current_neuron_id] = pt

        # 记录到撤销栈
        self.marker_undo_stack.append({
            'action': 'add',
            'neuron_id': self.current_neuron_id,
            'frame_idx': self.current_frame_idx,
            'point': pt,
        })

        self.update_marker_info()
        total = len(self.tracker.get_all_waypoints(self.current_neuron_id))
        self.status.set(
            f"N{self.current_neuron_id}: 添加 ({ix},{iy}) @ 帧{self.current_frame_idx} "
            f"[共{total}点]  Ctrl+Z可撤销"
        )
        self.update_display()
        self.request_auto_recompute()

    def on_right_click(self, e):
        """右键：删除当前帧最后一个标记，记录到撤销栈。"""
        if not self.tracker.frames:
            return

        removed = self.tracker.remove_last_marker(
            self.current_neuron_id, self.current_frame_idx
        )

        if removed is not None:
            # 记录到撤销栈（action='remove' 表示删除，撤销时要重新 add）
            self.marker_undo_stack.append({
                'action': 'remove',
                'neuron_id': self.current_neuron_id,
                'frame_idx': self.current_frame_idx,
                'point': removed,
            })
            self.status.set(
                f"N{self.current_neuron_id}: 删除 {removed} @ 帧{self.current_frame_idx}  "
                f"Ctrl+Z可撤销"
            )
            if self.first_marker_by_neuron.get(self.current_neuron_id) == removed:
                markers = self.tracker.get_neuron_markers(self.current_neuron_id)
                if markers:
                    first_frame = min(markers.keys())
                    pts = markers.get(first_frame, [])
                    if pts:
                        self.first_marker_by_neuron[self.current_neuron_id] = pts[0]
                    else:
                        self.first_marker_by_neuron.pop(self.current_neuron_id, None)
                else:
                    self.first_marker_by_neuron.pop(self.current_neuron_id, None)
        else:
            self.status.set(f"N{self.current_neuron_id}: 当前帧无标记可删")

        self.update_marker_info()
        self.update_display()
        if removed is not None:
            self.request_auto_recompute()

    def undo_last_marker(self):
        """撤销最近一次标记添加或删除操作（Ctrl+Z / 撤销按钮）。"""
        if not self.marker_undo_stack:
            self.status.set("撤销栈为空，无可撤销操作")
            return

        record = self.marker_undo_stack.pop()
        action = record['action']
        nid = record['neuron_id']
        frame_idx = record['frame_idx']
        pt = record['point']

        if nid != self.current_neuron_id:
            self.neuron_var.set(str(nid))
            self.on_neuron_change()

        if action == 'add':
            # 上次是"添加"，撤销 = 删除该精确点
            self.tracker.remove_specific_marker(nid, frame_idx, pt)
            if self.first_marker_by_neuron.get(nid) == pt:
                markers = self.tracker.get_neuron_markers(nid)
                if markers:
                    first_frame = min(markers.keys())
                    pts = markers.get(first_frame, [])
                    if pts:
                        self.first_marker_by_neuron[nid] = pts[0]
                    else:
                        self.first_marker_by_neuron.pop(nid, None)
                else:
                    self.first_marker_by_neuron.pop(nid, None)
            self.status.set(
                f"撤销：移除 N{nid} 帧{frame_idx} 的 {pt}  "
                f"(栈剩余:{len(self.marker_undo_stack)})"
            )
        else:
            # 上次是"删除"，撤销 = 重新添加
            self.tracker.add_marker(nid, frame_idx, pt[0], pt[1])
            if nid not in self.first_marker_by_neuron:
                self.first_marker_by_neuron[nid] = pt
            self.status.set(
                f"撤销：恢复 N{nid} 帧{frame_idx} 的 {pt}  "
                f"(栈剩余:{len(self.marker_undo_stack)})"
            )

        self.update_marker_info()
        self.update_display()
        self.request_auto_recompute()

    def request_auto_recompute(self):
        """
        如果当前神经元已经有计算结果（已确认或正在预览），则在加点/删点后自动触发一次重算预览。
        使用 after 做防抖，避免连续点击每次都立即重算。
        """
        if not self.auto_recompute_enabled:
            return
        if not self.tracker.frames:
            return

        nid = self.current_neuron_id
        has_any_result = (nid in self.tracker.tracking_results) or (self.preview_neuron_id == nid)
        if not has_any_result:
            return

        # 取消上一次排队任务（防抖）
        if self._auto_recompute_job is not None:
            try:
                self.root.after_cancel(self._auto_recompute_job)
            except Exception:
                pass
            self._auto_recompute_job = None

        self._auto_recompute_job = self.root.after(self.auto_recompute_delay_ms, self._auto_recompute_now)

    def _auto_recompute_now(self):
        """执行自动重算（生成 preview，不自动确认）。"""
        self._auto_recompute_job = None
        nid = self.current_neuron_id
        markers = self.tracker.get_neuron_markers(nid)
        if not markers:
            return
        # 直接复用现有“计算当前”逻辑：它会生成 preview，并在画面上显示
        self.compute_current_neuron()

    def clear_current_markers(self):
        self.tracker.clear_neuron_markers(self.current_neuron_id)
        self.first_marker_by_neuron.pop(self.current_neuron_id, None)
        # 清除当前神经元相关的撤销记录
        self.marker_undo_stack = [
            r for r in self.marker_undo_stack
            if r['neuron_id'] != self.current_neuron_id
        ]
        self.update_marker_info()
        self.status.set(f"已清除 N{self.current_neuron_id} 的所有标记")
        self.update_display()

    def clear_current_everything(self):
        """一键清除当前神经元：标记 + 已确认结果 + 预览状态。"""
        nid = self.current_neuron_id
        if not messagebox.askyesno("确认", f"清除 N{nid} 的标记与追踪结果？"):
            return

        # 清标记
        self.tracker.clear_neuron_markers(nid)
        self.first_marker_by_neuron.pop(nid, None)
        self.marker_undo_stack = [r for r in self.marker_undo_stack if r['neuron_id'] != nid]

        # 删已确认结果
        if nid in self.tracker.tracking_results:
            del self.tracker.tracking_results[nid]

        # 清预览（如果正在预览/编辑该神经元）
        if self.preview_neuron_id == nid:
            self.preview_result = None
            self.preview_neuron_id = None
            self.preview_backup_result = None
            self.preview_backup_neuron_id = None
            self.preview_is_edit_mode = False
            self.compute_info.set("")

        self.update_marker_info()
        self.update_result_list()
        self.status.set(f"已清除 N{nid} 的标记与结果")
        self.update_display()

    def clear_all_markers(self):
        if messagebox.askyesno("确认", "清除所有神经元的标记？"):
            self.tracker.markers.clear()
            self.tracker.first_marker_by_neuron.clear()
            self.marker_undo_stack.clear()
            self.first_marker_by_neuron.clear()
            self.update_marker_info()
            self.status.set("已清除所有标记")
            self.update_display()

    def apply_params(self):
        try:
            self.tracker.brightness_threshold = int(self.thresh_var.get())
            self.tracker.search_radius = int(self.radius_var.get())
            self.tracker.max_turn_angle = int(self.angle_var.get())
            self.tracker.max_step_distance = int(self.step_var.get())
            self.tracker._search_grids.clear()
            self.status.set(
                f"参数已更新: 转角≤{self.tracker.max_turn_angle}°, "
                f"步距≤{self.tracker.max_step_distance}"
            )
        except ValueError:
            pass

    def compute_current_neuron(self):
        """计算（或重算）当前神经元的完整轨迹。"""
        markers = self.tracker.get_neuron_markers(self.current_neuron_id)
        if not markers:
            messagebox.showwarning("警告", f"N{self.current_neuron_id} 没有标记点")
            return

        self.apply_params()
        self.status.set(f"计算 N{self.current_neuron_id} 轨迹（完整重算）...")
        self.root.update()

        if self.preview_neuron_id != self.current_neuron_id:
            self.preview_result = None
            self.preview_neuron_id = None
            self.preview_backup_result = None
            self.preview_backup_neuron_id = None
            self.preview_is_edit_mode = False

        start_time = time.time()
        result = self.tracker.compute_neuron_trajectory(self.current_neuron_id)
        elapsed = time.time() - start_time

        if result:
            self.preview_result = result
            self.preview_neuron_id = self.current_neuron_id

            init_len = len(result['initial_path'])
            final_tip = result['final_tip_x']

            self.compute_info.set(
                f"N{self.current_neuron_id}: 耗时{elapsed:.2f}秒\n"
                f"初始: {init_len}点, 末端x={final_tip}\n"
                f"Enter确认 / Esc取消"
            )
            self.status.set(f"预览就绪 (耗时{elapsed:.2f}秒)")
        else:
            self.compute_info.set("计算失败")
            self.status.set("未找到有效轨迹")

        self.update_display()

    def compute_all_neurons(self):
        neuron_ids = list(self.tracker.markers.keys())
        if not neuron_ids:
            messagebox.showwarning("警告", "没有任何标记")
            return

        self.apply_params()
        success = 0
        total_time = 0

        for nid in neuron_ids:
            self.status.set(f"计算 N{nid} ...")
            self.root.update()
            start = time.time()
            result = self.tracker.compute_neuron_trajectory(nid)
            total_time += time.time() - start
            if result:
                self.tracker.tracking_results[nid] = result
                success += 1

        self.update_result_list()
        self.status.set(
            f"完成: {success}/{len(neuron_ids)} 个神经元, 总耗时{total_time:.2f}秒"
        )
        self.update_display()

    def confirm_result(self):
        if self.preview_result and self.preview_neuron_id:
            self.tracker.tracking_results[self.preview_neuron_id] = self.preview_result
            self.update_result_list()
            self.status.set(f"已确认 N{self.preview_neuron_id}")
            self.preview_result = None
            self.preview_neuron_id = None
            self.preview_backup_result = None
            self.preview_backup_neuron_id = None
            self.preview_is_edit_mode = False
            self.compute_info.set("")
            self.update_display()

    def cancel_preview(self):
        if self.preview_is_edit_mode and self.preview_neuron_id is not None:
            if self.preview_backup_result is not None and self.preview_backup_neuron_id == self.preview_neuron_id:
                self.tracker.tracking_results[self.preview_neuron_id] = self.preview_backup_result
                self.update_result_list()
        self.preview_result = None
        self.preview_neuron_id = None
        self.preview_backup_result = None
        self.preview_backup_neuron_id = None
        self.preview_is_edit_mode = False
        self.compute_info.set("已取消")
        self.update_display()

    def update_result_list(self):
        self.result_list.delete(0, tk.END)
        for nid, result in sorted(self.tracker.tracking_results.items()):
            init_len = len(result.get('initial_path', []))
            final_tip = result.get('final_tip_x', 0)
            self.result_list.insert(tk.END, f"N{nid}: {init_len}点, 末x={final_tip}")

    def on_result_select(self, e):
        sel = self.result_list.curselection()
        if sel:
            text = self.result_list.get(sel[0])
            nid = int(text.split(':')[0][1:])
            self.neuron_var.set(str(nid))
            self.on_neuron_change()

    def delete_selected_result(self):
        sel = self.result_list.curselection()
        if sel:
            text = self.result_list.get(sel[0])
            nid = int(text.split(':')[0][1:])
            if nid in self.tracker.tracking_results:
                del self.tracker.tracking_results[nid]
            self.update_result_list()
            self.update_display()

    def clear_all_results(self):
        if messagebox.askyesno("确认", "清除所有追踪结果？"):
            self.tracker.tracking_results.clear()
            self.update_result_list()
            self.update_display()

    # ------------------------------------------------------------------ #
    #  编辑重算                                                             #
    # ------------------------------------------------------------------ #

    def edit_selected_result(self):
        """
        将已确认的结果移回 preview 状态：
        - 用户可继续添加/删除标记（支持撤销）
        - 点击「▶ 计算当前」会完整重算整条轨迹
        - Enter 确认新结果，Esc 取消（恢复旧结果）
        """
        sel = self.result_list.curselection()
        if not sel:
            messagebox.showinfo("提示", "请先在列表中选择一个神经元")
            return

        text = self.result_list.get(sel[0])
        nid = int(text.split(':')[0][1:])

        # 切换到该神经元
        self.neuron_var.set(str(nid))
        self.on_neuron_change()

        # 把已确认结果暂存为 preview，并从 confirmed 中移除
        if nid in self.tracker.tracking_results:
            self.preview_result = self.tracker.tracking_results.pop(nid)
            self.preview_neuron_id = nid
            self.preview_backup_result = self.preview_result
            self.preview_backup_neuron_id = nid
            self.preview_is_edit_mode = True
            self.update_result_list()

        # 清空该神经元相关的旧撤销记录，以免混淆
        self.marker_undo_stack = [
            r for r in self.marker_undo_stack if r['neuron_id'] != nid
        ]

        self.compute_info.set(
            f"✏️ 编辑模式 N{nid}\n"
            f"左键加点 / Ctrl+Z撤销\n"
            f"点击「▶ 计算当前」完整重算\n"
            f"Enter确认 / Esc取消"
        )
        self.status.set(f"N{nid} 编辑模式：添加标记后点「计算当前」重算全轨迹")
        self.update_display()

    # ------------------------------------------------------------------ #
    #  导出速度                                                             #
    # ------------------------------------------------------------------ #

    def export_speed_data(self):
        if not self.tracker.tracking_results:
            messagebox.showwarning("警告", "无追踪结果")
            return

        try:
            fps = float(self.fps_var.get())
            pixel_um = float(self.pixel_um_var.get())
        except ValueError:
            messagebox.showerror("错误", "FPS 和 像素/μm 必须是数字")
            return

        self.output_dir = self.output_entry.get()
        os.makedirs(self.output_dir, exist_ok=True)
        speed_path = os.path.join(self.output_dir, "neuron_speed.csv")
        tip_path = os.path.join(self.output_dir, "neuron_tip.csv")

        all_speeds = self.tracker.compute_all_speeds(fps=fps, pixel_um=pixel_um)
        all_tips = self.tracker.compute_all_tip_positions()

        with open(speed_path, 'w', encoding='utf-8') as f:
            f.write("neuron_id,frame,tip_x,tip_y,dx,dy,"
                    "speed_px_per_frame,speed_um_per_sec\n")
            for nid in sorted(all_speeds.keys()):
                for rec in all_speeds[nid]:
                    f.write(
                        f"{nid},{rec['frame']},{rec['tip_x']},{rec['tip_y']},"
                        f"{rec['dx']},{rec['dy']},"
                        f"{rec['speed_px_per_frame']},{rec['speed_um_per_sec']}\n"
                    )

        with open(tip_path, 'w', encoding='utf-8') as f:
            f.write("neuron_id,frame,tip_x,tip_y,path_points\n")
            for nid in sorted(all_tips.keys()):
                for rec in all_tips[nid]:
                    f.write(
                        f"{nid},{rec['frame']},{rec['tip_x']},{rec['tip_y']},{rec['path_points']}\n"
                    )

        # 状态栏显示各神经元平均速度摘要
        summary = []
        for nid, records in sorted(all_speeds.items()):
            if records:
                avg = sum(r['speed_um_per_sec'] for r in records) / len(records)
                summary.append(f"N{nid}均速{avg:.2f}μm/s")

        self.status.set("速度已保存: " + speed_path + "  |  " + "  ".join(summary))
        messagebox.showinfo(
            "完成",
            f"速度CSV导出完成！\n{speed_path}\n\n末端点CSV：\n{tip_path}\n\n" + "\n".join(summary)
        )

    # ------------------------------------------------------------------ #
    #  原有导出方法                                                         #
    # ------------------------------------------------------------------ #

    def export_video(self):
        if not self.tracker.tracking_results:
            messagebox.showwarning("警告", "无结果")
            return
        try:
            fps = float(self.fps_var.get())
        except ValueError:
            messagebox.showerror("错误", "FPS 必须是数字")
            return
        self.output_dir = self.output_entry.get()
        os.makedirs(self.output_dir, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        path = os.path.join(self.output_dir, "neuron_tracking_smooth.mp4")
        out = cv2.VideoWriter(path, fourcc, fps,
                              (self.tracker.width, self.tracker.height), True)
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

    def _get_first_anchor_point(self, neuron_id, result):
        if neuron_id in self.first_marker_by_neuron:
            return self.first_marker_by_neuron[neuron_id]
        markers = self.tracker.get_neuron_markers(neuron_id)
        if markers:
            first_frame = min(markers.keys())
            pts = markers.get(first_frame, [])
            if pts:
                return pts[0]
        waypoints = result.get('waypoints', [])
        if waypoints:
            return waypoints[0]
        return None

    def _orient_path_from_anchor(self, frame_path, anchor):
        if not frame_path:
            return []
        oriented = list(frame_path)
        if len(oriented) == 1 or anchor is None:
            return oriented
        ax, ay = anchor
        x0, y0 = oriented[0]
        x1, y1 = oriented[-1]
        d0 = (x0 - ax) ** 2 + (y0 - ay) ** 2
        d1 = (x1 - ax) ** 2 + (y1 - ay) ** 2
        if d1 < d0:
            oriented.reverse()
        return oriented

    def export_data(self):
        if not self.tracker.tracking_results:
            messagebox.showwarning("警告", "无结果")
            return
        self.output_dir = self.output_entry.get()
        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir, "neuron_paths.csv")

        with open(path, 'w', encoding='utf-8') as f:
            f.write("neuron_id,frame,path_index,x,y\n")
            for nid, result in sorted(self.tracker.tracking_results.items()):
                anchor = self._get_first_anchor_point(nid, result)
                paths_by_frame = result.get('paths_by_frame', {})
                for frame_idx, frame_path in sorted(paths_by_frame.items()):
                    oriented_path = self._orient_path_from_anchor(frame_path, anchor)
                    for idx, (x, y) in enumerate(oriented_path):
                        f.write(f"{nid},{frame_idx},{idx},{x},{y}\n")

        self.status.set(f"数据已保存: {path}")
        messagebox.showinfo("完成", f"导出完成!\n{path}")

    def render(self, idx, for_export=False):
        frame = self.tracker.frames_np[idx]
        vis = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        cv2.line(vis, (self.tracker.left_margin, 0),
                 (self.tracker.left_margin, self.tracker.height), (0, 100, 0), 1)
        cv2.line(vis, (self.tracker.width - self.tracker.right_margin, 0),
                 (self.tracker.width - self.tracker.right_margin, self.tracker.height), (0, 100, 0), 1)
        cv2.line(vis, (0, self.tracker.edge_margin),
                 (self.tracker.width, self.tracker.edge_margin), (0, 100, 0), 1)
        cv2.line(vis, (0, self.tracker.height - self.tracker.edge_margin),
                 (self.tracker.width, self.tracker.height - self.tracker.edge_margin), (0, 100, 0), 1)

        results_to_render = []
        for nid, result in self.tracker.tracking_results.items():
            if self.preview_neuron_id == nid:
                continue
            results_to_render.append((nid, result, False))
        if self.preview_result and self.preview_neuron_id:
            results_to_render.append((self.preview_neuron_id, self.preview_result, True))

        for nid, result, is_preview in results_to_render:
            color = self.get_neuron_color(nid)
            if is_preview:
                color = (0, 255, 255)

            paths_by_frame = result.get('paths_by_frame', {})
            waypoints = result.get('waypoints', [])
            initial_path = result.get('initial_path', [])

            if idx in paths_by_frame:
                frame_path = paths_by_frame[idx]
                init_len = len(initial_path)
                anchor = self._get_first_anchor_point(nid, result)
                oriented_path = self._orient_path_from_anchor(frame_path, anchor)
                if oriented_path:
                    start_pt = anchor if anchor is not None else oriented_path[0]
                    tip_pt = frame_path[-1] if frame_path else oriented_path[-1]
                else:
                    start_pt = None
                    tip_pt = None

                if for_export:
                    if start_pt is not None and tip_pt is not None:
                        # 对于导出视频：
                        # oriented_path 已经是通过追踪器拿到的“当前帧从起点到端点的完整轨迹”
                        # 确保完全不画首尾直连线，只画真实走过的连续轨迹
                        if len(oriented_path) >= 2:
                            cv2.polylines(vis, [np.array(oriented_path, np.int32)], False, color, 2)

                        # 画端点随时间移动轨迹（tip track，画到当前帧为止）
                        tip_track = result.get('tip_track')
                        if tip_track:
                            pts = [p for p in tip_track[:idx + 1] if p is not None]
                            if len(pts) >= 2:
                                track_color = tuple(min(255, c + 60) for c in color)
                                cv2.polylines(vis, [np.array(pts, np.int32)], False, track_color, 1)

                        # 画起点 (强制取用 _get_first_anchor_point 保证起点位置在导出时绝对统一)
                        first_marker = self._get_first_anchor_point(nid, result)
                        if first_marker:
                            cv2.circle(vis, first_marker, 6, (0, 255, 0), -1)
                        else:
                            cv2.circle(vis, start_pt, 6, (0, 255, 0), -1)

                        # 画当前末端点（也就是橙色点，严格跟着轨迹尽头）
                        cv2.circle(vis, tip_pt, 8, (0, 165, 255), -1)
                        cv2.putText(vis, f"N{nid}",
                                    (tip_pt[0] + 5, tip_pt[1] - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                else:
                    # 确保完全不画首尾直连线，用 polylines 绘制真实连贯轨迹
                    if len(oriented_path) >= 2:
                        cv2.polylines(vis, [np.array(oriented_path, np.int32)], False, color, 2)

                    if len(oriented_path) > init_len:
                        lighter_color = tuple(min(255, c + 60) for c in color)
                        start_j = max(init_len, 1)
                        if len(oriented_path[start_j - 1:]) >= 2:
                            cv2.polylines(vis, [np.array(oriented_path[start_j - 1:], np.int32)], False, lighter_color,
                                          2)

                    if start_pt is not None and tip_pt is not None:
                        first_marker = self._get_first_anchor_point(nid, result)
                        if first_marker:
                            cv2.circle(vis, first_marker, 6, (0, 255, 0), -1)
                        else:
                            cv2.circle(vis, start_pt, 6, (0, 255, 0), -1)

                        cv2.circle(vis, tip_pt, 6, (0, 165, 255), -1)
                        cv2.putText(vis, f"N{nid}",
                                    (tip_pt[0] + 5, tip_pt[1] - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            if not for_export:
                for wp in waypoints:
                    cv2.drawMarker(vis, wp, (255, 255, 255), cv2.MARKER_DIAMOND, 10, 2)

        if not for_export:
            for nid, frame_markers in self.tracker.markers.items():
                color = self.get_neuron_color(nid)
                all_pts = []
                for f, pts in frame_markers.items():
                    for pt in pts:
                        all_pts.append((pt, f == idx))

                for pt, is_current in all_pts:
                    if is_current:
                        cv2.circle(vis, pt, 10, color, -1)
                        cv2.circle(vis, pt, 10, (255, 255, 255), 2)
                    else:
                        cv2.circle(vis, pt, 6, color, 2)

                if all_pts:
                    rightmost = max(all_pts, key=lambda x: x[0][0])
                    cv2.putText(vis, f"N{nid}",
                                (rightmost[0][0] + 12, rightmost[0][1] - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        if for_export:
            cv2.putText(vis, f"Frame {idx}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            y_pos = 60
            for nid in sorted(self.tracker.tracking_results.keys()):
                color = self.get_neuron_color(nid)
                cv2.rectangle(vis, (10, y_pos - 12), (30, y_pos + 2), color, -1)
                cv2.putText(vis, f"N{nid}", (35, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_pos += 20

        return vis

    def update_display(self):
        if not self.tracker.frames:
            return
        vis = cv2.cvtColor(self.render(self.current_frame_idx), cv2.COLOR_BGR2RGB)
        nw = int(self.tracker.width * self.zoom_level)
        nh = int(self.tracker.height * self.zoom_level)
        if nw < 1 or nh < 1:
            return
        vis_scaled = cv2.resize(vis, (nw, nh))
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        if cw < 10:
            return
        self.update_pan_limits(nw, nh, cw, ch)
        canvas_img = np.zeros((ch, cw, 3), dtype=np.uint8)
        x0, y0 = int(self.pan_x), int(self.pan_y)
        sx1, sy1 = max(0, -x0), max(0, -y0)
        sx2, sy2 = min(nw, cw - x0), min(nh, ch - y0)
        dx1, dy1 = max(0, x0), max(0, y0)
        if sx2 > sx1 and sy2 > sy1:
            canvas_img[dy1:dy1 + (sy2 - sy1), dx1:dx1 + (sx2 - sx1)] = \
                vis_scaled[sy1:sy2, sx1:sx2]
        self.photo = ImageTk.PhotoImage(Image.fromarray(canvas_img))
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)

    def run(self):
        self.root.mainloop()
