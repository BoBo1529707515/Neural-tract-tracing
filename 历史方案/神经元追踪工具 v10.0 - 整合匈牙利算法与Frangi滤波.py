"""
神经元追踪工具 v10.0
整合：Frangi滤波 + 尖端检测 + 匈牙利算法匹配
"""

import cv2
import numpy as np
import json
import os
import heapq
from datetime import datetime
from collections import defaultdict

from skimage.morphology import skeletonize, remove_small_objects
from skimage.filters import frangi, hessian
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import convolve

try:
    import tkinter as tk
    from tkinter import filedialog

    HAS_TK = True
except ImportError:
    HAS_TK = False


# ============================================================================
#                              配置
# ============================================================================

class Config:
    MAX_NEURONS = 50
    DISPLAY_WIDTH = 1600
    DISPLAY_HEIGHT = 1000
    PANEL_HEIGHT = 130

    MIN_ZOOM, MAX_ZOOM, ZOOM_STEP = 0.2, 10.0, 1.4

    # 图像处理
    GREEN_THRESHOLD = 40
    CLAHE_CLIP_LIMIT = 3.0
    MEDIAN_KERNEL = 5
    MIN_OBJECT_SIZE = 50

    # Frangi 滤波参数
    FRANGI_SCALE_RANGE = (1, 5)
    FRANGI_SCALE_STEP = 1
    FRANGI_BETA1 = 0.5
    FRANGI_BETA2 = 15

    # 追踪参数
    MAX_MATCH_DIST = 50  # 最大匹配距离
    BIRTH_COST = 80  # 新生代价
    DEATH_COST = 80  # 消失代价
    DIRECTION_WEIGHT = 15.0  # 方向一致性权重
    PREDICTION_WEIGHT = 5.0  # 预测偏差权重
    Y_TOLERANCE = 35
    MAX_GAP = 15
    DIRECTION_HISTORY = 8

    # 颜色
    COLOR_BG = (30, 30, 30)
    COLOR_PANEL = (25, 25, 25)
    COLOR_BTN = (65, 65, 65)
    COLOR_BTN_HOVER = (85, 85, 85)
    COLOR_BTN_ACTIVE = (70, 130, 70)


def gen_colors(n):
    colors = []
    for i in range(n):
        h = int(180 * i / n)
        hsv = np.uint8([[[h, 255, 255]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
        colors.append((int(bgr[0]), int(bgr[1]), int(bgr[2])))
    return colors


# ============================================================================
#                              UI组件
# ============================================================================

class Button:
    def __init__(self, x, y, w, h, text, color=None):
        self.x, self.y, self.w, self.h = x, y, w, h
        self.text = text
        self.color = color or Config.COLOR_BTN
        self.hover = self.active = False

    def contains(self, px, py):
        return self.x <= px < self.x + self.w and self.y <= py < self.y + self.h

    def draw(self, img):
        bg = Config.COLOR_BTN_ACTIVE if self.active else (Config.COLOR_BTN_HOVER if self.hover else self.color)
        cv2.rectangle(img, (self.x, self.y), (self.x + self.w, self.y + self.h), bg, -1)
        cv2.rectangle(img, (self.x, self.y), (self.x + self.w, self.y + self.h), (100, 100, 100), 1)
        (tw, th), _ = cv2.getTextSize(self.text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.putText(img, self.text, (self.x + (self.w - tw) // 2, self.y + (self.h + th) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)


class InputBox:
    def __init__(self, x, y, w, h, label):
        self.x, self.y, self.w, self.h = x, y, w, h
        self.label = label
        self.text = ""
        self.active = False

    def contains(self, px, py):
        return self.x <= px < self.x + self.w and self.y <= py < self.y + self.h

    def draw(self, img, highlight=None):
        cv2.putText(img, self.label, (self.x, self.y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
        bg = (55, 55, 75) if self.active else (45, 45, 45)
        cv2.rectangle(img, (self.x, self.y), (self.x + self.w, self.y + self.h), bg, -1)
        border = (150, 180, 255) if self.active else (100, 100, 100)
        cv2.rectangle(img, (self.x, self.y), (self.x + self.w, self.y + self.h), border, 1)
        tx = self.x + 6
        if highlight:
            cv2.rectangle(img, (self.x + 4, self.y + 4), (self.x + 20, self.y + self.h - 4), highlight, -1)
            tx = self.x + 26
        cv2.putText(img, self.text, (tx, self.y + self.h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

    def handle_key(self, key):
        if key in [13, 10]: return 'confirm'
        if key == 27: return 'cancel'
        if key in [8, 127]:
            self.text = self.text[:-1]
        elif ord('0') <= key <= ord('9'):
            self.text += chr(key)
        return None


# ============================================================================
#                              视频处理
# ============================================================================

class VideoHandler:
    def __init__(self):
        self.cap = self.writer = None
        self.path = ""
        self.total = self.w = self.h = 0
        self.fps = 30
        self.frame = None
        self.idx = 0

    def load(self, path):
        self.path = path
        self.cap = cv2.VideoCapture(path)
        self.total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        print(f"✓ 视频: {self.w}x{self.h}, {self.fps:.1f}fps, {self.total}帧")

    def read(self, idx):
        idx = max(0, min(idx, self.total - 1))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, self.frame = self.cap.read()
        if ret: self.idx = idx
        return self.frame

    def start_write(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), int(self.fps), (self.w, self.h))

    def write(self, f):
        if self.writer: self.writer.write(f)

    def stop_write(self):
        if self.writer: self.writer.release(); self.writer = None

    def release(self):
        if self.cap: self.cap.release()
        self.stop_write()


# ============================================================================
#                     阶段一：图像增强与分割（Frangi滤波）
# ============================================================================

class ImageProcessor:
    """
    图像处理器：整合 Frangi 滤波增强管状结构
    """

    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=Config.CLAHE_CLIP_LIMIT, tileGridSize=(8, 8))
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    def is_green(self, f, y, x):
        """检查像素是否为绿色"""
        if not (0 <= y < f.shape[0] and 0 <= x < f.shape[1]):
            return False
        b, g, r = int(f[y, x, 0]), int(f[y, x, 1]), int(f[y, x, 2])
        return g > 50 and g - r > Config.GREEN_THRESHOLD and g - b > Config.GREEN_THRESHOLD

    def extract_green_channel(self, frame):
        """提取并增强绿色通道"""
        b, g, r = cv2.split(frame)
        b, g, r = b.astype(np.int16), g.astype(np.int16), r.astype(np.int16)

        # 绿色掩码
        mask = (g > 50) & ((g - r) > Config.GREEN_THRESHOLD) & ((g - b) > Config.GREEN_THRESHOLD)
        green_mask = mask.astype(np.uint8) * 255

        # CLAHE增强
        enhanced = self.clahe.apply(frame[:, :, 1])

        return enhanced, green_mask

    def apply_frangi(self, image):
        """
        应用 Frangi 滤波器增强管状结构

        Frangi 滤波基于 Hessian 矩阵特征值分析：
        - 对于管状结构：|λ1| ≈ 0, |λ2| >> 0
        - 输出响应强度
        """
        # 归一化到 [0, 1]
        img_norm = image.astype(np.float64) / 255.0

        # Frangi 滤波（黑色背景上的亮管）
        try:
            frangi_resp = frangi(
                img_norm,
                sigmas=range(Config.FRANGI_SCALE_RANGE[0],
                             Config.FRANGI_SCALE_RANGE[1],
                             Config.FRANGI_SCALE_STEP),
                black_ridges=False,
                beta=Config.FRANGI_BETA1,
                gamma=Config.FRANGI_BETA2
            )
        except TypeError:
            # 旧版本 skimage 兼容
            frangi_resp = frangi(
                img_norm,
                scale_range=Config.FRANGI_SCALE_RANGE,
                scale_step=Config.FRANGI_SCALE_STEP,
                black_ridges=False
            )

        # 归一化到 [0, 255]
        frangi_norm = (frangi_resp / (frangi_resp.max() + 1e-8) * 255).astype(np.uint8)

        return frangi_norm

    def process_frame(self, frame, use_frangi=True):
        """
        完整的图像处理流程

        返回：骨架图像 (bool array)
        """
        enhanced, green_mask = self.extract_green_channel(frame)

        if use_frangi:
            frangi_resp = self.apply_frangi(enhanced)
            combined = cv2.bitwise_and(frangi_resp, green_mask)
            _, binary = cv2.threshold(combined, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            denoised = cv2.medianBlur(enhanced, Config.MEDIAN_KERNEL)
            binary = cv2.adaptiveThreshold(
                denoised, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 15, -3
            )
            binary = cv2.bitwise_and(binary, green_mask)

        # 形态学清理
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, self.kernel, iterations=2)

        # 移除小对象（兼容新旧 skimage 版本）
        bool_mask = closed > 0
        try:
            # skimage >= 0.26: 使用 min_size（仍有效，只是警告）
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                cleaned = remove_small_objects(bool_mask, min_size=Config.MIN_OBJECT_SIZE)
        except Exception:
            cleaned = bool_mask

        # 骨架化
        skeleton = skeletonize(cleaned)

        return skeleton

    def nearest_skeleton_point(self, skeleton, frame, point, radius=30):
        """找到最近的骨架点"""
        h, w = skeleton.shape
        px, py = int(point[0]), int(point[1])
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


# ============================================================================
#                     阶段二：拓扑结构提取（尖端/分支点检测）
# ============================================================================

class TopologyExtractor:
    """
    拓扑结构提取器：从骨架中提取关键节点

    节点类型根据度(degree)分类：
    - 尖端(Tip): degree = 1
    - 路径(Path): degree = 2
    - 分支点(Branch): degree >= 3
    """

    # 8邻域偏移
    NEIGHBORS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    def __init__(self):
        # 用于计算度的卷积核
        self.degree_kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)

    def compute_degree_map(self, skeleton):
        """计算每个骨架像素的度（邻居数量）"""
        skel_uint8 = skeleton.astype(np.uint8)
        degree = convolve(skel_uint8, self.degree_kernel, mode='constant', cval=0)
        degree = degree * skel_uint8  # 只保留骨架上的点
        return degree

    def extract_tips(self, skeleton):
        """
        提取所有尖端点 (degree = 1)

        返回: [(y, x, direction), ...]
        direction 是指向内部的单位向量
        """
        degree = self.compute_degree_map(skeleton)
        tips = []

        # 找到所有 degree=1 的点
        tip_coords = np.argwhere(degree == 1)

        for (y, x) in tip_coords:
            # 计算方向（从尖端指向内部）
            direction = self._compute_tip_direction(skeleton, y, x)
            tips.append((y, x, direction))

        return tips

    def extract_branches(self, skeleton):
        """
        提取所有分支点 (degree >= 3)

        返回: [(y, x, degree), ...]
        """
        degree = self.compute_degree_map(skeleton)
        branch_coords = np.argwhere(degree >= 3)

        branches = []
        for (y, x) in branch_coords:
            branches.append((y, x, int(degree[y, x])))

        return branches

    def _compute_tip_direction(self, skeleton, y, x, depth=5):
        """
        计算尖端的方向向量

        沿着骨架追踪 depth 个像素，计算平均方向
        """
        h, w = skeleton.shape
        path = [(y, x)]
        visited = {(y, x)}

        current = (y, x)
        for _ in range(depth):
            found = False
            for dy, dx in self.NEIGHBORS:
                ny, nx = current[0] + dy, current[1] + dx
                if (0 <= ny < h and 0 <= nx < w and
                        skeleton[ny, nx] and (ny, nx) not in visited):
                    visited.add((ny, nx))
                    path.append((ny, nx))
                    current = (ny, nx)
                    found = True
                    break
            if not found:
                break

        if len(path) < 2:
            return (0, 1)  # 默认向右

        # 计算方向（从尖端到内部）
        dy = path[-1][0] - path[0][0]
        dx = path[-1][1] - path[0][1]
        length = np.sqrt(dy * dy + dx * dx)

        if length < 0.001:
            return (0, 1)

        return (dy / length, dx / length)

    def extract_all_nodes(self, skeleton):
        """
        提取所有关键节点

        返回: {
            'tips': [(y, x, direction), ...],
            'branches': [(y, x, degree), ...],
            'skeleton': skeleton
        }
        """
        return {
            'tips': self.extract_tips(skeleton),
            'branches': self.extract_branches(skeleton),
            'skeleton': skeleton
        }


# ============================================================================
#                     阶段三：时序关联（匈牙利算法）
# ============================================================================

class HungarianMatcher:
    """
    基于匈牙利算法的时序关联器

    代价函数: C_ij = α·d_spatial + β·d_direction + γ·d_prediction
    """

    def __init__(self, max_dist=None, birth_cost=None, death_cost=None):
        self.max_dist = max_dist or Config.MAX_MATCH_DIST
        self.birth_cost = birth_cost or Config.BIRTH_COST
        self.death_cost = death_cost or Config.DEATH_COST

    def compute_cost_matrix(self, tips_prev, tips_curr, trajectories=None):
        """
        计算增强版代价矩阵

        Args:
            tips_prev: 前一帧的尖端 [(y, x, direction), ...]
            tips_curr: 当前帧的尖端 [(y, x, direction), ...]
            trajectories: 历史轨迹 {id: [(y, x), ...]}

        Returns:
            代价矩阵 (n x m)
        """
        n, m = len(tips_prev), len(tips_curr)

        if n == 0 or m == 0:
            return np.zeros((n, m))

        cost = np.full((n, m), 1e9)

        for i, (y1, x1, dir1) in enumerate(tips_prev):
            for j, (y2, x2, dir2) in enumerate(tips_curr):

                # 1. 空间距离
                d_spatial = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)

                # 距离过大，不可能匹配
                if d_spatial > self.max_dist:
                    continue

                # 2. 方向一致性（点积）
                if dir1 is not None and dir2 is not None:
                    dir_consistency = dir1[0] * dir2[0] + dir1[1] * dir2[1]
                    d_direction = 1 - dir_consistency  # [0, 2]
                else:
                    d_direction = 0.5

                # 3. 运动预测偏差
                d_prediction = 0
                if trajectories is not None and i in trajectories:
                    traj = trajectories[i]
                    if len(traj) >= 3:
                        # 简单线性预测
                        vy = traj[-1][0] - traj[-2][0]
                        vx = traj[-1][1] - traj[-2][1]
                        pred_y, pred_x = y1 + vy, x1 + vx
                        d_prediction = np.sqrt((y2 - pred_y) ** 2 + (x2 - pred_x) ** 2)

                # 组合代价
                cost[i, j] = (
                        1.0 * d_spatial +
                        Config.DIRECTION_WEIGHT * d_direction +
                        Config.PREDICTION_WEIGHT * d_prediction
                )

        return cost

    def match_with_birth_death(self, tips_prev, tips_curr, trajectories=None):
        """
        带新生/消失处理的匈牙利匹配

        Args:
            tips_prev: 前一帧尖端
            tips_curr: 当前帧尖端
            trajectories: 历史轨迹

        Returns:
            matches: [(prev_idx, curr_idx), ...]
            births: [curr_idx, ...]  新出现的尖端
            deaths: [prev_idx, ...]  消失的尖端
        """
        n, m = len(tips_prev), len(tips_curr)

        if n == 0:
            return [], list(range(m)), []
        if m == 0:
            return [], [], list(range(n))

        # 计算原始代价矩阵
        cost = self.compute_cost_matrix(tips_prev, tips_curr, trajectories)

        # 扩展矩阵处理新生/消失
        # 大小: (n + m) x (m + n)
        extended = np.full((n + m, m + n), 1e9)

        # 左上: 原始代价
        extended[:n, :m] = cost

        # 右上: 消失代价（对角线）
        for i in range(n):
            extended[i, m + i] = self.death_cost

        # 左下: 新生代价（对角线）
        for j in range(m):
            extended[n + j, j] = self.birth_cost

        # 右下: 零代价
        extended[n:, m:] = 0

        # 匈牙利算法求解
        row_ind, col_ind = linear_sum_assignment(extended)

        # 解析结果
        matches = []
        births = []
        deaths = []

        for i, j in zip(row_ind, col_ind):
            if i < n and j < m:
                # 检查是否是有效匹配
                if cost[i, j] < 1e8:
                    matches.append((i, j))
                else:
                    deaths.append(i)
                    births.append(j)
            elif i < n and j >= m:
                deaths.append(i)
            elif i >= n and j < m:
                births.append(j)

        return matches, births, deaths


# ============================================================================
#                     综合追踪器
# ============================================================================

class NeuronTracker:
    """
    神经元追踪器：整合三阶段追踪
    """

    def __init__(self, img_processor):
        self.ip = img_processor
        self.topo = TopologyExtractor()
        self.matcher = HungarianMatcher()

        # 追踪状态
        self.trajectories = {}  # tip_id -> [(y, x), ...]
        self.tip_directions = {}  # tip_id -> (dy, dx)
        self.next_id = 0
        self.branch_tree = {}  # tip_id -> parent_id

    def reset(self):
        """重置追踪状态"""
        self.trajectories = {}
        self.tip_directions = {}
        self.next_id = 0
        self.branch_tree = {}

    def _compute_direction(self, traj, n=None):
        """计算轨迹最近n个点的方向"""
        if n is None:
            n = Config.DIRECTION_HISTORY
        if len(traj) < 2:
            return (0, 1)
        recent = traj[-min(n, len(traj)):]
        dy = recent[-1][0] - recent[0][0]
        dx = recent[-1][1] - recent[0][1]
        length = np.sqrt(dy * dy + dx * dx)
        return (dy / length, dx / length) if length > 0.001 else (0, 1)

    def initialize_from_markers(self, skeleton, frame, markers):
        """
        从用户标记初始化追踪

        Args:
            skeleton: 骨架图像
            frame: 原始帧
            markers: [{"x": x, "y": y}, ...]

        Returns:
            初始轨迹列表 [(y, x), ...]
        """
        if not markers:
            return []

        # 找到标记点对应的骨架点
        skel_points = []
        for m in markers:
            pt = self.ip.nearest_skeleton_point(skeleton, frame, (m["x"], m["y"]))
            if pt:
                skel_points.append(pt)

        if not skel_points:
            return []

        # 去重
        unique = []
        for p in skel_points:
            if not any(abs(p[0] - u[0]) < 5 and abs(p[1] - u[1]) < 5 for u in unique):
                unique.append(p)
        unique.sort(key=lambda p: p[1])  # 按x排序

        if len(unique) == 1:
            # 单点双向追踪
            return self._trace_bidirectional(skeleton, frame, unique[0])

        # 多点连接
        full_traj = []
        for i in range(len(unique) - 1):
            segment = self._astar_connect(skeleton, frame, unique[i], unique[i + 1])
            if segment:
                if full_traj and segment[0] == full_traj[-1]:
                    segment = segment[1:]
                full_traj.extend(segment)

        # 两端延伸
        if unique:
            left_ext = self._trace_single_direction(
                skeleton, frame, unique[0], unique[0][0],
                go_left=True, avoid=set(full_traj)
            )
            full_traj = left_ext[::-1] + full_traj

            right_ext = self._trace_single_direction(
                skeleton, frame, unique[-1], unique[-1][0],
                go_left=False, avoid=set(full_traj)
            )
            full_traj = full_traj + right_ext

        # 去重
        seen = set()
        result = []
        for p in full_traj:
            if p not in seen:
                seen.add(p)
                result.append(p)

        return sorted(result, key=lambda p: p[1])

    def track_growth(self, skeleton, frame, trajectory, y_center):
        """
        追踪神经元生长（延伸）

        Args:
            skeleton: 当前帧骨架
            frame: 当前帧
            trajectory: 当前轨迹
            y_center: Y方向中心约束

        Returns:
            新增的点列表
        """
        if not trajectory:
            return []

        return self._trace_single_direction(
            skeleton, frame,
            start=trajectory[-1],
            y_center=y_center,
            go_left=False,
            avoid=set(trajectory),
            history=trajectory
        )

    def track_frame_hungarian(self, tips_prev, tips_curr, trajectories):
        """
        使用匈牙利算法进行帧间匹配

        Args:
            tips_prev: 前一帧的尖端
            tips_curr: 当前帧的尖端
            trajectories: {nid: trajectory}

        Returns:
            matches, births, deaths
        """
        # 构建ID到索引的映射
        prev_ids = list(trajectories.keys())

        # 为前一帧尖端添加方向信息
        tips_prev_with_dir = []
        for i, nid in enumerate(prev_ids):
            traj = trajectories[nid]
            if traj:
                y, x = traj[-1]
                direction = self._compute_direction(traj)
                tips_prev_with_dir.append((y, x, direction))

        # 匹配
        matches, births, deaths = self.matcher.match_with_birth_death(
            tips_prev_with_dir, tips_curr,
            {i: trajectories[prev_ids[i]] for i in range(len(prev_ids))}
        )

        # 转换索引到ID
        match_results = [(prev_ids[i], j) for i, j in matches]
        death_results = [prev_ids[i] for i in deaths]

        return match_results, births, death_results

    def _trace_single_direction(self, skeleton, frame, start, y_center,
                                go_left=True, avoid=None, history=None):
        """单方向追踪"""
        if not start:
            return []

        h, w = skeleton.shape
        visited = set(avoid) if avoid else set()

        # 使用历史初始化方向
        if history and len(history) >= 2:
            dir_history = list(history[-10:])
        else:
            dir_history = [start]

        visited.add(start)
        current = start
        new_points = []

        while True:
            d = self._compute_direction(dir_history)
            candidates = []

            for search_radius in [1, 2, Config.MAX_GAP]:
                for dy in range(-search_radius, search_radius + 1):
                    for dx in range(-search_radius, search_radius + 1):
                        if dy == 0 and dx == 0:
                            continue

                        ny, nx = current[0] + dy, current[1] + dx

                        if not (0 <= ny < h and 0 <= nx < w):
                            continue
                        if (ny, nx) in visited:
                            continue
                        if not skeleton[ny, nx]:
                            continue
                        if not self.ip.is_green(frame, ny, nx):
                            continue
                        if abs(ny - y_center) > Config.Y_TOLERANCE * 2:
                            continue

                        dist = np.sqrt(dy * dy + dx * dx)
                        score = (-dx if go_left else dx) * 10 - abs(dy) * 3 - dist * 2

                        if dist > 0:
                            cand_dir = (dy / dist, dx / dist)
                            dir_score = d[0] * cand_dir[0] + d[1] * cand_dir[1]
                            score += dir_score * Config.DIRECTION_WEIGHT

                        candidates.append(((ny, nx), score))

                if candidates:
                    break

            if not candidates:
                break

            candidates.sort(key=lambda x: x[1], reverse=True)
            best = candidates[0][0]

            visited.add(best)
            new_points.append(best)
            dir_history.append(best)
            current = best

        return new_points

    def _trace_bidirectional(self, skeleton, frame, start):
        """双向追踪"""
        if not start:
            return []

        y_center = start[0]
        left = self._trace_single_direction(skeleton, frame, start, y_center, go_left=True)
        right = self._trace_single_direction(skeleton, frame, start, y_center, go_left=False)

        full = []
        if left:
            full = left[::-1]
        full.append(start)
        if right:
            full.extend(right)

        return full

    def _astar_connect(self, skeleton, frame, start, end):
        """A*路径连接"""
        h, w = skeleton.shape

        def heuristic(p):
            return np.sqrt((p[0] - end[0]) ** 2 + (p[1] - end[1]) ** 2)

        counter = 0
        heap = [(heuristic(start), counter, start, [start])]
        visited = {start}

        for _ in range(5000):
            if not heap:
                break

            _, _, current, path = heapq.heappop(heap)

            if current == end or heuristic(current) < 3:
                return path + [end] if current != end else path

            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    if dy == 0 and dx == 0:
                        continue

                    ny, nx = current[0] + dy, current[1] + dx

                    if not (0 <= ny < h and 0 <= nx < w):
                        continue
                    if (ny, nx) in visited:
                        continue
                    if not skeleton[ny, nx] or not self.ip.is_green(frame, ny, nx):
                        continue

                    visited.add((ny, nx))
                    counter += 1
                    heapq.heappush(heap, (
                        len(path) + heuristic((ny, nx)),
                        counter,
                        (ny, nx),
                        path + [(ny, nx)]
                    ))

        # A*失败，直接连接
        path = [start]
        steps = max(1, int(heuristic(start)))
        for i in range(1, steps + 1):
            t = i / steps
            y = int(start[0] + t * (end[0] - start[0]))
            x = int(start[1] + t * (end[1] - start[1]))
            if (y, x) != path[-1]:
                path.append((y, x))
        if path[-1] != end:
            path.append(end)

        return path


# ============================================================================
#                              数据管理
# ============================================================================

class DataManager:
    def __init__(self):
        self.marks = {}  # {nid: {"color": [...], "marks": [...]}}
        self.trajs = {}  # {nid: [(y,x), ...]}
        self.colors = gen_colors(Config.MAX_NEURONS)

    def add_mark(self, nid, fidx, x, y):
        if nid not in self.marks:
            self.marks[nid] = {"color": list(self.colors[nid % len(self.colors)]), "marks": []}
        for m in self.marks[nid]["marks"]:
            if m["frame"] == fidx and np.sqrt((m["x"] - x) ** 2 + (m["y"] - y) ** 2) < 5:
                m["x"], m["y"] = x, y
                return
        self.marks[nid]["marks"].append({"frame": fidx, "x": x, "y": y})

    def del_mark_near(self, nid, fidx, x, y, r=20):
        if nid not in self.marks:
            return
        ms = self.marks[nid]["marks"]
        best, bd = None, 1e9
        for i, m in enumerate(ms):
            if m["frame"] == fidx:
                d = np.sqrt((m["x"] - x) ** 2 + (m["y"] - y) ** 2)
                if d < r and d < bd:
                    bd, best = d, i
        if best is not None:
            ms.pop(best)

    def get_marks(self, nid):
        return self.marks[nid]["marks"] if nid in self.marks else []

    def color(self, nid):
        return tuple(self.marks[nid]["color"]) if nid in self.marks else tuple(self.colors[nid % len(self.colors)])

    def del_neuron(self, nid):
        self.marks.pop(nid, None)
        self.trajs.pop(nid, None)

    def new_id(self):
        ids = set(self.marks) | set(self.trajs)
        i = 0
        while i in ids:
            i += 1
        return i

    def save(self, path):
        d = {
            "version": "10.0",
            "marks": self.marks,
            "trajs": {str(k): [list(p) for p in v] for k, v in self.trajs.items()}
        }
        with open(path, 'w') as f:
            json.dump(d, f, indent=2)
        print(f"✓ 保存: {path}")

    def load(self, path):
        if not os.path.exists(path):
            return False
        with open(path) as f:
            d = json.load(f)
        self.marks = {int(k): v for k, v in d.get("marks", {}).items()}
        self.trajs = {int(k): [tuple(p) for p in v] for k, v in d.get("trajs", {}).items()}
        print(f"✓ 加载: {len(self.marks)}神经元")
        return True


# ============================================================================
#                              主程序
# ============================================================================

class NeuronTool:
    def __init__(self):
        self.vid = VideoHandler()
        self.ip = ImageProcessor()
        self.data = DataManager()
        self.tracker = NeuronTracker(self.ip)
        self.topo = TopologyExtractor()

        self.zoom = 1.0
        self.pan_x = self.pan_y = 0
        self.panning = False
        self.pan_start = (0, 0)

        self.cur_nid = 0
        self.mode = 'mark'
        self.use_frangi = True  # 是否使用Frangi滤波

        self.buttons = {}
        self.frame_inp = self.neuron_inp = self.active_inp = None
        self.mx = self.my = 0
        self.action = None
        self.data_path = self.out_path = None

        if HAS_TK:
            self.root = tk.Tk()
            self.root.withdraw()
        else:
            self.root = None

    @property
    def img_h(self):
        return Config.DISPLAY_HEIGHT - Config.PANEL_HEIGHT

    def _init_ui(self):
        y1, y2, y3 = self.img_h + 12, self.img_h + 55, self.img_h + 95
        bh = 36
        x = 15

        self.buttons['prev'] = Button(x, y1, 55, bh, '<Prev');
        x += 60
        self.buttons['next'] = Button(x, y1, 55, bh, 'Next>');
        x += 60
        self.buttons['p10'] = Button(x, y1, 40, bh, '-10');
        x += 45
        self.buttons['n10'] = Button(x, y1, 40, bh, '+10');
        x += 55

        self.frame_inp = InputBox(x, y1 + 2, 80, 32, "Frame:");
        x += 95

        self.buttons['zin'] = Button(x, y1, 35, bh, 'Z+');
        x += 40
        self.buttons['zout'] = Button(x, y1, 35, bh, 'Z-');
        x += 40
        self.buttons['zfit'] = Button(x, y1, 35, bh, 'Fit');
        x += 50

        self.neuron_inp = InputBox(x, y1 + 2, 60, 32, "Neuron:")
        self.neuron_inp.text = "0";
        x += 75

        self.buttons['new'] = Button(x, y1, 45, bh, 'New');
        x += 50
        self.buttons['deln'] = Button(x, y1, 50, bh, 'Del N');
        x += 55
        self.buttons['frangi'] = Button(x, y1, 60, bh, 'Frangi');
        x += 65

        x = 15
        self.buttons['mark'] = Button(x, y2, 60, bh, 'MARK');
        x += 65
        self.buttons['track'] = Button(x, y2, 60, bh, 'TRACK');
        x += 75
        self.buttons['run'] = Button(x, y2, 80, bh, '> RUN', (60, 110, 60));
        x += 85
        self.buttons['save'] = Button(x, y2, 50, bh, 'Save', (60, 110, 60));
        x += 55
        self.buttons['load'] = Button(x, y2, 50, bh, 'Load');
        x += 55
        self.buttons['clr'] = Button(x, y2, 70, bh, 'ClearAll', (60, 60, 110));
        x += 80
        self.buttons['tips'] = Button(x, y2, 55, bh, 'Tips');
        x += 60
        self.buttons['exit'] = Button(x, y2, 50, bh, 'EXIT', (60, 60, 110))

        for i in range(15):
            self.buttons[f'n{i}'] = Button(15 + i * 50, y3, 46, 28, f'N{i}')

    def s2i(self, sx, sy):
        dw, dh = int(self.vid.w * self.zoom), int(self.vid.h * self.zoom)
        ox = (Config.DISPLAY_WIDTH - dw) // 2 + self.pan_x
        oy = (self.img_h - dh) // 2 + self.pan_y
        return (sx - ox) / self.zoom, (sy - oy) / self.zoom

    def i2s(self, ix, iy):
        dw, dh = int(self.vid.w * self.zoom), int(self.vid.h * self.zoom)
        ox = (Config.DISPLAY_WIDTH - dw) // 2 + self.pan_x
        oy = (self.img_h - dh) // 2 + self.pan_y
        return int(ix * self.zoom + ox), int(iy * self.zoom + oy)

    def _fit(self):
        self.zoom = min((Config.DISPLAY_WIDTH - 40) / self.vid.w, (self.img_h - 40) / self.vid.h, 1.0)
        self.pan_x = self.pan_y = 0

    def _show_tips(self):
        """显示当前帧的尖端检测结果"""
        if self.vid.frame is None:
            return

        print("\n检测尖端...")
        skeleton = self.ip.process_frame(self.vid.frame, use_frangi=self.use_frangi)
        tips = self.topo.extract_tips(skeleton)
        branches = self.topo.extract_branches(skeleton)

        print(f"  尖端数: {len(tips)}")
        print(f"  分支点数: {len(branches)}")

        # 可视化
        vis = self.vid.frame.copy()

        # 画骨架
        skel_vis = (skeleton.astype(np.uint8) * 255)
        skel_bgr = cv2.cvtColor(skel_vis, cv2.COLOR_GRAY2BGR)
        vis = cv2.addWeighted(vis, 0.7, skel_bgr, 0.3, 0)

        # 画尖端（蓝色）
        for (y, x, direction) in tips:
            cv2.circle(vis, (x, y), 5, (255, 100, 0), -1)
            if direction:
                # 画方向箭头
                end_x = int(x + direction[1] * 20)
                end_y = int(y + direction[0] * 20)
                cv2.arrowedLine(vis, (x, y), (end_x, end_y), (255, 200, 0), 2)

        # 画分支点（红色）
        for (y, x, degree) in branches:
            cv2.circle(vis, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(vis, str(degree), (x + 7, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        cv2.putText(vis, f"Tips: {len(tips)} (blue) | Branches: {len(branches)} (red)",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow('Topology', vis)
        cv2.waitKey(1)

    def _btn_click(self, name):
        if name == 'prev':
            self.vid.read(self.vid.idx - 1)
            self.frame_inp.text = str(self.vid.idx + 1)
        elif name == 'next':
            self.vid.read(self.vid.idx + 1)
            self.frame_inp.text = str(self.vid.idx + 1)
        elif name == 'p10':
            self.vid.read(self.vid.idx - 10)
            self.frame_inp.text = str(self.vid.idx + 1)
        elif name == 'n10':
            self.vid.read(self.vid.idx + 10)
            self.frame_inp.text = str(self.vid.idx + 1)
        elif name == 'zin':
            self.zoom = min(Config.MAX_ZOOM, self.zoom * Config.ZOOM_STEP)
        elif name == 'zout':
            self.zoom = max(Config.MIN_ZOOM, self.zoom / Config.ZOOM_STEP)
        elif name == 'zfit':
            self._fit()
        elif name == 'new':
            self.cur_nid = self.data.new_id()
            self.neuron_inp.text = str(self.cur_nid)
        elif name == 'deln':
            self.data.del_neuron(self.cur_nid)
        elif name == 'frangi':
            self.use_frangi = not self.use_frangi
            print(f"Frangi滤波: {'启用' if self.use_frangi else '禁用'}")
        elif name == 'mark':
            self.mode = 'mark'
        elif name == 'track':
            self.mode = 'track'
        elif name == 'run':
            return 'run'
        elif name == 'save':
            return 'save'
        elif name == 'load':
            return 'load'
        elif name == 'tips':
            self._show_tips()
        elif name == 'clr':
            self.data.marks.clear()
            self.data.trajs.clear()
        elif name == 'exit':
            return 'exit'
        elif name.startswith('n') and name[1:].isdigit():
            self.cur_nid = int(name[1:])
            self.neuron_inp.text = str(self.cur_nid)

    def _confirm_inp(self):
        if self.frame_inp.active:
            try:
                self.vid.read(int(self.frame_inp.text) - 1)
            except:
                pass
            self.frame_inp.active = False
        if self.neuron_inp.active:
            try:
                self.cur_nid = max(0, min(int(self.neuron_inp.text), Config.MAX_NEURONS - 1))
            except:
                pass
            self.neuron_inp.text = str(self.cur_nid)
            self.neuron_inp.active = False
        self.active_inp = None

    def _mouse(self, ev, x, y, flags, _):
        self.mx, self.my = x, y
        in_panel = y >= self.img_h

        for b in self.buttons.values():
            b.hover = b.contains(x, y)
        for i in range(15):
            self.buttons[f'n{i}'].active = (i == self.cur_nid)
        self.buttons['mark'].active = (self.mode == 'mark')
        self.buttons['track'].active = (self.mode == 'track')
        self.buttons['frangi'].active = self.use_frangi

        if ev == cv2.EVENT_LBUTTONDOWN:
            if self.frame_inp.contains(x, y):
                self.active_inp = self.frame_inp
                self.frame_inp.active = True
                self.neuron_inp.active = False
                return
            if self.neuron_inp.contains(x, y):
                self.active_inp = self.neuron_inp
                self.neuron_inp.active = True
                self.frame_inp.active = False
                return
            self._confirm_inp()

            if in_panel:
                for nm, b in self.buttons.items():
                    if b.contains(x, y):
                        r = self._btn_click(nm)
                        if r:
                            self.action = r
                        return
            elif self.mode == 'mark':
                ix, iy = self.s2i(x, y)
                if 0 <= ix < self.vid.w and 0 <= iy < self.vid.h:
                    self.data.add_mark(self.cur_nid, self.vid.idx, int(ix), int(iy))
                    print(f"  + N{self.cur_nid} @ F{self.vid.idx + 1}: ({int(ix)},{int(iy)})")

        elif ev == cv2.EVENT_RBUTTONDOWN and not in_panel and self.mode == 'mark':
            ix, iy = self.s2i(x, y)
            self.data.del_mark_near(self.cur_nid, self.vid.idx, int(ix), int(iy))

        elif ev == cv2.EVENT_MBUTTONDOWN:
            self.panning = True
            self.pan_start = (x, y)
        elif ev == cv2.EVENT_MBUTTONUP:
            self.panning = False
        elif ev == cv2.EVENT_MOUSEMOVE and self.panning:
            self.pan_x += x - self.pan_start[0]
            self.pan_y += y - self.pan_start[1]
            self.pan_start = (x, y)
        elif ev == cv2.EVENT_MOUSEWHEEL and not in_panel:
            self.zoom = min(Config.MAX_ZOOM, self.zoom * 1.2) if flags > 0 else max(Config.MIN_ZOOM, self.zoom / 1.2)

    def _draw(self):
        if self.vid.frame is None:
            return None

        c = np.full((Config.DISPLAY_HEIGHT, Config.DISPLAY_WIDTH, 3), Config.COLOR_BG, np.uint8)

        dw, dh = int(self.vid.w * self.zoom), int(self.vid.h * self.zoom)
        interp = cv2.INTER_LINEAR if self.zoom < 1 else cv2.INTER_NEAREST
        resized = cv2.resize(self.vid.frame, (dw, dh), interpolation=interp)
        ox = (Config.DISPLAY_WIDTH - dw) // 2 + self.pan_x
        oy = (self.img_h - dh) // 2 + self.pan_y

        sx1, sy1 = max(0, -ox), max(0, -oy)
        sx2, sy2 = min(dw, Config.DISPLAY_WIDTH - ox), min(dh, self.img_h - oy)
        dx1, dy1 = max(0, ox), max(0, oy)
        dx2, dy2 = min(Config.DISPLAY_WIDTH, ox + dw), min(self.img_h, oy + dh)

        if sx2 > sx1 and sy2 > sy1:
            c[dy1:dy2, dx1:dx2] = resized[sy1:sy2, sx1:sx2]

        # 画标记（其他帧先画，当前帧后画）
        for nid, nd in self.data.marks.items():
            col = tuple(nd["color"])
            for m in nd["marks"]:
                sx, sy = self.i2s(m["x"], m["y"])
                if m["frame"] != self.vid.idx:
                    cv2.drawMarker(c, (sx, sy), col, cv2.MARKER_TILTED_CROSS, 8, 1)

        for nid, nd in self.data.marks.items():
            col = tuple(nd["color"])
            for m in nd["marks"]:
                if m["frame"] == self.vid.idx:
                    sx, sy = self.i2s(m["x"], m["y"])
                    if nid == self.cur_nid:
                        cv2.circle(c, (sx, sy), 10, (255, 255, 255), 2)
                    cv2.circle(c, (sx, sy), 6, col, -1)
                    cv2.putText(c, str(nid), (sx + 8, sy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)

        # track模式画轨迹
        if self.mode == 'track':
            for nid, traj in self.data.trajs.items():
                col = self.data.color(nid)
                for i in range(len(traj) - 1):
                    p1, p2 = self.i2s(traj[i][1], traj[i][0]), self.i2s(traj[i + 1][1], traj[i + 1][0])
                    if np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) < 50:
                        cv2.line(c, p1, p2, col, 2, cv2.LINE_AA)
                if traj:
                    cv2.circle(c, self.i2s(traj[-1][1], traj[-1][0]), 5, (0, 0, 255), -1)

        # 面板
        cv2.rectangle(c, (0, self.img_h), (Config.DISPLAY_WIDTH, Config.DISPLAY_HEIGHT), Config.COLOR_PANEL, -1)
        for b in self.buttons.values():
            b.draw(c)
        self.frame_inp.draw(c)
        self.neuron_inp.draw(c, self.data.color(self.cur_nid))

        # 信息
        frangi_str = "ON" if self.use_frangi else "OFF"
        info = f"Frame: {self.vid.idx + 1}/{self.vid.total} | Zoom: {self.zoom:.2f} | Mode: {self.mode.upper()} | N{self.cur_nid} | Frangi: {frangi_str}"
        cv2.putText(c, info, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 220, 0), 2, cv2.LINE_AA)

        return c

    # ==========================================================================
    #                    追踪核心（整合匈牙利算法）
    # ==========================================================================

    def run_tracking(self):
        """执行追踪（带调试）"""
        if not self.data.marks:
            print("⚠ 没有标记")
            return

        print("\n" + "=" * 60)
        print("神经元追踪 v10.0 (调试模式)")
        print("=" * 60)

        self.data.trajs.clear()

        # 找初始帧
        init_f = min(m["frame"] for nd in self.data.marks.values() for m in nd["marks"])
        print(f"  初始帧: {init_f + 1}")

        # ========== 调试：检查标记 ==========
        print("\n[DEBUG] 标记信息:")
        for nid, nd in self.data.marks.items():
            marks = nd["marks"]
            print(f"  N{nid}: {len(marks)} 个标记")
            for m in marks:
                print(f"    帧{m['frame'] + 1}: ({m['x']}, {m['y']})")

        history = {}

        for fidx in range(self.vid.total):
            self.vid.read(fidx)

            if fidx < init_f:
                history[fidx] = {}
                continue

            # 处理图像
            skeleton = self.ip.process_frame(self.vid.frame, use_frangi=self.use_frangi)

            # ========== 调试：检查骨架 ==========
            if fidx == init_f:
                skel_count = np.sum(skeleton)
                print(f"\n[DEBUG] 初始帧骨架:")
                print(f"  骨架像素数: {skel_count}")

                if skel_count == 0:
                    print("  ❌ 骨架为空！检查图像处理参数")
                    # 尝试不用Frangi
                    print("  尝试禁用Frangi...")
                    skeleton = self.ip.process_frame(self.vid.frame, use_frangi=False)
                    skel_count = np.sum(skeleton)
                    print(f"  禁用Frangi后骨架像素数: {skel_count}")

                # 初始化轨迹
                for nid in self.data.marks:
                    ms = self.data.get_marks(nid)
                    if not ms:
                        continue

                    print(f"\n[DEBUG] N{nid} 初始化:")
                    print(f"  标记数: {len(ms)}")

                    # 检查每个标记点
                    for m in ms:
                        pt = self.ip.nearest_skeleton_point(skeleton, self.vid.frame, (m["x"], m["y"]))
                        is_g = self.ip.is_green(self.vid.frame, m["y"], m["x"])
                        print(f"  标记({m['x']},{m['y']}): 绿色={is_g}, 最近骨架点={pt}")

                    traj = self.tracker.initialize_from_markers(skeleton, self.vid.frame, ms)
                    print(f"  初始轨迹长度: {len(traj)}")

                    if len(traj) >= 5:
                        traj = sorted(traj, key=lambda p: p[1])
                        self.data.trajs[nid] = traj
                        print(f"  ✓ N{nid}: 初始化成功 {len(traj)} 点")
                    else:
                        print(f"  ✗ N{nid}: 轨迹太短 ({len(traj)} < 5)，跳过")

                history[fidx] = {nid: list(t) for nid, t in self.data.trajs.items()}
            else:
                # 生长追踪
                for nid, traj in list(self.data.trajs.items()):
                    if not traj:
                        continue

                    y_center = np.mean([p[0] for p in traj])
                    new_pts = self.tracker.track_growth(skeleton, self.vid.frame, traj, y_center)

                    if new_pts:
                        traj.extend(new_pts)
                        self.data.trajs[nid] = sorted(traj, key=lambda p: p[1])

                history[fidx] = {nid: list(t) for nid, t in self.data.trajs.items()}

            if (fidx + 1) % 10 == 0:
                total_pts = sum(len(t) for t in self.data.trajs.values())
                print(f"  进度: {fidx + 1}/{self.vid.total} | 总点数: {total_pts}")

        print(f"\n✓ 追踪完成!")
        for nid, t in self.data.trajs.items():
            print(f"  N{nid}: {len(t)} 点")

        if not self.data.trajs:
            print("\n⚠ 没有成功初始化任何轨迹！")
            print("可能原因:")
            print("  1. 标记点不在绿色区域")
            print("  2. 骨架提取失败")
            print("  3. GREEN_THRESHOLD 设置过高")
            return

        self._play_animation(history)

    def _play_animation(self, history):
        """播放生长动画"""
        print("\n播放生长动画 (Q=退出 Space=暂停 A/D=逐帧 R=重置)")

        win = 'Growth Animation'
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 1280, 720)

        self.vid.start_write(self.out_path)

        delay = max(1, int(1000 / self.vid.fps))
        fidx = 0
        paused = False

        while True:
            self.vid.read(fidx)
            vis = self.vid.frame.copy()

            ft = history.get(fidx, {})

            for nid, traj in ft.items():
                if len(traj) < 2:
                    continue
                col = self.data.color(nid)

                pts = np.array([(int(p[1]), int(p[0])) for p in traj], np.int32)
                cv2.polylines(vis, [pts], False, col, 2, cv2.LINE_AA)

                cv2.circle(vis, (int(traj[0][1]), int(traj[0][0])), 5, (0, 255, 0), -1)
                cv2.circle(vis, (int(traj[-1][1]), int(traj[-1][0])), 7, (0, 0, 255), -1)
                cv2.putText(vis, f"N{nid}", (int(traj[-1][1]) + 10, int(traj[-1][0]) + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)

            total_pts = sum(len(t) for t in ft.values())
            info = f"Frame {fidx + 1}/{self.vid.total} | Neurons: {len(ft)} | Points: {total_pts}"
            cv2.putText(vis, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(vis, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

            self.vid.write(vis)

            scale = min(1.0, 1280 / self.vid.w)
            disp = cv2.resize(vis, None, fx=scale, fy=scale)
            st = "PAUSED" if paused else "PLAYING"
            cv2.putText(disp, f"{st} | Q=exit Space=pause A/D=step R=reset", (10, disp.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.imshow(win, disp)

            key = cv2.waitKey(delay if not paused else 30) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord(' '):
                paused = not paused
            elif key == ord('a'):
                fidx = max(0, fidx - 1)
                paused = True
            elif key == ord('d'):
                fidx = min(self.vid.total - 1, fidx + 1)
                paused = True
            elif key == ord('r'):
                fidx = 0

            if not paused:
                fidx += 1
                if fidx >= self.vid.total:
                    fidx = 0

        self.vid.stop_write()
        cv2.destroyWindow(win)
        print(f"\n✓ 视频已保存: {self.out_path}")

    # ==========================================================================
    #                    主循环
    # ==========================================================================

    def run(self, video_path, data_path=None):
        self.vid.load(video_path)
        self.data_path = data_path or os.path.splitext(video_path)[0] + "_data.json"
        self.out_path = os.path.splitext(video_path)[0] + "_tracked.mp4"

        if os.path.exists(self.data_path):
            self.data.load(self.data_path)

        self._init_ui()
        self.vid.read(0)
        self.frame_inp.text = "1"
        self._fit()

        win = 'Neuron Tool v10'
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, Config.DISPLAY_WIDTH, Config.DISPLAY_HEIGHT)
        cv2.setMouseCallback(win, self._mouse)

        print("\n" + "=" * 60)
        print("神经元追踪工具 v10.0")
        print("=" * 60)
        print("新特性: Frangi滤波 | 尖端检测 | 匈牙利匹配")
        print("-" * 60)
        print("左键=标记 | 右键=删除 | 滚轮=缩放 | 中键=平移")
        print("A/D=帧 | 0-9=神经元 | RUN=追踪 | Tips=查看尖端")
        print("=" * 60 + "\n")

        while True:
            disp = self._draw()
            if disp is not None:
                cv2.imshow(win, disp)

            if self.action == 'run':
                self.run_tracking()
                self.data.save(self.data_path)
                self.action = None
            elif self.action == 'save':
                self.data.save(self.data_path)
                self.action = None
            elif self.action == 'load':
                if HAS_TK:
                    p = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
                    if p:
                        self.data_path = p
                        self.data.load(p)
                self.action = None
            elif self.action == 'exit':
                break

            key = cv2.waitKey(30) & 0xFF

            if self.active_inp:
                r = self.active_inp.handle_key(key)
                if r == 'confirm':
                    self._confirm_inp()
                elif r == 'cancel':
                    self.active_inp.active = False
                    self.active_inp = None
                continue

            if key == 27:
                break
            elif key == ord('a'):
                self.vid.read(self.vid.idx - 1)
                self.frame_inp.text = str(self.vid.idx + 1)
            elif key == ord('d'):
                self.vid.read(self.vid.idx + 1)
                self.frame_inp.text = str(self.vid.idx + 1)
            elif key == ord('w'):
                self.vid.read(self.vid.idx + 10)
                self.frame_inp.text = str(self.vid.idx + 1)
            elif key == ord('s'):
                self.vid.read(self.vid.idx - 10)
                self.frame_inp.text = str(self.vid.idx + 1)
            elif ord('0') <= key <= ord('9'):
                self.cur_nid = key - ord('0')
                self.neuron_inp.text = str(self.cur_nid)
            elif key == ord('f'):
                self.use_frangi = not self.use_frangi
                print(f"Frangi滤波: {'启用' if self.use_frangi else '禁用'}")
            elif key == ord('t'):
                self._show_tips()

        cv2.destroyAllWindows()
        self.vid.release()
        if self.root:
            self.root.destroy()


# ============================================================================
#                              入口
# ============================================================================

if __name__ == "__main__":
    VIDEO_PATH = r"/neuron_growth_50.mp4"

    tool = NeuronTool()
    tool.run(VIDEO_PATH)
