"""
神经元追踪工具 v10.1
真正实现：帧间骨架匹配 + 轨迹连续延伸
"""

import cv2
import numpy as np
import json
import os
import heapq
import warnings
from collections import defaultdict

from skimage.morphology import skeletonize, remove_small_objects
from skimage.filters import frangi
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import convolve

warnings.filterwarnings('ignore', category=FutureWarning)

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
    GREEN_THRESHOLD = 30  # 降低阈值，更容易检测
    CLAHE_CLIP_LIMIT = 3.0
    MEDIAN_KERNEL = 5
    MIN_OBJECT_SIZE = 30

    # Frangi 滤波
    FRANGI_SCALE_RANGE = (1, 4)
    FRANGI_SCALE_STEP = 1

    # 时空关联参数
    MAX_MATCH_DIST = 60  # 最大匹配距离（像素）
    DIRECTION_WEIGHT = 10.0  # 方向一致性权重
    BIRTH_COST = 100  # 新生代价
    DEATH_COST = 100  # 消失代价

    # 骨架追踪
    MAX_GAP = 10
    Y_TOLERANCE = 40
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
#                              UI
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
#                              视频
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
#                     阶段一：图像处理
# ============================================================================

class ImageProcessor:
    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=Config.CLAHE_CLIP_LIMIT, tileGridSize=(8, 8))
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    def is_green(self, f, y, x):
        if not (0 <= y < f.shape[0] and 0 <= x < f.shape[1]):
            return False
        b, g, r = int(f[y, x, 0]), int(f[y, x, 1]), int(f[y, x, 2])
        return g > 40 and g - r > Config.GREEN_THRESHOLD and g - b > Config.GREEN_THRESHOLD

    def extract_skeleton(self, frame, use_frangi=True):
        """提取骨架"""
        # 分离通道
        b, g, r = cv2.split(frame)
        b, g, r = b.astype(np.int16), g.astype(np.int16), r.astype(np.int16)

        # 绿色掩码
        mask = (g > 40) & ((g - r) > Config.GREEN_THRESHOLD) & ((g - b) > Config.GREEN_THRESHOLD)
        green_mask = mask.astype(np.uint8) * 255

        # 增强
        enhanced = self.clahe.apply(frame[:, :, 1])

        if use_frangi:
            img_norm = enhanced.astype(np.float64) / 255.0
            try:
                frangi_resp = frangi(img_norm, sigmas=range(1, 4), black_ridges=False)
            except:
                frangi_resp = frangi(img_norm, scale_range=(1, 4), scale_step=1, black_ridges=False)
            frangi_norm = (frangi_resp / (frangi_resp.max() + 1e-8) * 255).astype(np.uint8)
            combined = cv2.bitwise_and(frangi_norm, green_mask)
            _, binary = cv2.threshold(combined, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            denoised = cv2.medianBlur(enhanced, Config.MEDIAN_KERNEL)
            binary = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, 15, -3)
            binary = cv2.bitwise_and(binary, green_mask)

        # 清理
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, self.kernel, iterations=2)
        cleaned = remove_small_objects(closed > 0, min_size=Config.MIN_OBJECT_SIZE)
        skeleton = skeletonize(cleaned)

        return skeleton, binary

    def find_nearest_skeleton(self, skeleton, frame, point, radius=35):
        """找最近的骨架点"""
        h, w = skeleton.shape
        px, py = int(point[0]), int(point[1])
        best, bd = None, float('inf')

        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                ny, nx = py + dy, px + dx
                if 0 <= ny < h and 0 <= nx < w and skeleton[ny, nx]:
                    if self.is_green(frame, ny, nx):
                        d = dy * dy + dx * dx
                        if d < bd:
                            bd, best = d, (ny, nx)
        return best


# ============================================================================
#                     阶段二：拓扑提取
# ============================================================================

class TopologyExtractor:
    """提取骨架的拓扑结构：尖端和分支点"""

    NEIGHBORS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    def __init__(self):
        self.degree_kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)

    def compute_degree(self, skeleton):
        """计算每个骨架像素的度"""
        skel = skeleton.astype(np.uint8)
        degree = convolve(skel, self.degree_kernel, mode='constant', cval=0)
        return degree * skel

    def extract_tips(self, skeleton):
        """提取尖端 (度=1的点)"""
        degree = self.compute_degree(skeleton)
        coords = np.argwhere(degree == 1)

        tips = []
        for (y, x) in coords:
            direction = self._compute_direction(skeleton, y, x)
            tips.append({'y': y, 'x': x, 'dir': direction})
        return tips

    def extract_branches(self, skeleton):
        """提取分支点 (度>=3的点)"""
        degree = self.compute_degree(skeleton)
        coords = np.argwhere(degree >= 3)
        return [{'y': y, 'x': x, 'degree': int(degree[y, x])} for (y, x) in coords]

    def _compute_direction(self, skeleton, y, x, depth=8):
        """计算尖端的方向向量"""
        h, w = skeleton.shape
        path = [(y, x)]
        visited = {(y, x)}
        current = (y, x)

        for _ in range(depth):
            found = False
            for dy, dx in self.NEIGHBORS:
                ny, nx = current[0] + dy, current[1] + dx
                if 0 <= ny < h and 0 <= nx < w and skeleton[ny, nx] and (ny, nx) not in visited:
                    visited.add((ny, nx))
                    path.append((ny, nx))
                    current = (ny, nx)
                    found = True
                    break
            if not found:
                break

        if len(path) < 2:
            return (0, 1)

        dy = path[-1][0] - path[0][0]
        dx = path[-1][1] - path[0][1]
        ln = np.sqrt(dy * dy + dx * dx)
        return (dy / ln, dx / ln) if ln > 0.001 else (0, 1)


# ============================================================================
#                     阶段三：时空关联追踪器（核心！）
# ============================================================================

class SpatiotemporalTracker:
    """
    时空关联追踪器

    核心算法：
    1. 每帧提取尖端
    2. 使用匈牙利算法匹配 t 帧和 t+1 帧的尖端
    3. 对于匹配的尖端，沿骨架追踪生长部分
    4. 轨迹 = 前一帧轨迹 + 新生长的点
    """

    def __init__(self, img_proc, topo):
        self.ip = img_proc
        self.topo = topo

        # 追踪状态
        self.trajectories = {}  # nid -> [(y, x), ...]
        self.active_tips = {}  # nid -> {'y': y, 'x': x, 'dir': direction}
        self.next_id = 0

    def reset(self):
        self.trajectories = {}
        self.active_tips = {}
        self.next_id = 0

    def initialize_from_markers(self, skeleton, frame, markers, nid):
        """
        从用户标记初始化一条轨迹
        """
        if not markers:
            return None

        # 找骨架点
        skel_pts = []
        for m in markers:
            pt = self.ip.find_nearest_skeleton(skeleton, frame, (m["x"], m["y"]))
            if pt:
                skel_pts.append(pt)

        if not skel_pts:
            return None

        # 去重排序
        unique = []
        for p in skel_pts:
            if not any(abs(p[0] - u[0]) < 5 and abs(p[1] - u[1]) < 5 for u in unique):
                unique.append(p)
        unique.sort(key=lambda p: p[1])  # 按x排序

        # 从标记点追踪完整轨迹
        if len(unique) == 1:
            traj = self._trace_bidirectional(skeleton, frame, unique[0])
        else:
            traj = self._connect_and_extend(skeleton, frame, unique)

        if len(traj) < 5:
            return None

        traj = sorted(traj, key=lambda p: p[1])

        # 存储轨迹和尖端
        self.trajectories[nid] = traj

        # 找到轨迹的右端尖端
        right_tip = traj[-1]
        direction = self._compute_traj_direction(traj)
        self.active_tips[nid] = {'y': right_tip[0], 'x': right_tip[1], 'dir': direction}

        return traj

    def track_frame(self, skeleton_prev, skeleton_curr, frame_curr, tips_curr):
        """
        帧间追踪（核心！）

        Args:
            skeleton_prev: 前一帧骨架（用于对比）
            skeleton_curr: 当前帧骨架
            frame_curr: 当前帧图像
            tips_curr: 当前帧的尖端列表

        Returns:
            更新后的轨迹
        """
        if not self.trajectories:
            return {}

        # ========== Step 1: 匈牙利匹配 ==========
        # 将前一帧的活跃尖端与当前帧的尖端匹配

        prev_tips = list(self.active_tips.items())  # [(nid, tip_info), ...]

        if not prev_tips or not tips_curr:
            # 没有可匹配的，直接沿骨架延伸
            return self._extend_all_trajectories(skeleton_curr, frame_curr)

        # 计算代价矩阵
        n = len(prev_tips)
        m = len(tips_curr)
        cost = np.full((n, m), 1e9)

        for i, (nid, prev_tip) in enumerate(prev_tips):
            traj = self.trajectories.get(nid, [])
            prev_dir = prev_tip['dir']

            for j, curr_tip in enumerate(tips_curr):
                # 空间距离
                dy = curr_tip['y'] - prev_tip['y']
                dx = curr_tip['x'] - prev_tip['x']
                d_spatial = np.sqrt(dy * dy + dx * dx)

                if d_spatial > Config.MAX_MATCH_DIST:
                    continue

                # 方向一致性
                curr_dir = curr_tip['dir']
                if prev_dir and curr_dir:
                    dot = prev_dir[0] * curr_dir[0] + prev_dir[1] * curr_dir[1]
                    d_dir = 1 - dot  # [0, 2]
                else:
                    d_dir = 0.5

                # 运动预测：尖端应该沿着生长方向移动
                if prev_dir and d_spatial > 0:
                    move_dir = (dy / d_spatial, dx / d_spatial)
                    align = prev_dir[0] * move_dir[0] + prev_dir[1] * move_dir[1]
                    d_motion = 1 - max(0, align)  # 不沿方向移动则惩罚
                else:
                    d_motion = 0

                cost[i, j] = d_spatial + Config.DIRECTION_WEIGHT * d_dir + 5 * d_motion

        # 扩展矩阵处理新生/消失
        extended = np.full((n + m, m + n), 1e9)
        extended[:n, :m] = cost
        for i in range(n):
            extended[i, m + i] = Config.DEATH_COST
        for j in range(m):
            extended[n + j, j] = Config.BIRTH_COST
        extended[n:, m:] = 0

        # 匈牙利算法
        row_ind, col_ind = linear_sum_assignment(extended)

        # 解析匹配结果
        matches = {}  # nid -> curr_tip_idx
        for i, j in zip(row_ind, col_ind):
            if i < n and j < m and cost[i, j] < 1e8:
                nid = prev_tips[i][0]
                matches[nid] = j

        # ========== Step 2: 更新轨迹 ==========
        for nid, traj in list(self.trajectories.items()):
            if nid in matches:
                # 匹配成功：追踪从旧尖端到新尖端的路径
                curr_tip = tips_curr[matches[nid]]
                new_tip_pos = (curr_tip['y'], curr_tip['x'])

                # 沿骨架从旧尖端追踪到新尖端
                old_tip = (self.active_tips[nid]['y'], self.active_tips[nid]['x'])
                growth = self._trace_growth(skeleton_curr, frame_curr, old_tip, new_tip_pos, traj)

                if growth:
                    traj.extend(growth)
                    self.trajectories[nid] = sorted(traj, key=lambda p: p[1])

                # 更新活跃尖端
                self.active_tips[nid] = curr_tip
            else:
                # 未匹配：尝试直接沿骨架延伸
                growth = self._extend_trajectory(skeleton_curr, frame_curr, traj)
                if growth:
                    traj.extend(growth)
                    self.trajectories[nid] = sorted(traj, key=lambda p: p[1])
                    # 更新尖端
                    new_tip = traj[-1]
                    direction = self._compute_traj_direction(traj)
                    self.active_tips[nid] = {'y': new_tip[0], 'x': new_tip[1], 'dir': direction}

        return self.trajectories

    def _trace_growth(self, skeleton, frame, start, end, existing_traj):
        """
        追踪从 start 到 end 的生长路径

        使用 A* 沿骨架搜索
        """
        h, w = skeleton.shape
        avoid = set(existing_traj)

        # 如果 end 不在骨架上，找最近的骨架点
        if not skeleton[end[0], end[1]]:
            nearest = self.ip.find_nearest_skeleton(skeleton, frame, (end[1], end[0]), radius=15)
            if nearest:
                end = nearest
            else:
                return []

        # A* 搜索
        def heur(p):
            return np.sqrt((p[0] - end[0]) ** 2 + (p[1] - end[1]) ** 2)

        heap = [(heur(start), 0, start, [])]
        visited = {start}

        for _ in range(3000):
            if not heap:
                break

            _, _, cur, path = heapq.heappop(heap)

            if cur == end or heur(cur) < 3:
                return path + [end] if cur != end else path

            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = cur[0] + dy, cur[1] + dx
                    if not (0 <= ny < h and 0 <= nx < w):
                        continue
                    if (ny, nx) in visited or (ny, nx) in avoid:
                        continue
                    if not skeleton[ny, nx]:
                        continue

                    visited.add((ny, nx))
                    new_path = path + [(ny, nx)]
                    heapq.heappush(heap, (len(new_path) + heur((ny, nx)), len(new_path), (ny, nx), new_path))

        # A* 失败，尝试直接延伸
        return self._extend_trajectory(skeleton, frame, existing_traj)

    def _extend_trajectory(self, skeleton, frame, traj):
        """直接沿骨架延伸轨迹"""
        if not traj:
            return []

        h, w = skeleton.shape
        avoid = set(traj)
        current = traj[-1]
        y_center = np.mean([p[0] for p in traj[-20:]])
        direction = self._compute_traj_direction(traj)

        growth = []

        for _ in range(200):  # 最多延伸200个点
            candidates = []

            for sr in [1, 2, Config.MAX_GAP]:
                for dy in range(-sr, sr + 1):
                    for dx in range(-sr, sr + 1):
                        if dy == 0 and dx == 0:
                            continue
                        ny, nx = current[0] + dy, current[1] + dx
                        if not (0 <= ny < h and 0 <= nx < w):
                            continue
                        if (ny, nx) in avoid:
                            continue
                        if not skeleton[ny, nx]:
                            continue
                        if not self.ip.is_green(frame, ny, nx):
                            continue
                        if abs(ny - y_center) > Config.Y_TOLERANCE * 2:
                            continue

                        dist = np.sqrt(dy * dy + dx * dx)
                        score = dx * 10 - abs(dy) * 3 - dist * 2  # 向右优先

                        if dist > 0 and direction:
                            cd = (dy / dist, dx / dist)
                            score += (direction[0] * cd[0] + direction[1] * cd[1]) * Config.DIRECTION_WEIGHT

                        candidates.append(((ny, nx), score))

                if candidates:
                    break

            if not candidates:
                break

            candidates.sort(key=lambda x: x[1], reverse=True)
            best = candidates[0][0]
            avoid.add(best)
            growth.append(best)
            current = best

            # 更新方向
            if len(growth) >= 3:
                direction = self._compute_traj_direction(traj + growth)

        return growth

    def _extend_all_trajectories(self, skeleton, frame):
        """延伸所有轨迹"""
        for nid, traj in list(self.trajectories.items()):
            growth = self._extend_trajectory(skeleton, frame, traj)
            if growth:
                traj.extend(growth)
                self.trajectories[nid] = sorted(traj, key=lambda p: p[1])
                new_tip = traj[-1]
                direction = self._compute_traj_direction(traj)
                self.active_tips[nid] = {'y': new_tip[0], 'x': new_tip[1], 'dir': direction}
        return self.trajectories

    def _trace_bidirectional(self, skeleton, frame, start):
        """双向追踪"""
        left = self._trace_one_direction(skeleton, frame, start, go_left=True)
        right = self._trace_one_direction(skeleton, frame, start, go_left=False)
        return left[::-1] + [start] + right

    def _trace_one_direction(self, skeleton, frame, start, go_left=True, avoid=None):
        """单方向追踪"""
        h, w = skeleton.shape
        visited = set(avoid) if avoid else set()
        visited.add(start)
        current = start
        path = []
        y_center = start[0]

        for _ in range(500):
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
                        if not self.ip.is_green(frame, ny, nx):
                            continue
                        if abs(ny - y_center) > Config.Y_TOLERANCE * 2:
                            continue

                        dist = np.sqrt(dy * dy + dx * dx)
                        score = (-dx if go_left else dx) * 10 - abs(dy) * 3 - dist * 2
                        candidates.append(((ny, nx), score))

                if candidates:
                    break

            if not candidates:
                break

            candidates.sort(key=lambda x: x[1], reverse=True)
            best = candidates[0][0]
            visited.add(best)
            path.append(best)
            current = best

        return path

    def _connect_and_extend(self, skeleton, frame, points):
        """连接多个点并延伸"""
        traj = []
        for i in range(len(points) - 1):
            seg = self._astar_path(skeleton, frame, points[i], points[i + 1])
            if traj and seg and seg[0] == traj[-1]:
                seg = seg[1:]
            traj.extend(seg)

        # 两端延伸
        left = self._trace_one_direction(skeleton, frame, points[0], go_left=True, avoid=set(traj))
        right = self._trace_one_direction(skeleton, frame, points[-1], go_left=False, avoid=set(traj))

        return left[::-1] + traj + right

    def _astar_path(self, skeleton, frame, start, end):
        """A*路径搜索"""
        h, w = skeleton.shape
        heur = lambda p: np.sqrt((p[0] - end[0]) ** 2 + (p[1] - end[1]) ** 2)

        heap = [(heur(start), 0, start, [start])]
        visited = {start}

        for _ in range(3000):
            if not heap:
                break
            _, _, cur, path = heapq.heappop(heap)
            if cur == end or heur(cur) < 3:
                return path + [end] if cur != end else path

            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = cur[0] + dy, cur[1] + dx
                    if not (0 <= ny < h and 0 <= nx < w):
                        continue
                    if (ny, nx) in visited:
                        continue
                    if not skeleton[ny, nx]:
                        continue
                    visited.add((ny, nx))
                    heapq.heappush(heap, (len(path) + heur((ny, nx)), len(path), (ny, nx), path + [(ny, nx)]))

        # 失败则直连
        return [start, end]

    def _compute_traj_direction(self, traj, n=10):
        """计算轨迹方向"""
        if len(traj) < 2:
            return (0, 1)
        recent = traj[-min(n, len(traj)):]
        dy = recent[-1][0] - recent[0][0]
        dx = recent[-1][1] - recent[0][1]
        ln = np.sqrt(dy * dy + dx * dx)
        return (dy / ln, dx / ln) if ln > 0.001 else (0, 1)


# ============================================================================
#                              数据管理
# ============================================================================

class DataManager:
    def __init__(self):
        self.marks = {}
        self.trajs = {}
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
        if nid not in self.marks: return
        ms = self.marks[nid]["marks"]
        best, bd = None, 1e9
        for i, m in enumerate(ms):
            if m["frame"] == fidx:
                d = np.sqrt((m["x"] - x) ** 2 + (m["y"] - y) ** 2)
                if d < r and d < bd: bd, best = d, i
        if best is not None: ms.pop(best)

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
        while i in ids: i += 1
        return i

    def save(self, path):
        d = {"version": "10.1", "marks": self.marks,
             "trajs": {str(k): [list(p) for p in v] for k, v in self.trajs.items()}}
        with open(path, 'w') as f: json.dump(d, f, indent=2)
        print(f"✓ 保存: {path}")

    def load(self, path):
        if not os.path.exists(path): return False
        with open(path) as f: d = json.load(f)
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
        self.topo = TopologyExtractor()
        self.data = DataManager()
        self.tracker = SpatiotemporalTracker(self.ip, self.topo)

        self.zoom = 1.0
        self.pan_x = self.pan_y = 0
        self.panning = False
        self.pan_start = (0, 0)

        self.cur_nid = 0
        self.mode = 'mark'
        self.use_frangi = True

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
        self.buttons['frangi'] = Button(x, y1, 60, bh, 'Frangi')

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
        self.buttons['debug'] = Button(x, y2, 55, bh, 'Debug');
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

    def _debug_frame(self):
        """调试当前帧"""
        if self.vid.frame is None:
            return

        print("\n" + "=" * 50)
        print(f"[DEBUG] 帧 {self.vid.idx + 1}")
        print("=" * 50)

        # 提取骨架
        skeleton, binary = self.ip.extract_skeleton(self.vid.frame, use_frangi=self.use_frangi)
        skel_count = np.sum(skeleton)
        print(f"  二值图白点数: {np.sum(binary > 0)}")
        print(f"  骨架像素数: {skel_count}")

        # 提取尖端
        tips = self.topo.extract_tips(skeleton)
        branches = self.topo.extract_branches(skeleton)
        print(f"  尖端数: {len(tips)}")
        print(f"  分支点数: {len(branches)}")

        # 检查标记
        for nid, nd in self.data.marks.items():
            for m in nd["marks"]:
                if m["frame"] == self.vid.idx:
                    pt = self.ip.find_nearest_skeleton(skeleton, self.vid.frame, (m["x"], m["y"]))
                    is_g = self.ip.is_green(self.vid.frame, m["y"], m["x"])
                    print(f"  标记N{nid}({m['x']},{m['y']}): 绿色={is_g}, 骨架点={pt}")

        # 可视化
        vis = self.vid.frame.copy()
        skel_vis = (skeleton.astype(np.uint8) * 255)
        skel_bgr = cv2.cvtColor(skel_vis, cv2.COLOR_GRAY2BGR)
        skel_bgr[:, :, 0] = 0  # 只保留绿色
        skel_bgr[:, :, 2] = 0
        vis = cv2.addWeighted(vis, 0.7, skel_bgr, 0.5, 0)

        for t in tips:
            cv2.circle(vis, (t['x'], t['y']), 5, (255, 100, 0), -1)
            if t['dir']:
                ex, ey = int(t['x'] + t['dir'][1] * 20), int(t['y'] + t['dir'][0] * 20)
                cv2.arrowedLine(vis, (t['x'], t['y']), (ex, ey), (255, 200, 0), 2)

        for b in branches:
            cv2.circle(vis, (b['x'], b['y']), 5, (0, 0, 255), -1)

        cv2.putText(vis, f"Skel: {skel_count} | Tips: {len(tips)} | Branches: {len(branches)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow('Debug', vis)
        cv2.waitKey(1)

    def _btn_click(self, name):
        if name == 'prev':
            self.vid.read(self.vid.idx - 1); self.frame_inp.text = str(self.vid.idx + 1)
        elif name == 'next':
            self.vid.read(self.vid.idx + 1); self.frame_inp.text = str(self.vid.idx + 1)
        elif name == 'p10':
            self.vid.read(self.vid.idx - 10); self.frame_inp.text = str(self.vid.idx + 1)
        elif name == 'n10':
            self.vid.read(self.vid.idx + 10); self.frame_inp.text = str(self.vid.idx + 1)
        elif name == 'zin':
            self.zoom = min(Config.MAX_ZOOM, self.zoom * Config.ZOOM_STEP)
        elif name == 'zout':
            self.zoom = max(Config.MIN_ZOOM, self.zoom / Config.ZOOM_STEP)
        elif name == 'zfit':
            self._fit()
        elif name == 'new':
            self.cur_nid = self.data.new_id(); self.neuron_inp.text = str(self.cur_nid)
        elif name == 'deln':
            self.data.del_neuron(self.cur_nid)
        elif name == 'frangi':
            self.use_frangi = not self.use_frangi; print(f"Frangi: {'ON' if self.use_frangi else 'OFF'}")
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
        elif name == 'debug':
            self._debug_frame()
        elif name == 'clr':
            self.data.marks.clear(); self.data.trajs.clear()
        elif name == 'exit':
            return 'exit'
        elif name.startswith('n') and name[1:].isdigit():
            self.cur_nid = int(name[1:]);
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

        for b in self.buttons.values(): b.hover = b.contains(x, y)
        for i in range(15): self.buttons[f'n{i}'].active = (i == self.cur_nid)
        self.buttons['mark'].active = (self.mode == 'mark')
        self.buttons['track'].active = (self.mode == 'track')
        self.buttons['frangi'].active = self.use_frangi

        if ev == cv2.EVENT_LBUTTONDOWN:
            if self.frame_inp.contains(x, y):
                self.active_inp = self.frame_inp;
                self.frame_inp.active = True;
                self.neuron_inp.active = False;
                return
            if self.neuron_inp.contains(x, y):
                self.active_inp = self.neuron_inp;
                self.neuron_inp.active = True;
                self.frame_inp.active = False;
                return
            self._confirm_inp()
            if in_panel:
                for nm, b in self.buttons.items():
                    if b.contains(x, y):
                        r = self._btn_click(nm)
                        if r: self.action = r
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
            self.panning = True;
            self.pan_start = (x, y)
        elif ev == cv2.EVENT_MBUTTONUP:
            self.panning = False
        elif ev == cv2.EVENT_MOUSEMOVE and self.panning:
            self.pan_x += x - self.pan_start[0];
            self.pan_y += y - self.pan_start[1];
            self.pan_start = (x, y)
        elif ev == cv2.EVENT_MOUSEWHEEL and not in_panel:
            self.zoom = min(Config.MAX_ZOOM, self.zoom * 1.2) if flags > 0 else max(Config.MIN_ZOOM, self.zoom / 1.2)

    def _draw(self):
        if self.vid.frame is None: return None
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

        # 画标记
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
                    if nid == self.cur_nid: cv2.circle(c, (sx, sy), 10, (255, 255, 255), 2)
                    cv2.circle(c, (sx, sy), 6, col, -1)
                    cv2.putText(c, str(nid), (sx + 8, sy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)

        # 画轨迹
        if self.mode == 'track':
            for nid, traj in self.data.trajs.items():
                col = self.data.color(nid)
                for i in range(len(traj) - 1):
                    p1, p2 = self.i2s(traj[i][1], traj[i][0]), self.i2s(traj[i + 1][1], traj[i + 1][0])
                    if np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) < 50:
                        cv2.line(c, p1, p2, col, 2, cv2.LINE_AA)
                if traj:
                    cv2.circle(c, self.i2s(traj[0][1], traj[0][0]), 4, (0, 255, 0), -1)
                    cv2.circle(c, self.i2s(traj[-1][1], traj[-1][0]), 5, (0, 0, 255), -1)

        # 面板
        cv2.rectangle(c, (0, self.img_h), (Config.DISPLAY_WIDTH, Config.DISPLAY_HEIGHT), Config.COLOR_PANEL, -1)
        for b in self.buttons.values(): b.draw(c)
        self.frame_inp.draw(c)
        self.neuron_inp.draw(c, self.data.color(self.cur_nid))

        info = f"Frame: {self.vid.idx + 1}/{self.vid.total} | Zoom: {self.zoom:.2f} | Mode: {self.mode.upper()} | N{self.cur_nid} | Frangi: {'ON' if self.use_frangi else 'OFF'}"
        cv2.putText(c, info, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 220, 0), 2, cv2.LINE_AA)
        return c

    # ==========================================================================
    #                    追踪核心（时空关联版）
    # ==========================================================================

    def run_tracking(self):
        """时空关联追踪"""
        if not self.data.marks:
            print("⚠ 没有标记")
            return

        print("\n" + "=" * 60)
        print("神经元追踪 v10.1 (时空关联)")
        print("=" * 60)

        self.data.trajs.clear()
        self.tracker.reset()

        # 找初始帧
        init_f = min(m["frame"] for nd in self.data.marks.values() for m in nd["marks"])
        print(f"  初始帧: {init_f + 1}")

        # 轨迹历史
        history = {}
        skeleton_prev = None

        for fidx in range(self.vid.total):
            self.vid.read(fidx)

            if fidx < init_f:
                history[fidx] = {}
                continue

            # 提取骨架和尖端
            skeleton, _ = self.ip.extract_skeleton(self.vid.frame, use_frangi=self.use_frangi)
            tips = self.topo.extract_tips(skeleton)

            if fidx == init_f:
                # ===== 初始化 =====
                print(f"\n[初始化] 骨架: {np.sum(skeleton)} 像素, 尖端: {len(tips)}")

                for nid in self.data.marks:
                    ms = self.data.get_marks(nid)
                    if not ms:
                        continue

                    traj = self.tracker.initialize_from_markers(skeleton, self.vid.frame, ms, nid)
                    if traj:
                        self.data.trajs[nid] = traj
                        print(f"  ✓ N{nid}: {len(traj)} 点")
                    else:
                        print(f"  ✗ N{nid}: 初始化失败")

                history[fidx] = {nid: list(t) for nid, t in self.tracker.trajectories.items()}
            else:
                # ===== 时空关联追踪 =====
                self.tracker.track_frame(skeleton_prev, skeleton, self.vid.frame, tips)

                # 更新数据
                for nid, traj in self.tracker.trajectories.items():
                    self.data.trajs[nid] = traj

                history[fidx] = {nid: list(t) for nid, t in self.tracker.trajectories.items()}

            skeleton_prev = skeleton

            if (fidx + 1) % 10 == 0:
                total = sum(len(t) for t in self.tracker.trajectories.values())
                print(f"  进度: {fidx + 1}/{self.vid.total} | 总点数: {total}")

        print(f"\n✓ 追踪完成!")
        for nid, t in self.data.trajs.items():
            print(f"  N{nid}: {len(t)} 点")

        if not self.data.trajs:
            print("\n⚠ 追踪失败！请按 Debug 检查骨架提取")
            return

        self._play_animation(history)

    def _play_animation(self, history):
        """播放动画"""
        print("\n播放 (Q=退出 Space=暂停 A/D=逐帧)")

        win = 'Growth'
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
                if len(traj) < 2: continue
                col = self.data.color(nid)

                pts = np.array([(int(p[1]), int(p[0])) for p in traj], np.int32)
                cv2.polylines(vis, [pts], False, col, 2, cv2.LINE_AA)

                cv2.circle(vis, (int(traj[0][1]), int(traj[0][0])), 5, (0, 255, 0), -1)
                cv2.circle(vis, (int(traj[-1][1]), int(traj[-1][0])), 7, (0, 0, 255), -1)
                cv2.putText(vis, f"N{nid}", (int(traj[-1][1]) + 10, int(traj[-1][0]) + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)

            total = sum(len(t) for t in ft.values())
            info = f"Frame {fidx + 1}/{self.vid.total} | Neurons: {len(ft)} | Points: {total}"
            cv2.putText(vis, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(vis, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

            self.vid.write(vis)

            scale = min(1.0, 1280 / self.vid.w)
            disp = cv2.resize(vis, None, fx=scale, fy=scale)
            st = "PAUSED" if paused else "PLAYING"
            cv2.putText(disp, f"{st} | Q=exit Space=pause A/D=step", (10, disp.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.imshow(win, disp)

            key = cv2.waitKey(delay if not paused else 30) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord(' '):
                paused = not paused
            elif key == ord('a'):
                fidx = max(0, fidx - 1); paused = True
            elif key == ord('d'):
                fidx = min(self.vid.total - 1, fidx + 1); paused = True
            elif key == ord('r'):
                fidx = 0

            if not paused:
                fidx += 1
                if fidx >= self.vid.total: fidx = 0

        self.vid.stop_write()
        cv2.destroyWindow(win)
        print(f"\n✓ 视频: {self.out_path}")

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

        win = 'Neuron Tool v10.1'
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, Config.DISPLAY_WIDTH, Config.DISPLAY_HEIGHT)
        cv2.setMouseCallback(win, self._mouse)

        print("\n" + "=" * 60)
        print("神经元追踪工具 v10.1 (时空关联)")
        print("=" * 60)
        print("左键=标记 | 右键=删除 | 滚轮=缩放 | 中键=平移")
        print("A/D=帧 | Debug=调试 | RUN=追踪")
        print("=" * 60 + "\n")

        while True:
            disp = self._draw()
            if disp is not None: cv2.imshow(win, disp)

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
                    if p: self.data_path = p; self.data.load(p)
                self.action = None
            elif self.action == 'exit':
                break

            key = cv2.waitKey(30) & 0xFF

            if self.active_inp:
                r = self.active_inp.handle_key(key)
                if r == 'confirm':
                    self._confirm_inp()
                elif r == 'cancel':
                    self.active_inp.active = False; self.active_inp = None
                continue

            if key == 27:
                break
            elif key == ord('a'):
                self.vid.read(self.vid.idx - 1); self.frame_inp.text = str(self.vid.idx + 1)
            elif key == ord('d'):
                self.vid.read(self.vid.idx + 1); self.frame_inp.text = str(self.vid.idx + 1)
            elif key == ord('w'):
                self.vid.read(self.vid.idx + 10); self.frame_inp.text = str(self.vid.idx + 1)
            elif key == ord('s'):
                self.vid.read(self.vid.idx - 10); self.frame_inp.text = str(self.vid.idx + 1)
            elif ord('0') <= key <= ord('9'):
                self.cur_nid = key - ord('0');
                self.neuron_inp.text = str(self.cur_nid)
            elif key == ord('f'):
                self.use_frangi = not self.use_frangi;
                print(f"Frangi: {'ON' if self.use_frangi else 'OFF'}")

        cv2.destroyAllWindows()
        self.vid.release()
        if self.root: self.root.destroy()


if __name__ == "__main__":
    VIDEO_PATH = r"/neuron_growth_50.mp4"
    tool = NeuronTool()
    tool.run(VIDEO_PATH)
