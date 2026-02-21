import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
from math import sqrt, cos, sin, radians
from collections import defaultdict
import colorsys

DEFAULT_INPUT_DIR = r"F:\工作文件\RA\python\项目汇总\神经图像\05_二次空间去噪\frames_final"
DEFAULT_OUTPUT_DIR = r"F:\工作文件\RA\python\项目汇总\神经图像\07_生长追踪"


def generate_distinct_colors(n=99):
    colors = []
    for i in range(n):
        hue = i / n
        saturation = 0.8 + (i % 3) * 0.1
        value = 0.9 - (i % 2) * 0.15
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        bgr = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
        colors.append(bgr)
    return colors


NEURON_COLORS = generate_distinct_colors(99)


class NeuronTracker:
    """神经元追踪 - 前半段向左统一 + 后半段向右生长"""

    def __init__(self):
        self.frames = []
        self.height = 0
        self.width = 0

        self.brightness_threshold = 30
        self.search_radius = 30
        self.linearity_weight = 2.0
        self.left_margin = 3
        self.edge_margin = 3
        self.right_margin = 3  # 右边界
        self.max_turn_angle = 60
        self.max_search_radius = 150
        self.search_radius_step = 20

        self.waypoint_capture_radius = 25

        self.markers = defaultdict(lambda: defaultdict(list))
        self.tracking_results = {}

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

    def add_marker(self, neuron_id, frame_idx, x, y):
        self.markers[neuron_id][frame_idx].append((int(x), int(y)))

    def remove_last_marker(self, neuron_id, frame_idx):
        if neuron_id in self.markers and frame_idx in self.markers[neuron_id]:
            if self.markers[neuron_id][frame_idx]:
                self.markers[neuron_id][frame_idx].pop()
                if not self.markers[neuron_id][frame_idx]:
                    del self.markers[neuron_id][frame_idx]
                if not self.markers[neuron_id]:
                    del self.markers[neuron_id]

    def clear_neuron_markers(self, neuron_id):
        if neuron_id in self.markers:
            del self.markers[neuron_id]

    def get_neuron_markers(self, neuron_id):
        return dict(self.markers.get(neuron_id, {}))

    def get_all_waypoints(self, neuron_id):
        markers = self.get_neuron_markers(neuron_id)
        all_points = []
        for pts in markers.values():
            all_points.extend(pts)
        unique = []
        for p in all_points:
            is_dup = False
            for u in unique:
                if sqrt((p[0] - u[0]) ** 2 + (p[1] - u[1]) ** 2) < 5:
                    is_dup = True
                    break
            if not is_dup:
                unique.append(p)
        return unique

    def is_bright(self, frame_idx, x, y):
        if frame_idx < 0 or frame_idx >= len(self.frames):
            return False
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return False
        return self.frames[frame_idx][int(y), int(x)] > self.brightness_threshold

    def check_left_boundary(self, x, y):
        """检查是否到达左/上/下边界"""
        if x <= self.left_margin:
            return True, 'left'
        if y <= self.edge_margin:
            return True, 'top'
        if y >= self.height - self.edge_margin:
            return True, 'bottom'
        return False, None

    def check_right_boundary(self, x, y):
        """检查是否到达右边界"""
        if x >= self.width - self.right_margin:
            return True, 'right'
        if y <= self.edge_margin:
            return True, 'top'
        if y >= self.height - self.edge_margin:
            return True, 'bottom'
        return False, None

    def find_best_frame(self, neuron_id, waypoints):
        markers = self.get_neuron_markers(neuron_id)
        marked_frames = list(markers.keys())

        best_frame = marked_frames[0] if marked_frames else 0
        best_score = 0

        check_frames = set()
        for f in marked_frames:
            for df in range(-10, 11):
                if 0 <= f + df < len(self.frames):
                    check_frames.add(f + df)

        for f in check_frames:
            frame = self.frames[f]
            score = 0
            for wp in waypoints:
                x, y = wp
                for dy in range(-5, 6):
                    for dx in range(-5, 6):
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < self.width and 0 <= ny < self.height:
                            score += frame[ny, nx]
            if score > best_score:
                best_score = score
                best_frame = f

        return best_frame

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

    def find_candidates_leftward(self, frame_idx, cx, cy, path, direction, visited, target=None, radius=None):
        """向左寻找候选点"""
        if radius is None:
            radius = self.search_radius
        frame = self.frames[frame_idx]
        candidates = []
        search_range = int(radius)

        allow_right = target is not None and target[0] > cx

        for dy in range(-search_range, search_range + 1):
            dx_end = search_range if allow_right else 1
            for dx in range(-search_range, dx_end):
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

                linearity = self.compute_linearity(path, (nx, ny))
                score = brightness * (linearity ** self.linearity_weight) / (dist + 0.5)

                if dx < 0:
                    score *= (1 + abs(dx) / (dist + 0.1))

                if direction:
                    dir_x, dir_y = direction
                    dir_len = sqrt(dir_x ** 2 + dir_y ** 2)
                    if dir_len > 0:
                        cos_a = (dx * dir_x + dy * dir_y) / (dist * dir_len)
                        score *= (1 + (cos_a + 1) / 2)

                if target:
                    tx, ty = target
                    to_target_dx = tx - cx
                    to_target_dy = ty - cy
                    to_target_len = sqrt(to_target_dx ** 2 + to_target_dy ** 2)
                    if to_target_len > 0:
                        cos_to_target = (dx * to_target_dx + dy * to_target_dy) / (dist * to_target_len)
                        score *= (1 + 3.0 * (cos_to_target + 1) / 2)

                candidates.append((nx, ny, score, dx, dy))

        return candidates

    def find_candidates_rightward(self, frame_idx, cx, cy, path, direction, visited, radius=None):
        """向右寻找候选点（只能向右或垂直，不能向左）"""
        if radius is None:
            radius = self.search_radius
        frame = self.frames[frame_idx]
        candidates = []
        search_range = int(radius)

        for dy in range(-search_range, search_range + 1):
            # dx >= 0：只能向右或垂直
            for dx in range(0, search_range + 1):
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

                linearity = self.compute_linearity(path, (nx, ny))
                score = brightness * (linearity ** self.linearity_weight) / (dist + 0.5)

                # 向右加分
                if dx > 0:
                    right_bonus = dx / (dist + 0.1)
                    score *= (1 + right_bonus)

                if direction:
                    dir_x, dir_y = direction
                    dir_len = sqrt(dir_x ** 2 + dir_y ** 2)
                    if dir_len > 0:
                        cos_a = (dx * dir_x + dy * dir_y) / (dist * dir_len)
                        score *= (1 + (cos_a + 1) / 2)

                candidates.append((nx, ny, score, dx, dy))

        return candidates

    def trace_segment_left(self, frame_idx, start_x, start_y, target=None, initial_direction=None, max_steps=3000):
        """向左追踪一段路径"""
        path = [(int(start_x), int(start_y))]
        visited = set()
        visited.add((int(start_x), int(start_y)))
        cx, cy = int(start_x), int(start_y)
        direction = initial_direction if initial_direction else (-1, 0)
        boundary_reached = None
        reached_target = False

        for step in range(max_steps):
            if target:
                dist_to_target = sqrt((cx - target[0]) ** 2 + (cy - target[1]) ** 2)
                if dist_to_target <= self.waypoint_capture_radius:
                    reached_target = True
                    break

            reached, boundary_type = self.check_left_boundary(cx, cy)
            if reached:
                boundary_reached = boundary_type
                break

            candidates = self.find_candidates_leftward(
                frame_idx, cx, cy, path, direction, visited, target=target
            )

            if not candidates:
                for radius in range(self.search_radius, self.max_search_radius + 1, self.search_radius_step):
                    candidates = self.find_candidates_leftward(
                        frame_idx, cx, cy, path, direction, visited, target=target, radius=radius
                    )
                    if candidates:
                        break

            if not candidates:
                break

            candidates.sort(key=lambda x: -x[2])
            chosen = candidates[0]
            nx, ny = chosen[0], chosen[1]
            dx, dy = chosen[3], chosen[4]

            if target is None and dx > 0:
                left_only = [c for c in candidates if c[3] <= 0]
                if left_only:
                    chosen = max(left_only, key=lambda x: x[2])
                    nx, ny = chosen[0], chosen[1]
                    dx, dy = chosen[3], chosen[4]
                else:
                    break

            if (nx, ny) in visited:
                remaining = [c for c in candidates if (c[0], c[1]) not in visited]
                if target is None:
                    remaining = [c for c in remaining if c[3] <= 0]
                if remaining:
                    chosen = max(remaining, key=lambda x: x[2])
                    nx, ny = chosen[0], chosen[1]
                    dx, dy = chosen[3], chosen[4]
                else:
                    break

            path.append((nx, ny))
            visited.add((nx, ny))

            dist = sqrt(dx ** 2 + dy ** 2)
            if dist > 0:
                new_dir = (dx / dist, dy / dist)
                alpha = 0.3
                direction = (
                    direction[0] * (1 - alpha) + new_dir[0] * alpha,
                    direction[1] * (1 - alpha) + new_dir[1] * alpha
                )
                dlen = sqrt(direction[0] ** 2 + direction[1] ** 2)
                if dlen > 0:
                    direction = (direction[0] / dlen, direction[1] / dlen)

            cx, cy = nx, ny

        return path, boundary_reached, reached_target, direction

    def trace_segment_right(self, frame_idx, start_x, start_y, initial_direction=None, max_steps=500):
        """向右追踪一段路径（用于生长追踪）"""
        path = [(int(start_x), int(start_y))]
        visited = set()
        visited.add((int(start_x), int(start_y)))
        cx, cy = int(start_x), int(start_y)
        direction = initial_direction if initial_direction else (1, 0)
        boundary_reached = None

        for step in range(max_steps):
            reached, boundary_type = self.check_right_boundary(cx, cy)
            if reached:
                boundary_reached = boundary_type
                break

            candidates = self.find_candidates_rightward(
                frame_idx, cx, cy, path, direction, visited
            )

            if not candidates:
                for radius in range(self.search_radius, self.max_search_radius + 1, self.search_radius_step):
                    candidates = self.find_candidates_rightward(
                        frame_idx, cx, cy, path, direction, visited, radius=radius
                    )
                    if candidates:
                        break

            if not candidates:
                break

            candidates.sort(key=lambda x: -x[2])
            chosen = candidates[0]
            nx, ny = chosen[0], chosen[1]
            dx, dy = chosen[3], chosen[4]

            # 强制向右：如果dx < 0就跳过（但find_candidates_rightward已经保证dx >= 0）
            if dx < 0:
                right_only = [c for c in candidates if c[3] >= 0]
                if right_only:
                    chosen = max(right_only, key=lambda x: x[2])
                    nx, ny = chosen[0], chosen[1]
                    dx, dy = chosen[3], chosen[4]
                else:
                    break

            if (nx, ny) in visited:
                remaining = [c for c in candidates if (c[0], c[1]) not in visited and c[3] >= 0]
                if remaining:
                    chosen = max(remaining, key=lambda x: x[2])
                    nx, ny = chosen[0], chosen[1]
                    dx, dy = chosen[3], chosen[4]
                else:
                    break

            path.append((nx, ny))
            visited.add((nx, ny))

            dist = sqrt(dx ** 2 + dy ** 2)
            if dist > 0:
                new_dir = (dx / dist, dy / dist)
                alpha = 0.3
                direction = (
                    direction[0] * (1 - alpha) + new_dir[0] * alpha,
                    direction[1] * (1 - alpha) + new_dir[1] * alpha
                )
                dlen = sqrt(direction[0] ** 2 + direction[1] ** 2)
                if dlen > 0:
                    direction = (direction[0] / dlen, direction[1] / dlen)

            cx, cy = nx, ny

        return path, boundary_reached, direction

    def connect_points_directly(self, frame_idx, start, end):
        """简单连接两点"""
        path = [start]
        cx, cy = start
        ex, ey = end

        for _ in range(2000):
            if sqrt((cx - ex) ** 2 + (cy - ey) ** 2) <= 2:
                path.append(end)
                break

            dx = ex - cx
            dy = ey - cy
            dist = sqrt(dx ** 2 + dy ** 2)
            if dist == 0:
                break

            dx, dy = dx / dist, dy / dist

            best_pt = None
            best_score = -1
            frame = self.frames[frame_idx]

            for step_len in [1, 2]:
                for angle_offset in [0, 0.15, -0.15, 0.3, -0.3]:
                    ndx = dx * cos(angle_offset) - dy * sin(angle_offset)
                    ndy = dx * sin(angle_offset) + dy * cos(angle_offset)
                    nx = int(cx + ndx * step_len)
                    ny = int(cy + ndy * step_len)

                    if 0 <= nx < self.width and 0 <= ny < self.height:
                        brightness = frame[ny, nx]
                        score = brightness - abs(angle_offset) * 100
                        if score > best_score:
                            best_score = score
                            best_pt = (nx, ny)

            if best_pt and best_pt != (cx, cy):
                path.append(best_pt)
                cx, cy = best_pt
            else:
                nx = int(cx + dx)
                ny = int(cy + dy)
                if (nx, ny) != (cx, cy):
                    path.append((nx, ny))
                    cx, cy = nx, ny
                else:
                    break

        return path

    def trace_left_unified(self, frame_idx, waypoints):
        """向左追踪经过所有必经点的统一路径"""
        if not waypoints:
            return [], None

        full_path = []
        boundary_reached = None
        direction = (-1, 0)

        for i in range(len(waypoints)):
            start = waypoints[i]
            target = waypoints[i + 1] if i + 1 < len(waypoints) else None

            segment, boundary, reached_target, direction = self.trace_segment_left(
                frame_idx, start[0], start[1], target=target, initial_direction=direction
            )

            if full_path:
                if segment and segment[0] == full_path[-1]:
                    segment = segment[1:]
            full_path.extend(segment)

            if target and not reached_target:
                last_pt = full_path[-1] if full_path else start
                if last_pt != target:
                    connect_path = self.connect_points_directly(frame_idx, last_pt, target)
                    if connect_path:
                        full_path.extend(connect_path[1:])

            if boundary:
                boundary_reached = boundary
                break

        for wp in waypoints:
            found = False
            for i, pt in enumerate(full_path):
                if sqrt((pt[0] - wp[0]) ** 2 + (pt[1] - wp[1]) ** 2) < 3:
                    found = True
                    break
            if not found:
                min_dist = float('inf')
                insert_idx = len(full_path)
                for idx, pt in enumerate(full_path):
                    dist = sqrt((pt[0] - wp[0]) ** 2 + (pt[1] - wp[1]) ** 2)
                    if dist < min_dist:
                        min_dist = dist
                        insert_idx = idx
                full_path.insert(insert_idx, wp)

        return full_path, boundary_reached

    def smooth_path_preserve_waypoints(self, path, waypoints, window=3):
        if len(path) <= window:
            return path

        waypoint_indices = set()
        for wp in waypoints:
            for i, pt in enumerate(path):
                if sqrt((pt[0] - wp[0]) ** 2 + (pt[1] - wp[1]) ** 2) < 3:
                    waypoint_indices.add(i)
                    break

        smoothed = [path[0]]
        for i in range(1, len(path) - 1):
            if i in waypoint_indices:
                smoothed.append(path[i])
            else:
                start = max(0, i - window // 2)
                end = min(len(path), i + window // 2 + 1)
                avg_x = sum(p[0] for p in path[start:end]) / (end - start)
                avg_y = sum(p[1] for p in path[start:end]) / (end - start)
                smoothed.append((int(round(avg_x)), int(round(avg_y))))
        smoothed.append(path[-1])

        return smoothed

    def find_growth_start_point(self, frame_idx, prev_end_point, search_radius=20):
        """在当前帧找到最接近上一帧末端的亮点作为生长起点"""
        px, py = prev_end_point
        best_pt = None
        best_score = -1
        frame = self.frames[frame_idx]

        for dy in range(-search_radius, search_radius + 1):
            for dx in range(-search_radius, search_radius + 1):
                nx, ny = px + dx, py + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    brightness = frame[ny, nx]
                    if brightness > self.brightness_threshold:
                        dist = sqrt(dx ** 2 + dy ** 2)
                        # 优先选择靠右且靠近的点
                        score = brightness * (1 + dx * 0.1) / (dist + 1)
                        if score > best_score:
                            best_score = score
                            best_pt = (nx, ny)

        return best_pt if best_pt else prev_end_point

    def compute_neuron_trajectory(self, neuron_id):
        """
        计算神经元轨迹：
        1. 前半段：向左追踪到边界（统一轨迹）
        2. 后半段：向右逐帧生长
        """
        markers = self.get_neuron_markers(neuron_id)
        if not markers:
            return None

        all_waypoints = self.get_all_waypoints(neuron_id)
        if not all_waypoints:
            return None

        # 按x从大到小排序（从右到左）
        all_waypoints = sorted(all_waypoints, key=lambda p: -p[0])

        # 找标记帧范围
        marked_frames = sorted(markers.keys())
        first_marked_frame = min(marked_frames)
        last_marked_frame = max(marked_frames)

        print(f"\n{'=' * 60}")
        print(f"计算神经元 N{neuron_id} 轨迹")
        print(f"标记帧: {marked_frames}")
        print(f"必经点({len(all_waypoints)}个): {all_waypoints}")
        print(f"{'=' * 60}")

        # ========== 前半段：向左追踪到边界 ==========
        best_frame = self.find_best_frame(neuron_id, all_waypoints)
        print(f"\n[前半段] 代表帧: {best_frame}, 向左追踪")

        left_path, left_boundary = self.trace_left_unified(best_frame, all_waypoints)

        if not left_path:
            print("  前半段追踪失败！")
            return None

        if len(left_path) > 5:
            left_path = self.smooth_path_preserve_waypoints(left_path, all_waypoints)

        # 确保路径方向（从右到左）
        if len(left_path) >= 2 and left_path[0][0] < left_path[-1][0]:
            left_path = left_path[::-1]

        # 前半段的起点（最右边的点）作为后半段的基准
        growth_base_point = left_path[0]  # 最右端

        print(f"  前半段: {len(left_path)}点, 左端到达{left_boundary or '未知'}")
        print(f"  生长基准点: {growth_base_point}")

        # ========== 后半段：向右逐帧生长 ==========
        print(f"\n[后半段] 从帧{first_marked_frame}开始向右生长")

        # 存储每帧的路径
        paths_by_frame = {}

        # 初始方向（从前半段末端推断）
        if len(left_path) >= 2:
            dx = left_path[0][0] - left_path[1][0]
            dy = left_path[0][1] - left_path[1][1]
            dist = sqrt(dx ** 2 + dy ** 2)
            initial_direction = (dx / dist, dy / dist) if dist > 0 else (1, 0)
        else:
            initial_direction = (1, 0)

        # 第一个标记帧：只有前半段
        current_growth_tip = growth_base_point
        current_direction = initial_direction

        for frame_idx in range(len(self.frames)):
            if frame_idx < first_marked_frame:
                # 标记帧之前：只有前半段（静态）
                paths_by_frame[frame_idx] = {
                    'left_path': left_path.copy(),
                    'right_path': [],
                    'full_path': left_path.copy()
                }
            else:
                # 标记帧及之后：前半段固定 + 后半段向右生长

                # 在当前帧寻找生长点
                growth_start = self.find_growth_start_point(frame_idx, current_growth_tip)

                # 向右追踪生长
                right_segment, right_boundary, new_direction = self.trace_segment_right(
                    frame_idx, growth_start[0], growth_start[1],
                    initial_direction=current_direction,
                    max_steps=200  # 每帧最多生长200步
                )

                # 更新生长末端
                if right_segment:
                    current_growth_tip = right_segment[-1]
                    current_direction = new_direction

                # 组合完整路径
                # 注意：left_path是从右到左，right_segment是从左到右
                # 完整路径应该是：左边界 ← left_path ← growth_base → right_segment → 右边界

                full_path = left_path.copy()  # 前半段（从右到左）

                if right_segment and len(right_segment) > 1:
                    # 跳过第一个点（与growth_base重复）
                    full_path = right_segment[::-1] + left_path[1:]  # 反转right放前面

                paths_by_frame[frame_idx] = {
                    'left_path': left_path.copy(),
                    'right_path': right_segment if right_segment else [],
                    'full_path': full_path,
                    'growth_tip': current_growth_tip
                }

                if frame_idx <= first_marked_frame + 5 or frame_idx % 20 == 0:
                    tip_x = current_growth_tip[0] if current_growth_tip else 0
                    print(f"  帧{frame_idx}: 生长末端 x={tip_x}")

        # ========== 生成最终结果 ==========
        # 计算最大生长范围
        max_growth_x = max(
            p['growth_tip'][0] if 'growth_tip' in p and p['growth_tip'] else growth_base_point[0]
            for p in paths_by_frame.values()
        )

        result = {
            'left_path': left_path,  # 前半段统一路径
            'growth_base': growth_base_point,  # 生长基准点
            'paths_by_frame': paths_by_frame,  # 每帧的路径
            'waypoints': all_waypoints,
            'left_boundary': left_boundary,
            'first_marked_frame': first_marked_frame,
            'max_growth_x': max_growth_x
        }

        self.tracking_results[neuron_id] = result

        print(f"\n完成 N{neuron_id}:")
        print(f"  前半段: {len(left_path)}点 → {left_boundary or '边界'}")
        print(f"  后半段: 从帧{first_marked_frame}开始生长")
        print(f"  最大生长: x={max_growth_x}")

        return result


class TrackerGUI:
    """GUI - 前半段统一 + 后半段生长"""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("神经元追踪 - 向左统一 + 向右生长")
        self.root.geometry("1550x980")

        self.tracker = NeuronTracker()
        self.current_frame_idx = 0
        self.zoom_level = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.dragging = False
        self.drag_start = None

        self.current_neuron_id = 1
        self.preview_result = None
        self.preview_neuron_id = None

        self.input_dir = DEFAULT_INPUT_DIR
        self.output_dir = DEFAULT_OUTPUT_DIR

        self.setup_ui()
        self.bind_events()

    def get_neuron_color(self, neuron_id):
        return NEURON_COLORS[(neuron_id - 1) % len(NEURON_COLORS)]

    def setup_ui(self):
        self.main_paned = ttk.PanedWindow(self.root, orient="horizontal")
        self.main_paned.pack(fill="both", expand=True, padx=5, pady=5)

        left = ttk.Frame(self.main_paned)
        self.main_paned.add(left, weight=3)
        left.columnconfigure(0, weight=1)
        left.rowconfigure(0, weight=1)
        self.canvas = tk.Canvas(left, bg='black')
        self.canvas.grid(row=0, column=0, sticky="nsew")

        right = ttk.Frame(self.main_paned, width=400)
        self.main_paned.add(right, weight=0)

        self.ctrl_canvas = tk.Canvas(right, width=380)
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
        self.input_entry = ttk.Entry(f1, width=26)
        self.input_entry.insert(0, self.input_dir)
        self.input_entry.grid(row=0, column=1, sticky="ew")
        ttk.Button(f1, text="...", command=self.browse_input, width=3).grid(row=0, column=2)
        ttk.Label(f1, text="输出:").grid(row=1, column=0, sticky="w", pady=2)
        self.output_entry = ttk.Entry(f1, width=26)
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

        # 神经元选择
        f4 = ttk.LabelFrame(p, text="🧠 神经元选择", padding=5)
        f4.pack(fill="x", pady=pad, padx=3)

        row_n = ttk.Frame(f4)
        row_n.pack(fill="x")
        ttk.Label(row_n, text="当前神经元:").pack(side="left")
        self.neuron_var = tk.StringVar(value="1")
        self.neuron_spinbox = ttk.Spinbox(row_n, from_=1, to=99, width=5, textvariable=self.neuron_var,
                                          command=self.on_neuron_change)
        self.neuron_spinbox.pack(side="left", padx=3)
        self.neuron_spinbox.bind('<Return>', lambda e: self.on_neuron_change())

        self.neuron_color_label = tk.Label(row_n, text="  ████  ", bg='red', width=6)
        self.neuron_color_label.pack(side="left", padx=5)
        self.update_neuron_color_display()

        ttk.Label(f4, text="数字键1-9快速切换").pack(anchor="w")

        # 追踪模式说明
        f_mode = ttk.LabelFrame(p, text="📐 追踪模式", padding=5)
        f_mode.pack(fill="x", pady=pad, padx=3)
        ttk.Label(f_mode,
                  text="前半段: 向左追踪到边界（统一）\n后半段: 随时间向右生长（逐帧）\n\n时间增加 → 轨迹只能向右延伸",
                  font=("", 9), justify="left", foreground="darkblue").pack(anchor="w")

        # 标记操作
        f5 = ttk.LabelFrame(p, text="📍 标记必经点", padding=5)
        f5.pack(fill="x", pady=pad, padx=3)

        ttk.Label(f5, text="左键：添加必经点\n右键：删除当前帧最后标记",
                  font=("", 9), justify="left").pack(anchor="w")

        self.marker_info = tk.StringVar(value="当前神经元无标记")
        ttk.Label(f5, textvariable=self.marker_info, foreground="blue", wraplength=340).pack(anchor="w", pady=3)

        btn_row = ttk.Frame(f5)
        btn_row.pack(fill="x", pady=3)
        ttk.Button(btn_row, text="清除当前神经元", command=self.clear_current_markers, width=14).pack(side="left",
                                                                                                      padx=2)
        ttk.Button(btn_row, text="清除全部", command=self.clear_all_markers, width=10).pack(side="left", padx=2)

        # 计算轨迹
        f6 = ttk.LabelFrame(p, text="🔬 计算轨迹", padding=5)
        f6.pack(fill="x", pady=pad, padx=3)

        btn_row2 = ttk.Frame(f6)
        btn_row2.pack(fill="x", pady=3)
        ttk.Button(btn_row2, text="▶ 计算当前神经元", command=self.compute_current_neuron, width=16).pack(side="left",
                                                                                                          padx=2)
        ttk.Button(btn_row2, text="▶▶ 计算全部", command=self.compute_all_neurons, width=12).pack(side="left", padx=2)

        self.compute_info = tk.StringVar(value="")
        ttk.Label(f6, textvariable=self.compute_info, foreground="green", wraplength=340).pack(anchor="w", pady=3)

        btn_row3 = ttk.Frame(f6)
        btn_row3.pack(fill="x", pady=3)
        ttk.Button(btn_row3, text="✓ 确认", command=self.confirm_result, width=10).pack(side="left", padx=2)
        ttk.Button(btn_row3, text="✗ 取消", command=self.cancel_preview, width=10).pack(side="left", padx=2)

        # 参数
        f7 = ttk.LabelFrame(p, text="⚙️ 参数", padding=5)
        f7.pack(fill="x", pady=pad, padx=3)
        g = ttk.Frame(f7)
        g.pack(fill="x")

        ttk.Label(g, text="亮度阈值:").grid(row=0, column=0, sticky="w")
        self.thresh_var = tk.StringVar(value="30")
        ttk.Entry(g, textvariable=self.thresh_var, width=5).grid(row=0, column=1)
        ttk.Label(g, text="搜索半径:").grid(row=0, column=2, sticky="w", padx=(8, 0))
        self.radius_var = tk.StringVar(value="30")
        ttk.Entry(g, textvariable=self.radius_var, width=5).grid(row=0, column=3)

        ttk.Button(f7, text="应用参数", command=self.apply_params).pack(pady=3)

        # 结果列表
        f8 = ttk.LabelFrame(p, text="📊 追踪结果", padding=5)
        f8.pack(fill="x", pady=pad, padx=3)

        self.result_list = tk.Listbox(f8, height=5, font=("Consolas", 9))
        self.result_list.pack(fill="x")
        self.result_list.bind('<<ListboxSelect>>', self.on_result_select)

        btn_row4 = ttk.Frame(f8)
        btn_row4.pack(fill="x", pady=3)
        ttk.Button(btn_row4, text="删除选中", command=self.delete_selected_result, width=10).pack(side="left", padx=2)
        ttk.Button(btn_row4, text="清空全部", command=self.clear_all_results, width=10).pack(side="left", padx=2)

        # 导出
        f9 = ttk.LabelFrame(p, text="💾 导出", padding=5)
        f9.pack(fill="x", pady=pad, padx=3)
        btn_row5 = ttk.Frame(f9)
        btn_row5.pack(fill="x")
        ttk.Button(btn_row5, text="导出视频", command=self.export_video, width=10).pack(side="left", padx=2)
        ttk.Button(btn_row5, text="截图", command=self.screenshot, width=8).pack(side="left", padx=2)
        ttk.Button(btn_row5, text="导出CSV", command=self.export_data, width=10).pack(side="left", padx=2)

        # 状态
        f10 = ttk.LabelFrame(p, text="📌 状态", padding=5)
        f10.pack(fill="x", pady=pad, padx=3)
        self.status = tk.StringVar(value="请加载帧")
        ttk.Label(f10, textvariable=self.status, wraplength=340, foreground="blue").pack(fill="x")
        self.progress = ttk.Progressbar(f10, mode='determinate')
        self.progress.pack(fill="x", pady=3)

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
        except:
            pass

    def update_marker_info(self):
        markers = self.tracker.get_neuron_markers(self.current_neuron_id)
        waypoints = self.tracker.get_all_waypoints(self.current_neuron_id)
        if not markers:
            self.marker_info.set(f"N{self.current_neuron_id}: 无标记")
        else:
            frames = sorted(markers.keys())
            self.marker_info.set(
                f"N{self.current_neuron_id}: {len(waypoints)}个必经点\n"
                f"标记帧: {frames}"
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

        self.tracker.add_marker(self.current_neuron_id, self.current_frame_idx, ix, iy)
        self.update_marker_info()

        total = len(self.tracker.get_all_waypoints(self.current_neuron_id))
        self.status.set(
            f"N{self.current_neuron_id}: 添加必经点 ({ix}, {iy}) @ 帧{self.current_frame_idx} [共{total}点]")
        self.update_display()

    def on_right_click(self, e):
        if not self.tracker.frames: return
        self.tracker.remove_last_marker(self.current_neuron_id, self.current_frame_idx)
        self.update_marker_info()
        self.status.set(f"N{self.current_neuron_id}: 删除标记 @ 帧{self.current_frame_idx}")
        self.update_display()

    def clear_current_markers(self):
        self.tracker.clear_neuron_markers(self.current_neuron_id)
        self.update_marker_info()
        self.status.set(f"已清除 N{self.current_neuron_id} 的所有标记")
        self.update_display()

    def clear_all_markers(self):
        if messagebox.askyesno("确认", "清除所有神经元的标记？"):
            self.tracker.markers.clear()
            self.update_marker_info()
            self.status.set("已清除所有标记")
            self.update_display()

    def apply_params(self):
        try:
            self.tracker.brightness_threshold = int(self.thresh_var.get())
            self.tracker.search_radius = int(self.radius_var.get())
        except:
            pass

    def compute_current_neuron(self):
        markers = self.tracker.get_neuron_markers(self.current_neuron_id)
        if not markers:
            messagebox.showwarning("警告", f"N{self.current_neuron_id} 没有标记点")
            return

        self.apply_params()
        self.status.set(f"计算 N{self.current_neuron_id} 轨迹...")
        self.root.update()

        result = self.tracker.compute_neuron_trajectory(self.current_neuron_id)

        if result:
            self.preview_result = result
            self.preview_neuron_id = self.current_neuron_id

            left_len = len(result['left_path'])
            wp_count = len(result['waypoints'])
            max_x = result['max_growth_x']
            first_frame = result['first_marked_frame']

            self.compute_info.set(
                f"N{self.current_neuron_id}:\n"
                f"前半段: {left_len}点 → {result['left_boundary'] or '边界'}\n"
                f"后半段: 从帧{first_frame}开始生长, 最远x={max_x}\n"
                f"Enter确认 / Esc取消"
            )
            self.status.set("预览就绪")
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

        for nid in neuron_ids:
            self.status.set(f"计算 N{nid} ...")
            self.root.update()
            result = self.tracker.compute_neuron_trajectory(nid)
            if result:
                success += 1

        self.update_result_list()
        self.status.set(f"完成: {success}/{len(neuron_ids)} 个神经元")
        self.update_display()

    def confirm_result(self):
        if self.preview_result and self.preview_neuron_id:
            self.tracker.tracking_results[self.preview_neuron_id] = self.preview_result
            self.update_result_list()
            self.status.set(f"已确认 N{self.preview_neuron_id}")
            self.preview_result = None
            self.preview_neuron_id = None
            self.compute_info.set("")
            self.update_display()

    def cancel_preview(self):
        if self.preview_neuron_id:
            if self.preview_neuron_id in self.tracker.tracking_results:
                del self.tracker.tracking_results[self.preview_neuron_id]
        self.preview_result = None
        self.preview_neuron_id = None
        self.compute_info.set("已取消")
        self.update_display()

    def update_result_list(self):
        self.result_list.delete(0, tk.END)
        for nid, result in sorted(self.tracker.tracking_results.items()):
            left_len = len(result.get('left_path', []))
            max_x = result.get('max_growth_x', 0)
            lb = result.get('left_boundary', '')
            self.result_list.insert(tk.END, f"N{nid}: 左{left_len}点→{lb}, 右生长→{max_x}")

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

    def export_video(self):
        if not self.tracker.tracking_results:
            messagebox.showwarning("警告", "无结果")
            return
        self.output_dir = self.output_entry.get()
        os.makedirs(self.output_dir, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        path = os.path.join(self.output_dir, "neuron_growth_tracking.mp4")
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

    def export_data(self):
        if not self.tracker.tracking_results:
            messagebox.showwarning("警告", "无结果")
            return
        self.output_dir = self.output_entry.get()
        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir, "neuron_growth_paths.csv")

        with open(path, 'w') as f:
            f.write("neuron_id,frame,path_index,x,y,segment\n")
            for nid, result in sorted(self.tracker.tracking_results.items()):
                paths_by_frame = result.get('paths_by_frame', {})
                for frame_idx, frame_data in sorted(paths_by_frame.items()):
                    # 前半段
                    for idx, (x, y) in enumerate(frame_data.get('left_path', [])):
                        f.write(f"{nid},{frame_idx},{idx},{x},{y},left\n")
                    # 后半段
                    left_len = len(frame_data.get('left_path', []))
                    for idx, (x, y) in enumerate(frame_data.get('right_path', [])):
                        f.write(f"{nid},{frame_idx},{left_len + idx},{x},{y},right\n")

        self.status.set(f"数据已保存: {path}")
        messagebox.showinfo("完成", f"导出完成!\n{path}")

    def render(self, idx, for_export=False):
        frame = self.tracker.frames[idx]
        vis = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        # 边界线
        cv2.line(vis, (self.tracker.left_margin, 0),
                 (self.tracker.left_margin, self.tracker.height), (0, 100, 0), 1)
        cv2.line(vis, (self.tracker.width - self.tracker.right_margin, 0),
                 (self.tracker.width - self.tracker.right_margin, self.tracker.height), (0, 100, 0), 1)
        cv2.line(vis, (0, self.tracker.edge_margin),
                 (self.tracker.width, self.tracker.edge_margin), (0, 100, 0), 1)
        cv2.line(vis, (0, self.tracker.height - self.tracker.edge_margin),
                 (self.tracker.width, self.tracker.height - self.tracker.edge_margin), (0, 100, 0), 1)

        # 渲染结果或预览
        results_to_render = []

        for nid, result in self.tracker.tracking_results.items():
            results_to_render.append((nid, result, False))

        if self.preview_result and self.preview_neuron_id:
            results_to_render.append((self.preview_neuron_id, self.preview_result, True))

        for nid, result, is_preview in results_to_render:
            color = self.get_neuron_color(nid)
            if is_preview:
                color = (0, 255, 255)  # 预览用黄色

            paths_by_frame = result.get('paths_by_frame', {})
            waypoints = result.get('waypoints', [])

            if idx in paths_by_frame:
                frame_data = paths_by_frame[idx]

                # 绘制前半段（统一部分）
                left_path = frame_data.get('left_path', [])
                if left_path:
                    for j in range(1, len(left_path)):
                        cv2.line(vis, left_path[j - 1], left_path[j], color, 2)
                    # 左端点
                    cv2.circle(vis, left_path[-1], 5, (0, 255, 0), -1)

                # 绘制后半段（生长部分）- 用稍浅的颜色
                right_path = frame_data.get('right_path', [])
                if right_path:
                    lighter_color = tuple(min(255, c + 50) for c in color)
                    for j in range(1, len(right_path)):
                        cv2.line(vis, right_path[j - 1], right_path[j], lighter_color, 2)
                    # 生长末端
                    cv2.circle(vis, right_path[-1], 6, (0, 165, 255), -1)  # 橙色

                # 标注
                if left_path:
                    cv2.putText(vis, f"N{nid}", (left_path[0][0] + 5, left_path[0][1] - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            # 必经点
            if not for_export:
                for wp in waypoints:
                    cv2.drawMarker(vis, wp, (255, 255, 255), cv2.MARKER_DIAMOND, 10, 2)

        # 标记点（未计算的）
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
                    cv2.putText(vis, f"N{nid}", (rightmost[0][0] + 12, rightmost[0][1] - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        if for_export:
            cv2.putText(vis, f"Frame {idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            y_pos = 60
            for nid in sorted(self.tracker.tracking_results.keys()):
                color = self.get_neuron_color(nid)
                cv2.rectangle(vis, (10, y_pos - 12), (30, y_pos + 2), color, -1)
                cv2.putText(vis, f"N{nid}", (35, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_pos += 20

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
    print("启动神经元追踪 - 向左统一 + 向右生长")
    TrackerGUI().run()
