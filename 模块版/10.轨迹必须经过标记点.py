import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
from math import sqrt, cos, radians
from collections import defaultdict
import colorsys

DEFAULT_INPUT_DIR = r"F:\工作文件\RA\python\项目汇总\神经图像\05_二次空间去噪\frames_final"
DEFAULT_OUTPUT_DIR = r"F:\工作文件\RA\python\项目汇总\神经图像\07_生长追踪"


def generate_distinct_colors(n=99):
    """生成N个视觉上易区分的颜色"""
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
    """神经元追踪 - 强制经过标记点版"""

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
        self.intersection_score_ratio = 0.7
        self.intersection_angle_threshold = 0.87

        # 强制经过标记点的距离阈值
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

    def check_boundary_reached(self, x, y):
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

    def compute_smoothness_score(self, path, candidate):
        if len(path) < 2:
            return 1.0
        n = min(8, len(path))
        recent = path[-n:]
        dx_avg, dy_avg = 0, 0
        for i in range(1, len(recent)):
            dx_avg += recent[i][0] - recent[i - 1][0]
            dy_avg += recent[i][1] - recent[i - 1][1]
        dir_len = sqrt(dx_avg ** 2 + dy_avg ** 2)
        if dir_len == 0:
            return 1.0
        last = path[-1]
        cx, cy = candidate[0], candidate[1]
        dx_new = cx - last[0]
        dy_new = cy - last[1]
        new_len = sqrt(dx_new ** 2 + dy_new ** 2)
        if new_len == 0:
            return 1.0
        cos_angle = (dx_avg * dx_new + dy_avg * dy_new) / (dir_len * new_len)
        smoothness = (cos_angle + 1) / 2
        return smoothness

    def detect_intersection(self, candidates):
        if len(candidates) < 2:
            return False, candidates
        sorted_cands = sorted(candidates, key=lambda x: -x[2])
        best_score = sorted_cands[0][2]
        high_score_cands = [c for c in sorted_cands if c[2] >= best_score * self.intersection_score_ratio]
        if len(high_score_cands) < 2:
            return False, sorted_cands
        directions = []
        for c in high_score_cands[:3]:
            dx, dy = c[3], c[4]
            dist = sqrt(dx ** 2 + dy ** 2)
            if dist > 0:
                directions.append((dx / dist, dy / dist))
        if len(directions) >= 2:
            cos_angle = directions[0][0] * directions[1][0] + directions[0][1] * directions[1][1]
            if cos_angle < self.intersection_angle_threshold:
                return True, high_score_cands
        return False, sorted_cands

    def select_smoothest_at_intersection(self, path, candidates):
        if not candidates:
            return None
        best_candidate = None
        best_smoothness = -1
        for c in candidates:
            smoothness = self.compute_smoothness_score(path, c)
            if smoothness > best_smoothness:
                best_smoothness = smoothness
                best_candidate = c
        return best_candidate

    def find_candidates_toward_target(self, frame_idx, cx, cy, path, direction, visited, target=None, radius=None):
        """
        寻找候选点，如果有目标点则优先向目标方向
        """
        if radius is None:
            radius = self.search_radius
        frame = self.frames[frame_idx]
        candidates = []
        search_range = int(radius)

        # 如果有目标，放宽方向限制
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

                # 向左加分
                if dx < 0:
                    left_bonus = abs(dx) / (dist + 0.1)
                    score *= (1 + left_bonus)

                if direction:
                    dir_x, dir_y = direction
                    dir_len = sqrt(dir_x ** 2 + dir_y ** 2)
                    if dir_len > 0:
                        cos_a = (dx * dir_x + dy * dir_y) / (dist * dir_len)
                        score *= (1 + (cos_a + 1) / 2)

                # 如果有目标点，向目标方向大幅加分
                if target:
                    tx, ty = target
                    to_target_dx = tx - cx
                    to_target_dy = ty - cy
                    to_target_len = sqrt(to_target_dx ** 2 + to_target_dy ** 2)
                    if to_target_len > 0:
                        # 计算候选点方向与目标方向的一致性
                        cos_to_target = (dx * to_target_dx + dy * to_target_dy) / (dist * to_target_len)
                        # 大幅加分
                        score *= (1 + 3.0 * (cos_to_target + 1) / 2)

                candidates.append((nx, ny, score, dx, dy))

        return candidates

    def trace_segment(self, frame_idx, start_x, start_y, target=None, initial_direction=None, max_steps=3000):
        """
        追踪一段路径
        如果有target，追踪到target附近就停止
        如果没有target，追踪到边界
        """
        path = [(int(start_x), int(start_y))]
        visited = set()
        visited.add((int(start_x), int(start_y)))
        cx, cy = int(start_x), int(start_y)
        direction = initial_direction if initial_direction else (-1, 0)
        boundary_reached = None
        reached_target = False

        for step in range(max_steps):
            # 检查是否到达目标
            if target:
                dist_to_target = sqrt((cx - target[0]) ** 2 + (cy - target[1]) ** 2)
                if dist_to_target <= self.waypoint_capture_radius:
                    reached_target = True
                    break

            # 检查边界
            reached, boundary_type = self.check_boundary_reached(cx, cy)
            if reached:
                boundary_reached = boundary_type
                break

            candidates = self.find_candidates_toward_target(
                frame_idx, cx, cy, path, direction, visited, target=target
            )

            if not candidates:
                for radius in range(self.search_radius, self.max_search_radius + 1, self.search_radius_step):
                    candidates = self.find_candidates_toward_target(
                        frame_idx, cx, cy, path, direction, visited, target=target, radius=radius
                    )
                    if candidates:
                        break

            if not candidates:
                break

            is_intersection, filtered_cands = self.detect_intersection(candidates)

            if is_intersection:
                chosen = self.select_smoothest_at_intersection(path, filtered_cands)
            else:
                filtered_cands.sort(key=lambda x: -x[2])
                chosen = filtered_cands[0]

            if chosen is None:
                break

            nx, ny = chosen[0], chosen[1]
            dx, dy = chosen[3], chosen[4]

            # 如果没有目标，强制向左
            if target is None and dx > 0:
                left_only = [c for c in filtered_cands if c[3] <= 0]
                if left_only:
                    if is_intersection:
                        chosen = self.select_smoothest_at_intersection(path, left_only)
                    else:
                        chosen = max(left_only, key=lambda x: x[2])
                    if chosen:
                        nx, ny = chosen[0], chosen[1]
                        dx, dy = chosen[3], chosen[4]
                    else:
                        break
                else:
                    break

            if (nx, ny) in visited:
                remaining = [c for c in filtered_cands if (c[0], c[1]) not in visited]
                if target is None:
                    remaining = [c for c in remaining if c[3] <= 0]
                if remaining:
                    if is_intersection:
                        chosen = self.select_smoothest_at_intersection(path, remaining)
                    else:
                        chosen = max(remaining, key=lambda x: x[2])
                    if chosen:
                        nx, ny = chosen[0], chosen[1]
                        dx, dy = chosen[3], chosen[4]
                    else:
                        break
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

    def connect_waypoints_directly(self, frame_idx, start, end):
        """
        在两个标记点之间直接连接（用于追踪失败时的备选）
        使用简单的亮度跟随
        """
        path = [start]
        cx, cy = start
        ex, ey = end

        for _ in range(2000):
            if sqrt((cx - ex) ** 2 + (cy - ey) ** 2) <= 3:
                path.append(end)
                break

            # 计算方向
            dx = ex - cx
            dy = ey - cy
            dist = sqrt(dx ** 2 + dy ** 2)
            if dist == 0:
                break

            # 归一化
            dx, dy = dx / dist, dy / dist

            # 步进
            best_pt = None
            best_score = -1

            for step_len in [1, 2, 3]:
                for angle_offset in [0, 0.2, -0.2, 0.4, -0.4]:
                    ndx = dx * cos(angle_offset) - dy * sin(angle_offset) if angle_offset != 0 else dx
                    ndy = dx * sin(angle_offset) + dy * cos(angle_offset) if angle_offset != 0 else dy
                    nx = int(cx + ndx * step_len)
                    ny = int(cy + ndy * step_len)

                    if 0 <= nx < self.width and 0 <= ny < self.height:
                        brightness = self.frames[frame_idx][ny, nx]
                        score = brightness - abs(angle_offset) * 50
                        if score > best_score:
                            best_score = score
                            best_pt = (nx, ny)

            if best_pt and best_pt != (cx, cy):
                path.append(best_pt)
                cx, cy = best_pt
            else:
                # 直接步进
                nx = int(cx + dx)
                ny = int(cy + dy)
                if (nx, ny) != (cx, cy):
                    path.append((nx, ny))
                    cx, cy = nx, ny
                else:
                    break

        return path

    def trace_through_waypoints(self, frame_idx, waypoints):
        """
        追踪经过所有路标点的路径
        waypoints: [(x,y), ...] 按从右到左排序
        """
        if not waypoints:
            return [], None

        full_path = []
        boundary_reached = None
        direction = (-1, 0)

        for i in range(len(waypoints)):
            start = waypoints[i]
            target = waypoints[i + 1] if i + 1 < len(waypoints) else None

            # 追踪这一段
            segment, boundary, reached_target, direction = self.trace_segment(
                frame_idx, start[0], start[1], target=target, initial_direction=direction
            )

            # 添加到完整路径
            if full_path:
                # 避免重复起点
                if segment and segment[0] == full_path[-1]:
                    segment = segment[1:]
            full_path.extend(segment)

            # 如果有目标但没到达，尝试直接连接
            if target and not reached_target:
                last_pt = full_path[-1] if full_path else start
                # 直接添加目标点，确保经过
                if last_pt != target:
                    # 尝试用简单方法连接
                    connect_path = self.connect_waypoints_directly(frame_idx, last_pt, target)
                    if connect_path:
                        full_path.extend(connect_path[1:])  # 跳过起点

            if boundary:
                boundary_reached = boundary
                break

        # 确保所有waypoint都在路径中
        for wp in waypoints:
            if wp not in full_path:
                # 找最近的位置插入
                min_dist = float('inf')
                insert_idx = len(full_path)
                for idx, pt in enumerate(full_path):
                    dist = sqrt((pt[0] - wp[0]) ** 2 + (pt[1] - wp[1]) ** 2)
                    if dist < min_dist:
                        min_dist = dist
                        insert_idx = idx
                full_path.insert(insert_idx, wp)

        return full_path, boundary_reached

    def trace_single_frame_through_markers(self, frame_idx, all_markers_for_neuron):
        """
        在单帧中追踪，强制经过所有相关标记点
        all_markers_for_neuron: 该神经元在所有帧的标记点
        """
        # 收集该帧和之前帧的标记点作为必经点
        waypoints = []

        # 首先添加该帧的标记
        if frame_idx in all_markers_for_neuron:
            waypoints.extend(all_markers_for_neuron[frame_idx])

        # 如果该帧没有标记，使用最近帧的标记作为参考起点
        if not waypoints:
            for f in range(frame_idx, -1, -1):
                if f in all_markers_for_neuron:
                    # 找该帧中能在当前帧找到信号的点
                    for pt in all_markers_for_neuron[f]:
                        if self.has_signal_near(frame_idx, pt[0], pt[1]):
                            waypoints.append(pt)
                    if waypoints:
                        break

        if not waypoints:
            return [], None

        # 按x坐标从大到小排序（从右到左）
        waypoints = sorted(set(waypoints), key=lambda p: -p[0])

        # 追踪经过所有路标点
        path, boundary = self.trace_through_waypoints(frame_idx, waypoints)

        return path, boundary

    def compute_frame_weight(self, frame_idx, earliest_frame, latest_frame):
        if earliest_frame == latest_frame:
            return 1.0
        position = (frame_idx - earliest_frame) / (latest_frame - earliest_frame)
        weight = self.weight_decay ** position
        return weight

    def compute_weighted_path(self, paths_by_frame, earliest_frame, latest_frame, required_waypoints=None):
        """
        计算加权路径，确保包含所有必经点
        """
        if not paths_by_frame:
            return []

        weights = {}
        total_weight = 0
        for f in paths_by_frame:
            w = self.compute_frame_weight(f, earliest_frame, latest_frame)
            weights[f] = w
            total_weight += w

        if total_weight > 0:
            for f in weights:
                weights[f] /= total_weight

        x_to_weighted_y = defaultdict(list)

        for frame_idx, path in paths_by_frame.items():
            weight = weights.get(frame_idx, 1.0)
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

        # 确保必经点在路径中
        if required_waypoints:
            for wp in required_waypoints:
                if wp not in final_path:
                    # 找最近位置插入
                    min_dist = float('inf')
                    insert_idx = 0
                    for idx, pt in enumerate(final_path):
                        if pt[0] <= wp[0]:
                            insert_idx = idx
                            break
                    final_path.insert(insert_idx, wp)

        if len(final_path) > 3:
            final_path = self.smooth_path_preserve_waypoints(final_path, required_waypoints)

        return final_path

    def smooth_path_preserve_waypoints(self, path, waypoints=None, window=3):
        """平滑路径，但保留必经点"""
        if len(path) <= window:
            return path

        waypoint_set = set(waypoints) if waypoints else set()

        smoothed = [path[0]]
        for i in range(1, len(path) - 1):
            if path[i] in waypoint_set:
                # 必经点不平滑
                smoothed.append(path[i])
            else:
                start = max(0, i - window // 2)
                end = min(len(path), i + window // 2 + 1)
                avg_x = sum(p[0] for p in path[start:end]) / (end - start)
                avg_y = sum(p[1] for p in path[start:end]) / (end - start)
                smoothed.append((int(round(avg_x)), int(round(avg_y))))
        smoothed.append(path[-1])

        return smoothed

    def compute_neuron_trajectory(self, neuron_id):
        """
        基于标记点计算神经元轨迹
        确保轨迹经过所有标记点
        """
        markers = self.get_neuron_markers(neuron_id)
        if not markers:
            return None

        print(f"\n{'=' * 60}")
        print(f"计算神经元 N{neuron_id} 轨迹 (强制经过标记点)")
        print(f"标记帧: {list(markers.keys())}")
        for f, pts in sorted(markers.items()):
            print(f"  帧{f}: {pts}")
        print(f"{'=' * 60}")

        # 收集所有标记点
        all_waypoints = []
        for pts in markers.values():
            all_waypoints.extend(pts)
        all_waypoints = list(set(all_waypoints))

        # 帧范围
        all_marked_frames = sorted(markers.keys())
        earliest_marked = min(all_marked_frames)
        latest_marked = max(all_marked_frames)

        # 向前找信号
        earliest_signal = earliest_marked
        for f in range(earliest_marked - 1, max(-1, earliest_marked - self.max_backtrack_check), -1):
            has_signal = False
            for wp in all_waypoints:
                if self.has_signal_near(f, wp[0], wp[1]):
                    has_signal = True
                    break
            if has_signal:
                earliest_signal = f
            else:
                break

        latest_signal = latest_marked

        print(f"信号帧范围: {earliest_signal} ~ {latest_signal}")
        print(f"必经点: {all_waypoints}")

        # 对每帧独立追踪
        paths_by_frame = {}

        for frame_idx in range(earliest_signal, latest_signal + 1):
            path, boundary = self.trace_single_frame_through_markers(frame_idx, markers)

            if len(path) > 3:
                paths_by_frame[frame_idx] = path
                left_x = min(p[0] for p in path)
                boundary_str = f"→{boundary}" if boundary else ""

                # 检查经过了多少必经点
                passed = sum(1 for wp in all_waypoints if wp in path or
                             any(sqrt((p[0] - wp[0]) ** 2 + (p[1] - wp[1]) ** 2) < 5 for p in path))

                print(
                    f"  帧 {frame_idx}: {len(path)}点, 左x={left_x}, 经过{passed}/{len(all_waypoints)}点 {boundary_str}")

        if not paths_by_frame:
            print("  没有找到有效路径！")
            return None

        # 计算加权叠加路径
        print(f"\n计算加权叠加...")
        final_path = self.compute_weighted_path(
            paths_by_frame, earliest_signal, latest_signal,
            required_waypoints=all_waypoints
        )

        if not final_path:
            return None

        if len(final_path) >= 2 and final_path[0][0] > final_path[-1][0]:
            final_path = final_path[::-1]

        # 验证必经点
        passed_count = sum(1 for wp in all_waypoints if wp in final_path or
                           any(sqrt((p[0] - wp[0]) ** 2 + (p[1] - wp[1]) ** 2) < 5 for p in final_path))

        print(f"  最终路径: {len(final_path)}点, 经过{passed_count}/{len(all_waypoints)}个必经点")

        # 判断边界
        end_pt = final_path[0]
        final_boundary_reached, final_boundary_type = self.check_boundary_reached(end_pt[0], end_pt[1])

        if final_boundary_reached:
            print(f"  ✓ 到达{final_boundary_type}边界")

        # 生成所有帧的路径
        all_paths = {}
        for frame_idx in range(earliest_signal, latest_signal + 1):
            if frame_idx in paths_by_frame:
                all_paths[frame_idx] = paths_by_frame[frame_idx]
            else:
                all_paths[frame_idx] = final_path.copy()

        for frame_idx in range(0, earliest_signal):
            all_paths[frame_idx] = final_path.copy()

        result = {
            'paths': all_paths,
            'final_path': final_path,
            'signal_range': (earliest_signal, latest_signal),
            'boundary_info': (final_boundary_reached, final_boundary_type),
            'markers': dict(markers),
            'waypoints': all_waypoints
        }

        self.tracking_results[neuron_id] = result

        print(f"\n完成 N{neuron_id}: 覆盖 {len(all_paths)} 帧")

        return result


class TrackerGUI:
    """GUI - 强制经过标记点版"""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("神经元追踪 - 强制经过标记点")
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

        right = ttk.Frame(self.main_paned, width=380)
        self.main_paned.add(right, weight=0)

        self.ctrl_canvas = tk.Canvas(right, width=360)
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
        self.input_entry = ttk.Entry(f1, width=24)
        self.input_entry.insert(0, self.input_dir)
        self.input_entry.grid(row=0, column=1, sticky="ew")
        ttk.Button(f1, text="...", command=self.browse_input, width=3).grid(row=0, column=2)
        ttk.Label(f1, text="输出:").grid(row=1, column=0, sticky="w", pady=2)
        self.output_entry = ttk.Entry(f1, width=24)
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

        # 标记操作
        f5 = ttk.LabelFrame(p, text="📍 标记操作 (必经点)", padding=5)
        f5.pack(fill="x", pady=pad, padx=3)

        ttk.Label(f5, text="左键：添加必经点\n右键：删除最后标记\n\n⚠️ 轨迹将强制经过所有标记点",
                  font=("", 9), justify="left").pack(anchor="w")

        self.marker_info = tk.StringVar(value="当前神经元无标记")
        ttk.Label(f5, textvariable=self.marker_info, foreground="blue", wraplength=320).pack(anchor="w", pady=3)

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
        ttk.Label(f6, textvariable=self.compute_info, foreground="green", wraplength=320).pack(anchor="w", pady=3)

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

        ttk.Label(g, text="捕获半径:").grid(row=1, column=0, sticky="w", pady=2)
        self.capture_var = tk.StringVar(value="25")
        ttk.Entry(g, textvariable=self.capture_var, width=5).grid(row=1, column=1)
        ttk.Label(g, text="(到达必经点距离)").grid(row=1, column=2, columnspan=2, sticky="w", padx=(5, 0))

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
        ttk.Label(f10, textvariable=self.status, wraplength=320, foreground="blue").pack(fill="x")
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
        if not markers:
            self.marker_info.set(f"N{self.current_neuron_id}: 无必经点")
        else:
            total_pts = sum(len(pts) for pts in markers.values())
            frames = sorted(markers.keys())
            self.marker_info.set(f"N{self.current_neuron_id}: {total_pts}个必经点 @ 帧{frames}")

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
        self.status.set(f"N{self.current_neuron_id}: 添加必经点 ({ix}, {iy}) @ 帧{self.current_frame_idx}")
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
            self.tracker.waypoint_capture_radius = int(self.capture_var.get())
        except:
            pass

    def compute_current_neuron(self):
        markers = self.tracker.get_neuron_markers(self.current_neuron_id)
        if not markers:
            messagebox.showwarning("警告", f"N{self.current_neuron_id} 没有标记点")
            return

        self.apply_params()
        self.status.set(f"计算 N{self.current_neuron_id} 轨迹中...")
        self.root.update()

        result = self.tracker.compute_neuron_trajectory(self.current_neuron_id)

        if result:
            self.preview_result = result
            self.preview_neuron_id = self.current_neuron_id

            sr = result['signal_range']
            br, bt = result['boundary_info']
            boundary_str = f"→{bt}" if br else ""
            wp_count = len(result.get('waypoints', []))

            self.compute_info.set(
                f"N{self.current_neuron_id}: 帧{sr[0]}~{sr[1]}\n"
                f"{len(result['final_path'])}点, {wp_count}个必经点 {boundary_str}\n"
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
            sr = result['signal_range']
            br, bt = result['boundary_info']
            bstr = f"→{bt}" if br else ""
            wp = len(result.get('waypoints', []))
            self.result_list.insert(tk.END, f"N{nid}: 帧{sr[0]}~{sr[1]} {wp}点 {bstr}")

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
        path = os.path.join(self.output_dir, "neuron_tracking.mp4")
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
        path = os.path.join(self.output_dir, "neuron_paths.csv")

        with open(path, 'w') as f:
            f.write("neuron_id,frame,x,y,is_waypoint\n")
            for nid, result in sorted(self.tracker.tracking_results.items()):
                waypoints = set(result.get('waypoints', []))
                for frame_idx, path_pts in sorted(result['paths'].items()):
                    for x, y in path_pts:
                        is_wp = 1 if (x, y) in waypoints else 0
                        f.write(f"{nid},{frame_idx},{x},{y},{is_wp}\n")

        self.status.set(f"数据已保存: {path}")
        messagebox.showinfo("完成", f"导出完成!\n{path}")

    def render(self, idx, for_export=False):
        frame = self.tracker.frames[idx]
        vis = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        # 边界线
        cv2.line(vis, (self.tracker.left_margin, 0),
                 (self.tracker.left_margin, self.tracker.height), (0, 100, 0), 1)
        cv2.line(vis, (0, self.tracker.edge_margin),
                 (self.tracker.width, self.tracker.edge_margin), (0, 100, 0), 1)
        cv2.line(vis, (0, self.tracker.height - self.tracker.edge_margin),
                 (self.tracker.width, self.tracker.height - self.tracker.edge_margin), (0, 100, 0), 1)

        # 已确认的结果
        for nid, result in self.tracker.tracking_results.items():
            color = self.get_neuron_color(nid)
            paths = result['paths']
            waypoints = set(result.get('waypoints', []))

            if idx in paths:
                path = paths[idx]
                for j in range(1, len(path)):
                    cv2.line(vis, path[j - 1], path[j], color, 2)
                if path:
                    cv2.circle(vis, path[0], 5, (0, 255, 0), -1)
                    cv2.circle(vis, path[-1], 5, color, -1)
                    cv2.putText(vis, f"N{nid}", (path[-1][0] + 5, path[-1][1] - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

                # 标记必经点
                for wp in waypoints:
                    if wp in path or any(sqrt((p[0] - wp[0]) ** 2 + (p[1] - wp[1]) ** 2) < 10 for p in path):
                        cv2.drawMarker(vis, wp, (255, 255, 255), cv2.MARKER_DIAMOND, 12, 2)

        # 预览
        if not for_export and self.preview_result and self.preview_neuron_id:
            color = self.get_neuron_color(self.preview_neuron_id)
            paths = self.preview_result['paths']
            waypoints = set(self.preview_result.get('waypoints', []))

            if idx in paths:
                path = paths[idx]
                for j in range(1, len(path)):
                    cv2.line(vis, path[j - 1], path[j], (0, 255, 255), 2)
                if path:
                    cv2.circle(vis, path[0], 6, (0, 255, 0), -1)
                    cv2.circle(vis, path[-1], 6, (0, 255, 255), -1)

                # 预览中的必经点
                for wp in waypoints:
                    cv2.drawMarker(vis, wp, (0, 255, 255), cv2.MARKER_DIAMOND, 14, 2)

        # 标记点（未计算的）
        if not for_export:
            for nid, frame_markers in self.tracker.markers.items():
                color = self.get_neuron_color(nid)
                # 当前帧的标记
                if idx in frame_markers:
                    for pt in frame_markers[idx]:
                        cv2.circle(vis, pt, 10, color, -1)
                        cv2.circle(vis, pt, 10, (255, 255, 255), 2)
                        cv2.putText(vis, f"N{nid}", (pt[0] + 12, pt[1] - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                # 其他帧的标记（小圆）
                for f, pts in frame_markers.items():
                    if f != idx:
                        for pt in pts:
                            cv2.circle(vis, pt, 4, color, 1)

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


# 需要导入sin函数
from math import sin

if __name__ == "__main__":
    print("启动神经元追踪 - 强制经过标记点")
    TrackerGUI().run()
