import cv2
import numpy as np
import os
from math import sqrt, cos, sin, radians
from collections import defaultdict
from scipy.ndimage import distance_transform_edt
import time


class NeuronTracker:
    """神经元追踪 - 走廊端点检测版（严格单调搜索）"""

    def __init__(self):
        self.frames = []
        self.frames_np = None
        self.height = 0
        self.width = 0

        # ── 参数 ──
        self.brightness_threshold = 30
        self.search_radius = 30
        self.linearity_weight = 2.0
        self.left_margin = 3
        self.edge_margin = 3
        self.right_margin = 3
        self.max_turn_angle = 60
        self.max_search_radius = 150
        self.search_radius_step = 20
        self.max_step_distance = 15
        self.min_step_distance = 1
        self.max_dark_gap = 5
        self.waypoint_capture_radius = 25

        # ── 端点检测参数 ──
        self.tip_corridor_radius = 3  # 沿路径检测时的邻域半径

        # ── 内部状态 ──
        self.markers = defaultdict(lambda: defaultdict(list))
        self.tracking_results = {}
        self.first_marker_by_neuron = {}
        self._search_grids = {}
        self._visited_matrix = None

    # ================================================================== #
    #  visited 矩阵操作                                                    #
    # ================================================================== #

    def _init_visited_matrix(self):
        self._visited_matrix = np.zeros((self.height, self.width), dtype=bool)

    def _mark_visited(self, x, y):
        self._visited_matrix[int(y), int(x)] = True

    def _mark_visited_points(self, points):
        for x, y in points:
            self._visited_matrix[int(y), int(x)] = True

    def _check_not_visited_batch(self, nx_arr, ny_arr):
        return ~self._visited_matrix[ny_arr.astype(int), nx_arr.astype(int)]

    # ================================================================== #
    #  帧加载                                                              #
    # ================================================================== #

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

        self.frames_np = np.array(self.frames, dtype=np.uint8)
        if progress_callback:
            progress_callback(1.0)
        return True

    # ================================================================== #
    #  搜索网格（预排序）                                                    #
    # ================================================================== #

    def _get_search_grid(self, radius, direction='left', max_dist=None):
        if max_dist is None:
            max_dist = radius
        key = (radius, direction, max_dist)
        if key not in self._search_grids:
            if direction == 'left':
                dx_range = np.arange(-radius, 1)
            else:
                dx_range = np.arange(0, radius + 1)
            dy_range = np.arange(-radius, radius + 1)
            dx_grid, dy_grid = np.meshgrid(dx_range, dy_range)
            dx_flat = dx_grid.flatten()
            dy_flat = dy_grid.flatten()
            dist_flat = np.sqrt(dx_flat ** 2 + dy_flat ** 2)
            valid = (dist_flat <= radius) & (dist_flat > 0) & (dist_flat <= max_dist)
            order = np.argsort(dist_flat[valid])
            self._search_grids[key] = {
                'dx': dx_flat[valid][order],
                'dy': dy_flat[valid][order],
                'dist': dist_flat[valid][order]
            }
        return self._search_grids[key]

    # ================================================================== #
    #  标记管理                                                            #
    # ================================================================== #

    def _recompute_first_marker(self, neuron_id):
        """
        重新计算“起始标记点”。
        为了避免用户后续补点（更靠左/更接近起点）被 _enforce_start_at_first_marker 截断丢失，
        这里将 first_marker 定义为：该神经元所有标记点中 X 最小的点（若相同取更早帧）。
        这也更符合本项目默认的“从左到右生长”的假设。
        """
        markers = self.markers.get(neuron_id)
        if not markers:
            self.first_marker_by_neuron.pop(neuron_id, None)
            return None

        best = None  # (x, y, frame)
        for fidx, pts in markers.items():
            for (x, y) in pts:
                cand = (x, y, fidx)
                if best is None:
                    best = cand
                else:
                    # 先比 x，再比帧号
                    if cand[0] < best[0] or (cand[0] == best[0] and cand[2] < best[2]):
                        best = cand
        if best is None:
            self.first_marker_by_neuron.pop(neuron_id, None)
            return None
        pt = (int(best[0]), int(best[1]))
        self.first_marker_by_neuron[neuron_id] = pt
        return pt

    def add_marker(self, neuron_id, frame_idx, x, y):
        pt = (int(x), int(y))
        self.markers[neuron_id][frame_idx].append(pt)
        # 起点使用“最靠左”的标记点（可随着用户补点而更新）
        if neuron_id not in self.first_marker_by_neuron:
            self.first_marker_by_neuron[neuron_id] = pt
        else:
            cur = self.first_marker_by_neuron[neuron_id]
            if pt[0] < cur[0]:
                self.first_marker_by_neuron[neuron_id] = pt

    def remove_last_marker(self, neuron_id, frame_idx):
        if neuron_id in self.markers and frame_idx in self.markers[neuron_id]:
            if self.markers[neuron_id][frame_idx]:
                pt = self.markers[neuron_id][frame_idx].pop()
                if self.first_marker_by_neuron.get(neuron_id) == pt:
                    self._recompute_first_marker(neuron_id)
                if not self.markers[neuron_id][frame_idx]:
                    del self.markers[neuron_id][frame_idx]
                if not self.markers[neuron_id]:
                    del self.markers[neuron_id]
                    self.first_marker_by_neuron.pop(neuron_id, None)
                return pt
        return None

    def remove_specific_marker(self, neuron_id, frame_idx, point):
        if neuron_id in self.markers and frame_idx in self.markers[neuron_id]:
            try:
                self.markers[neuron_id][frame_idx].remove(point)
                if self.first_marker_by_neuron.get(neuron_id) == point:
                    self._recompute_first_marker(neuron_id)
                if not self.markers[neuron_id][frame_idx]:
                    del self.markers[neuron_id][frame_idx]
                if not self.markers[neuron_id]:
                    del self.markers[neuron_id]
                    self.first_marker_by_neuron.pop(neuron_id, None)
            except ValueError:
                pass

    def clear_neuron_markers(self, neuron_id):
        if neuron_id in self.markers:
            del self.markers[neuron_id]
        self.first_marker_by_neuron.pop(neuron_id, None)

    def _refresh_first_marker(self, neuron_id):
        # 保留函数名兼容旧调用，但实际逻辑改为“最靠左点”
        self._recompute_first_marker(neuron_id)

    def _get_first_marker_point(self, neuron_id):
        if neuron_id in self.first_marker_by_neuron:
            return self.first_marker_by_neuron[neuron_id]
        markers = self.get_neuron_markers(neuron_id)
        if markers:
            # 懒加载：重新计算一次，确保是最靠左点
            pt = self._recompute_first_marker(neuron_id)
            if pt is not None:
                return pt
        return None

    def _enforce_start_at_first_marker(self, neuron_id, path):
        if not path:
            return path
        first_marker = self._get_first_marker_point(neuron_id)
        if first_marker is None:
            return path
        arr = np.array(path)
        dists = (arr[:, 0] - first_marker[0]) ** 2 + (arr[:, 1] - first_marker[1]) ** 2
        nearest_idx = int(np.argmin(dists))
        trimmed = path[nearest_idx:]
        if not trimmed:
            return [first_marker]
        trimmed[0] = first_marker
        return trimmed

    def get_neuron_markers(self, neuron_id):
        return dict(self.markers.get(neuron_id, {}))

    def get_all_waypoints(self, neuron_id):
        markers = self.get_neuron_markers(neuron_id)
        all_points = []
        for pts in markers.values():
            all_points.extend(pts)
        if not all_points:
            return []
        # 旧逻辑会按 5x5 网格去重，可能导致“用户明明点了多个点，但轨迹只保留其中一部分”。
        # 这里改为：仅对完全相同的坐标去重，保留用户标记的每一个不同点。
        seen = set()
        unique = []
        for p in all_points:
            if p not in seen:
                seen.add(p)
                unique.append(p)
        return unique

    # ================================================================== #
    #  边界 / 亮度检查                                                      #
    # ================================================================== #

    def is_bright(self, frame_idx, x, y):
        if frame_idx < 0 or frame_idx >= len(self.frames):
            return False
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return False
        return self.frames_np[frame_idx, int(y), int(x)] > self.brightness_threshold

    def check_left_boundary(self, x, y):
        if x <= self.left_margin:
            return True, 'left'
        if y <= self.edge_margin:
            return True, 'top'
        if y >= self.height - self.edge_margin:
            return True, 'bottom'
        return False, None

    def check_right_boundary(self, x, y):
        if x >= self.width - self.right_margin:
            return True, 'right'
        if y <= self.edge_margin:
            return True, 'top'
        if y >= self.height - self.edge_margin:
            return True, 'bottom'
        return False, None

    # ================================================================== #
    #  合并向量化指标计算                                                    #
    # ================================================================== #

    def compute_all_metrics_vectorized(self, path, cand_x, cand_y):
        n_cand = len(cand_x)
        result = {
            'is_smooth': np.ones(n_cand, dtype=bool),
            'smoothness': np.ones(n_cand, dtype=np.float64),
            'linearity': np.ones(n_cand, dtype=np.float64),
        }
        if len(path) < 2 or n_cand == 0:
            return result

        last_x, last_y = path[-1]
        dx_new = cand_x - last_x
        dy_new = cand_y - last_y
        new_len = np.sqrt(dx_new * dx_new + dy_new * dy_new)
        new_len_safe = np.where(new_len == 0, 1.0, new_len)

        # 最近5个点 → is_smooth
        n5 = min(5, len(path))
        recent5 = path[-n5:]
        dx5, dy5 = 0.0, 0.0
        for i in range(1, len(recent5)):
            dx5 += recent5[i][0] - recent5[i - 1][0]
            dy5 += recent5[i][1] - recent5[i - 1][1]
        len5 = sqrt(dx5 * dx5 + dy5 * dy5)
        min_cos = cos(radians(self.max_turn_angle))
        if len5 > 0:
            cos5 = (dx5 * dx_new + dy5 * dy_new) / (len5 * new_len_safe)
            result['is_smooth'] = (cos5 >= min_cos) | (new_len == 0)

        # 最近8个点 → smoothness + linearity
        n8 = min(8, len(path))
        recent8 = path[-n8:]
        dx8, dy8 = 0.0, 0.0
        for i in range(1, len(recent8)):
            dx8 += recent8[i][0] - recent8[i - 1][0]
            dy8 += recent8[i][1] - recent8[i - 1][1]
        len8 = sqrt(dx8 * dx8 + dy8 * dy8)
        if len8 > 0:
            cos8 = (dx8 * dx_new + dy8 * dy_new) / (len8 * new_len_safe)
            result['smoothness'] = (cos8 + 1.0) / 2.0
            result['linearity'] = (cos8 + 1.0) / 2.0

        return result

    # ================================================================== #
    #  候选点搜索（向左）                                                    #
    # ================================================================== #

    def find_candidates_leftward_vectorized(self, frame_idx, cx, cy, path, direction,
                                            visited_unused,
                                            target=None, radius=None, max_dist=None):
        if radius is None:
            radius = self.search_radius
        if max_dist is None:
            max_dist = self.max_step_distance

        frame = self.frames_np[frame_idx]
        grid = self._get_search_grid(radius, 'left', max_dist)
        dx = grid['dx']
        dy = grid['dy']
        nx = cx + dx
        ny = cy + dy

        valid_bounds = (nx >= 0) & (nx < self.width) & (ny >= 0) & (ny < self.height)
        if not np.any(valid_bounds):
            return []

        nx_v = nx[valid_bounds].astype(np.intp)
        ny_v = ny[valid_bounds].astype(np.intp)
        brightness = frame[ny_v, nx_v]
        bright_mask = brightness > self.brightness_threshold
        if not np.any(bright_mask):
            return []

        nx_b = nx_v[bright_mask]
        ny_b = ny_v[bright_mask]
        bright_b = brightness[bright_mask].astype(np.float64)

        not_vis = self._check_not_visited_batch(nx_b, ny_b)
        if not np.any(not_vis):
            return []

        nx_f = nx_b[not_vis].astype(np.float64)
        ny_f = ny_b[not_vis].astype(np.float64)
        bright_f = bright_b[not_vis]
        dx_f = nx_f - cx
        dy_f = ny_f - cy
        dist_f = np.sqrt(dx_f * dx_f + dy_f * dy_f)
        dist_f_safe = np.where(dist_f == 0, 1.0, dist_f)

        metrics = self.compute_all_metrics_vectorized(path, nx_f, ny_f)
        is_smooth = metrics['is_smooth']
        if not np.any(is_smooth):
            threshold = np.percentile(metrics['smoothness'], 50)
            is_smooth = metrics['smoothness'] >= threshold

        m = is_smooth
        nx_s, ny_s = nx_f[m], ny_f[m]
        bright_s = bright_f[m]
        dx_s, dy_s = dx_f[m], dy_f[m]
        dist_s = dist_f_safe[m]
        lin_s = metrics['linearity'][m]
        smo_s = metrics['smoothness'][m]

        if len(nx_s) == 0:
            return []

        scores = bright_s * (lin_s ** self.linearity_weight) / (dist_s + 0.5)
        scores *= (1.0 + smo_s)
        scores *= (1.0 + np.abs(dx_s) / (dist_s + 0.1))

        if direction:
            dir_x, dir_y = direction
            dir_len = sqrt(dir_x ** 2 + dir_y ** 2)
            if dir_len > 0:
                cos_a = (dx_s * dir_x + dy_s * dir_y) / (dist_s * dir_len)
                scores *= (1.0 + (cos_a + 1.0) / 2.0)

        if target:
            tx, ty = target
            tdx = tx - cx
            tdy = ty - cy
            tlen = sqrt(tdx ** 2 + tdy ** 2)
            if tlen > 0:
                cos_t = (dx_s * tdx + dy_s * tdy) / (dist_s * tlen)
                scores *= (1.0 + 3.0 * (cos_t + 1.0) / 2.0)

        return [(int(nx_s[i]), int(ny_s[i]), float(scores[i]),
                 int(dx_s[i]), int(dy_s[i])) for i in range(len(nx_s))]

    # ================================================================== #
    #  候选点搜索（向右）                                                    #
    # ================================================================== #

    def find_candidates_rightward_vectorized(self, frame_idx, cx, cy, path, direction,
                                             visited_unused, radius=None):
        if radius is None:
            radius = self.search_radius

        frame = self.frames_np[frame_idx]
        effective_radius = min(radius, self.max_step_distance)
        grid = self._get_search_grid(effective_radius, 'right', self.max_step_distance)
        dx = grid['dx']
        dy = grid['dy']
        nx = cx + dx
        ny = cy + dy

        valid_bounds = (nx >= 0) & (nx < self.width) & (ny >= 0) & (ny < self.height)
        if not np.any(valid_bounds):
            return []

        nx_v = nx[valid_bounds].astype(np.intp)
        ny_v = ny[valid_bounds].astype(np.intp)
        brightness = frame[ny_v, nx_v]
        bright_mask = brightness > self.brightness_threshold
        if not np.any(bright_mask):
            return []

        nx_b = nx_v[bright_mask]
        ny_b = ny_v[bright_mask]
        bright_b = brightness[bright_mask].astype(np.float64)

        not_vis = self._check_not_visited_batch(nx_b, ny_b)
        if not np.any(not_vis):
            return []

        nx_f = nx_b[not_vis].astype(np.float64)
        ny_f = ny_b[not_vis].astype(np.float64)
        bright_f = bright_b[not_vis]
        dx_f = nx_f - cx
        dy_f = ny_f - cy
        dist_f = np.sqrt(dx_f * dx_f + dy_f * dy_f)
        dist_f_safe = np.where(dist_f == 0, 1.0, dist_f)

        metrics = self.compute_all_metrics_vectorized(path, nx_f, ny_f)
        is_smooth = metrics['is_smooth']
        if not np.any(is_smooth):
            if len(metrics['smoothness']) > 0:
                is_smooth = metrics['smoothness'] >= np.percentile(metrics['smoothness'], 50)
            else:
                return []

        m = is_smooth
        nx_s, ny_s = nx_f[m], ny_f[m]
        bright_s = bright_f[m]
        dx_s, dy_s = dx_f[m], dy_f[m]
        dist_s = dist_f_safe[m]
        lin_s = metrics['linearity'][m]
        smo_s = metrics['smoothness'][m]

        if len(nx_s) == 0:
            return []

        scores = bright_s * (lin_s ** self.linearity_weight) / (dist_s + 0.5)
        scores *= (1.0 + 2.0 * smo_s)
        scores *= (1.0 + np.maximum(0, dx_s) / (dist_s + 0.1))

        if direction:
            dir_x, dir_y = direction
            dir_len = sqrt(dir_x ** 2 + dir_y ** 2)
            if dir_len > 0:
                cos_a = (dx_s * dir_x + dy_s * dir_y) / (dist_s * dir_len)
                scores *= (1.0 + (cos_a + 1.0) / 2.0)

        optimal_dist = self.max_step_distance * 0.5
        dist_penalty = np.abs(dist_s - optimal_dist) / optimal_dist
        scores *= (1.0 - 0.3 * dist_penalty)

        return [(int(nx_s[i]), int(ny_s[i]), float(scores[i]),
                 int(dx_s[i]), int(dy_s[i])) for i in range(len(nx_s))]

    # ================================================================== #
    #  段追踪（向左）                                                       #
    # ================================================================== #

    def trace_segment_left(self, frame_idx, start_x, start_y, target=None,
                           initial_direction=None, max_steps=3000):
        path = [(int(start_x), int(start_y))]
        self._init_visited_matrix()
        self._mark_visited(int(start_x), int(start_y))
        cx, cy = int(start_x), int(start_y)
        direction = initial_direction if initial_direction else (-1, 0)
        boundary_reached = None
        reached_target = False

        for step in range(max_steps):
            if target:
                if sqrt((cx - target[0]) ** 2 + (cy - target[1]) ** 2) <= self.waypoint_capture_radius:
                    reached_target = True
                    break

            reached, btype = self.check_left_boundary(cx, cy)
            if reached:
                boundary_reached = btype
                break

            candidates = self.find_candidates_leftward_vectorized(
                frame_idx, cx, cy, path, direction, None, target=target)

            if not candidates:
                for r in range(self.search_radius, self.max_search_radius + 1,
                               self.search_radius_step):
                    candidates = self.find_candidates_leftward_vectorized(
                        frame_idx, cx, cy, path, direction, None,
                        target=target, radius=r, max_dist=r)
                    if candidates:
                        break

            if not candidates:
                break

            candidates.sort(key=lambda x: -x[2])
            chosen = candidates[0]
            nx, ny, _, dxc, dyc = chosen

            if target is None and dxc > 0:
                left_only = [c for c in candidates if c[3] <= 0]
                if left_only:
                    chosen = max(left_only, key=lambda x: x[2])
                    nx, ny, _, dxc, dyc = chosen
                else:
                    break

            if self._visited_matrix[ny, nx]:
                remaining = [c for c in candidates if not self._visited_matrix[c[1], c[0]]]
                if target is None:
                    remaining = [c for c in remaining if c[3] <= 0]
                if remaining:
                    chosen = max(remaining, key=lambda x: x[2])
                    nx, ny, _, dxc, dyc = chosen
                else:
                    break

            path.append((nx, ny))
            self._mark_visited(nx, ny)

            dist = sqrt(dxc ** 2 + dyc ** 2)
            if dist > 0:
                nd = (dxc / dist, dyc / dist)
                a = 0.3
                direction = (direction[0] * (1 - a) + nd[0] * a,
                             direction[1] * (1 - a) + nd[1] * a)
                dl = sqrt(direction[0] ** 2 + direction[1] ** 2)
                if dl > 0:
                    direction = (direction[0] / dl, direction[1] / dl)
            cx, cy = nx, ny

        return path, boundary_reached, reached_target, direction

    # ================================================================== #
    #  两点直连（numpy 插值 + 局部最亮微调）                                  #
    # ================================================================== #

    def connect_points_directly(self, frame_idx, start, end):
        x1, y1 = start
        x2, y2 = end
        dist = max(abs(x2 - x1), abs(y2 - y1))
        if dist == 0:
            return [start]
        frame = self.frames_np[frame_idx]
        t = np.linspace(0, 1, dist + 1)
        xs = np.clip(np.round(x1 + t * (x2 - x1)).astype(int), 0, self.width - 1)
        ys = np.clip(np.round(y1 + t * (y2 - y1)).astype(int), 0, self.height - 1)
        path = []
        for i in range(len(xs)):
            bx, by = int(xs[i]), int(ys[i])
            y_lo, y_hi = max(0, by - 1), min(self.height, by + 2)
            x_lo, x_hi = max(0, bx - 1), min(self.width, bx + 2)
            patch = frame[y_lo:y_hi, x_lo:x_hi]
            lm = np.unravel_index(patch.argmax(), patch.shape)
            path.append((x_lo + int(lm[1]), y_lo + int(lm[0])))
        deduped = [path[0]]
        for p in path[1:]:
            if p != deduped[-1]:
                deduped.append(p)
        return deduped

    # ================================================================== #
    #  向右生长                                                            #
    # ================================================================== #

    def grow_rightward(self, frame_idx, prev_path, prev_direction, max_steps=300):
        if not prev_path:
            return [], None, None
        sx, sy = prev_path[-1]
        reached, btype = self.check_right_boundary(sx, sy)
        if reached:
            return [], btype, prev_direction

        new_points = []
        self._init_visited_matrix()
        self._mark_visited_points(prev_path)
        cx, cy = sx, sy
        direction = prev_direction if prev_direction else (1, 0)
        boundary_reached = None
        context = list(prev_path[-10:])
        consec_fail = 0

        for step in range(max_steps):
            reached, btype = self.check_right_boundary(cx, cy)
            if reached:
                boundary_reached = btype
                break

            cands = self.find_candidates_rightward_vectorized(
                frame_idx, cx, cy, context + new_points, direction, None)
            if not cands:
                for r in range(self.search_radius, self.max_search_radius + 1,
                               self.search_radius_step):
                    cands = self.find_candidates_rightward_vectorized(
                        frame_idx, cx, cy, context + new_points, direction, None, radius=r)
                    if cands:
                        break
            if not cands:
                consec_fail += 1
                if consec_fail >= 5:
                    break
                continue
            consec_fail = 0

            cands.sort(key=lambda x: -x[2])
            chosen = cands[0]
            nx, ny, _, dxc, dyc = chosen

            if dxc < 0:
                right_only = [c for c in cands if c[3] >= 0]
                if right_only:
                    chosen = max(right_only, key=lambda x: x[2])
                    nx, ny, _, dxc, dyc = chosen
                else:
                    consec_fail += 1
                    if consec_fail >= 5:
                        break
                    continue

            if self._visited_matrix[ny, nx]:
                remaining = [c for c in cands
                             if not self._visited_matrix[c[1], c[0]] and c[3] >= 0]
                if remaining:
                    chosen = max(remaining, key=lambda x: x[2])
                    nx, ny, _, dxc, dyc = chosen
                else:
                    consec_fail += 1
                    if consec_fail >= 5:
                        break
                    continue

            new_points.append((nx, ny))
            self._mark_visited(nx, ny)
            dist = sqrt(dxc ** 2 + dyc ** 2)
            if dist > 0:
                nd = (dxc / dist, dyc / dist)
                a = 0.3
                direction = (direction[0] * (1 - a) + nd[0] * a,
                             direction[1] * (1 - a) + nd[1] * a)
                dl = sqrt(direction[0] ** 2 + direction[1] ** 2)
                if dl > 0:
                    direction = (direction[0] / dl, direction[1] / dl)
            cx, cy = nx, ny

        return new_points, boundary_reached, direction

    # ================================================================== #
    #  走廊参考路径构建                                                     #
    # ================================================================== #

    def trace_left_unified(self, frame_idx, waypoints):
        if not waypoints:
            return [], None

        sorted_wp = sorted(waypoints, key=lambda p: p[0])

        # 骨架连线
        skeleton = []
        for i in range(len(sorted_wp) - 1):
            x1, y1 = sorted_wp[i]
            x2, y2 = sorted_wp[i + 1]
            dx, dy = x2 - x1, y2 - y1
            dist = max(abs(dx), abs(dy))
            if dist == 0:
                if not skeleton or skeleton[-1] != (x1, y1):
                    skeleton.append((x1, y1))
                continue
            for j in range(dist):
                pt = (int(round(x1 + j * dx / dist)), int(round(y1 + j * dy / dist)))
                if not skeleton or skeleton[-1] != pt:
                    skeleton.append(pt)
        if not skeleton or skeleton[-1] != sorted_wp[-1]:
            skeleton.append(sorted_wp[-1])

        # 距离变换 → 走廊距离图
        skeleton_mask = np.zeros((self.height, self.width), dtype=bool)
        for sx, sy in skeleton:
            if 0 <= sx < self.width and 0 <= sy < self.height:
                skeleton_mask[sy, sx] = True
        corridor_dist_map = distance_transform_edt(~skeleton_mask).astype(np.float32)

        CORRIDOR_RADIUS = 15
        frame = self.frames_np[frame_idx]

        full_path = [sorted_wp[0]]
        self._init_visited_matrix()
        self._mark_visited(sorted_wp[0][0], sorted_wp[0][1])
        cx, cy = sorted_wp[0]
        direction = (1, 0)
        # 依次确保路径经过每一个 waypoint（用户标记点）
        next_wp_idx = 1  # 下一个目标 waypoint 的索引
        steps_since_progress = 0
        no_dist_progress_steps = 0
        last_best_dist_to_tgt = None

        for step in range(5000):
            # 已经把所有 waypoint 都纳入路径：结束
            if next_wp_idx >= len(sorted_wp):
                break

            # 如果已经接近下一个 waypoint，则强制把路径连到该点，确保“必经点必经”
            tgt = sorted_wp[next_wp_idx]
            cur_dist_to_tgt = sqrt((cx - tgt[0]) ** 2 + (cy - tgt[1]) ** 2)
            if last_best_dist_to_tgt is None:
                last_best_dist_to_tgt = cur_dist_to_tgt

            if sqrt((cx - tgt[0]) ** 2 + (cy - tgt[1]) ** 2) <= self.waypoint_capture_radius:
                seg = self.connect_points_directly(frame_idx, (cx, cy), tgt)
                if seg:
                    if seg[0] == full_path[-1]:
                        seg = seg[1:]
                    for px, py in seg:
                        full_path.append((px, py))
                        self._mark_visited(px, py)
                    # 强制把 waypoint 本身放进路径，避免 connect_points_directly 的“亮点微调”偏离目标点
                    if full_path[-1] != tgt:
                        full_path.append(tgt)
                        self._mark_visited(tgt[0], tgt[1])
                    cx, cy = tgt
                else:
                    # 兜底：至少把 waypoint 写入路径
                    if full_path[-1] != tgt:
                        full_path.append(tgt)
                        self._mark_visited(tgt[0], tgt[1])
                    cx, cy = tgt
                next_wp_idx += 1
                steps_since_progress = 0
                no_dist_progress_steps = 0
                last_best_dist_to_tgt = None
                continue

            eff_r = min(self.search_radius, self.max_step_distance)
            grid = self._get_search_grid(eff_r, 'right', self.max_step_distance)
            nx = cx + grid['dx']
            ny = cy + grid['dy']
            vb = (nx >= 0) & (nx < self.width) & (ny >= 0) & (ny < self.height)
            if not np.any(vb):
                break

            nx_v = nx[vb].astype(np.intp)
            ny_v = ny[vb].astype(np.intp)
            brightness = frame[ny_v, nx_v]

            not_vis = self._check_not_visited_batch(nx_v, ny_v)
            if not np.any(not_vis):
                break

            nx_f = nx_v[not_vis].astype(np.float64)
            ny_f = ny_v[not_vis].astype(np.float64)
            bright_f = brightness[not_vis].astype(np.float64)
            dx_f = nx_f - cx
            dy_f = ny_f - cy
            dist_f = np.sqrt(dx_f * dx_f + dy_f * dy_f)
            dist_f_safe = np.where(dist_f == 0, 1.0, dist_f)

            metrics = self.compute_all_metrics_vectorized(full_path, nx_f, ny_f)
            is_smooth = metrics['is_smooth']
            if not np.any(is_smooth):
                if len(metrics['smoothness']) > 0:
                    is_smooth = metrics['smoothness'] >= np.percentile(metrics['smoothness'], 50)
                else:
                    break

            nx_s = nx_f[is_smooth].astype(np.intp)
            ny_s = ny_f[is_smooth].astype(np.intp)
            dist_s = dist_f_safe[is_smooth]
            bright_s = bright_f[is_smooth]

            if len(nx_s) == 0:
                steps_since_progress += 1
                if steps_since_progress >= 30:
                    # 走廊/阈值导致无法继续推进：直接连到目标 waypoint，保证强制必经
                    seg = self.connect_points_directly(frame_idx, (cx, cy), tgt)
                    if seg:
                        if seg[0] == full_path[-1]:
                            seg = seg[1:]
                        for px, py in seg:
                            full_path.append((px, py))
                            self._mark_visited(px, py)
                    if full_path[-1] != tgt:
                        full_path.append(tgt)
                        self._mark_visited(tgt[0], tgt[1])
                    cx, cy = tgt
                    next_wp_idx += 1
                    steps_since_progress = 0
                    no_dist_progress_steps = 0
                    last_best_dist_to_tgt = None
                    continue
                break

            cdists = corridor_dist_map[ny_s, nx_s]
            in_corr = cdists <= CORRIDOR_RADIUS

            if not np.any(in_corr):
                next_sk = None
                for sk_pt in skeleton:
                    if sk_pt[0] > cx:
                        next_sk = sk_pt
                        break
                if next_sk:
                    best_pt = next_sk
                else:
                    steps_since_progress += 1
                    if steps_since_progress >= 30:
                        seg = self.connect_points_directly(frame_idx, (cx, cy), tgt)
                        if seg:
                            if seg[0] == full_path[-1]:
                                seg = seg[1:]
                            for px, py in seg:
                                full_path.append((px, py))
                                self._mark_visited(px, py)
                        if full_path[-1] != tgt:
                            full_path.append(tgt)
                            self._mark_visited(tgt[0], tgt[1])
                        cx, cy = tgt
                        next_wp_idx += 1
                        steps_since_progress = 0
                        no_dist_progress_steps = 0
                        last_best_dist_to_tgt = None
                        continue
                    break
            else:
                nx_c = nx_s[in_corr].astype(np.float64)
                ny_c = ny_s[in_corr].astype(np.float64)
                dist_c = dist_s[in_corr]
                bright_c = bright_s[in_corr]
                cdist_c = cdists[in_corr].astype(np.float64)
                lin_c = metrics['linearity'][is_smooth][in_corr]
                smo_c = metrics['smoothness'][is_smooth][in_corr]

                scores = bright_c * (lin_c ** self.linearity_weight) / (dist_c + 0.5)
                scores *= (1.0 + 2.0 * smo_c)
                scores *= (1.0 - 0.8 * cdist_c / CORRIDOR_RADIUS)

                # 额外拉向下一个 waypoint（避免走廊内“绕开必经点”）
                tx, ty = tgt
                tdx = float(tx - cx)
                tdy = float(ty - cy)
                tlen = sqrt(tdx * tdx + tdy * tdy)
                if tlen > 1e-6:
                    cos_t = (dx_f[is_smooth][in_corr] * tdx + dy_f[is_smooth][in_corr] * tdy) / (dist_c * tlen)
                    scores *= (1.0 + 1.5 * (cos_t + 1.0) / 2.0)

                best_idx = int(np.argmax(scores))
                best_pt = (int(nx_c[best_idx]), int(ny_c[best_idx]))

            full_path.append(best_pt)
            self._mark_visited(best_pt[0], best_pt[1])
            steps_since_progress = 0

            # 如果路径在“错误分支”上一直走但并没有靠近目标 waypoint，则强制拉回（硬约束优先）
            new_dist_to_tgt = sqrt((best_pt[0] - tgt[0]) ** 2 + (best_pt[1] - tgt[1]) ** 2)
            if new_dist_to_tgt < (last_best_dist_to_tgt - 0.5):
                last_best_dist_to_tgt = new_dist_to_tgt
                no_dist_progress_steps = 0
            else:
                no_dist_progress_steps += 1
                if no_dist_progress_steps >= 120:
                    seg = self.connect_points_directly(frame_idx, (best_pt[0], best_pt[1]), tgt)
                    if seg:
                        if seg[0] == full_path[-1]:
                            seg = seg[1:]
                        for px, py in seg:
                            full_path.append((px, py))
                            self._mark_visited(px, py)
                    if full_path[-1] != tgt:
                        full_path.append(tgt)
                        self._mark_visited(tgt[0], tgt[1])
                    cx, cy = tgt
                    next_wp_idx += 1
                    steps_since_progress = 0
                    no_dist_progress_steps = 0
                    last_best_dist_to_tgt = None
                    continue

            dx = best_pt[0] - cx
            dy = best_pt[1] - cy
            dlen = sqrt(dx ** 2 + dy ** 2)
            if dlen > 0:
                nd = (dx / dlen, dy / dlen)
                direction = (direction[0] * 0.7 + nd[0] * 0.3,
                             direction[1] * 0.7 + nd[1] * 0.3)
                dl2 = sqrt(direction[0] ** 2 + direction[1] ** 2)
                direction = (direction[0] / dl2, direction[1] / dl2)
            cx, cy = best_pt

        # 若由于步数限制/中途 break 导致仍有未到达的 waypoint：强制逐个直连补齐
        while next_wp_idx < len(sorted_wp):
            tgt = sorted_wp[next_wp_idx]
            seg = self.connect_points_directly(frame_idx, (cx, cy), tgt)
            if seg:
                if seg[0] == full_path[-1]:
                    seg = seg[1:]
                for px, py in seg:
                    full_path.append((px, py))
            if full_path[-1] != tgt:
                full_path.append(tgt)
            cx, cy = tgt
            next_wp_idx += 1

        # 最终兜底：确保每个 waypoint 坐标都出现在路径里（完全优先于搜索）
        for wp in sorted_wp:
            if wp in full_path:
                continue
            arr = np.array(full_path, dtype=np.int32)
            dists = (arr[:, 0] - wp[0]) ** 2 + (arr[:, 1] - wp[1]) ** 2
            insert_idx = int(np.argmin(dists))
            full_path.insert(insert_idx + 1, wp)

        return full_path, None

    # ================================================================== #
    #  平滑                                                                #
    # ================================================================== #

    def smooth_path_preserve_waypoints(self, path, waypoints, window=3):
        if len(path) <= window:
            return path
        path_arr = np.array(path)
        wp_indices = set()
        for wp in waypoints:
            dists = (path_arr[:, 0] - wp[0]) ** 2 + (path_arr[:, 1] - wp[1]) ** 2
            idx = int(np.argmin(dists))
            if dists[idx] < 9:
                wp_indices.add(idx)
        smoothed = [path[0]]
        hw = window // 2
        for i in range(1, len(path) - 1):
            if i in wp_indices:
                smoothed.append(path[i])
            else:
                s = max(0, i - hw)
                e = min(len(path), i + hw + 1)
                avg_x = sum(p[0] for p in path[s:e]) / (e - s)
                avg_y = sum(p[1] for p in path[s:e]) / (e - s)
                smoothed.append((int(round(avg_x)), int(round(avg_y))))
        smoothed.append(path[-1])
        return smoothed

    # ================================================================== #
    #  最佳帧选择                                                          #
    # ================================================================== #

    def find_best_frame(self, neuron_id, waypoints):
        markers = self.get_neuron_markers(neuron_id)
        marked_frames = list(markers.keys())
        best_frame = min(marked_frames) if marked_frames else 0
        best_score = 0
        for f in marked_frames:
            frame = self.frames_np[f]
            score = 0
            for wp in waypoints:
                x, y = wp
                y1, y2 = max(0, y - 5), min(self.height, y + 6)
                x1, x2 = max(0, x - 5), min(self.width, x + 6)
                score += np.sum(frame[y1:y2, x1:x2])
            if score > best_score:
                best_score = score
                best_frame = f
        return best_frame

    @staticmethod
    def _nearest_idx(path, pt):
        arr = np.array(path)
        dists = (arr[:, 0] - pt[0]) ** 2 + (arr[:, 1] - pt[1]) ** 2
        return int(np.argmin(dists))

    def _force_waypoints_into_path(self, frame_idx, path, waypoints):
        """
        强制硬必经：无论走廊/评分/搜索结果如何，最终 path 必须包含 waypoints 中每个点（坐标级别）。
        做法：若某 waypoint 不在 path 中，则在其最近的路径位置附近插入一段连接，使 polyline 必然穿过该点。
        """
        if not path or not waypoints:
            return path

        out = list(path)
        for wp in waypoints:
            if wp in out:
                continue

            # 选择插入位置：离 waypoint 最近的路径点索引
            ins = self._nearest_idx(out, wp)
            ins = max(0, min(ins, len(out) - 1))

            if ins >= len(out) - 1:
                # 插到末尾
                seg = self.connect_points_directly(frame_idx, out[-1], wp)
                if seg:
                    if seg[0] == out[-1]:
                        seg = seg[1:]
                    out.extend(seg)
                if out[-1] != wp:
                    out.append(wp)
                continue

            prev_pt = out[ins]
            next_pt = out[ins + 1]

            seg1 = self.connect_points_directly(frame_idx, prev_pt, wp) or [prev_pt, wp]
            seg2 = self.connect_points_directly(frame_idx, wp, next_pt) or [wp, next_pt]

            # 拼接两段，去重端点
            merged = []
            for p in seg1:
                if not merged or merged[-1] != p:
                    merged.append(p)
            for p in seg2:
                if not merged or merged[-1] != p:
                    merged.append(p)

            # 硬插入：prev_pt ... wp ... next_pt
            # out[:ins+1] 以 prev_pt 结尾；merged 以 prev_pt 开头
            if merged and merged[0] == prev_pt:
                merged = merged[1:]
            if merged and merged[-1] == next_pt:
                merged = merged[:-1]
            out = out[:ins + 1] + merged + out[ins + 1:]

            # 再保证 wp 坐标一定存在
            if wp not in out:
                out.insert(ins + 1, wp)

        return out

    def _compute_required_tip_by_frame(self, neuron_id, reference_path):
        """
        根据用户标记点（带帧号）生成“硬约束 tip”：
        - 若用户在帧 f 标了某个点 p，则在显示/计算帧 f 及之后，路径至少要包含该点。
        - 做法：将 p 映射到 reference_path 上的索引 idx（最近点/精确点），然后要求 tip_idx >= idx+1。
        这能保证：黄色预览（paths_by_frame[current_frame]）一定经过用户在该帧标记的所有点。
        """
        n_frames = len(self.frames)
        required = np.zeros(n_frames, dtype=np.int32)
        markers = self.get_neuron_markers(neuron_id)  # {frame: [pt,...]}
        if not markers or not reference_path:
            return required

        # 逐帧累积最大要求（帧 f 的点在 f 及之后都必须包含）
        for f, pts in markers.items():
            if f < 0 or f >= n_frames:
                continue
            for pt in pts:
                idx = self._nearest_idx(reference_path, pt)
                required[f] = max(required[f], idx + 1)

        # 前向传播：后续帧也必须满足前面帧的约束
        for f in range(1, n_frames):
            if required[f] < required[f - 1]:
                required[f] = required[f - 1]
        return required

    def _local_max_brightness(self, frame_idx, x, y, radius=3):
        """取 (x,y) 邻域内最大亮度（用于判断该帧是否支持把 tip 拉到某个位置）。"""
        if self.frames_np is None:
            return 0
        x = int(x)
        y = int(y)
        y1 = max(0, y - radius)
        y2 = min(self.height, y + radius + 1)
        x1 = max(0, x - radius)
        x2 = min(self.width, x + radius + 1)
        patch = self.frames_np[frame_idx, y1:y2, x1:x2]
        if patch.size == 0:
            return 0
        return int(patch.max())

    def _build_paths_with_growth(self, neuron_id, reference_path, tips, required_tip, max_steps=600):
        """
        从 reference_path 构建每帧路径：
        - reference 前缀长度由 max(tips, required_tip) 控制（硬约束）
        - grow_rightward 产生 tail，仅用于该次前向生长；当 reference 前缀变长时会丢弃 tail
        """
        n_frames = len(self.frames)
        paths_by_frame = {}
        tip_track = []

        current_ref_end = max(1, min(max(int(tips[0]), int(required_tip[0])), len(reference_path)))
        current_tail = []
        base_prefix = list(reference_path[:current_ref_end])
        current_path = base_prefix
        current_dir = self._estimate_direction_from_path(current_path, fallback=(1.0, 0.0))

        paths_by_frame[0] = list(current_path)
        tip_track.append(current_path[-1] if current_path else None)

        for f in range(1, n_frames):
            desired_end = max(1, min(max(int(tips[f]), int(required_tip[f])), len(reference_path)))
            if desired_end > current_ref_end:
                current_ref_end = desired_end
                current_tail = []  # 新硬约束出现：丢弃旧 tail，避免锁死

            base_prefix = list(reference_path[:current_ref_end])
            current_path = base_prefix + current_tail
            if current_tail and base_prefix and current_path[len(base_prefix) - 1] == current_tail[0]:
                current_path = base_prefix + current_tail[1:]

            current_dir = self._estimate_direction_from_path(current_path, fallback=current_dir)
            new_points, _, new_dir = self.grow_rightward(
                frame_idx=f,
                prev_path=current_path,
                prev_direction=current_dir,
                max_steps=max_steps,
            )
            if new_points:
                current_tail = current_tail + new_points
                current_path = base_prefix + current_tail
                if new_dir is not None:
                    current_dir = new_dir

            paths_by_frame[f] = list(current_path)
            tip_track.append(current_path[-1] if current_path else None)

        return paths_by_frame, tip_track

    def _backward_correct_required_tip(self, reference_path, tips, required_tip, tip_track,
                                       jump_px=25, idx_jump=40, brightness_ratio=0.9):
        """
        后向纠错（用户选择的方案）：
        - 如果某帧 tip 明显“跳错分支”（与下一帧 tip 差异大），但下一帧又回到正确路，
          则尝试用下一帧的 tip 在 reference_path 上的位置来抬高当前帧 required_tip，
          从而让当前帧的黄色预览也落到正确分支上。
        - 只有当该帧在该位置附近确实“够亮”时才抬高，避免强行拉到不存在信号的位置。
        """
        n_frames = len(self.frames)
        if n_frames < 2 or not reference_path:
            return required_tip

        ref_len = len(reference_path)
        proj_idx = np.zeros(n_frames, dtype=np.int32)
        for f in range(n_frames):
            pt = tip_track[f] if f < len(tip_track) else None
            if pt is None:
                proj_idx[f] = int(tips[f])
            else:
                proj_idx[f] = min(ref_len - 1, max(0, self._nearest_idx(reference_path, pt)))

        req = np.array(required_tip, dtype=np.int32).copy()
        for f in range(n_frames - 2, -1, -1):
            p0 = tip_track[f]
            p1 = tip_track[f + 1]
            if p0 is None or p1 is None:
                continue

            dx = float(p0[0] - p1[0])
            dy = float(p0[1] - p1[1])
            dist = sqrt(dx * dx + dy * dy)
            idx_diff = abs(int(proj_idx[f + 1]) - int(proj_idx[f]))

            if dist < jump_px and idx_diff < idx_jump:
                continue

            candidate_end = min(ref_len, int(proj_idx[f + 1]) + 1)
            if candidate_end <= int(req[f]):
                continue

            cx, cy = reference_path[candidate_end - 1]
            b = self._local_max_brightness(f, cx, cy, radius=self.tip_corridor_radius)
            if b >= int(self.brightness_threshold * brightness_ratio):
                req[f] = candidate_end

        # 确保单调：后续帧也必须满足前面帧的约束
        for f in range(1, n_frames):
            if req[f] < req[f - 1]:
                req[f] = req[f - 1]
        return req

    # ================================================================== #
    # ★★★  核心：全帧端点检测（严格从上一帧端点继续）  ★★★                     #
    # ================================================================== #

    def _build_neighborhood_offsets(self, radius):
        """生成圆形邻域的 (dy, dx) 偏移列表"""
        offsets = []
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx * dx + dy * dy <= radius * radius:
                    offsets.append((dy, dx))
        return offsets

    def _detect_tips_all_frames(self, reference_path):
        """
        逐帧顺序检测端点，严格从上一帧的端点位置继续向前搜索。

        规则：
          - 帧0：从路径起点(索引0)开始向前扫描
          - 帧f：从帧f-1的端点索引开始，只往前找，不回头
          - 遇到连续暗区 > max_dark_gap → 停止
          - 端点只增不减（严格时间单调性）

        返回：
          tips: np.array shape=(n_frames,)，每帧端点在参考路径中的索引
        """
        n_frames = len(self.frames)
        n_path = len(reference_path)
        radius = self.tip_corridor_radius

        # ── 1. 预提取路径坐标 ──
        path_xs = np.array([p[0] for p in reference_path], dtype=np.int32)
        path_ys = np.array([p[1] for p in reference_path], dtype=np.int32)

        # ── 2. 建立邻域偏移 ──
        offsets = self._build_neighborhood_offsets(radius)
        offset_dy = np.array([o[0] for o in offsets], dtype=np.int32)
        offset_dx = np.array([o[1] for o in offsets], dtype=np.int32)

        # ── 3. 预计算所有采样坐标: (n_offsets, n_path) ──
        sample_ys = np.clip(path_ys[None, :] + offset_dy[:, None], 0, self.height - 1)
        sample_xs = np.clip(path_xs[None, :] + offset_dx[:, None], 0, self.width - 1)

        # ── 4. 批量读取所有帧的亮度: (n_frames, n_offsets, n_path) ──
        brightness_3d = self.frames_np[:, sample_ys, sample_xs]

        # ── 5. 邻域取最大值 → (n_frames, n_path) ──
        max_brightness = brightness_3d.max(axis=1)

        # ── 6. 二值化 ──
        is_bright = max_brightness >= self.brightness_threshold  # (n_frames, n_path)

        # ── 7. 逐帧顺序搜索端点（从上一帧端点开始，只增不减） ──
        tips = np.zeros(n_frames, dtype=np.int32)
        current_tip = 0  # 帧0从索引0开始

        for f in range(n_frames):
            # 从 current_tip 开始往前扫描
            if current_tip >= n_path:
                tips[f] = current_tip
                continue

            bright_slice = is_bright[f, current_tip:]

            if len(bright_slice) == 0:
                tips[f] = current_tip
                continue

            # 逐步向前，允许 max_dark_gap 个暗点
            dark_count = 0
            new_tip = current_tip  # 至少保持不退

            for j in range(len(bright_slice)):
                if bright_slice[j]:
                    new_tip = current_tip + j + 1  # 这个点亮了，端点推进到它之后
                    dark_count = 0
                else:
                    dark_count += 1
                    if dark_count > self.max_dark_gap:
                        break  # 连续暗区过长，停止

            current_tip = new_tip
            tips[f] = current_tip

        return tips

    # ================================================================== #
    # ★★★  主入口：计算轨迹  ★★★                                           #
    # ================================================================== #

    @staticmethod
    def _estimate_direction_from_path(path, fallback=(1.0, 0.0)):
        """从路径末端估计生长方向（单位向量）。"""
        if not path or len(path) < 2:
            return fallback
        x0, y0 = path[-2]
        x1, y1 = path[-1]
        dx, dy = (x1 - x0), (y1 - y0)
        d = sqrt(dx * dx + dy * dy)
        if d <= 1e-6:
            return fallback
        return (dx / d, dy / d)

    def compute_neuron_trajectory(self, neuron_id, locked_frame=None):
        """
        完整流程：
          1. 在最优帧上构建走廊参考路径（经过所有 waypoint）
          2. 尝试向右生长延伸参考路径
          3. 对所有帧逐帧检测端点（严格从上一帧端点继续搜索）
          4. 输出 paths_by_frame
        """
        start_time = time.time()

        markers = self.get_neuron_markers(neuron_id)
        if not markers:
            return None

        all_waypoints = self.get_all_waypoints(neuron_id)
        if not all_waypoints:
            return None

        all_wp_sorted = sorted(all_waypoints, key=lambda p: p[0])
        marked_frames = sorted(markers.keys())
        first_marked_frame = min(marked_frames)

        print(f"\n{'=' * 60}")
        print(f"计算神经元 N{neuron_id} 轨迹 (走廊端点检测 - 严格单调)")
        print(f"标记帧: {marked_frames}, 首帧: {first_marked_frame}")
        print(f"必经点({len(all_wp_sorted)}个): {all_wp_sorted}")
        print(f"{'=' * 60}")

        # ── 步骤1：选帧 + 建参考路径 ──
        t1 = time.time()

        if locked_frame is not None and 0 <= locked_frame < len(self.frames):
            best_frame = locked_frame
            print(f"\n[锁定参考帧 {best_frame}]")
        else:
            best_frame = self.find_best_frame(neuron_id, all_wp_sorted)
            print(f"\n[自动选帧 {best_frame}]")

        reference_path, _ = self.trace_left_unified(best_frame, all_wp_sorted)

        if not reference_path:
            print("  追踪失败！")
            return None

        # 保证从左到右
        if len(reference_path) >= 2 and reference_path[0][0] > reference_path[-1][0]:
            reference_path = reference_path[::-1]
        reference_path = self._enforce_start_at_first_marker(neuron_id, reference_path)
        # 最终硬必经兜底：无论走廊怎么跑，必须把用户点全部塞进参考路径
        reference_path = self._force_waypoints_into_path(best_frame, reference_path, all_waypoints)

        t2 = time.time()
        print(f"  参考路径: {len(reference_path)}点, 耗时{t2 - t1:.3f}秒")

        # ── 步骤2：向右生长延伸参考路径 ──
        t3 = time.time()
        growth_count = 0

        # 检查最后一帧信号是否接近参考路径尾端
        last_frame_np = self.frames_np[-1]
        path_xs = np.array([p[0] for p in reference_path], dtype=np.int32)
        path_ys = np.array([p[1] for p in reference_path], dtype=np.int32)
        last_bright = last_frame_np[
            np.clip(path_ys, 0, self.height - 1),
            np.clip(path_xs, 0, self.width - 1)
        ]
        if len(last_bright) >= 10 and np.sum(last_bright[-10:] >= self.brightness_threshold) > 5:
            if len(reference_path) >= 2:
                dx = reference_path[-1][0] - reference_path[-2][0]
                dy = reference_path[-1][1] - reference_path[-2][1]
                d = sqrt(dx ** 2 + dy ** 2)
                cur_dir = (dx / d, dy / d) if d > 0 else (1, 0)
            else:
                cur_dir = (1, 0)

            for grow_frame in [len(self.frames) - 1,
                               len(self.frames) - 2,
                               len(self.frames) - 3]:
                if grow_frame < 0:
                    continue
                new_pts, _, new_dir = self.grow_rightward(
                    grow_frame, reference_path, cur_dir)
                if new_pts:
                    reference_path.extend(new_pts)
                    growth_count += len(new_pts)
                    if new_dir:
                        cur_dir = new_dir
                    print(f"  帧{grow_frame} 向右生长 +{len(new_pts)}点")

        t4 = time.time()
        print(f"  向右生长: +{growth_count}点, 耗时{t4 - t3:.3f}秒")
        print(f"  最终参考路径: {len(reference_path)}点, "
              f"X范围[{reference_path[0][0]}, {reference_path[-1][0]}]")
        # 生长后再次确保硬必经（避免延伸/截断等逻辑导致极端情况下丢点）
        reference_path = self._force_waypoints_into_path(best_frame, reference_path, all_waypoints)

        # ── 步骤3：全帧端点检测（严格从上一帧继续） ──
        t5 = time.time()
        tips = self._detect_tips_all_frames(reference_path)
        t6 = time.time()

        tip_min = int(tips.min())
        tip_max = int(tips.max())
        print(f"\n  全帧端点检测耗时: {t6 - t5:.3f}秒")
        print(f"  端点索引范围: [{tip_min}, {tip_max}]")
        print(f"  端点X范围: [{reference_path[max(0, tip_min)][0]}, "
              f"{reference_path[min(len(reference_path) - 1, tip_max)][0]}]")

        # ── 重要：用户标记点 = 硬约束，任何时候都必须经过 ──
        # 依据“标记帧”生成每帧最小 tip 要求，确保黄色预览必经所有已标记点
        required_tip = self._compute_required_tip_by_frame(neuron_id, reference_path)

        # ── 步骤4：先前向生长，再做“后向纠错”，最后重建一次 paths_by_frame ──
        paths_by_frame, tip_track = self._build_paths_with_growth(
            neuron_id, reference_path, tips, required_tip, max_steps=600
        )
        required_tip = self._backward_correct_required_tip(
            reference_path, tips, required_tip, tip_track,
            jump_px=25, idx_jump=40, brightness_ratio=0.9
        )
        paths_by_frame, tip_track = self._build_paths_with_growth(
            neuron_id, reference_path, tips, required_tip, max_steps=600
        )

        n_frames = len(self.frames)
        max_path = paths_by_frame[n_frames - 1]

        result = {
            'initial_path': reference_path,
            'reference_path': reference_path,
            'paths_by_frame': paths_by_frame,
            'waypoints': all_waypoints,
            'left_boundary': None,
            'first_marked_frame': first_marked_frame,
            'best_frame': best_frame,
            'final_tip_x': max_path[-1][0] if max_path else 0,
            'tips': tips,
            'tip_track': tip_track,  # [(x,y)|None] 每帧端点位置，用于视频/分析展示
            'required_tip': required_tip,  # 调试/可视化用：每帧硬约束 tip
        }

        self.tracking_results[neuron_id] = result

        total = time.time() - start_time
        print(f"\n完成 N{neuron_id}: 总耗时{total:.3f}秒")
        print(f"  帧0端点: idx={tips[0]}, pos={paths_by_frame[0][-1]}")
        print(f"  帧{n_frames - 1}端点: idx={tips[-1]}, pos={max_path[-1]}")

        return result

    # ================================================================== #
    #  局部重算                                                             #
    # ================================================================== #

    def local_recompute_with_waypoints(self, neuron_id, new_waypoints,
                                       locked_frame=None):
        """
        添加新 waypoint 后局部重算：
          - 只重建参考路径的受影响段
          - 然后重新做全帧端点检测（numpy 批量，极快）
        """
        result = self.tracking_results.get(neuron_id)
        if not result or not result.get('reference_path'):
            print(f"N{neuron_id} 无原始结果，完整重算")
            return self.compute_neuron_trajectory(neuron_id, locked_frame)

        best_frame = locked_frame if locked_frame is not None else result.get('best_frame', 0)
        base_path = list(result['reference_path'])
        first_marked = result.get('first_marked_frame', 0)

        all_wp_sorted = sorted(self.get_all_waypoints(neuron_id), key=lambda p: p[0])
        # 关键修复：old_wp 不能包含 new_waypoints，否则“左端/右端/中间”的分类会被干扰
        #（尤其新点距离旧点很近时，原本的距离过滤会误删旧点，导致走错分支）
        existing_waypoints = [w for w in all_wp_sorted if w not in new_waypoints]
        if not existing_waypoints:
            existing_waypoints = all_wp_sorted

        print(f"\n局部重算 N{neuron_id}，参考帧={best_frame}")
        print(f"全部必经点: {all_wp_sorted}")
        print(f"新点: {new_waypoints}")

        def nearest_idx(path, pt):
            return self._nearest_idx(path, pt)

        def build_segment(start_pt, targets, end_pt):
            seg = [start_pt]
            cur = start_pt
            idx = nearest_idx(base_path, start_pt)
            look_fwd = min(idx + 5, len(base_path) - 1)
            dx0 = base_path[look_fwd][0] - start_pt[0]
            dy0 = base_path[look_fwd][1] - start_pt[1]
            d0 = sqrt(dx0 ** 2 + dy0 ** 2)
            trace_dir = (dx0 / d0, dy0 / d0) if d0 > 0 else (1, 0)

            for tgt in list(targets) + [end_pt]:
                sub, _, reached, trace_dir = self.trace_segment_left(
                    best_frame, cur[0], cur[1],
                    target=tgt, initial_direction=trace_dir)
                if len(sub) >= 2 and sub[0][0] > sub[-1][0]:
                    sub = sub[::-1]
                if sub and sub[0] == seg[-1]:
                    sub = sub[1:]
                if not reached:
                    direct = self.connect_points_directly(best_frame, cur, tgt)
                    if direct and direct[0] == seg[-1]:
                        direct = direct[1:]
                    sub = direct
                seg.extend(sub)
                cur = seg[-1] if seg else tgt
            return seg

        # ── 逐个处理新点，修改 base_path ──
        for new_pt in new_waypoints:
            nx_new = new_pt[0]
            old_wp = list(existing_waypoints)
            if not old_wp:
                return self.compute_neuron_trajectory(neuron_id, locked_frame)

            leftmost_wp = min(old_wp, key=lambda p: p[0])
            rightmost_wp = max(old_wp, key=lambda p: p[0])

            # ── 情况1：新点在最左端之外 ──
            if nx_new <= leftmost_wp[0]:
                anchor_idx = nearest_idx(base_path, leftmost_wp)
                anchor_point = base_path[anchor_idx]

                print(f"  新点{new_pt} 在左端外，锚点={anchor_point}(idx={anchor_idx})")

                seg_left, _, _, _ = self.trace_segment_left(
                    best_frame, new_pt[0], new_pt[1],
                    target=None, initial_direction=(-1, 0))
                if len(seg_left) >= 2 and seg_left[0][0] > seg_left[-1][0]:
                    seg_left = seg_left[::-1]

                seg_right, _, reached_r, _ = self.trace_segment_left(
                    best_frame, new_pt[0], new_pt[1],
                    target=anchor_point, initial_direction=(1, 0))
                if len(seg_right) >= 2 and seg_right[0][0] > seg_right[-1][0]:
                    seg_right = seg_right[::-1]
                if not reached_r:
                    seg_right = self.connect_points_directly(best_frame, new_pt, anchor_point)

                if seg_right and seg_left and seg_right[0] == seg_left[-1]:
                    seg_right = seg_right[1:]
                new_left = seg_left + seg_right
                if not new_left or new_left[-1] != anchor_point:
                    new_left.append(anchor_point)
                if len(new_left) > 5:
                    new_left = self.smooth_path_preserve_waypoints(new_left, [new_pt])

                base_path = new_left + base_path[anchor_idx + 1:]
                print(f"  左端重追完成，段长={len(new_left)}")

            # ── 情况2：新点在最右端之外 ──
            elif nx_new >= rightmost_wp[0]:
                anchor_idx = nearest_idx(base_path, rightmost_wp)
                anchor_point = base_path[anchor_idx]
                before = base_path[:anchor_idx]

                print(f"  新点{new_pt} 在右端外，锚点={anchor_point}(idx={anchor_idx})")

                seg, _, reached, _ = self.trace_segment_left(
                    best_frame, anchor_point[0], anchor_point[1],
                    target=new_pt, initial_direction=(1, 0))
                if len(seg) >= 2 and seg[0][0] > seg[-1][0]:
                    seg = seg[::-1]
                if not reached:
                    seg = self.connect_points_directly(best_frame, anchor_point, new_pt)
                if not seg or seg[0] != anchor_point:
                    seg = [anchor_point] + (seg or [])
                if len(seg) > 5:
                    seg = self.smooth_path_preserve_waypoints(seg, [new_pt])

                base_path = before + seg
                print(f"  右端重追完成，段长={len(seg)}")

            # ── 情况3：新点在两个 waypoint 之间 ──
            else:
                left_wp = max((w for w in old_wp if w[0] <= nx_new), key=lambda p: p[0])
                right_wp = min((w for w in old_wp if w[0] > nx_new), key=lambda p: p[0])
                left_idx = nearest_idx(base_path, left_wp)
                right_idx = nearest_idx(base_path, right_wp)
                if left_idx > right_idx:
                    left_idx, right_idx = right_idx, left_idx

                anchor_start = base_path[left_idx]
                anchor_end = base_path[right_idx]
                before = base_path[:left_idx]
                after = base_path[right_idx + 1:]

                print(f"  新点{new_pt} 夹在 {left_wp}(idx={left_idx}) "
                      f"和 {right_wp}(idx={right_idx}) 之间")

                seg_mid_wp = sorted(
                    [w for w in old_wp if left_wp[0] < w[0] < right_wp[0]] + [new_pt],
                    key=lambda p: p[0])
                print(f"  段内必经点(含新): {seg_mid_wp}")

                full_seg = build_segment(anchor_start, seg_mid_wp, anchor_end)
                if not full_seg or full_seg[0] != anchor_start:
                    full_seg = [anchor_start] + (full_seg or [])
                if full_seg[-1] != anchor_end:
                    full_seg.append(anchor_end)
                if len(full_seg) > 5:
                    full_seg = self.smooth_path_preserve_waypoints(full_seg, seg_mid_wp)

                base_path = before + full_seg + after
                print(f"  中间段替换完成，段长={len(full_seg)}")

        base_path = self._enforce_start_at_first_marker(neuron_id, base_path)
        # 局部重算同样做硬必经兜底
        base_path = self._force_waypoints_into_path(best_frame, base_path, self.get_all_waypoints(neuron_id))

        # ── 参考路径更新完毕，重新做全帧端点检测 ──
        print(f"  参考路径更新完毕，{len(base_path)}点，重新检测端点...")
        tips = self._detect_tips_all_frames(base_path)

        # 用户标记点硬约束（局部重算也必须保持必经）
        required_tip = self._compute_required_tip_by_frame(neuron_id, base_path)

        # ── 前向生长 → 后向纠错 → 重建（与 compute_neuron_trajectory 对齐） ──
        paths_by_frame, tip_track = self._build_paths_with_growth(
            neuron_id, base_path, tips, required_tip, max_steps=600
        )
        required_tip = self._backward_correct_required_tip(
            base_path, tips, required_tip, tip_track,
            jump_px=25, idx_jump=40, brightness_ratio=0.9
        )
        paths_by_frame, tip_track = self._build_paths_with_growth(
            neuron_id, base_path, tips, required_tip, max_steps=600
        )

        max_path = paths_by_frame.get(n_frames - 1, base_path)

        new_result = {
            'initial_path': base_path,
            'reference_path': base_path,
            'paths_by_frame': paths_by_frame,
            'waypoints': self.get_all_waypoints(neuron_id),
            'left_boundary': None,
            'first_marked_frame': first_marked,
            'best_frame': best_frame,
            'final_tip_x': max_path[-1][0] if max_path else 0,
            'tips': tips,
            'tip_track': tip_track,
            'required_tip': required_tip,
        }

        self.tracking_results[neuron_id] = new_result
        print(f"局部重算完成，参考路径{len(base_path)}点，最终{len(max_path)}点")
        return new_result

    # ================================================================== #
    #  速度计算                                                            #
    # ================================================================== #

    def compute_neuron_speed(self, neuron_id, fps=10.0, pixel_um=1.0):
        result = self.tracking_results.get(neuron_id)
        if not result:
            return []

        paths_by_frame = result.get('paths_by_frame', {})
        sorted_frames = sorted(paths_by_frame.keys())
        if len(sorted_frames) < 2:
            return []

        tips = []
        for f in sorted_frames:
            p = paths_by_frame[f]
            if p:
                tips.append((f, p[-1][0], p[-1][1]))
        if len(tips) < 2:
            return []

        tips_arr = np.array(tips, dtype=np.float64)
        dx = np.diff(tips_arr[:, 1])
        dy = np.diff(tips_arr[:, 2])
        dist_px = np.sqrt(dx * dx + dy * dy)
        speed_um = dist_px * pixel_um * fps

        return [{
            'frame': int(tips_arr[i + 1, 0]),
            'tip_x': int(tips_arr[i + 1, 1]),
            'tip_y': int(tips_arr[i + 1, 2]),
            'dx': int(dx[i]),
            'dy': int(dy[i]),
            'speed_px_per_frame': round(float(dist_px[i]), 4),
            'speed_um_per_sec': round(float(speed_um[i]), 4),
        } for i in range(len(dx))]

    def compute_all_speeds(self, fps=10.0, pixel_um=1.0):
        return {nid: self.compute_neuron_speed(nid, fps, pixel_um)
                for nid in self.tracking_results}

    # ================================================================== #
    #  端点位置                                                            #
    # ================================================================== #

    def compute_neuron_tip_positions(self, neuron_id):
        result = self.tracking_results.get(neuron_id)
        if not result:
            return []

        paths_by_frame = result.get('paths_by_frame', {})
        tips = []
        for f in sorted(paths_by_frame.keys()):
            p = paths_by_frame[f]
            if p:
                tips.append({
                    'frame': f,
                    'tip_x': p[-1][0],
                    'tip_y': p[-1][1],
                    'path_points': len(p),
                })
        return tips

    def compute_all_tip_positions(self):
        return {nid: self.compute_neuron_tip_positions(nid)
                for nid in self.tracking_results}
