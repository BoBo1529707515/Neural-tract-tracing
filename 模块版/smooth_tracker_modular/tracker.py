import cv2
import numpy as np
import os
from math import sqrt, cos, sin, radians
from collections import defaultdict
import time


class NeuronTracker:
    """神经元追踪 - 平滑约束优化版"""

    def __init__(self):
        self.frames = []
        self.frames_np = None
        self.height = 0
        self.width = 0

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

        self.waypoint_capture_radius = 25

        self.markers = defaultdict(lambda: defaultdict(list))
        self.tracking_results = {}

        self._search_grids = {}

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

            self._search_grids[key] = {
                'dx': dx_flat[valid],
                'dy': dy_flat[valid],
                'dist': dist_flat[valid]
            }

        return self._search_grids[key]

    def add_marker(self, neuron_id, frame_idx, x, y):
        self.markers[neuron_id][frame_idx].append((int(x), int(y)))

    def remove_last_marker(self, neuron_id, frame_idx):
        if neuron_id in self.markers and frame_idx in self.markers[neuron_id]:
            if self.markers[neuron_id][frame_idx]:
                pt = self.markers[neuron_id][frame_idx].pop()
                if not self.markers[neuron_id][frame_idx]:
                    del self.markers[neuron_id][frame_idx]
                if not self.markers[neuron_id]:
                    del self.markers[neuron_id]
                return pt
        return None

    def remove_specific_marker(self, neuron_id, frame_idx, point):
        if neuron_id in self.markers and frame_idx in self.markers[neuron_id]:
            try:
                self.markers[neuron_id][frame_idx].remove(point)
                if not self.markers[neuron_id][frame_idx]:
                    del self.markers[neuron_id][frame_idx]
                if not self.markers[neuron_id]:
                    del self.markers[neuron_id]
            except ValueError:
                pass

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

    def is_smooth_transition(self, path, new_point, direction=None):
        if len(path) < 2:
            return True, 1.0

        n = min(5, len(path))
        recent = path[-n:]
        dx_prev, dy_prev = 0, 0
        for i in range(1, len(recent)):
            dx_prev += recent[i][0] - recent[i - 1][0]
            dy_prev += recent[i][1] - recent[i - 1][1]

        prev_len = sqrt(dx_prev ** 2 + dy_prev ** 2)
        if prev_len == 0:
            return True, 1.0

        last = path[-1]
        dx_new = new_point[0] - last[0]
        dy_new = new_point[1] - last[1]
        new_len = sqrt(dx_new ** 2 + dy_new ** 2)

        if new_len == 0:
            return True, 1.0

        cos_angle = (dx_prev * dx_new + dy_prev * dy_new) / (prev_len * new_len)
        min_cos = cos(radians(self.max_turn_angle))

        return cos_angle >= min_cos, cos_angle

    def compute_smoothness_score(self, path, candidate_x, candidate_y):
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
        dx_new = candidate_x - last[0]
        dy_new = candidate_y - last[1]
        new_len = sqrt(dx_new ** 2 + dy_new ** 2)

        if new_len == 0:
            return 1.0

        cos_angle = (dx_avg * dx_new + dy_avg * dy_new) / (dir_len * new_len)
        smoothness = (cos_angle + 1) / 2

        return smoothness

    def compute_smoothness_vectorized(self, path, candidate_coords):
        if len(path) < 2 or len(candidate_coords) == 0:
            return np.ones(len(candidate_coords))

        n = min(8, len(path))
        recent = path[-n:]

        dx_avg, dy_avg = 0, 0
        for i in range(1, len(recent)):
            dx_avg += recent[i][0] - recent[i - 1][0]
            dy_avg += recent[i][1] - recent[i - 1][1]

        dir_len = sqrt(dx_avg ** 2 + dy_avg ** 2)
        if dir_len == 0:
            return np.ones(len(candidate_coords))

        last_x, last_y = path[-1]

        cand_arr = np.array(candidate_coords)
        dx_new = cand_arr[:, 0] - last_x
        dy_new = cand_arr[:, 1] - last_y
        new_len = np.sqrt(dx_new ** 2 + dy_new ** 2)
        new_len = np.where(new_len == 0, 1, new_len)

        cos_angle = (dx_avg * dx_new + dy_avg * dy_new) / (dir_len * new_len)
        smoothness = (cos_angle + 1) / 2

        return smoothness

    def check_smooth_transition_vectorized(self, path, candidate_coords):
        if len(path) < 2 or len(candidate_coords) == 0:
            return np.ones(len(candidate_coords), dtype=bool)

        n = min(5, len(path))
        recent = path[-n:]

        dx_prev, dy_prev = 0, 0
        for i in range(1, len(recent)):
            dx_prev += recent[i][0] - recent[i - 1][0]
            dy_prev += recent[i][1] - recent[i - 1][1]

        prev_len = sqrt(dx_prev ** 2 + dy_prev ** 2)
        if prev_len == 0:
            return np.ones(len(candidate_coords), dtype=bool)

        last_x, last_y = path[-1]

        cand_arr = np.array(candidate_coords)
        dx_new = cand_arr[:, 0] - last_x
        dy_new = cand_arr[:, 1] - last_y
        new_len = np.sqrt(dx_new ** 2 + dy_new ** 2)
        new_len_safe = np.where(new_len == 0, 1, new_len)

        cos_angle = (dx_prev * dx_new + dy_prev * dy_new) / (prev_len * new_len_safe)
        min_cos = cos(radians(self.max_turn_angle))

        is_smooth = (cos_angle >= min_cos) | (new_len == 0)

        return is_smooth

    def compute_linearity_vectorized(self, path, candidate_coords):
        if len(path) < 2 or len(candidate_coords) == 0:
            return np.ones(len(candidate_coords))

        n = min(5, len(path))
        recent = path[-n:]

        dx_sum, dy_sum = 0, 0
        for i in range(1, len(recent)):
            dx_sum += recent[i][0] - recent[i - 1][0]
            dy_sum += recent[i][1] - recent[i - 1][1]

        dir_len = sqrt(dx_sum ** 2 + dy_sum ** 2)
        if dir_len == 0:
            return np.ones(len(candidate_coords))

        last_x, last_y = path[-1]

        cand_arr = np.array(candidate_coords)
        new_dx = cand_arr[:, 0] - last_x
        new_dy = cand_arr[:, 1] - last_y
        new_len = np.sqrt(new_dx ** 2 + new_dy ** 2)
        new_len = np.where(new_len == 0, 1, new_len)

        cos_angle = (dx_sum * new_dx + dy_sum * new_dy) / (dir_len * new_len)
        linearity = (cos_angle + 1) / 2

        return linearity

    def find_candidates_leftward_vectorized(self, frame_idx, cx, cy, path, direction, visited,
                                            target=None, radius=None):
        if radius is None:
            radius = self.search_radius

        frame = self.frames_np[frame_idx]
        grid = self._get_search_grid(radius, 'left', self.max_step_distance)

        dx = grid['dx']
        dy = grid['dy']
        dist = grid['dist']

        nx = cx + dx
        ny = cy + dy

        valid_bounds = (nx >= 0) & (nx < self.width) & (ny >= 0) & (ny < self.height)
        if not np.any(valid_bounds):
            return []

        nx_valid = nx[valid_bounds].astype(int)
        ny_valid = ny[valid_bounds].astype(int)
        brightness = frame[ny_valid, nx_valid]

        bright_mask = brightness > self.brightness_threshold
        if not np.any(bright_mask):
            return []

        nx_bright = nx_valid[bright_mask]
        ny_bright = ny_valid[bright_mask]
        brightness_bright = brightness[bright_mask]

        dx_bright = nx_bright - cx
        dy_bright = ny_bright - cy
        dist_bright = np.sqrt(dx_bright ** 2 + dy_bright ** 2)

        not_visited = np.array([
            (int(nx_bright[i]), int(ny_bright[i])) not in visited
            for i in range(len(nx_bright))
        ])
        if not np.any(not_visited):
            return []

        nx_filter = nx_bright[not_visited]
        ny_filter = ny_bright[not_visited]
        brightness_filter = brightness_bright[not_visited]
        dx_filter = dx_bright[not_visited]
        dy_filter = dy_bright[not_visited]
        dist_filter = dist_bright[not_visited]

        coords = list(zip(nx_filter, ny_filter))
        is_smooth = self.check_smooth_transition_vectorized(path, coords)

        if not np.any(is_smooth):
            smoothness = self.compute_smoothness_vectorized(path, coords)
            threshold = np.percentile(smoothness, 50)
            is_smooth = smoothness >= threshold

        nx_final = nx_filter[is_smooth]
        ny_final = ny_filter[is_smooth]
        brightness_final = brightness_filter[is_smooth]
        dx_final = dx_filter[is_smooth]
        dy_final = dy_filter[is_smooth]
        dist_final = dist_filter[is_smooth]

        if len(nx_final) == 0:
            return []

        coords_final = list(zip(nx_final, ny_final))
        linearity = self.compute_linearity_vectorized(path, coords_final)
        smoothness = self.compute_smoothness_vectorized(path, coords_final)

        scores = brightness_final * (linearity ** self.linearity_weight) / (dist_final + 0.5)
        scores *= (1 + smoothness)

        left_bonus = np.abs(dx_final) / (dist_final + 0.1)
        scores *= (1 + left_bonus)

        if direction:
            dir_x, dir_y = direction
            dir_len = sqrt(dir_x ** 2 + dir_y ** 2)
            if dir_len > 0:
                cos_a = (dx_final * dir_x + dy_final * dir_y) / (dist_final * dir_len)
                scores *= (1 + (cos_a + 1) / 2)

        if target:
            tx, ty = target
            to_target_dx = tx - cx
            to_target_dy = ty - cy
            to_target_len = sqrt(to_target_dx ** 2 + to_target_dy ** 2)
            if to_target_len > 0:
                cos_to_target = (dx_final * to_target_dx + dy_final * to_target_dy) / (
                        dist_final * to_target_len)
                scores *= (1 + 3.0 * (cos_to_target + 1) / 2)

        candidates = [
            (int(nx_final[i]), int(ny_final[i]), float(scores[i]),
             int(dx_final[i]), int(dy_final[i]))
            for i in range(len(nx_final))
        ]

        return candidates

    def find_candidates_rightward_vectorized(self, frame_idx, cx, cy, path, direction, visited,
                                             radius=None):
        if radius is None:
            radius = self.search_radius

        frame = self.frames_np[frame_idx]

        effective_radius = min(radius, self.max_step_distance)
        grid = self._get_search_grid(effective_radius, 'right', self.max_step_distance)

        dx = grid['dx']
        dy = grid['dy']
        dist = grid['dist']

        nx = cx + dx
        ny = cy + dy

        valid_bounds = (nx >= 0) & (nx < self.width) & (ny >= 0) & (ny < self.height)
        if not np.any(valid_bounds):
            return []

        nx_valid = nx[valid_bounds].astype(int)
        ny_valid = ny[valid_bounds].astype(int)
        dist_valid = dist[valid_bounds]
        brightness = frame[ny_valid, nx_valid]

        bright_mask = brightness > self.brightness_threshold
        if not np.any(bright_mask):
            return []

        nx_bright = nx_valid[bright_mask]
        ny_bright = ny_valid[bright_mask]
        brightness_bright = brightness[bright_mask]
        dist_bright = dist_valid[bright_mask]

        dx_bright = nx_bright - cx
        dy_bright = ny_bright - cy

        speed_mask = dist_bright <= self.max_step_distance
        if not np.any(speed_mask):
            return []

        nx_speed = nx_bright[speed_mask]
        ny_speed = ny_bright[speed_mask]
        brightness_speed = brightness_bright[speed_mask]
        dx_speed = dx_bright[speed_mask]
        dy_speed = dy_bright[speed_mask]
        dist_speed = dist_bright[speed_mask]

        not_visited = np.array([
            (int(nx_speed[i]), int(ny_speed[i])) not in visited
            for i in range(len(nx_speed))
        ])
        if not np.any(not_visited):
            return []

        nx_filter = nx_speed[not_visited]
        ny_filter = ny_speed[not_visited]
        brightness_filter = brightness_speed[not_visited]
        dx_filter = dx_speed[not_visited]
        dy_filter = dy_speed[not_visited]
        dist_filter = dist_speed[not_visited]

        coords = list(zip(nx_filter, ny_filter))
        is_smooth = self.check_smooth_transition_vectorized(path, coords)

        if not np.any(is_smooth):
            smoothness = self.compute_smoothness_vectorized(path, coords)
            if len(smoothness) > 0:
                threshold = np.percentile(smoothness, 50)
                is_smooth = smoothness >= threshold
            else:
                return []

        nx_final = nx_filter[is_smooth]
        ny_final = ny_filter[is_smooth]
        brightness_final = brightness_filter[is_smooth]
        dx_final = dx_filter[is_smooth]
        dy_final = dy_filter[is_smooth]
        dist_final = dist_filter[is_smooth]

        if len(nx_final) == 0:
            return []

        coords_final = list(zip(nx_final, ny_final))
        linearity = self.compute_linearity_vectorized(path, coords_final)
        smoothness = self.compute_smoothness_vectorized(path, coords_final)

        scores = brightness_final * (linearity ** self.linearity_weight) / (dist_final + 0.5)
        scores *= (1 + 2.0 * smoothness)

        right_bonus = np.maximum(0, dx_final) / (dist_final + 0.1)
        scores *= (1 + right_bonus)

        if direction:
            dir_x, dir_y = direction
            dir_len = sqrt(dir_x ** 2 + dir_y ** 2)
            if dir_len > 0:
                cos_a = (dx_final * dir_x + dy_final * dir_y) / (dist_final * dir_len)
                scores *= (1 + (cos_a + 1) / 2)

        optimal_dist = self.max_step_distance * 0.5
        dist_penalty = np.abs(dist_final - optimal_dist) / optimal_dist
        scores *= (1 - 0.3 * dist_penalty)

        candidates = [
            (int(nx_final[i]), int(ny_final[i]), float(scores[i]),
             int(dx_final[i]), int(dy_final[i]))
            for i in range(len(nx_final))
        ]

        return candidates

    def trace_segment_left(self, frame_idx, start_x, start_y, target=None,
                           initial_direction=None, max_steps=3000):
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

            candidates = self.find_candidates_leftward_vectorized(
                frame_idx, cx, cy, path, direction, visited, target=target
            )

            if not candidates:
                for radius in range(self.search_radius, self.max_search_radius + 1,
                                    self.search_radius_step):
                    candidates = self.find_candidates_leftward_vectorized(
                        frame_idx, cx, cy, path, direction, visited,
                        target=target, radius=radius
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

    def grow_rightward(self, frame_idx, prev_path, prev_direction, max_steps=300):
        if not prev_path:
            return [], None, None

        start_x, start_y = prev_path[-1]

        reached, boundary_type = self.check_right_boundary(start_x, start_y)
        if reached:
            return [], boundary_type, prev_direction

        new_points = []
        visited = set(prev_path)
        cx, cy = start_x, start_y
        direction = prev_direction if prev_direction else (1, 0)
        boundary_reached = None

        context_path = list(prev_path[-10:])
        consecutive_failures = 0
        max_consecutive_failures = 5

        for step in range(max_steps):
            reached, boundary_type = self.check_right_boundary(cx, cy)
            if reached:
                boundary_reached = boundary_type
                break

            candidates = self.find_candidates_rightward_vectorized(
                frame_idx, cx, cy, context_path + new_points, direction, visited
            )

            if not candidates:
                for radius in range(self.search_radius, self.max_search_radius + 1,
                                    self.search_radius_step):
                    candidates = self.find_candidates_rightward_vectorized(
                        frame_idx, cx, cy, context_path + new_points,
                        direction, visited, radius=radius
                    )
                    if candidates:
                        break

            if not candidates:
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    break
                continue

            consecutive_failures = 0

            candidates.sort(key=lambda x: -x[2])
            chosen = candidates[0]
            nx, ny = chosen[0], chosen[1]
            dx, dy = chosen[3], chosen[4]

            if dx < 0:
                right_only = [c for c in candidates if c[3] >= 0]
                if right_only:
                    chosen = max(right_only, key=lambda x: x[2])
                    nx, ny = chosen[0], chosen[1]
                    dx, dy = chosen[3], chosen[4]
                else:
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        break
                    continue

            if (nx, ny) in visited:
                remaining = [c for c in candidates if (c[0], c[1]) not in visited and c[3] >= 0]
                if remaining:
                    chosen = max(remaining, key=lambda x: x[2])
                    nx, ny = chosen[0], chosen[1]
                    dx, dy = chosen[3], chosen[4]
                else:
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        break
                    continue

            new_points.append((nx, ny))
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

        return new_points, boundary_reached, direction

    def connect_points_directly(self, frame_idx, start, end):
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
            frame = self.frames_np[frame_idx]

            for step_len in [1, 2, min(3, self.max_step_distance)]:
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
            for pt in full_path:
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

    def compute_neuron_trajectory(self, neuron_id, locked_frame=None):
        start_time = time.time()

        markers = self.get_neuron_markers(neuron_id)
        if not markers:
            return None

        all_waypoints = self.get_all_waypoints(neuron_id)
        if not all_waypoints:
            return None

        all_waypoints = sorted(all_waypoints, key=lambda p: -p[0])

        marked_frames = sorted(markers.keys())
        first_marked_frame = min(marked_frames)

        print(f"\n{'=' * 60}")
        print(f"计算神经元 N{neuron_id} 轨迹 (平滑约束版)")
        print(f"标记帧: {marked_frames}, 首帧: {first_marked_frame}")
        print(f"必经点({len(all_waypoints)}个): {all_waypoints}")
        print(f"最大转角: {self.max_turn_angle}°, 最大步距: {self.max_step_distance}")
        print(f"{'=' * 60}")

        t1 = time.time()

        if locked_frame is not None and 0 <= locked_frame < len(self.frames):
            best_frame = locked_frame
            print(f"\n[锁定参考帧{best_frame}] 向左追踪")
        else:
            best_frame = self.find_best_frame(neuron_id, all_waypoints)
            print(f"\n[自动选帧{best_frame}] 向左追踪")

        initial_path, left_boundary = self.trace_left_unified(best_frame, all_waypoints)

        t2 = time.time()
        print(f"  向左追踪耗时: {t2 - t1:.3f}秒")

        if not initial_path:
            print("  追踪失败！")
            return None

        if len(initial_path) > 5:
            initial_path = self.smooth_path_preserve_waypoints(initial_path, all_waypoints)

        if len(initial_path) >= 2 and initial_path[0][0] > initial_path[-1][0]:
            initial_path = initial_path[::-1]

        print(f"  初始路径: {len(initial_path)}点, 左端到达{left_boundary or '边界'}")

        if len(initial_path) >= 2:
            dx = initial_path[-1][0] - initial_path[-2][0]
            dy = initial_path[-1][1] - initial_path[-2][1]
            dist = sqrt(dx ** 2 + dy ** 2)
            current_direction = (dx / dist, dy / dist) if dist > 0 else (1, 0)
        else:
            current_direction = (1, 0)

        t3 = time.time()
        paths_by_frame = {}

        for frame_idx in range(first_marked_frame):
            paths_by_frame[frame_idx] = initial_path.copy()

        current_path = initial_path.copy()

        print(f"\n[向右生长] 从帧{first_marked_frame}开始")

        growth_count = 0
        for frame_idx in range(first_marked_frame, len(self.frames)):
            new_points, right_boundary, new_direction = self.grow_rightward(
                frame_idx, current_path, current_direction
            )

            if new_points:
                current_path = current_path + new_points
                current_direction = new_direction
                growth_count += len(new_points)

            paths_by_frame[frame_idx] = current_path.copy()

            if right_boundary:
                print(f"  帧{frame_idx}: 到达右边界 {right_boundary}")

        t4 = time.time()
        print(f"  向右生长耗时: {t4 - t3:.3f}秒, 共生长{growth_count}点")

        max_path = paths_by_frame[len(self.frames) - 1]

        result = {
            'initial_path':       initial_path,
            'paths_by_frame':     paths_by_frame,
            'waypoints':          all_waypoints,
            'left_boundary':      left_boundary,
            'first_marked_frame': first_marked_frame,
            'best_frame':         best_frame,
            'final_tip_x':        max_path[-1][0] if max_path else 0
        }

        self.tracking_results[neuron_id] = result

        total_time = time.time() - start_time
        print(f"\n完成 N{neuron_id}: 总耗时{total_time:.3f}秒, "
              f"初始{len(initial_path)}点, 最终{len(max_path)}点")

        return result

    # ------------------------------------------------------------------ #
    #  局部重算                                                             #
    # ------------------------------------------------------------------ #

    def local_recompute_with_waypoints(self, neuron_id, new_waypoints,
                                       locked_frame=None):
        """
        三种情况全部只替换对应段，grow_rightward 一律不重跑：

        - 新点 X 在两个已有 waypoint 之间
            → 只替换夹它的两锚点之间的路径段
        - 新点 X 小于最左 waypoint
            → 只重追 [路径起点 ~ 最左锚点] 这一段
        - 新点 X 大于最右 waypoint
            → 只重追 [最右锚点 ~ 路径末尾] 这一段
        全程遵守白线 + 平滑（转角）约束，其余路径严格不动。
        """
        result = self.tracking_results.get(neuron_id)
        if not result or not result.get('initial_path'):
            print(f"N{neuron_id} 无原始结果，转为完整重算")
            return self.compute_neuron_trajectory(neuron_id, locked_frame)

        best_frame         = locked_frame if locked_frame is not None else result.get('best_frame', 0)
        base_path          = list(result['initial_path'])
        first_marked       = result.get('first_marked_frame', 0)
        old_paths_by_frame = {f: list(p) for f, p in result['paths_by_frame'].items()}

        all_wp_sorted = sorted(self.get_all_waypoints(neuron_id), key=lambda p: p[0])

        print(f"\n局部重算 N{neuron_id}，参考帧={best_frame}")
        print(f"全部必经点(按X排序): {all_wp_sorted}")
        print(f"新点: {new_waypoints}")

        # ── 辅助：找路径中离某点最近的索引 ────────────────────────────────
        def nearest_idx(path, pt):
            best_i, best_d = 0, float('inf')
            for i, p in enumerate(path):
                d = sqrt((p[0] - pt[0]) ** 2 + (p[1] - pt[1]) ** 2)
                if d < best_d:
                    best_d, best_i = d, i
            return best_i

        # ── 辅助：从 start_pt 依次经过 targets，最终到达 end_pt ───────────
        def build_segment(start_pt, targets, end_pt):
            """首点强制为 start_pt，遵守白线 + 平滑约束向右追踪。"""
            seg = [start_pt]
            cur = start_pt

            idx      = nearest_idx(base_path, start_pt)
            look_fwd = min(idx + 5, len(base_path) - 1)
            dx0 = base_path[look_fwd][0] - start_pt[0]
            dy0 = base_path[look_fwd][1] - start_pt[1]
            d0  = sqrt(dx0 ** 2 + dy0 ** 2)
            cur_dir   = (dx0 / d0, dy0 / d0) if d0 > 0 else (1, 0)
            trace_dir = (-cur_dir[0], -cur_dir[1])

            for tgt in list(targets) + [end_pt]:
                sub, _, reached, trace_dir = self.trace_segment_left(
                    best_frame, cur[0], cur[1],
                    target=tgt, initial_direction=trace_dir,
                )
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

        # ── 辅助：只替换 paths_by_frame 中的指定段，其余完全不动 ──────────
        def patch_paths_by_frame(old_pf, left_idx, right_idx, new_seg):
            """
            对所有帧的路径，将 [left_idx : right_idx+1] 替换为 new_seg。
            - 路径还没长到该段：不动
            - 路径进入了段但未到右锚点：替换已有部分
            - 路径已经过了右锚点：完整替换并拼回右侧
            """
            patched = {}
            for f, fpath in old_pf.items():
                if len(fpath) <= left_idx:
                    patched[f] = fpath
                elif len(fpath) <= right_idx:
                    patched[f] = fpath[:left_idx] + new_seg
                else:
                    patched[f] = fpath[:left_idx] + new_seg + fpath[right_idx + 1:]
            return patched

        # ── 依次处理每个新点 ──────────────────────────────────────────────
        paths_by_frame = old_paths_by_frame

        for new_pt in new_waypoints:
            nx_new = new_pt[0]

            old_wp = [w for w in all_wp_sorted
                      if sqrt((w[0] - new_pt[0]) ** 2 + (w[1] - new_pt[1]) ** 2) >= 8]

            if not old_wp:
                return self.compute_neuron_trajectory(neuron_id, locked_frame)

            leftmost_wp  = min(old_wp, key=lambda p: p[0])
            rightmost_wp = max(old_wp, key=lambda p: p[0])

            # ── 情况 1：新点在最左端之外 → 只替换左侧段，grow 完全不跑 ──────
            if nx_new <= leftmost_wp[0]:
                anchor_idx   = nearest_idx(base_path, leftmost_wp)
                anchor_point = base_path[anchor_idx]

                print(f"  新点{new_pt} 在左端外，锚点={anchor_point}(idx={anchor_idx})")

                # 从新点向左追到边界
                seg_left, _, _, _ = self.trace_segment_left(
                    best_frame, new_pt[0], new_pt[1],
                    target=None, initial_direction=(-1, 0),
                )
                if len(seg_left) >= 2 and seg_left[0][0] > seg_left[-1][0]:
                    seg_left = seg_left[::-1]

                # 从新点向右追到 anchor_point
                seg_right, _, reached_r, _ = self.trace_segment_left(
                    best_frame, new_pt[0], new_pt[1],
                    target=anchor_point, initial_direction=(1, 0),
                )
                if len(seg_right) >= 2 and seg_right[0][0] > seg_right[-1][0]:
                    seg_right = seg_right[::-1]
                if not reached_r:
                    seg_right = self.connect_points_directly(best_frame, new_pt, anchor_point)

                if seg_right and seg_left and seg_right[0] == seg_left[-1]:
                    seg_right = seg_right[1:]
                new_left = seg_left + seg_right

                # 强制尾点对齐 anchor_point
                if not new_left or new_left[-1] != anchor_point:
                    new_left = new_left + [anchor_point]
                if len(new_left) > 5:
                    new_left = self.smooth_path_preserve_waypoints(new_left, [new_pt])

                # anchor_point 之后严格不动
                after     = base_path[anchor_idx + 1:]
                base_path = new_left + after

                # ★ 只替换各帧中 [0 : anchor_idx+1] 的部分，grow 完全不跑
                paths_by_frame = patch_paths_by_frame(
                    paths_by_frame, 0, anchor_idx, new_left
                )

                print(f"  左端重追完成，段长={len(new_left)}，"
                      f"新initial_path长={len(base_path)}，grow_rightward未重跑")

            # ── 情况 2：新点在最右端之外 → 只替换右侧段，grow 完全不跑 ──────
            elif nx_new >= rightmost_wp[0]:
                anchor_idx   = nearest_idx(base_path, rightmost_wp)
                anchor_point = base_path[anchor_idx]

                # ★ anchor 之前绝对不动
                before = base_path[:anchor_idx]

                print(f"  新点{new_pt} 在右端外，锚点={anchor_point}(idx={anchor_idx})")

                seg, _, reached, _ = self.trace_segment_left(
                    best_frame, anchor_point[0], anchor_point[1],
                    target=new_pt, initial_direction=(1, 0),
                )
                if len(seg) >= 2 and seg[0][0] > seg[-1][0]:
                    seg = seg[::-1]
                if not reached:
                    seg = self.connect_points_directly(best_frame, anchor_point, new_pt)

                # 强制首点为 anchor_point
                if not seg or seg[0] != anchor_point:
                    seg = [anchor_point] + (seg or [])
                if len(seg) > 5:
                    seg = self.smooth_path_preserve_waypoints(seg, [new_pt])

                base_path = before + seg

                # ★ 只替换各帧中 anchor_idx 之后的部分，grow 完全不跑
                paths_by_frame = patch_paths_by_frame(
                    paths_by_frame, anchor_idx, len(base_path) + 99999, seg
                )

                print(f"  右端重追完成，段长={len(seg)}，"
                      f"新initial_path长={len(base_path)}，grow_rightward未重跑")

            # ── 情况 3：新点在两个 waypoint 之间 → 只替换该段，grow 完全不跑 ─
            else:
                left_wp  = max((w for w in old_wp if w[0] <= nx_new), key=lambda p: p[0])
                right_wp = min((w for w in old_wp if w[0] >  nx_new), key=lambda p: p[0])

                left_idx  = nearest_idx(base_path, left_wp)
                right_idx = nearest_idx(base_path, right_wp)
                if left_idx > right_idx:
                    left_idx, right_idx = right_idx, left_idx

                anchor_start = base_path[left_idx]
                anchor_end   = base_path[right_idx]

                # ★ 左侧 before 和右侧 after 严格不动
                before = base_path[:left_idx]
                after  = base_path[right_idx + 1:]

                print(f"  新点{new_pt} 夹在 {left_wp}(idx={left_idx}) "
                      f"和 {right_wp}(idx={right_idx}) 之间")

                seg_mid_wp = sorted(
                    [w for w in old_wp if left_wp[0] < w[0] < right_wp[0]] + [new_pt],
                    key=lambda p: p[0]
                )
                print(f"  段内必经点(含新): {seg_mid_wp}")

                full_seg = build_segment(anchor_start, seg_mid_wp, anchor_end)

                # 强制首尾精确对齐锚点
                if not full_seg or full_seg[0] != anchor_start:
                    full_seg = [anchor_start] + (full_seg or [])
                if full_seg[-1] != anchor_end:
                    full_seg = full_seg + [anchor_end]
                if len(full_seg) > 5:
                    full_seg = self.smooth_path_preserve_waypoints(full_seg, seg_mid_wp)

                base_path = before + full_seg + after

                # ★ 只替换各帧路径中对应的段，grow_rightward 完全不重跑
                paths_by_frame = patch_paths_by_frame(
                    paths_by_frame, left_idx, right_idx, full_seg
                )

                print(f"  中间段替换完成，段长={len(full_seg)}，"
                      f"新initial_path长={len(base_path)}，grow_rightward未重跑")

        # ── 汇总结果 ──────────────────────────────────────────────────────
        max_path      = paths_by_frame.get(len(self.frames) - 1, base_path)
        all_waypoints = self.get_all_waypoints(neuron_id)

        new_result = {
            'initial_path':       base_path,
            'paths_by_frame':     paths_by_frame,
            'waypoints':          all_waypoints,
            'left_boundary':      result.get('left_boundary'),
            'first_marked_frame': first_marked,
            'best_frame':         best_frame,
            'final_tip_x':        max_path[-1][0] if max_path else 0,
        }

        self.tracking_results[neuron_id] = new_result
        print(f"局部重算完成，initial_path={len(base_path)}点，"
              f"最终={len(max_path)}点，grow_rightward全程未重跑")
        return new_result

    # ------------------------------------------------------------------ #
    #  速度计算                                                             #
    # ------------------------------------------------------------------ #

    def compute_neuron_speed(self, neuron_id, fps=10.0, pixel_um=1.0):
        result = self.tracking_results.get(neuron_id)
        if not result:
            return []

        paths_by_frame = result.get('paths_by_frame', {})
        sorted_frames  = sorted(paths_by_frame.keys())

        speeds = []
        for i in range(1, len(sorted_frames)):
            f_prev = sorted_frames[i - 1]
            f_curr = sorted_frames[i]

            path_prev = paths_by_frame.get(f_prev, [])
            path_curr = paths_by_frame.get(f_curr, [])

            if not path_prev or not path_curr:
                continue

            tip_prev = path_prev[-1]
            tip_curr = path_curr[-1]

            dx       = tip_curr[0] - tip_prev[0]
            dy       = tip_curr[1] - tip_prev[1]
            dist_px  = sqrt(dx ** 2 + dy ** 2)
            speed_um = dist_px * pixel_um * fps

            speeds.append({
                'frame':              f_curr,
                'tip_x':              tip_curr[0],
                'tip_y':              tip_curr[1],
                'dx':                 dx,
                'dy':                 dy,
                'speed_px_per_frame': round(dist_px, 4),
                'speed_um_per_sec':   round(speed_um, 4),
            })

        return speeds

    def compute_all_speeds(self, fps=10.0, pixel_um=1.0):
        return {
            nid: self.compute_neuron_speed(nid, fps, pixel_um)
            for nid in self.tracking_results
        }
