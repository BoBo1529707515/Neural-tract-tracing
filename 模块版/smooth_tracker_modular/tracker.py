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

    def find_candidates_leftward_vectorized(self, frame_idx, cx, cy, path, direction, visited, target=None,
                                            radius=None):
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
                cos_to_target = (dx_final * to_target_dx + dy_final * to_target_dy) / (dist_final * to_target_len)
                scores *= (1 + 3.0 * (cos_to_target + 1) / 2)

        candidates = [
            (int(nx_final[i]), int(ny_final[i]), float(scores[i]),
             int(dx_final[i]), int(dy_final[i]))
            for i in range(len(nx_final))
        ]

        return candidates

    def find_candidates_rightward_vectorized(self, frame_idx, cx, cy, path, direction, visited, radius=None):
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

    def trace_segment_left(self, frame_idx, start_x, start_y, target=None, initial_direction=None, max_steps=3000):
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
                for radius in range(self.search_radius, self.max_search_radius + 1, self.search_radius_step):
                    candidates = self.find_candidates_leftward_vectorized(
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
                for radius in range(self.search_radius, self.max_search_radius + 1, self.search_radius_step):
                    candidates = self.find_candidates_rightward_vectorized(
                        frame_idx, cx, cy, context_path + new_points, direction, visited, radius=radius
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

    def compute_neuron_trajectory(self, neuron_id):
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
        best_frame = self.find_best_frame(neuron_id, all_waypoints)
        print(f"\n[初始帧{best_frame}] 向左追踪")

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

        print(f"\n[向右生长] 从帧{first_marked_frame}开始 (平滑约束)")

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
            'initial_path': initial_path,
            'paths_by_frame': paths_by_frame,
            'waypoints': all_waypoints,
            'left_boundary': left_boundary,
            'first_marked_frame': first_marked_frame,
            'final_tip_x': max_path[-1][0] if max_path else 0
        }

        self.tracking_results[neuron_id] = result

        total_time = time.time() - start_time
        print(f"\n完成 N{neuron_id}:")
        print(f"  总耗时: {total_time:.3f}秒")
        print(f"  初始路径: {len(initial_path)}点")
        print(f"  最终路径: {len(max_path)}点")

        return result
