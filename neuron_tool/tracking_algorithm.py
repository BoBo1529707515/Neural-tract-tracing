"""
tracking_algorithm.py（改进版）
利用多标记点约束的追踪算法
"""

import numpy as np
from collections import deque
import heapq


class NeuronTracker:
    """
    改进的神经元追踪器

    新增功能：
        - 多标记点约束
        - 分段追踪（点到点）
        - A*路径搜索
    """

    def __init__(self, image_processor, config=None):
        """
        初始化追踪器

        参数:
            image_processor: 图像处理器（用于绿色检测）
            config: 配置参数
        """
        self.img_proc = image_processor

        # 默认参数
        self.max_gap = 15
        self.y_tolerance = 35
        self.direction_weight = 15
        self.direction_history = 8

    # ================================================================
    #                    核心改进：多点约束追踪
    # ================================================================

    def trace_with_markers(self, skeleton, frame, markers):
        """
        使用多个标记点约束的追踪

        参数:
            skeleton: 骨架图像
            frame: 原始BGR图像
            markers: 标记点列表 [{"frame": f, "x": x, "y": y}, ...]
                     同一帧的多个点 或 不同帧的点（使用第一帧的位置）

        返回:
            list: 完整轨迹 [(y, x), ...]
        """
        if not markers:
            return []

        # 1. 将标记点转换为骨架点，并按X坐标排序
        skeleton_points = []
        for mark in markers:
            pt = self._find_nearest_skeleton(skeleton, frame, (mark["x"], mark["y"]))
            if pt:
                skeleton_points.append(pt)

        if not skeleton_points:
            return []

        # 按X坐标排序（从左到右）
        skeleton_points = sorted(skeleton_points, key=lambda p: p[1])

        # 2. 如果只有一个点，使用普通双向追踪
        if len(skeleton_points) == 1:
            return self.trace_bidirectional(skeleton, frame, skeleton_points[0])

        # 3. 多个点：分段追踪，连接相邻标记点
        full_trajectory = []

        for i in range(len(skeleton_points) - 1):
            start_pt = skeleton_points[i]
            end_pt = skeleton_points[i + 1]

            # 用A*算法找两点之间的最短路径
            segment = self._find_path_between(skeleton, frame, start_pt, end_pt)

            if segment:
                # 避免重复添加连接点
                if full_trajectory and segment[0] == full_trajectory[-1]:
                    segment = segment[1:]
                full_trajectory.extend(segment)

        # 4. 从最左点继续向左追踪
        leftmost = skeleton_points[0]
        left_extension = self._trace_single_direction(
            skeleton, frame, leftmost,
            y_center=leftmost[0],
            go_left=True,
            avoid_points=set(full_trajectory)
        )

        # 5. 从最右点继续向右追踪
        rightmost = skeleton_points[-1]
        right_extension = self._trace_single_direction(
            skeleton, frame, rightmost,
            y_center=rightmost[0],
            go_left=False,
            avoid_points=set(full_trajectory)
        )

        # 6. 合并：左延伸（倒序） + 主轨迹 + 右延伸
        final_trajectory = left_extension[::-1] + full_trajectory + right_extension

        # 去重并按X排序
        final_trajectory = list(dict.fromkeys(final_trajectory))
        final_trajectory = sorted(final_trajectory, key=lambda p: p[1])

        return final_trajectory

    def _find_path_between(self, skeleton, frame, start, end, max_iterations=5000):
        """
        A*算法找两点之间的最短路径

        参数:
            skeleton: 骨架图像
            frame: 原始图像
            start: 起点 (y, x)
            end: 终点 (y, x)
            max_iterations: 最大迭代次数

        返回:
            list: 路径点列表，失败返回空列表
        """
        h, w = skeleton.shape

        def heuristic(p):
            """启发式函数：到终点的欧氏距离"""
            return np.sqrt((p[0] - end[0])**2 + (p[1] - end[1])**2)

        # 优先队列：(f_score, counter, point, path)
        # counter用于打破f_score相同时的顺序
        counter = 0
        open_set = [(heuristic(start), counter, start, [start])]
        heapq.heapify(open_set)

        visited = set()
        visited.add(start)

        iterations = 0

        while open_set and iterations < max_iterations:
            iterations += 1

            f, _, current, path = heapq.heappop(open_set)

            # 到达终点
            if current == end:
                return path

            # 检查是否足够接近终点（允许小偏差）
            if heuristic(current) < 3:
                # 直接连接到终点
                return path + [end]

            # 搜索邻域
            cy, cx = current
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    if dy == 0 and dx == 0:
                        continue

                    ny, nx = cy + dy, cx + dx

                    if not (0 <= ny < h and 0 <= nx < w):
                        continue

                    if (ny, nx) in visited:
                        continue

                    # 必须在骨架上且是绿色
                    if not skeleton[ny, nx]:
                        continue

                    if not self.img_proc.is_green_pixel(frame, ny, nx):
                        continue

                    visited.add((ny, nx))

                    # 计算代价
                    g = len(path)  # 已走路径长度
                    h_val = heuristic((ny, nx))
                    f_new = g + h_val

                    counter += 1
                    heapq.heappush(open_set, (f_new, counter, (ny, nx), path + [(ny, nx)]))

        # 未找到路径，尝试宽松搜索（允许跳过非骨架点）
        print(f"    ⚠ A*未找到路径，尝试直接连接")
        return self._direct_connect(skeleton, frame, start, end)

    def _direct_connect(self, skeleton, frame, start, end):
        """
        直接连接两点（当A*失败时的备选方案）
        沿直线方向搜索，允许小范围偏离

        参数:
            skeleton, frame: 图像
            start, end: 起点和终点

        返回:
            list: 路径点列表
        """
        h, w = skeleton.shape
        path = [start]
        current = start

        max_steps = int(np.sqrt((end[0]-start[0])**2 + (end[1]-start[1])**2) * 2)

        for _ in range(max_steps):
            if current == end:
                break

            cy, cx = current

            # 计算到终点的方向
            dy_target = end[0] - cy
            dx_target = end[1] - cx
            dist = np.sqrt(dy_target**2 + dx_target**2)

            if dist < 2:
                path.append(end)
                break

            dy_norm = dy_target / dist
            dx_norm = dx_target / dist

            # 在目标方向附近搜索最佳下一步
            best_next = None
            best_score = float('-inf')

            for dy in range(-3, 4):
                for dx in range(-3, 4):
                    if dy == 0 and dx == 0:
                        continue

                    ny, nx = cy + dy, cx + dx

                    if not (0 <= ny < h and 0 <= nx < w):
                        continue

                    if (ny, nx) in path:
                        continue

                    # 评分：优先骨架点 + 方向一致性
                    score = 0

                    if skeleton[ny, nx] and self.img_proc.is_green_pixel(frame, ny, nx):
                        score += 100  # 骨架点大加分

                    # 方向一致性
                    step_dist = np.sqrt(dy**2 + dx**2)
                    if step_dist > 0:
                        dir_score = (dy * dy_norm + dx * dx_norm) / step_dist
                        score += dir_score * 10

                    # 距离终点
                    new_dist = np.sqrt((ny - end[0])**2 + (nx - end[1])**2)
                    score -= new_dist * 0.1

                    if score > best_score:
                        best_score = score
                        best_next = (ny, nx)

            if best_next is None:
                break

            path.append(best_next)
            current = best_next

        return path

    def _trace_single_direction(self, skeleton, frame, start, y_center,
                                 go_left=True, avoid_points=None):
        """
        单向追踪（带方向动量）

        参数:
            skeleton: 骨架图像
            frame: 原始图像
            start: 起点
            y_center: Y中心（限制偏移）
            go_left: True=向左, False=向右
            avoid_points: 要避开的点集合

        返回:
            list: 轨迹点列表
        """
        if start is None:
            return []

        h, w = skeleton.shape
        visited = set(avoid_points) if avoid_points else set()

        traj = [start]
        visited.add(start)
        current = start

        while True:
            # 计算当前方向
            current_dir = self._get_direction(traj)

            # 搜索候选点
            candidates = []

            for search_r in [1, 2, self.max_gap]:
                for dy in range(-search_r, search_r + 1):
                    for dx in range(-search_r, search_r + 1):
                        if dy == 0 and dx == 0:
                            continue

                        ny, nx = current[0] + dy, current[1] + dx

                        if not (0 <= ny < h and 0 <= nx < w):
                            continue

                        if (ny, nx) in visited:
                            continue

                        if not skeleton[ny, nx]:
                            continue

                        if not self.img_proc.is_green_pixel(frame, ny, nx):
                            continue

                        if abs(ny - y_center) > self.y_tolerance * 2:
                            continue

                        # 计算得分
                        dist = np.sqrt(dy**2 + dx**2)

                        if go_left:
                            base_score = -dx * 10 - abs(dy) * 3 - dist * 2
                        else:
                            base_score = dx * 10 - abs(dy) * 3 - dist * 2

                        # 方向一致性
                        if dist > 0:
                            cand_dir = (dy / dist, dx / dist)
                            dir_score = current_dir[0] * cand_dir[0] + current_dir[1] * cand_dir[1]
                            base_score += dir_score * self.direction_weight

                        candidates.append(((ny, nx), base_score))

                if candidates:
                    break

            if not candidates:
                break

            # 选最佳候选点
            candidates.sort(key=lambda x: x[1], reverse=True)
            best = candidates[0][0]

            visited.add(best)
            traj.append(best)
            current = best

        return traj

    def _get_direction(self, trajectory, num_points=None):
        """计算轨迹方向"""
        if num_points is None:
            num_points = self.direction_history

        if len(trajectory) < 2:
            return (0, 1)

        recent = trajectory[-min(num_points, len(trajectory)):]

        dy = recent[-1][0] - recent[0][0]
        dx = recent[-1][1] - recent[0][1]

        length = np.sqrt(dy**2 + dx**2)
        if length < 0.001:
            return (0, 1)

        return (dy / length, dx / length)

    def _find_nearest_skeleton(self, skeleton, frame, point, radius=30):
        """在点附近找最近的骨架点"""
        h, w = skeleton.shape
        px, py = int(point[0]), int(point[1])

        best, best_d = None, float('inf')
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                ny, nx = py + dy, px + dx
                if 0 <= ny < h and 0 <= nx < w:
                    if skeleton[ny, nx] and self.img_proc.is_green_pixel(frame, ny, nx):
                        d = dy**2 + dx**2
                        if d < best_d:
                            best_d, best = d, (ny, nx)
        return best

    # ================================================================
    #                    原有方法（保留兼容）
    # ================================================================

    def trace_bidirectional(self, skeleton, frame, start_point):
        """双向追踪（原有方法）"""
        if start_point is None:
            return []

        y_center = start_point[0]

        left = self._trace_single_direction(skeleton, frame, start_point, y_center, go_left=True)
        right = self._trace_single_direction(skeleton, frame, start_point, y_center, go_left=False)

        full = left[::-1] + right[1:] if len(right) > 1 else left[::-1]
        return full

    def find_growth(self, skeleton, frame, trajectory, y_center):
        """在生长端找新点（原有方法）"""
        if not trajectory:
            return []

        return self._trace_single_direction(
            skeleton, frame,
            trajectory[-1],
            y_center,
            go_left=False,
            avoid_points=set(trajectory)
        )
