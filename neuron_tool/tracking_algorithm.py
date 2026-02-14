"""
追踪算法模块
包含神经元追踪的核心算法
"""

import numpy as np
from .config import Config


class NeuronTracker:
    """
    神经元追踪器

    功能:
        - 单向追踪（向左或向右）
        - 双向追踪
        - 生长点搜索
        - 方向动量（交叉点处理）
    """

    def __init__(self, image_processor):
        """
        初始化追踪器

        参数:
            image_processor: ImageProcessor实例，用于绿色检测
        """
        self.img_proc = image_processor

    def get_direction(self, trajectory, num_points=None):
        """
        计算轨迹的当前方向向量

        参数:
            trajectory: 轨迹点列表 [(y, x), ...]
            num_points: 使用的历史点数量，默认使用配置值

        返回:
            tuple: 归一化方向向量 (dy, dx)
        """
        if num_points is None:
            num_points = Config.DIRECTION_HISTORY

        if len(trajectory) < 2:
            return (0, 1)  # 默认向右

        # 取最近的几个点
        recent = trajectory[-min(num_points, len(trajectory)):]

        if len(recent) < 2:
            return (0, 1)

        # 计算方向（从第一个点到最后一个点）
        dy = recent[-1][0] - recent[0][0]
        dx = recent[-1][1] - recent[0][1]

        length = np.sqrt(dy * dy + dx * dx)
        if length < 0.001:
            return (0, 1)

        return (dy / length, dx / length)

    def direction_score(self, current_dir, candidate_dir):
        """
        计算方向一致性得分

        参数:
            current_dir: 当前方向 (dy, dx)
            candidate_dir: 候选方向 (dy, dx)

        返回:
            float: 方向一致性得分，范围[-1, 1]，越大越一致
        """
        # 使用点积衡量方向一致性
        return current_dir[0] * candidate_dir[0] + current_dir[1] * candidate_dir[1]

    def trace_direction(self, skeleton, frame, start, y_center, go_left=True, initial_traj=None):
        """
        带方向动量的单向追踪

        参数:
            skeleton: 骨架图像
            frame: 原始BGR图像
            start: 起始点 (y, x)
            y_center: Y轴中心位置（用于限制偏移）
            go_left: True向左追踪，False向右追踪
            initial_traj: 初始轨迹（可选，用于继承方向）

        返回:
            list: 轨迹点列表 [(y, x), ...]
        """
        if start is None:
            return []

        h, w = skeleton.shape
        visited = set()

        # 初始化轨迹
        traj = list(initial_traj) if initial_traj else [start]
        visited.update(traj)
        current = start

        while True:
            # 计算当前方向
            current_dir = self.get_direction(traj)

            # 收集候选点
            candidates = []

            # 多级搜索半径
            for search_radius in [1, 2, Config.MAX_GAP]:
                for dy in range(-search_radius, search_radius + 1):
                    for dx in range(-search_radius, search_radius + 1):
                        if dy == 0 and dx == 0:
                            continue

                        ny, nx = current[0] + dy, current[1] + dx

                        # 边界检查
                        if not (0 <= ny < h and 0 <= nx < w):
                            continue

                        # 是否为骨架点
                        if not skeleton[ny, nx]:
                            continue

                        # 是否已访问
                        if (ny, nx) in visited:
                            continue

                        # 是否为绿色
                        if not self.img_proc.is_green_pixel(frame, ny, nx):
                            continue

                        # Y方向偏移限制
                        if abs(ny - y_center) > Config.Y_TOLERANCE * 2:
                            continue

                        # 计算得分
                        dist = np.sqrt(dy * dy + dx * dx)

                        # 基础得分：优先向左或向右
                        if go_left:
                            base_score = -dx * 10 - abs(dy) * 3 - dist * 2
                        else:
                            base_score = dx * 10 - abs(dy) * 3 - dist * 2

                        # 方向一致性得分
                        if dist > 0:
                            cand_dir = (dy / dist, dx / dist)
                            dir_score = self.direction_score(current_dir, cand_dir)
                            direction_bonus = dir_score * Config.DIRECTION_WEIGHT
                        else:
                            direction_bonus = 0

                        total_score = base_score + direction_bonus
                        candidates.append(((ny, nx), total_score, dist))

                # 如果找到候选点就停止扩大搜索
                if candidates:
                    break

            # 没有候选点，结束追踪
            if not candidates:
                break

            # 按得分排序
            candidates.sort(key=lambda x: x[1], reverse=True)

            # 交叉点处理：如果多个候选点得分接近，选方向最一致的
            best = candidates[0]
            if len(candidates) > 1:
                top_score = candidates[0][1]
                close_candidates = [c for c in candidates if top_score - c[1] < 3]

                if len(close_candidates) > 1:
                    # 多个得分接近的候选点 = 可能是交叉点
                    best_dir_score = -float('inf')
                    for cand, score, dist in close_candidates:
                        if dist > 0:
                            dy = cand[0] - current[0]
                            dx = cand[1] - current[1]
                            cand_dir = (dy / dist, dx / dist)
                            ds = self.direction_score(current_dir, cand_dir)
                            if ds > best_dir_score:
                                best_dir_score = ds
                                best = (cand, score, dist)

            # 添加最优点
            next_pt = best[0]
            visited.add(next_pt)
            traj.append(next_pt)
            current = next_pt

        return traj

    def trace_bidirectional(self, skeleton, frame, start_point):
        """
        双向追踪：从起始点向左和向右追踪

        参数:
            skeleton: 骨架图像
            frame: 原始BGR图像
            start_point: 起始点 (y, x)

        返回:
            list: 完整轨迹，按x排序
        """
        if start_point is None:
            return []

        y_center = start_point[0]

        # 向左追踪
        left_traj = self.trace_direction(skeleton, frame, start_point, y_center, go_left=True)

        # 向右追踪
        right_traj = self.trace_direction(skeleton, frame, start_point, y_center, go_left=False)

        # 合并轨迹（左侧倒序 + 右侧）
        if len(right_traj) > 1:
            full = left_traj[::-1] + right_traj[1:]
        else:
            full = left_traj[::-1]

        return full

    def find_growth(self, skeleton, frame, trajectory, y_center):
        """
        在轨迹末端搜索生长点

        参数:
            skeleton: 骨架图像
            frame: 原始BGR图像
            trajectory: 当前轨迹
            y_center: Y轴中心位置

        返回:
            list: 新增的点列表
        """
        if not trajectory:
            return []

        tip = trajectory[-1]  # 当前末端
        h, w = skeleton.shape

        # 获取当前方向
        current_dir = self.get_direction(trajectory, num_points=10)

        # 已访问点
        visited = set(trajectory)
        new_points = []

        # BFS搜索
        queue = [(tip, 0)]  # (point, depth)

        while queue:
            cur, depth = queue.pop(0)

            if depth > Config.GROWTH_MAX_DEPTH:
                continue

            cy, cx = cur
            candidates = []

            # 搜索邻域
            r = Config.GROWTH_SEARCH_RADIUS
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    if dy == 0 and dx == 0:
                        continue

                    ny, nx = cy + dy, cx + dx

                    if not (0 <= ny < h and 0 <= nx < w):
                        continue

                    if (ny, nx) in visited:
                        continue

                    if not skeleton[ny, nx]:
                        continue

                    if not self.img_proc.is_green_pixel(frame, ny, nx):
                        continue

                    if abs(ny - y_center) >= Config.Y_TOLERANCE * 2:
                        continue

                    # 计算得分
                    dist = np.sqrt(dy * dy + dx * dx)
                    if dist > 0:
                        cand_dir = (dy / dist, dx / dist)
                        dir_score = self.direction_score(current_dir, cand_dir)
                    else:
                        dir_score = 0

                    # 优先向右 + 方向一致
                    score = dx * 5 + dir_score * Config.DIRECTION_WEIGHT
                    candidates.append(((ny, nx), score))

            # 按得分排序
            candidates.sort(key=lambda x: x[1], reverse=True)

            # 添加候选点
            for (ny, nx), _ in candidates:
                if (ny, nx) not in visited:
                    visited.add((ny, nx))
                    new_points.append((ny, nx))
                    queue.append(((ny, nx), depth + 1))

                    # 定期更新方向
                    if len(new_points) % 5 == 0:
                        temp_traj = list(trajectory) + new_points
                        current_dir = self.get_direction(temp_traj, num_points=10)

        return new_points
