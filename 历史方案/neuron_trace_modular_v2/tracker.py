"""
追踪算法模块
"""
import numpy as np
import heapq
from .config import Config

class Tracker:
    def __init__(self, image_proc):
        self.ip = image_proc

    def _dir(self, t, n=8):
        """计算轨迹最近n个点的平均方向"""
        if len(t) < 2:
            return (0, 1)
        r = t[-min(n, len(t)):]
        dy = r[-1][0] - r[0][0]
        dx = r[-1][1] - r[0][1]
        ln = np.sqrt(dy * dy + dx * dx)
        return (dy / ln, dx / ln) if ln > 0.001 else (0, 1)

    def _trace1(self, skel, f, start, yc, left, avoid=None, history=None):
        """
        单向追踪 (严格限制单次生长长度)
        """
        if not start:
            return []

        h, w = skel.shape
        vis = set(avoid) if avoid else set()

        # 使用历史轨迹初始化方向
        if history and len(history) >= 2:
            dir_history = list(history[-10:])
        else:
            dir_history = [start]

        vis.add(start)
        cur = start
        new_points = []
        total_growth = 0 # 记录当前帧生长总长度
        
        # 预处理图像用于快速亮度检查
        enh = self.ip._preprocess(f)

        while True:
            # 检查生长极限
            if total_growth > Config.MAX_GROWTH_PER_FRAME:
                break
                
            d = self._dir(dir_history)
            cands = []

            for sr in [1, 2, Config.MAX_GAP]:
                for dy in range(-sr, sr + 1):
                    for dx in range(-sr, sr + 1):
                        if dy == 0 and dx == 0:
                            continue
                        ny, nx = cur[0] + dy, cur[1] + dx

                        if not (0 <= ny < h and 0 <= nx < w):
                            continue
                        if (ny, nx) in vis:
                            continue
                        if not skel[ny, nx]:
                            continue
                        
                        # 检查亮度
                        if enh[ny, nx] < Config.NEURON_BRIGHTNESS_THRESHOLD:
                            continue
                            
                        # Y轴严格约束
                        if abs(ny - yc) > Config.Y_TOLERANCE:
                            continue

                        # 距离约束
                        dist = np.sqrt(dy * dy + dx * dx)
                        if dist > Config.MAX_LINK_DISTANCE:
                            continue
                        
                        # 方向一致性约束 (加强版)
                        sc = (-dx if left else dx) * 10 - abs(dy) * 5 - dist * 3
                        
                        if dist > 0:
                            cd = (dy / dist, dx / dist)
                            # 严惩反向和大幅折返
                            dot_prod = d[0] * cd[0] + d[1] * cd[1]
                            if dot_prod < 0: # 只要是钝角就禁止，防止锯齿
                                continue
                            sc += dot_prod * Config.DIRECTION_WEIGHT

                        cands.append(((ny, nx), sc))

                if cands:
                    break

            if not cands:
                break

            cands.sort(key=lambda x: x[1], reverse=True)
            best = cands[0][0]
            
            # 累加生长长度
            step_dist = np.sqrt((best[0]-cur[0])**2 + (best[1]-cur[1])**2)
            total_growth += step_dist

            vis.add(best)
            new_points.append(best)
            dir_history.append(best)
            cur = best

        return new_points

    def _bidir(self, skel, f, start):
        """双向追踪"""
        if not start:
            return []
        yc = start[0]
        left = self._trace1(skel, f, start, yc, left=True)
        right = self._trace1(skel, f, start, yc, left=False)

        full = []
        if left:
            full = left[::-1]
        full.append(start)
        if right:
            full.extend(right)
        return full

    def grow(self, skel, f, traj, yc):
        """生长追踪"""
        if not traj:
            return []
        return self._trace1(
            skel, f,
            start=traj[-1],
            yc=yc,
            left=False,
            avoid=set(traj),
            history=traj
        )

    def _astar(self, skel, f, start, end):
        """A*路径搜索"""
        h, w = skel.shape
        heur = lambda p: np.sqrt((p[0] - end[0]) ** 2 + (p[1] - end[1]) ** 2)
        cnt = 0
        heap = [(heur(start), cnt, start, [start])]
        vis = {start}
        
        # 预处理图像
        enh = self.ip._preprocess(f)

        for _ in range(5000):
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
                    if (ny, nx) in vis:
                        continue
                    
                    # 骨架 + 亮度检查
                    if not skel[ny, nx] or enh[ny, nx] < Config.NEURON_BRIGHTNESS_THRESHOLD:
                        continue
                        
                    vis.add((ny, nx))
                    cnt += 1
                    heapq.heappush(heap, (len(path) + heur((ny, nx)), cnt, (ny, nx), path + [(ny, nx)]))

        # A*失败，直线连接
        path = [start]
        steps = max(1, int(heur(start)))
        for i in range(1, steps + 1):
            t = i / steps
            y = int(start[0] + t * (end[0] - start[0]))
            x = int(start[1] + t * (end[1] - start[1]))
            if (y, x) != path[-1]:
                path.append((y, x))
        if path[-1] != end:
            path.append(end)
        return path

    def trace(self, skel, f, markers):
        """使用标记点进行初始追踪"""
        if not markers:
            return []

        pts = []
        for m in markers:
            p = self.ip.nearest_skel(skel, f, (m["x"], m["y"]))
            if p:
                pts.append(p)

        if not pts:
            return []

        # 去重
        uniq = []
        for p in pts:
            if not any(abs(p[0] - u[0]) < 5 and abs(p[1] - u[1]) < 5 for u in uniq):
                uniq.append(p)
        uniq.sort(key=lambda p: p[1])

        if len(uniq) == 1:
            return self._bidir(skel, f, uniq[0])

        # 多点A*连接
        traj = []
        for i in range(len(uniq) - 1):
            seg = self._astar(skel, f, uniq[i], uniq[i + 1])
            if seg:
                if traj and seg[0] == traj[-1]:
                    seg = seg[1:]
                traj.extend(seg)

        # 两端延伸
        if uniq:
            left = self._trace1(skel, f, uniq[0], uniq[0][0], left=True, avoid=set(traj))
            traj = left[::-1] + traj
            right = self._trace1(skel, f, uniq[-1], uniq[-1][0], left=False, avoid=set(traj))
            traj = traj + right

        # 去重
        seen = set()
        return [p for p in traj if not (p in seen or seen.add(p))]
