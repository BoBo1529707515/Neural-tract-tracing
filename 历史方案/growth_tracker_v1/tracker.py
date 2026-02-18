import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment
from collections import deque

class KalmanPointTracker:
    """
    基于 OpenCV 卡尔曼滤波的点跟踪器
    用于单个生长点的状态预测
    """
    count = 0
    def __init__(self, initial_point):
        """
        初始化跟踪器
        :param initial_point: (x, y) 初始坐标
        """
        # 定义恒速模型 (Constant Velocity Model)
        # 状态向量: [x, y, dx, dy]
        self.kf = cv2.KalmanFilter(4, 2)
        
        # 状态转移矩阵 (F)
        # x' = x + dx
        # y' = y + dy
        # dx' = dx
        # dy' = dy
        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], np.float32)

        # 测量矩阵 (H) - 我们只能观测到位置 (x, y)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], np.float32)

        # 过程噪声协方差 (Q) - 模型的不确定性
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        
        # 测量噪声协方差 (R) - 观测的不确定性
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1
        
        # 误差协方差 (P) - 初始状态的不确定性
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 1.0

        # 初始化状态
        self.kf.statePost = np.array([
            [np.float32(initial_point[0])], 
            [np.float32(initial_point[1])], 
            [0], 
            [0]
        ], np.float32)

        self.time_since_update = 0
        self.id = KalmanPointTracker.count
        KalmanPointTracker.count += 1
        
        self.history = deque(maxlen=50) # 记录历史轨迹
        self.hits = 0 # 连续匹配次数
        self.hit_streak = 0
        self.age = 0
        
        # 记录初始位置
        self.history.append(initial_point)

    def update(self, detection):
        """
        使用观测值更新状态
        :param detection: (x, y) 观测坐标
        """
        self.time_since_update = 0
        self.history.append(detection)
        self.hits += 1
        self.hit_streak += 1
        
        # 将观测值转为 OpenCV 需要的格式
        measurement = np.array([[np.float32(detection[0])], [np.float32(detection[1])]])
        self.kf.correct(measurement)

    def predict(self):
        """
        预测下一帧状态
        :return: (x, y) 预测坐标
        """
        # 如果长时间未更新，增加预测的不确定性
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.age += 1
        
        prediction = self.kf.predict()
        return prediction[0, 0], prediction[1, 0]

    def get_state(self):
        """
        获取当前估计状态
        :return: [x, y, dx, dy]
        """
        return self.kf.statePost.flatten()
    
    def get_velocity_direction(self):
        """
        获取归一化的速度方向向量
        :return: (vx_norm, vy_norm)
        """
        vx = self.kf.statePost[2, 0]
        vy = self.kf.statePost[3, 0]
        norm = np.sqrt(vx**2 + vy**2)
        if norm < 1e-6:
            return (0, 0)
        return (vx/norm, vy/norm)


class DirectionalSort:
    """
    改进的 SORT 跟踪算法
    核心改进：代价矩阵结合了距离和方向一致性
    """
    def __init__(self, max_age=10, min_hits=3, dist_threshold=50, direction_weight=0.6):
        """
        :param max_age: 允许丢失的最大帧数
        :param min_hits: 确认为有效轨迹所需的最小连续匹配数
        :param dist_threshold: 允许匹配的最大距离
        :param direction_weight: 方向一致性权重 (0-1)
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.dist_threshold = dist_threshold
        self.direction_weight = direction_weight
        self.trackers = []
        self.frame_count = 0

    def update(self, detections):
        """
        处理当前帧的检测结果
        :param detections: list of (x, y)
        :return: list of active tracks info
        """
        self.frame_count += 1
        
        # 1. 获取所有跟踪器的预测位置
        predictions = []
        to_del = []
        for i, trk in enumerate(self.trackers):
            pos = trk.predict() # 预测下一帧位置
            # 如果预测位置非法（比如NaN），标记删除
            if np.any(np.isnan(pos)):
                to_del.append(i)
            predictions.append(pos)
            
        for i in reversed(to_del):
            self.trackers.pop(i)
            predictions.pop(i)

        # 2. 数据关联：计算代价矩阵
        # 行：现有跟踪器 (Tracks)
        # 列：当前检测点 (Detections)
        if len(self.trackers) > 0 and len(detections) > 0:
            cost_matrix = self._calculate_cost_matrix(self.trackers, detections, predictions)
            
            # 使用匈牙利算法求解最小代价匹配
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            
            unmatched_trackers = []
            for i in range(len(self.trackers)):
                if i not in row_indices:
                    unmatched_trackers.append(i)
            
            unmatched_detections = []
            for i in range(len(detections)):
                if i not in col_indices:
                    unmatched_detections.append(i)
            
            matches = []
            for r, c in zip(row_indices, col_indices):
                # 检查代价阈值，过大的代价视为不匹配
                if cost_matrix[r, c] > self.dist_threshold: # 这里简单用距离阈值做个兜底
                    unmatched_trackers.append(r)
                    unmatched_detections.append(c)
                else:
                    matches.append((r, c))
        else:
            matches = []
            unmatched_trackers = list(range(len(self.trackers)))
            unmatched_detections = list(range(len(detections)))

        # 3. 更新匹配的跟踪器
        for t_idx, d_idx in matches:
            self.trackers[t_idx].update(detections[d_idx])

        # 4. 为未匹配的检测点创建新跟踪器
        for d_idx in unmatched_detections:
            trk = KalmanPointTracker(detections[d_idx])
            self.trackers.append(trk)

        # 5. 管理跟踪器生命周期
        ret = []
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            i -= 1
            # 如果丢失时间过长，删除
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
                continue
            
            # 只有足够稳定的轨迹才输出显示
            # 或者如果是新创建的轨迹且还没有丢失过
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                # 输出格式: [x, y, id]
                state = trk.get_state()
                ret.append(np.concatenate((state[:2], [trk.id])))
                
        return ret

    def _calculate_cost_matrix(self, trackers, detections, predictions):
        """
        核心逻辑：计算代价矩阵
        Cost = w1 * 距离 + w2 * 方向惩罚
        """
        num_tracks = len(trackers)
        num_dets = len(detections)
        cost_matrix = np.zeros((num_tracks, num_dets))
        
        for t, trk in enumerate(trackers):
            # 获取预测位置
            pred_x, pred_y = predictions[t]
            # 获取当前轨迹的速度方向
            v_dir = trk.get_velocity_direction() # (vx, vy) normalized
            
            for d, det in enumerate(detections):
                det_x, det_y = det
                
                # 1. 距离代价 (欧氏距离)
                dist = np.sqrt((pred_x - det_x)**2 + (pred_y - det_y)**2)
                
                # 2. 方向代价 (余弦相似度)
                # 计算从"预测位置"指向"检测位置"的向量
                # 注意：这里我们比较的是 "历史速度方向" 和 "当前位移方向"
                # 实际上，如果卡尔曼滤波预测得准，pred应该很接近det
                # 更好的方向约束是：轨迹的历史运动方向 vs (检测点 - 轨迹上一帧位置) 的方向
                
                # 获取上一帧位置（从KF状态反推或者直接用history）
                last_pos = trk.history[-1] if trk.history else (pred_x, pred_y)
                move_vec = (det_x - last_pos[0], det_y - last_pos[1])
                move_dist = np.sqrt(move_vec[0]**2 + move_vec[1]**2)
                
                dir_cost = 0
                if move_dist > 1e-6 and (v_dir[0] != 0 or v_dir[1] != 0):
                    move_dir = (move_vec[0]/move_dist, move_vec[1]/move_dist)
                    # 余弦相似度: dot(v_dir, move_dir) -> range [-1, 1]
                    # 1 表示同向, -1 表示反向
                    cos_sim = v_dir[0]*move_dir[0] + v_dir[1]*move_dir[1]
                    
                    # 将相似度转换为代价: 
                    # 我们希望同向(1)代价为0，反向(-1)代价最大
                    # cost = 0.5 * (1 - cos_sim) -> range [0, 1]
                    dir_cost = 0.5 * (1 - cos_sim) * 100 # 放大系数，使其与像素距离量级相当
                
                # 综合代价
                # 如果是静止点或刚开始运动，方向权重应降低
                current_w = self.direction_weight
                if move_dist < 2 or (v_dir[0] == 0 and v_dir[1] == 0):
                    current_w = 0.1
                
                total_cost = (1 - current_w) * dist + current_w * dir_cost
                cost_matrix[t, d] = total_cost
                
        return cost_matrix
