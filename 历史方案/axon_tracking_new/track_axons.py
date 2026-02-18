import numpy as np
import tifffile
import cv2
import pandas as pd
from skimage import filters, feature, io
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

class AxonTracker:
    def __init__(self, 
                 pixel_size=1.0,           # 每个像素对应的微米数
                 max_linking_distance=30,  # 允许连接的最大像素距离
                 direction_weight=0.6,     # 方向一致性权重 (0-1)
                 min_track_length=5):      # 最小轨迹长度（帧数）
        self.pixel_size = pixel_size
        self.max_linking_distance = max_linking_distance
        self.direction_weight = direction_weight
        self.min_track_length = min_track_length
        self.tracks = {}  # {track_id: [list of (y, x, frame_idx)]}
        self.next_track_id = 0
        self.active_tracks = {}  # {track_id: last_point}
        self.track_directions = {} # {track_id: (dy, dx)} normalized

    def load_tiff_sequence(self, path):
        """加载TIFF序列或多页TIFF"""
        print(f"正在加载图像: {path}")
        if os.path.isdir(path):
            files = sorted([os.path.join(path, f) for f in os.listdir(path) if f.endswith(('.tif', '.tiff'))])
            stack = np.array([tifffile.imread(f) for f in files])
        else:
            stack = tifffile.imread(path)
        
        # 确保是3D数组 (Time, Height, Width)
        if stack.ndim == 2:
            stack = stack[np.newaxis, ...]
            
        print(f"加载完成，共 {stack.shape[0]} 帧，尺寸 {stack.shape[1]}x{stack.shape[2]}")
        return stack

    def preprocess_frame(self, frame):
        """预处理：Frangi Vesselness Filter 增强管状结构"""
        # 归一化到0-1
        frame_norm = (frame - frame.min()) / (frame.max() - frame.min() + 1e-8)
        
        # Frangi滤波增强血管/轴突结构
        # sigmas控制检测的尺度范围，beta控制对blob状结构的抑制程度
        vesselness = filters.frangi(frame_norm, sigmas=range(1, 4), beta=0.5)
        
        # 再次归一化便于显示和后续处理
        vesselness = (vesselness - vesselness.min()) / (vesselness.max() - vesselness.min() + 1e-8)
        return vesselness

    def detect_tips(self, vesselness_map, threshold=0.1):
        """检测：LoG 检测轴突尖端"""
        # LoG检测斑点 (y, x, sigma)
        blobs = feature.blob_log(vesselness_map, 
                               min_sigma=1, 
                               max_sigma=4, 
                               num_sigma=10, 
                               threshold=threshold)
        
        if len(blobs) == 0:
            return []
            
        # 返回 (y, x) 坐标
        return blobs[:, :2]

    def _calculate_cost_matrix(self, active_points, detections):
        """计算代价矩阵：结合距离和方向一致性"""
        if not active_points or len(detections) == 0:
            return None
            
        # 1. 欧氏距离矩阵
        active_coords = np.array(list(active_points.values()))
        dist_matrix = cdist(active_coords, detections)
        
        # 2. 方向代价矩阵
        dir_cost_matrix = np.zeros_like(dist_matrix)
        active_ids = list(active_points.keys())
        
        for i, track_id in enumerate(active_ids):
            last_pos = active_points[track_id]
            current_dir = self.track_directions.get(track_id, (0, 0))
            
            # 如果没有历史方向（新轨迹），方向代价为0
            # 兼容 tuple 和 numpy array 的比较
            is_zero_dir = False
            if isinstance(current_dir, tuple):
                if current_dir == (0, 0):
                    is_zero_dir = True
            else:
                if np.all(current_dir == 0):
                    is_zero_dir = True
            
            if is_zero_dir:
                continue
                
            for j, det in enumerate(detections):
                # 计算从上一帧位置到当前检测点的向量
                move_vec = det - last_pos
                dist = np.linalg.norm(move_vec)
                
                if dist > 0:
                    move_dir = move_vec / dist
                    # 计算余弦相似度: 1表示同向，-1表示反向
                    cos_sim = np.dot(current_dir, move_dir)
                    # 将余弦相似度转换为代价 (0到1之间，同向代价低)
                    # cos_sim范围[-1, 1] -> cost范围[1, 0]
                    dir_cost = 0.5 * (1 - cos_sim)
                    dir_cost_matrix[i, j] = dir_cost * dist # 结合距离加权
        
        # 3. 综合代价
        # Cost = (1 - w) * dist + w * direction_cost
        # 注意：这里简单的加权求和，需要量级大致匹配
        total_cost = (1 - self.direction_weight) * dist_matrix + \
                     self.direction_weight * (dir_cost_matrix * 10) # 乘以系数平衡量级
                     
        # 将超过最大距离的设为无穷大
        total_cost[dist_matrix > self.max_linking_distance] = 1e6
        
        return total_cost

    def update_tracks(self, detections, frame_idx):
        """追踪：LAP算法更新轨迹"""
        detections = np.array(detections)
        active_ids = list(self.active_tracks.keys())
        
        # 如果没有活跃轨迹，所有检测点都作为新轨迹
        if not active_ids:
            for det in detections:
                self._start_new_track(det, frame_idx)
            return

        # 如果没有检测点，所有活跃轨迹暂时标记为丢失（这里简化处理，不做复杂丢失恢复）
        if len(detections) == 0:
            # 可以在这里增加逻辑处理丢失
            return

        # 计算代价矩阵
        cost_matrix = self._calculate_cost_matrix(self.active_tracks, detections)
        
        # LAP求解
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        assigned_detections = set()
        assigned_tracks = set()
        
        for r, c in zip(row_ind, col_ind):
            cost = cost_matrix[r, c]
            
            # 只有在允许距离内的才算匹配
            if cost < 1e5:
                track_id = active_ids[r]
                detection = detections[c]
                
                self._extend_track(track_id, detection, frame_idx)
                
                assigned_tracks.add(track_id)
                assigned_detections.add(c)
        
        # 处理未匹配的检测点 -> 新轨迹
        for i in range(len(detections)):
            if i not in assigned_detections:
                self._start_new_track(detections[i], frame_idx)
                
        # 处理未匹配的轨迹 -> 结束或保留（这里简化为不更新，下帧继续尝试匹配）
        # 实际应用中可能需要设置“最大丢失帧数”来决定何时终止轨迹
        pass

    def _start_new_track(self, point, frame_idx):
        track_id = self.next_track_id
        self.next_track_id += 1
        
        self.tracks[track_id] = [(point[0], point[1], frame_idx)]
        self.active_tracks[track_id] = point
        self.track_directions[track_id] = (0, 0) # 初始无方向

    def _extend_track(self, track_id, point, frame_idx):
        # 更新轨迹点
        self.tracks[track_id].append((point[0], point[1], frame_idx))
        
        # 更新方向
        last_pos = self.active_tracks[track_id]
        move_vec = point - last_pos
        dist = np.linalg.norm(move_vec)
        
        if dist > 0:
            # 简单的动量更新：平滑方向
            new_dir = move_vec / dist
            old_dir = self.track_directions[track_id]
            
            is_zero = False
            if isinstance(old_dir, tuple):
                if old_dir == (0, 0):
                    is_zero = True
            elif np.all(old_dir == 0):
                is_zero = True

            if is_zero:
                self.track_directions[track_id] = new_dir
            else:
                # 动量因子 0.7
                smooth_dir = 0.7 * np.array(old_dir) + 0.3 * new_dir
                norm = np.linalg.norm(smooth_dir)
                if norm > 0:
                    self.track_directions[track_id] = smooth_dir / norm
        
        self.active_tracks[track_id] = point

    def run(self, input_path, output_dir):
        """运行完整追踪流程"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        stack = self.load_tiff_sequence(input_path)
        
        # 初始化视频写入
        height, width = stack.shape[1], stack.shape[2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_path = os.path.join(output_dir, 'axon_tracking_result.mp4')
        out = cv2.VideoWriter(video_path, fourcc, 10, (width, height), True) # 10 FPS
        
        # 颜色映射
        np.random.seed(42)
        colors = np.random.randint(0, 255, (1000, 3))
        
        print("开始追踪...")
        
        for frame_idx, frame in tqdm(enumerate(stack), total=len(stack)):
            # 1. 预处理
            vesselness = self.preprocess_frame(frame)
            
            # 2. 检测
            detections = self.detect_tips(vesselness)
            
            # 3. 追踪
            self.update_tracks(detections, frame_idx)
            
            # 4. 可视化
            # 转为RGB以便绘制彩色轨迹
            frame_disp = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            frame_color = cv2.cvtColor(frame_disp, cv2.COLOR_GRAY2BGR)
            
            # 绘制所有轨迹
            for track_id, points in self.tracks.items():
                if len(points) < 2:
                    continue
                    
                color = colors[track_id % len(colors)].tolist()
                
                # 绘制历史轨迹线
                pts = np.array([(int(p[1]), int(p[0])) for p in points], np.int32)
                cv2.polylines(frame_color, [pts], False, color, 2)
                
                # 绘制当前点
                if points[-1][2] == frame_idx:
                    cv2.circle(frame_color, (pts[-1][0], pts[-1][1]), 4, color, -1)
                    cv2.putText(frame_color, f"ID:{track_id}", (pts[-1][0]+5, pts[-1][1]-5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            out.write(frame_color)
            
            # 实时显示
            cv2.namedWindow("Axon Tracking", cv2.WINDOW_NORMAL)
            cv2.imshow("Axon Tracking", frame_color)
            if cv2.waitKey(100) & 0xFF == ord('q'):
                print("\n用户终止追踪")
                break
            
        out.release()
        cv2.destroyAllWindows()
        print(f"视频已保存至: {video_path}")
        
        # 5. 量化分析与导出
        self._export_data(output_dir)

    def _export_data(self, output_dir):
        """导出CSV数据：长度随时间变化"""
        print("正在计算量化指标...")
        data = []
        
        for track_id, points in self.tracks.items():
            if len(points) < self.min_track_length:
                continue
                
            # 计算每帧的累积长度
            cum_length = 0.0
            for i in range(len(points)):
                y, x, frame = points[i]
                
                if i > 0:
                    prev_y, prev_x, _ = points[i-1]
                    dist = np.sqrt((y - prev_y)**2 + (x - prev_x)**2) * self.pixel_size
                    cum_length += dist
                
                data.append({
                    'TrackID': track_id,
                    'Frame': frame,
                    'X': x,
                    'Y': y,
                    'Length_um': cum_length
                })
                
        df = pd.DataFrame(data)
        csv_path = os.path.join(output_dir, 'axon_tracking_data.csv')
        df.to_csv(csv_path, index=False)
        print(f"数据已保存至: {csv_path}")
        
        # 简单绘图
        if not df.empty:
            plt.figure(figsize=(10, 6))
            for track_id in df['TrackID'].unique():
                track_data = df[df['TrackID'] == track_id]
                plt.plot(track_data['Frame'], track_data['Length_um'], label=f'ID {track_id}')
            
            plt.xlabel('Frame')
            plt.ylabel('Length (um)')
            plt.title('Axon Growth Over Time')
            # 如果轨迹太多，就不显示图例了
            if len(df['TrackID'].unique()) < 20:
                plt.legend()
            plt.savefig(os.path.join(output_dir, 'growth_plot.png'))
            plt.close()

if __name__ == "__main__":
    # 真实数据路径
    real_data_path = r"/recover_0-72h_XY2.tif"
    
    if os.path.exists(real_data_path):
        print(f"检测到真实数据: {real_data_path}")
        # 运行追踪器
        # 注意：根据实际数据可能需要调整参数
        # 例如：pixel_size（像素大小），max_linking_distance（最大移动距离）
        tracker = AxonTracker(pixel_size=0.65, max_linking_distance=50, direction_weight=0.7)
        tracker.run(real_data_path, "tracking_output_real")
        print("真实数据处理完成！请查看 tracking_output_real 文件夹。")
    else:
        # ===== 生成测试数据 =====
        print("未找到真实数据，生成测试数据...")
        height, width = 512, 512
        num_frames = 50
        synthetic_stack = np.zeros((num_frames, height, width), dtype=np.uint8)
        
        # 模拟两个交叉生长的轴突
        # 轴突1: 左上 -> 右下
        start1 = (100, 100)
        # 轴突2: 左下 -> 右上
        start2 = (400, 100)
        
        for f in range(num_frames):
            # 添加背景噪声
            noise = np.random.normal(0, 5, (height, width)).astype(np.uint8)
            synthetic_stack[f] = noise
            
            # 轴突1生长
            len1 = 50 + f * 5
            end1 = (int(start1[0] + len1 * 0.5), int(start1[1] + len1 * 0.8))
            cv2.line(synthetic_stack[f], (start1[1], start1[0]), (end1[1], end1[0]), 200, 3)
            
            # 轴突2生长
            len2 = 50 + f * 6
            end2 = (int(start2[0] - len2 * 0.4), int(start2[1] + len2 * 0.8))
            cv2.line(synthetic_stack[f], (start2[1], start2[0]), (end2[1], end2[0]), 180, 3)
            
            # 高斯模糊模拟显微镜效果
            synthetic_stack[f] = cv2.GaussianBlur(synthetic_stack[f], (5, 5), 0)

        test_tif_path = "synthetic_axons.tif"
        tifffile.imwrite(test_tif_path, synthetic_stack)
        print(f"测试数据已保存: {test_tif_path}")

        # ===== 运行追踪器 =====
        tracker = AxonTracker(pixel_size=0.5, direction_weight=0.7)
        tracker.run(test_tif_path, "tracking_output")
        
        print("演示完成！请查看 tracking_output 文件夹。")
