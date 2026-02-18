import numpy as np
import cv2

class Simulation:
    def __init__(self, width=640, height=480, duration=100):
        self.width = width
        self.height = height
        self.duration = duration
        self.frames = []
        
    def generate_crossing_paths(self):
        """
        生成两个交叉生长的点
        Path 1: 左上 -> 右下
        Path 2: 左下 -> 右上
        """
        path1 = []
        path2 = []
        
        # 定义起点
        start1 = np.array([100.0, 100.0])
        start2 = np.array([100.0, 380.0])
        
        # 定义速度 (每帧移动像素)
        vel1 = np.array([3.0, 2.0])  # 向右下
        vel2 = np.array([3.0, -2.0]) # 向右上
        
        detections = [] # list of lists per frame [[(x1,y1), (x2,y2)], ...]
        
        for t in range(self.duration):
            # 计算当前位置 (加一点随机噪声模拟检测误差)
            pos1 = start1 + vel1 * t + np.random.normal(0, 1.0, 2)
            pos2 = start2 + vel2 * t + np.random.normal(0, 1.0, 2)
            
            # 记录真实轨迹
            path1.append(pos1)
            path2.append(pos2)
            
            # 模拟检测器输出 (只给坐标，不给ID)
            # 在交叉点附近，检测器可能会把两个点合并成一个，或者位置发生偏移
            # 这里简单模拟两个点都检测到了
            current_dets = [pos1, pos2]
            
            # 模拟偶尔漏检 (10% 概率)
            if np.random.rand() < 0.1:
                if np.random.rand() < 0.5:
                    current_dets = [pos1]
                else:
                    current_dets = [pos2]
            
            detections.append(current_dets)
            
        return detections, (path1, path2)

    def visualize_frame(self, frame_idx, detections, tracks, img=None):
        if img is None:
            img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            
        # 绘制检测点 (绿色圆圈)
        for det in detections:
            cv2.circle(img, (int(det[0]), int(det[1])), 5, (0, 255, 0), 2)
            
        # 绘制跟踪轨迹 (带ID)
        # tracks: [x, y, id]
        colors = [(255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
        
        for trk in tracks:
            x, y, tid = trk
            color = colors[int(tid) % len(colors)]
            cv2.circle(img, (int(x), int(y)), 8, color, -1)
            cv2.putText(img, f"ID:{int(tid)}", (int(x)+10, int(y)-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # 绘制方向箭头
            # (这里简化，只画当前点)
            
        return img
