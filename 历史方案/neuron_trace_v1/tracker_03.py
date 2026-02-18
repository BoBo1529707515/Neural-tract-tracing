import cv2
import numpy as np

标识_03_跟踪器 = "03_跟踪器"

class PointTracker:
    def __init__(self, all_neuron_marks):
        """
        all_neuron_marks: {neuron_id: {frame_idx: (x, y)}}
        """
        self.all_marks = all_neuron_marks
        self.roi_size = 40  # 增大搜索框
        
        # 模板匹配相关参数
        self.template_size = 20 # 模板大小 (20x20)
        self.search_radius = 40 # 局部搜索半径

    def track(self, frames_processed, start_frame_idx):
        if len(frames_processed) == 0:
            return {}

        all_trajectories = {} # {neuron_id: [(x, y, frame_idx), ...]}

        # 对每个神经元分别进行跟踪
        for nid, marks in self.all_marks.items():
            print(f"Tracking Neuron {nid}...")
            
            # 找到该神经元最早的标记帧
            if not marks:
                continue
            
            # 按照帧号排序，找到起始帧
            sorted_frames = sorted(marks.keys())
            first_frame = sorted_frames[0]
            
            # 轨迹初始化
            trajectory = []
            
            # 当前位置
            curr_pos = marks[first_frame]
            trajectory.append((int(curr_pos[0]), int(curr_pos[1]), first_frame))
            
            # 逐帧向后跟踪
            for i in range(first_frame + 1, len(frames_processed)):
                prev_img = frames_processed[i-1]
                curr_img = frames_processed[i]
                
                # 1. 检查是否有人工标记
                if i in marks:
                    curr_pos = marks[i] # 强制校正
                    trajectory.append((int(curr_pos[0]), int(curr_pos[1]), i))
                    continue
                
                # 2. 自动跟踪：使用局部模板匹配 (NCC)
                # 提取上一帧的模板
                px, py = int(curr_pos[0]), int(curr_pos[1])
                
                # 边界检查
                h, w = prev_img.shape
                x1 = max(0, px - self.template_size)
                y1 = max(0, py - self.template_size)
                x2 = min(w, px + self.template_size)
                y2 = min(h, py + self.template_size)
                
                template = prev_img[y1:y2, x1:x2]
                
                if template.size == 0:
                    trajectory.append((px, py, i))
                    continue
                
                # 在当前帧的局部区域搜索
                sx1 = max(0, px - self.search_radius)
                sy1 = max(0, py - self.search_radius)
                sx2 = min(w, px + self.search_radius)
                sy2 = min(h, py + self.search_radius)
                
                search_region = curr_img[sy1:sy2, sx1:sx2]
                
                if search_region.shape[0] < template.shape[0] or search_region.shape[1] < template.shape[1]:
                     trajectory.append((px, py, i))
                     continue

                # 模板匹配
                res = cv2.matchTemplate(search_region, template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(res)
                
                # 如果匹配度太低，认为跟丢了，保持不动
                if max_val < 0.5:
                    # 尝试简单的亮度重心法作为备选
                    # ... (此处省略，简单保持上一帧位置)
                    trajectory.append((px, py, i))
                else:
                    # 更新位置
                    # max_loc 是在 search_region 中的坐标
                    # new_x = sx1 + max_loc[0] + template_w/2
                    dx = max_loc[0] + (template.shape[1] // 2)
                    dy = max_loc[1] + (template.shape[0] // 2)
                    
                    new_x = sx1 + dx
                    new_y = sy1 + dy
                    
                    # 约束移动距离（生长不会突变）
                    dist = np.sqrt((new_x - px)**2 + (new_y - py)**2)
                    if dist > self.search_radius:
                        # 移动过大，可能是错误匹配
                        trajectory.append((px, py, i))
                    else:
                        curr_pos = (new_x, new_y)
                        trajectory.append((int(new_x), int(new_y), i))
            
            all_trajectories[nid] = trajectory
            
        return all_trajectories
