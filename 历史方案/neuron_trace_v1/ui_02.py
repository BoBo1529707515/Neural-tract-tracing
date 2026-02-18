import cv2
import numpy as np

标识_02_交互界面 = "02_交互界面"

class InteractiveUI:
    def __init__(self, frames, window_name):
        self.frames = frames
        self.total_frames = len(frames)
        self.current_idx = 0
        self.window_name = window_name
        
        # 交互状态
        self.points = {}  # {neuron_id: {frame_idx: (x, y)}}
        self.current_neuron_id = 1
        
        self.zoom_level = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.is_dragging = False
        self.last_mouse_pos = (0, 0)
        self.input_mode = False
        self.input_buffer = ""
        
        # UI配置
        self.bg_color = (40, 40, 40)
        self.text_color = (220, 220, 220)
        self.accent_color = (0, 255, 200)
        
        # 颜色映射 (ID -> Color)
        self.colors = {}

    def _get_color(self, neuron_id):
        if neuron_id not in self.colors:
            np.random.seed(neuron_id)
            self.colors[neuron_id] = tuple(map(int, np.random.randint(50, 255, 3)))
        return self.colors[neuron_id]

    def _transform_point(self, x, y, inverse=False):
        """坐标变换：屏幕坐标 <-> 图像坐标"""
        h, w = self.frames[0].shape[:2]
        center_x, center_y = w / 2, h / 2
        
        if inverse: # 屏幕 -> 图像
            img_x = (x - center_x) / self.zoom_level + center_x - self.offset_x
            img_y = (y - center_y) / self.zoom_level + center_y - self.offset_y
            return int(img_x), int(img_y)
        else: # 图像 -> 屏幕
            scr_x = (x - center_x + self.offset_x) * self.zoom_level + center_x
            scr_y = (y - center_y + self.offset_y) * self.zoom_level + center_y
            return int(scr_x), int(scr_y)

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if not self.is_dragging:
                # 转换为图像坐标并记录
                img_x, img_y = self._transform_point(x, y, inverse=True)
                
                if self.current_neuron_id not in self.points:
                    self.points[self.current_neuron_id] = {}
                
                # 记录该神经元在当前帧的位置（覆盖旧点）
                self.points[self.current_neuron_id][self.current_idx] = (img_x, img_y)
                
        elif event == cv2.EVENT_RBUTTONDOWN:
            # 删除当前神经元在当前帧的点
            if self.current_neuron_id in self.points:
                if self.current_idx in self.points[self.current_neuron_id]:
                    del self.points[self.current_neuron_id][self.current_idx]
                
        elif event == cv2.EVENT_MBUTTONDOWN:
            self.is_dragging = True
            self.last_mouse_pos = (x, y)
            
        elif event == cv2.EVENT_MBUTTONUP:
            self.is_dragging = False
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.is_dragging:
                dx = x - self.last_mouse_pos[0]
                dy = y - self.last_mouse_pos[1]
                self.offset_x += dx / self.zoom_level
                self.offset_y += dy / self.zoom_level
                self.last_mouse_pos = (x, y)
                
        elif event == cv2.EVENT_MOUSEWHEEL:
            if flags > 0:
                self.zoom_level *= 1.1
            else:
                self.zoom_level /= 1.1
            self.zoom_level = max(0.1, min(10.0, self.zoom_level))

    def draw_ui(self, img):
        h, w = img.shape[:2]
        
        # 顶部信息栏背景
        cv2.rectangle(img, (0, 0), (w, 60), self.bg_color, -1)
        
        # 帧信息
        info = f"Frame: {self.current_idx + 1}/{self.total_frames}"
        cv2.putText(img, info, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.text_color, 2)
        
        # 神经元 ID
        neuron_info = f"Neuron ID: {self.current_neuron_id}"
        color = self._get_color(self.current_neuron_id)
        cv2.putText(img, neuron_info, (250, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # 缩放信息
        zoom_info = f"Zoom: {self.zoom_level:.1f}x"
        cv2.putText(img, zoom_info, (500, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.text_color, 2)

        # 输入模式提示
        if self.input_mode:
            input_box = f"Goto: {self.input_buffer}_"
            cv2.rectangle(img, (w-200, 10), (w-10, 50), (100, 100, 100), -1)
            cv2.putText(img, input_box, (w-190, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
        # 底部操作提示
        help_text = "L-Click:Set Point | R-Click:Del Point | 1-9:Switch ID | Space:Finish | G:Goto"
        cv2.putText(img, help_text, (20, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    def run(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.on_mouse)
        
        while True:
            # 获取当前帧并应用变换
            frame = self.frames[self.current_idx].copy()
            h, w = frame.shape[:2]
            
            # 创建显示画布
            display = np.zeros_like(frame)
            if len(display.shape) == 2:
                display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)
            
            # 计算变换矩阵
            center_x, center_y = w / 2, h / 2
            M = np.float32([
                [self.zoom_level, 0, (1-self.zoom_level)*center_x + self.offset_x*self.zoom_level],
                [0, self.zoom_level, (1-self.zoom_level)*center_y + self.offset_y*self.zoom_level]
            ])
            
            # 应用仿射变换显示图像
            # 注意：如果是灰度图，先转BGR以便绘制彩色标记
            if len(frame.shape) == 2:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            else:
                frame_bgr = frame
                
            display = cv2.warpAffine(frame_bgr, M, (w, h))
            
            # 绘制所有神经元的点（需要变换坐标）
            for nid, frames_pts in self.points.items():
                if self.current_idx in frames_pts:
                    pt = frames_pts[self.current_idx]
                    scr_pt = self._transform_point(pt[0], pt[1])
                    color = self._get_color(nid)
                    
                    # 当前选中的神经元画大一点
                    radius = 8 if nid == self.current_neuron_id else 5
                    thickness = -1
                    
                    cv2.circle(display, scr_pt, radius, color, thickness)
                    cv2.putText(display, str(nid), (scr_pt[0]+10, scr_pt[1]-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # 绘制UI层
            self.draw_ui(display)
            
            cv2.imshow(self.window_name, display)
            
            key = cv2.waitKey(20) & 0xFF
            
            if self.input_mode:
                if key == 13: # Enter
                    if self.input_buffer.isdigit():
                        idx = int(self.input_buffer) - 1
                        if 0 <= idx < self.total_frames:
                            self.current_idx = idx
                    self.input_mode = False
                    self.input_buffer = ""
                elif key == 27: # Esc
                    self.input_mode = False
                    self.input_buffer = ""
                elif key == 8: # Backspace
                    self.input_buffer = self.input_buffer[:-1]
                elif 48 <= key <= 57: # Numbers
                    self.input_buffer += chr(key)
                continue
            
            if key in [ord("q"), 27]:
                return None
            elif key in [13, 32]: # Enter or Space
                # 必须至少选了一个点
                if any(self.points.values()):
                    return self.points
            elif key in [ord("a"), 81]: # Left arrow
                self.current_idx = max(0, self.current_idx - 1)
            elif key in [ord("d"), 83]: # Right arrow
                self.current_idx = min(self.total_frames - 1, self.current_idx + 1)
            elif key in [ord("g"), ord("G")]:
                self.input_mode = True
            elif 49 <= key <= 57: # 1-9 Switch Neuron ID
                self.current_neuron_id = key - 48

def select_neuron_point(frame_list, window_name):
    ui = InteractiveUI(frame_list, window_name)
    return ui.run()
