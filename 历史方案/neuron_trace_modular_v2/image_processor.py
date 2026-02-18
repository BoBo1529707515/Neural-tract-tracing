"""
图像处理模块
"""
import cv2
import numpy as np
from skimage.morphology import skeletonize, remove_small_objects
from .config import Config

class ImageProc:
    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=Config.CLAHE_CLIP_LIMIT, tileGridSize=(8,8))
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

    def _preprocess(self, f):
        """预处理：转灰度 -> CLAHE"""
        if len(f.shape) == 3:
            gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        else:
            gray = f
        enh = self.clahe.apply(gray)
        return enh

    def is_neuron(self, f, y, x):
        """判断某点是否属于神经元（亮度阈值）"""
        if not (0 <= y < f.shape[0] and 0 <= x < f.shape[1]): return False
        
        # 获取灰度值
        if len(f.shape) == 3:
            val = int(f[y, x].mean()) # 取RGB平均或者直接用灰度图
        else:
            val = f[y, x]
            
        # 简单亮度阈值判断 (可结合局部自适应)
        return val > Config.NEURON_BRIGHTNESS_THRESHOLD

    def _remove_lines(self, bw):
        """去除水平背景伪影（微流控通道边缘等）"""
        if not Config.REMOVE_HORIZONTAL_LINES:
            return bw
            
        # 1. 识别长水平线
        h, w = bw.shape
        line_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 20, 1))
        lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, line_kernel)
        
        # 2. 从二值图中减去这些线
        # 注意：只减去那些比较直且粗的结构，保留细小的神经元
        # 如果神经元也是水平生长的，这步要小心。这里假设伪影比神经元更“直”更“长”
        clean = cv2.bitwise_and(bw, cv2.bitwise_not(lines))
        return clean

    def skeleton(self, f):
        """生成骨架图"""
        # 1. 预处理
        enh = self._preprocess(f)
        
        # 2. 中值滤波去噪
        den = cv2.medianBlur(enh, Config.MEDIAN_KERNEL)
        
        # 3. 自适应二值化 (针对灰白前景)
        bs = Config.ADAPTIVE_BLOCK_SIZE
        if bs % 2 == 0: bs += 1
        
        bw = cv2.adaptiveThreshold(
            den, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 
            bs, -3 
        )
        
        # 4. 去除背景平行线伪影 (NEW)
        bw_no_lines = self._remove_lines(bw)
        
        # 5. 形态学闭运算连接断点
        cl = cv2.morphologyEx(bw_no_lines, cv2.MORPH_CLOSE, self.kernel, iterations=2)
        
        # 6. 去除小噪点 (更新为使用 Config.AREA_THRESHOLD)
        clean = remove_small_objects(cl > 0, min_size=Config.AREA_THRESHOLD)
        
        # 7. 骨架化
        return skeletonize(clean)

    def nearest_skel(self, skel, f, pt, r=30):
        """在骨架上寻找最近的神经元点"""
        h, w = skel.shape
        px, py = int(pt[0]), int(pt[1])
        best, bd = None, 1e9
        
        # 获取预处理后的图像用于亮度判断
        enh = self._preprocess(f)
        
        for dy in range(-r, r+1):
            for dx in range(-r, r+1):
                ny, nx = py+dy, px+dx
                # 检查边界 + 骨架点 + 亮度足够
                if 0 <= ny < h and 0 <= nx < w and skel[ny, nx]:
                    # 双重验证：必须在骨架上，且原图亮度不能太暗
                    if enh[ny, nx] > Config.NEURON_BRIGHTNESS_THRESHOLD:
                        d = dy*dy + dx*dx
                        if d < bd: bd, best = d, (ny, nx)
        return best
