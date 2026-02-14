"""
图像处理模块
包含骨架化、绿色检测、预处理等功能
"""

import cv2
import numpy as np
from skimage.morphology import skeletonize, remove_small_objects
from .config import Config


class ImageProcessor:
    """
    图像处理类

    功能:
        - 绿色像素检测
        - 绿色掩码提取
        - 图像预处理（增强、二值化、骨架化）
    """

    def __init__(self):
        """初始化图像处理器"""
        # CLAHE对比度增强器
        self.clahe = cv2.createCLAHE(
            clipLimit=Config.CLAHE_CLIP_LIMIT,
            tileGridSize=Config.CLAHE_GRID_SIZE
        )

        # 形态学操作核
        self.morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (Config.MORPH_KERNEL_SIZE, Config.MORPH_KERNEL_SIZE)
        )

    def is_green_pixel(self, frame, y, x):
        """
        检测指定位置是否为绿色像素

        参数:
            frame: BGR图像
            y, x: 像素坐标

        返回:
            bool: 是否为绿色像素
        """
        h, w = frame.shape[:2]

        # 边界检查
        if not (0 <= y < h and 0 <= x < w):
            return False

        b, g, r = int(frame[y, x, 0]), int(frame[y, x, 1]), int(frame[y, x, 2])

        # 亮度检查
        if g < 50:
            return False

        # 绿色通道优势检查
        if g - r < Config.GREEN_THRESHOLD:
            return False
        if g - b < Config.GREEN_THRESHOLD:
            return False

        # 排除白色/灰色
        if r > 150 and b > 150 and g > 150:
            if abs(r - g) < 50 and abs(b - g) < 50:
                return False

        return True

    def extract_green_mask(self, frame):
        """
        提取绿色区域掩码

        参数:
            frame: BGR图像

        返回:
            numpy.ndarray: 二值掩码图像
        """
        b, g, r = cv2.split(frame)
        b, g, r = b.astype(np.int16), g.astype(np.int16), r.astype(np.int16)

        # 绿色判断条件
        bright_enough = g > 50
        green_over_red = (g - r) > Config.GREEN_THRESHOLD
        green_over_blue = (g - b) > Config.GREEN_THRESHOLD

        # 排除白色/灰色
        not_white = ~(
                (r > 150) & (b > 150) & (g > 150) &
                (np.abs(r - g) < 50) & (np.abs(b - g) < 50)
        )

        # 合并条件
        mask = bright_enough & green_over_red & green_over_blue & not_white
        return mask.astype(np.uint8) * 255

    def preprocess(self, frame):
        """
        预处理帧图像，生成骨架

        处理流程:
            1. 提取绿色掩码
            2. CLAHE对比度增强
            3. 中值滤波去噪
            4. 自适应阈值二值化
            5. 形态学闭操作
            6. 去除小对象
            7. 骨架化

        参数:
            frame: BGR图像

        返回:
            numpy.ndarray: 骨架图像（布尔数组）
        """
        # 1. 提取绿色掩码
        green_mask = self.extract_green_mask(frame)

        # 2. 提取绿色通道并增强
        green_channel = frame[:, :, 1]
        enhanced = self.clahe.apply(green_channel)

        # 3. 中值滤波去噪
        denoised = cv2.medianBlur(enhanced, Config.MEDIAN_KERNEL)

        # 4. 自适应阈值二值化
        binary = cv2.adaptiveThreshold(
            denoised, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            Config.ADAPTIVE_BLOCK_SIZE,
            Config.ADAPTIVE_C
        )

        # 5. 与绿色掩码合并
        binary = cv2.bitwise_and(binary, green_mask)

        # 6. 形态学闭操作（填充小孔）
        closed = cv2.morphologyEx(
            binary, cv2.MORPH_CLOSE,
            self.morph_kernel,
            iterations=Config.MORPH_ITERATIONS
        )

        # 7. 去除小对象
        cleaned = remove_small_objects(
            closed > 0,
            min_size=Config.MIN_OBJECT_SIZE,
            connectivity=1
        )

        # 8. 骨架化
        skeleton = skeletonize(cleaned)

        return skeleton

    def find_nearest_skeleton_point(self, skeleton, frame, point, radius=30):
        """
        在指定点附近找最近的骨架点（且为绿色）

        参数:
            skeleton: 骨架图像
            frame: 原始BGR图像
            point: 目标点 (x, y)
            radius: 搜索半径

        返回:
            tuple: 最近的骨架点坐标 (y, x)，未找到返回None
        """
        h, w = skeleton.shape
        px, py = int(point[0]), int(point[1])

        best_point = None
        best_dist = float('inf')

        # 在半径内搜索
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                ny, nx = py + dy, px + dx

                if 0 <= ny < h and 0 <= nx < w:
                    if skeleton[ny, nx]:  # 是骨架点
                        if self.is_green_pixel(frame, ny, nx):  # 是绿色
                            dist = dy * dy + dx * dx
                            if dist < best_dist:
                                best_dist = dist
                                best_point = (ny, nx)

        return best_point
