"""
配置参数模块
存放所有可调节的参数，方便统一修改
"""


class Config:
    """全局配置类"""

    # ==================== 神经元参数 ====================
    MAX_NEURONS = 50  # 最大神经元数量

    # ==================== 显示参数 ====================
    DISPLAY_WIDTH = 1600  # 显示窗口宽度
    DISPLAY_HEIGHT = 1000  # 显示窗口高度
    PANEL_HEIGHT = 130  # 控制面板高度

    # ==================== 缩放参数 ====================
    MIN_ZOOM = 0.2  # 最小缩放倍数
    MAX_ZOOM = 10.0  # 最大缩放倍数
    ZOOM_STEP = 1.4  # 缩放步长

    # ==================== 标记参数 ====================
    MIN_MARK_RADIUS = 1  # 最小标记半径
    MAX_MARK_RADIUS = 20  # 最大标记半径
    DEFAULT_MARK_RADIUS = 1  # 默认标记半径

    # ==================== 图像处理参数 ====================
    GREEN_THRESHOLD = 40  # 绿色检测阈值
    CLAHE_CLIP_LIMIT = 3.0  # CLAHE对比度限制
    CLAHE_GRID_SIZE = (8, 8)  # CLAHE网格大小
    MEDIAN_KERNEL = 5  # 中值滤波核大小
    ADAPTIVE_BLOCK_SIZE = 15  # 自适应阈值块大小
    ADAPTIVE_C = -3  # 自适应阈值常数
    MORPH_KERNEL_SIZE = 3  # 形态学操作核大小
    MORPH_ITERATIONS = 2  # 形态学操作迭代次数
    MIN_OBJECT_SIZE = 50  # 最小对象大小（去噪）

    # ==================== 追踪参数 ====================
    MAX_GAP = 15  # 最大跨越间隙
    Y_TOLERANCE = 35  # Y方向容差
    DIRECTION_WEIGHT = 15  # 方向一致性权重（交叉点处理）
    DIRECTION_HISTORY = 8  # 计算方向时使用的历史点数
    GROWTH_SEARCH_RADIUS = 3  # 生长搜索半径
    GROWTH_MAX_DEPTH = 30  # 生长搜索最大深度

    # ==================== 颜色定义 ====================
    # UI颜色 (BGR格式)
    COLOR_BACKGROUND = (30, 30, 30)
    COLOR_PANEL = (25, 25, 25)
    COLOR_BUTTON = (70, 70, 70)
    COLOR_BUTTON_HOVER = (100, 100, 100)
    COLOR_BUTTON_ACTIVE = (70, 140, 70)
    COLOR_BUTTON_DANGER = (60, 60, 120)
    COLOR_BUTTON_SUCCESS = (50, 110, 50)
    COLOR_TEXT = (255, 255, 255)
    COLOR_TEXT_INFO = (0, 220, 220)
    COLOR_TEXT_DIM = (180, 180, 180)
    COLOR_BORDER = (140, 140, 140)
    COLOR_INPUT_ACTIVE = (60, 60, 90)
    COLOR_INPUT_BORDER_ACTIVE = (100, 150, 255)

    # 标记颜色
    COLOR_TRAJECTORY_END = (0, 0, 255)  # 轨迹终点（红色）
    COLOR_MARK_HIGHLIGHT = (255, 255, 255)  # 标记高亮（白色）


def generate_neuron_colors(n):
    """
    生成n种不同的神经元颜色（基于HSV色环）

    参数:
        n: 颜色数量

    返回:
        颜色列表，每个颜色为(B, G, R)元组
    """
    import cv2
    import numpy as np

    colors = []
    for i in range(n):
        # 在HSV色环上均匀分布
        hue = int(180 * i / n)  # OpenCV的H范围是0-180
        hsv = np.uint8([[[hue, 255, 255]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
        colors.append((int(bgr[0]), int(bgr[1]), int(bgr[2])))
    return colors
