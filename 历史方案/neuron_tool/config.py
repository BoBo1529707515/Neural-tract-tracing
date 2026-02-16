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
    # UI颜色 (BGR格式) - 现代深色主题
    COLOR_BACKGROUND = (32, 32, 36)      # 主背景：深碳色
    COLOR_PANEL = (40, 44, 50)           # 面板背景：稍亮的蓝灰色
    
    # 按钮颜色
    COLOR_BUTTON = (55, 60, 68)          # 默认按钮：中性灰
    COLOR_BUTTON_HOVER = (75, 80, 90)    # 悬停：提亮
    COLOR_BUTTON_ACTIVE = (200, 140, 60) # 激活/选中：醒目的蓝色 (BGR: 60, 140, 200)
    
    # 功能按钮特化颜色
    COLOR_BUTTON_DANGER = (80, 70, 180)  # 危险操作：柔和红 (BGR)
    COLOR_BUTTON_SUCCESS = (80, 160, 80) # 成功/运行：柔和绿 (BGR)
    
    # 文本与边框
    COLOR_TEXT = (230, 230, 230)         # 主要文字：柔和白
    COLOR_TEXT_INFO = (255, 180, 80)     # 信息文字：淡蓝/青色 (BGR)
    COLOR_TEXT_DIM = (160, 160, 160)     # 次要文字：灰色
    COLOR_BORDER = (70, 75, 85)          # 边框：低对比度灰
    
    # 输入框
    COLOR_INPUT_BG = (25, 28, 32)        # 输入框背景：深黑
    COLOR_INPUT_ACTIVE = (45, 50, 60)    # 输入框激活背景
    COLOR_INPUT_BORDER_ACTIVE = (200, 140, 60) # 输入框激活边框 (同强调色)

    # 标记颜色
    COLOR_TRAJECTORY_END = (50, 50, 255)    # 轨迹终点（鲜红）
    COLOR_MARK_HIGHLIGHT = (255, 255, 255)  # 标记高亮（纯白）


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
