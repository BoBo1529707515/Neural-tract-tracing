"""
配置文件
"""

class Config:
    MAX_NEURONS = 50
    DISPLAY_WIDTH = 1600
    DISPLAY_HEIGHT = 1000
    PANEL_HEIGHT = 130

    MIN_ZOOM, MAX_ZOOM, ZOOM_STEP = 0.2, 10.0, 1.4
    DEFAULT_MARK_RADIUS = 1

    # 图像处理参数 (针对灰白神经元)
    CLAHE_CLIP_LIMIT = 3.0
    MEDIAN_KERNEL = 5
    ADAPTIVE_BLOCK_SIZE = 15
    MIN_OBJECT_SIZE = 50
    
    # 灰度阈值 (用于简单判断像素点是否属于神经元)
    NEURON_BRIGHTNESS_THRESHOLD = 80 
    
    # 伪影去除
    REMOVE_HORIZONTAL_LINES = True
    LINE_THICKNESS_THRESHOLD = 3 # 超过这个宽度的横线认为是背景

    # 追踪参数
    MAX_GAP = 15
    MAX_LINK_DISTANCE = 50 # 最大连接距离（像素），超过此距离判定为断连
    MAX_GROWTH_PER_FRAME = 20 # 单帧最大生长长度（像素），防止瞬间“瞬移”
    DIRECTION_WEIGHT = 20 # 提高方向权重，严惩锯齿
    AREA_THRESHOLD = 5 
    Y_TOLERANCE = 35

    # 界面颜色
    COLOR_BG = (30, 30, 30)
    COLOR_PANEL = (25, 25, 25)
    COLOR_BTN = (65, 65, 65)
    COLOR_BTN_HOVER = (85, 85, 85)
    COLOR_BTN_ACTIVE = (70, 130, 70)
