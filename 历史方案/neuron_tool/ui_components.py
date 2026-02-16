"""
UI组件模块
包含按钮、输入框等界面元素（美化版）
"""

import cv2
import numpy as np
from .config import Config


def draw_rounded_rect(img, pt1, pt2, color, thickness=1, radius=5):
    """
    绘制圆角矩形
    
    参数:
        img: 目标图像
        pt1: 左上角坐标 (x1, y1)
        pt2: 右下角坐标 (x2, y2)
        color: 颜色
        thickness: 线宽，-1表示填充
        radius: 圆角半径
    """
    x1, y1 = pt1
    x2, y2 = pt2
    
    # 限制半径大小
    w = x2 - x1
    h = y2 - y1
    radius = min(radius, w // 2, h // 2)
    
    # 填充模式
    if thickness == -1:
        # 绘制中间的十字形矩形
        cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)
        
        # 绘制四个角的圆
        cv2.circle(img, (x1 + radius, y1 + radius), radius, color, -1)
        cv2.circle(img, (x2 - radius, y1 + radius), radius, color, -1)
        cv2.circle(img, (x2 - radius, y2 - radius), radius, color, -1)
        cv2.circle(img, (x1 + radius, y2 - radius), radius, color, -1)
    else:
        # 线条模式
        # 直线段
        cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
        cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
        cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
        cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness)
        
        # 弧线段
        cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)
        cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)


class Button:
    """
    按钮组件
    
    功能:
        - 绘制按钮（圆角、阴影、悬停效果）
        - 检测点击
        - hover效果
        - active状态
    """

    def __init__(self, x, y, w, h, text,
                 color=Config.COLOR_BUTTON,
                 text_color=Config.COLOR_TEXT):
        """
        初始化按钮
        
        参数:
            x, y: 左上角坐标
            w, h: 宽度和高度
            text: 按钮文字
            color: 背景颜色
            text_color: 文字颜色
        """
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.text = text
        self.base_color = color  # 基础颜色
        self.text_color = text_color
        self.hover = False      # 鼠标悬停状态
        self.active = False     # 激活状态（如当前选中）
        self.visible = True     # 是否可见
        self.radius = 6         # 圆角半径

    def contains(self, px, py):
        """
        检测点(px, py)是否在按钮范围内
        """
        if not self.visible:
            return False
        return self.x <= px < self.x + self.w and self.y <= py < self.y + self.h

    def draw(self, img):
        """
        在图像上绘制按钮
        """
        if not self.visible:
            return

        x, y, w, h = self.x, self.y, self.w, self.h
        
        # 1. 确定背景颜色
        if self.active:
            bg_color = Config.COLOR_BUTTON_ACTIVE
            # 激活状态文字颜色变白（如果背景很亮）或保持
            text_color = (255, 255, 255)
        else:
            bg_color = self.base_color
            text_color = self.text_color

        # 2. 绘制阴影 (仅在非激活且非hover时明显，或者一直有)
        # 简单的阴影：向右下偏移2像素的黑色半透明
        if not self.active:
            shadow_offset = 2
            shadow_alpha = 0.3
            # 提取阴影区域
            sx1, sy1 = x + shadow_offset, y + shadow_offset
            sx2, sy2 = sx1 + w, sy1 + h
            
            # 确保不越界
            if sx2 < img.shape[1] and sy2 < img.shape[0]:
                shadow_roi = img[sy1:sy2, sx1:sx2]
                shadow_layer = shadow_roi.copy()
                # 绘制黑色圆角矩形
                draw_rounded_rect(shadow_layer, (0, 0), (w, h), (0, 0, 0), -1, self.radius)
                # 混合
                cv2.addWeighted(shadow_layer, shadow_alpha, shadow_roi, 1 - shadow_alpha, 0, shadow_roi)
                img[sy1:sy2, sx1:sx2] = shadow_roi

        # 3. 绘制按钮主体
        draw_rounded_rect(img, (x, y), (x + w, y + h), bg_color, -1, self.radius)
        
        # 4. Hover效果：叠加白色半透明层
        if self.hover and not self.active:
            overlay = img[y:y+h, x:x+w].copy()
            # 绘制白色蒙版
            draw_rounded_rect(overlay, (0, 0), (w, h), (255, 255, 255), -1, self.radius)
            # 混合: 原图 0.9 + 白色 0.1
            cv2.addWeighted(overlay, 0.1, img[y:y+h, x:x+w], 0.9, 0, img[y:y+h, x:x+w])

        # 5. 绘制边框
        border_color = Config.COLOR_BORDER
        if self.active:
            border_color = (min(255, bg_color[0]+30), min(255, bg_color[1]+30), min(255, bg_color[2]+30))
        
        draw_rounded_rect(img, (x, y), (x + w, y + h), border_color, 1, self.radius)

        # 6. 绘制文字（居中）
        font_scale = 0.45
        thickness = 1
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        (tw, th), baseline = cv2.getTextSize(self.text, font, font_scale, thickness)
        tx = x + (w - tw) // 2
        ty = y + (h + th) // 2
        
        # 文字阴影（可选，增加立体感）
        if not self.active:
            cv2.putText(img, self.text, (tx+1, ty+1), font, font_scale, (0,0,0), thickness, cv2.LINE_AA)
            
        cv2.putText(img, self.text, (tx, ty), font, font_scale, text_color, thickness, cv2.LINE_AA)


class InputBox:
    """
    输入框组件（美化版）
    
    功能:
        - 显示标签和输入内容
        - 处理输入状态
        - 光标闪烁效果
        - 现代风格外观
    """

    def __init__(self, x, y, w, h, label, default=""):
        """
        初始化输入框
        """
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.label = label
        self.text = default         # 当前输入内容
        self.active = False         # 是否处于输入状态
        self.cursor_timer = 0       # 光标闪烁计时器
        self.radius = 4             # 圆角半径

    def contains(self, px, py):
        """检测点击是否在输入框范围内"""
        return self.x <= px < self.x + self.w and self.y <= py < self.y + self.h

    def draw(self, img, highlight_color=None):
        """
        绘制输入框
        
        参数:
            img: 目标图像
            highlight_color: 可选的高亮颜色块（用于显示当前神经元颜色）
        """
        # 1. 绘制标签
        cv2.putText(img, self.label, (self.x, self.y - 6),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, Config.COLOR_TEXT_DIM, 1, cv2.LINE_AA)

        # 2. 背景颜色
        if self.active:
            bg = Config.COLOR_INPUT_ACTIVE
            border = Config.COLOR_INPUT_BORDER_ACTIVE
            thickness = 2
        else:
            bg = Config.COLOR_INPUT_BG
            border = Config.COLOR_BORDER
            thickness = 1

        # 绘制背景
        draw_rounded_rect(img, (self.x, self.y), (self.x + self.w, self.y + self.h), bg, -1, self.radius)
        
        # 绘制边框
        draw_rounded_rect(img, (self.x, self.y), (self.x + self.w, self.y + self.h), border, thickness, self.radius)

        # 3. 颜色高亮块（如神经元颜色）
        text_x = self.x + 8
        if highlight_color:
            # 左侧画一个小圆点或方块
            cv2.circle(img, (self.x + 12, self.y + self.h // 2), 5, highlight_color, -1)
            text_x = self.x + 24

        # 4. 显示文字
        display_text = self.text
        
        # 光标逻辑
        if self.active:
            self.cursor_timer = (self.cursor_timer + 1) % 30
            if self.cursor_timer < 18: # 稍微调整闪烁频率
                display_text += "|"
        
        text_color = (255, 255, 255) if self.active else Config.COLOR_TEXT
        
        # 垂直居中
        font_scale = 0.5
        (tw, th), _ = cv2.getTextSize("Tg", cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        ty = self.y + (self.h + th) // 2 - 2
        
        cv2.putText(img, display_text, (text_x, ty),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1, cv2.LINE_AA)

    def handle_key(self, key):
        """
        处理键盘输入
        """
        if key == 13 or key == 10:  # Enter
            return 'confirm'
        elif key == 27:  # Esc
            return 'cancel'
        elif key == 8 or key == 127:  # Backspace
            self.text = self.text[:-1]
        elif ord('0') <= key <= ord('9'):
            # 限制长度
            if len(self.text) < 6:
                self.text += chr(key)
        return None
