"""
UI组件模块
包含按钮、输入框等界面元素
"""

import cv2
import numpy as np
from .config import Config


class Button:
    """
    按钮组件

    功能:
        - 绘制按钮
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
        self.color = color
        self.text_color = text_color
        self.hover = False      # 鼠标悬停状态
        self.active = False     # 激活状态（如当前选中）
        self.visible = True     # 是否可见

    def contains(self, px, py):
        """
        检测点(px, py)是否在按钮范围内

        参数:
            px, py: 点击坐标

        返回:
            bool: 是否在范围内
        """
        if not self.visible:
            return False
        return self.x <= px < self.x + self.w and self.y <= py < self.y + self.h

    def draw(self, img):
        """
        在图像上绘制按钮

        参数:
            img: 目标图像(numpy数组)
        """
        if not self.visible:
            return

        # 根据状态选择背景颜色
        if self.active:
            bg = Config.COLOR_BUTTON_ACTIVE
        elif self.hover:
            bg = Config.COLOR_BUTTON_HOVER
        else:
            bg = self.color

        # 绘制背景
        cv2.rectangle(img, (self.x, self.y),
                     (self.x + self.w, self.y + self.h), bg, -1)
        # 绘制边框
        cv2.rectangle(img, (self.x, self.y),
                     (self.x + self.w, self.y + self.h), Config.COLOR_BORDER, 1)

        # 绘制文字（居中）
        font_scale = 0.5
        thickness = 2
        (tw, th), _ = cv2.getTextSize(self.text, cv2.FONT_HERSHEY_SIMPLEX,
                                       font_scale, thickness)
        tx = self.x + (self.w - tw) // 2
        ty = self.y + (self.h + th) // 2
        cv2.putText(img, self.text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                   font_scale, self.text_color, thickness, cv2.LINE_AA)


class InputBox:
    """
    输入框组件

    功能:
        - 显示标签和输入内容
        - 处理输入状态
        - 光标闪烁效果
        - 可选颜色高亮
    """

    def __init__(self, x, y, w, h, label, default=""):
        """
        初始化输入框

        参数:
            x, y: 左上角坐标
            w, h: 宽度和高度
            label: 标签文字
            default: 默认值
        """
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.label = label
        self.text = default         # 当前输入内容
        self.active = False         # 是否处于输入状态
        self.cursor_timer = 0       # 光标闪烁计时器

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
        # 绘制标签
        cv2.putText(img, self.label, (self.x, self.y - 6),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

        # 背景颜色（激活时不同）
        bg = Config.COLOR_INPUT_ACTIVE if self.active else (50, 50, 50)
        cv2.rectangle(img, (self.x, self.y),
                     (self.x + self.w, self.y + self.h), bg, -1)

        # 边框颜色
        border = Config.COLOR_INPUT_BORDER_ACTIVE if self.active else Config.COLOR_BORDER
        cv2.rectangle(img, (self.x, self.y),
                     (self.x + self.w, self.y + self.h), border, 2)

        # 颜色高亮块（如神经元颜色）
        text_x = self.x + 8
        if highlight_color:
            cv2.rectangle(img, (self.x + 4, self.y + 4),
                         (self.x + 22, self.y + self.h - 4), highlight_color, -1)
            text_x = self.x + 28

        # 显示文字（激活时显示光标）
        display_text = self.text
        if self.active:
            self.cursor_timer = (self.cursor_timer + 1) % 30
            if self.cursor_timer < 20:
                display_text += "_"

        text_color = (255, 255, 100) if self.active else (220, 220, 220)
        cv2.putText(img, display_text, (text_x, self.y + self.h - 8),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_color, 2, cv2.LINE_AA)

    def handle_key(self, key):
        """
        处理键盘输入

        参数:
            key: 按键码

        返回:
            str: 'confirm'表示确认, 'cancel'表示取消, None表示继续输入
        """
        if key == 13 or key == 10:  # Enter
            return 'confirm'
        elif key == 27:  # Esc
            return 'cancel'
        elif key == 8 or key == 127:  # Backspace
            self.text = self.text[:-1]
        elif ord('0') <= key <= ord('9'):
            self.text += chr(key)
        return None
