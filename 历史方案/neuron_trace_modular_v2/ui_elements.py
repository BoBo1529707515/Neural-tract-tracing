"""
UI元素模块
"""
import cv2
from .config import Config

class Button:
    def __init__(self, x, y, w, h, text, color=None):
        self.x, self.y, self.w, self.h = x, y, w, h
        self.text = text
        self.color = color or Config.COLOR_BTN
        self.hover = self.active = False

    def contains(self, px, py):
        return self.x <= px < self.x + self.w and self.y <= py < self.y + self.h

    def draw(self, img):
        bg = Config.COLOR_BTN_ACTIVE if self.active else (Config.COLOR_BTN_HOVER if self.hover else self.color)
        cv2.rectangle(img, (self.x, self.y), (self.x+self.w, self.y+self.h), bg, -1)
        cv2.rectangle(img, (self.x, self.y), (self.x+self.w, self.y+self.h), (100,100,100), 1)
        (tw, th), _ = cv2.getTextSize(self.text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.putText(img, self.text, (self.x+(self.w-tw)//2, self.y+(self.h+th)//2),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)


class InputBox:
    def __init__(self, x, y, w, h, label):
        self.x, self.y, self.w, self.h = x, y, w, h
        self.label = label
        self.text = ""
        self.active = False

    def contains(self, px, py):
        return self.x <= px < self.x + self.w and self.y <= py < self.y + self.h

    def draw(self, img, highlight=None):
        cv2.putText(img, self.label, (self.x, self.y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180,180,180), 1)
        bg = (55, 55, 75) if self.active else (45, 45, 45)
        cv2.rectangle(img, (self.x, self.y), (self.x+self.w, self.y+self.h), bg, -1)
        border = (150, 180, 255) if self.active else (100, 100, 100)
        cv2.rectangle(img, (self.x, self.y), (self.x+self.w, self.y+self.h), border, 1)
        tx = self.x + 6
        if highlight:
            cv2.rectangle(img, (self.x+4, self.y+4), (self.x+20, self.y+self.h-4), highlight, -1)
            tx = self.x + 26
        cv2.putText(img, self.text, (tx, self.y+self.h-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220,220,220), 1)

    def handle_key(self, key):
        if key in [13, 10]: return 'confirm'
        if key == 27: return 'cancel'
        if key in [8, 127]: self.text = self.text[:-1]
        elif ord('0') <= key <= ord('9'): self.text += chr(key)
        return None
