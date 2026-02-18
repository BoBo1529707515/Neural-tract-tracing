"""
视频处理模块
"""
import cv2
import os

class VideoHandler:
    def __init__(self):
        self.cap = self.writer = None
        self.path = ""
        self.total = self.w = self.h = 0
        self.fps = 30
        self.frame = None
        self.idx = 0

    def load(self, path):
        self.path = path
        self.cap = cv2.VideoCapture(path)
        self.total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        print(f"✓ 视频加载: {self.w}x{self.h}, {self.fps:.1f}fps, {self.total}帧")

    def read(self, idx):
        idx = max(0, min(idx, self.total - 1))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, self.frame = self.cap.read()
        if ret: self.idx = idx
        return self.frame

    def start_write(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), int(self.fps), (self.w, self.h))

    def write(self, f):
        if self.writer: self.writer.write(f)

    def stop_write(self):
        if self.writer: self.writer.release(); self.writer = None

    def release(self):
        if self.cap: self.cap.release()
        self.stop_write()
