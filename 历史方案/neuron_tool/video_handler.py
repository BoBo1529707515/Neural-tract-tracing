"""
视频处理模块
负责视频的读取、帧获取、视频写入等
"""

import cv2
import os


class VideoHandler:
    """
    视频处理类

    功能:
        - 加载视频文件
        - 读取指定帧
        - 获取视频信息
        - 写入输出视频
    """

    def __init__(self):
        """初始化视频处理器"""
        self.cap = None  # 视频捕获对象
        self.writer = None  # 视频写入对象
        self.video_path = ""  # 视频路径
        self.total_frames = 0  # 总帧数
        self.frame_width = 0  # 帧宽度
        self.frame_height = 0  # 帧高度
        self.fps = 30  # 帧率
        self.current_frame = None  # 当前帧
        self.current_frame_idx = 0  # 当前帧索引

    def load(self, video_path):
        """
        加载视频文件

        参数:
            video_path: 视频文件路径

        返回:
            dict: 视频信息字典

        异常:
            ValueError: 无法打开视频时抛出
        """
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")

        # 获取视频属性
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30

        print(f"✓ 加载视频: {self.frame_width}x{self.frame_height}, "
              f"{self.fps:.1f}fps, {self.total_frames}帧")

        return self.get_info()

    def get_info(self):
        """
        获取视频信息字典

        返回:
            dict: 包含path, width, height, fps, total_frames
        """
        return {
            "path": self.video_path,
            "width": self.frame_width,
            "height": self.frame_height,
            "fps": self.fps,
            "total_frames": self.total_frames
        }

    def read_frame(self, frame_idx):
        """
        读取指定帧

        参数:
            frame_idx: 帧索引（0开始）

        返回:
            numpy.ndarray: 帧图像，失败返回None
        """
        if self.cap is None:
            return None

        # 限制范围
        frame_idx = max(0, min(frame_idx, self.total_frames - 1))

        # 跳转到指定帧
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()

        if ret:
            self.current_frame_idx = frame_idx
            self.current_frame = frame
            return frame
        return None

    def start_writer(self, output_path):
        """
        开始写入视频

        参数:
            output_path: 输出视频路径
        """
        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(
            output_path, fourcc, int(self.fps),
            (self.frame_width, self.frame_height)
        )
        print(f"✓ 开始写入视频: {output_path}")

    def write_frame(self, frame):
        """
        写入一帧

        参数:
            frame: 要写入的帧
        """
        if self.writer is not None:
            self.writer.write(frame)

    def stop_writer(self):
        """停止写入并释放资源"""
        if self.writer is not None:
            self.writer.release()
            self.writer = None
            print("✓ 视频写入完成")

    def release(self):
        """释放所有资源"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.stop_writer()
