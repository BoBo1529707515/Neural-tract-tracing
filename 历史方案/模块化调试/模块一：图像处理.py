"""
神经元追踪 - 模块一：图像处理（简化版）
"""

import cv2
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from skimage.morphology import skeletonize, remove_small_objects
from skimage.filters import frangi


class ImageProcessor:
    """图像处理器"""

    def __init__(self):
        # 可调参数
        self.green_threshold = 25
        self.min_object_size = 30
        self.use_frangi = True

        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    def process(self, frame):
        """完整处理流水线"""
        # 1. 绿色掩码
        b, g, r = cv2.split(frame)
        b, g, r = b.astype(np.int16), g.astype(np.int16), r.astype(np.int16)
        mask = (g > 40) & ((g - r) > self.green_threshold) & ((g - b) > self.green_threshold)
        green_mask = mask.astype(np.uint8) * 255

        # 2. 增强
        enhanced = self.clahe.apply(frame[:, :, 1])

        # 3. Frangi滤波（可选）
        if self.use_frangi:
            img_norm = enhanced.astype(np.float64) / 255.0
            try:
                resp = frangi(img_norm, sigmas=[1, 2, 3], black_ridges=False)
            except:
                resp = frangi(img_norm, scale_range=(1, 3), scale_step=1, black_ridges=False)
            filtered = (resp / (resp.max() + 1e-8) * 255).astype(np.uint8)
        else:
            filtered = enhanced

        # 4. 二值化
        combined = cv2.bitwise_and(filtered, green_mask)
        _, binary = cv2.threshold(combined, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 5. 骨架化
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, self.kernel, iterations=2)
        cleaned = remove_small_objects(closed > 0, min_size=self.min_object_size)
        skeleton = skeletonize(cleaned)

        return skeleton, green_mask, binary


def main(video_path):
    """主程序"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开: {video_path}")
        return

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"视频: {w}x{h}, {total}帧")

    proc = ImageProcessor()
    idx = 0
    view_mode = 0  # 0=叠加, 1=掩码, 2=二值, 3=骨架

    print("\n操作: A/D=帧 F=Frangi V=切换视图 1-5=阈值 Q=退出\n")

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break

        # 处理
        skeleton, green_mask, binary = proc.process(frame)

        # 显示
        if view_mode == 0:
            vis = frame.copy()
            vis[skeleton] = [0, 255, 255]
            title = "Overlay"
        elif view_mode == 1:
            vis = cv2.cvtColor(green_mask, cv2.COLOR_GRAY2BGR)
            title = "Green Mask"
        elif view_mode == 2:
            vis = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
            title = "Binary"
        else:
            vis = np.zeros((h, w, 3), np.uint8)
            vis[skeleton] = [255, 255, 255]
            title = "Skeleton"

        # 信息
        info = f"[{idx+1}/{total}] {title} | Frangi:{'ON' if proc.use_frangi else 'OFF'} Thresh:{proc.green_threshold} Skel:{np.sum(skeleton)}"
        cv2.putText(vis, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(vis, "A/D:frame F:frangi V:view 1-5:thresh Q:quit", (10, h-15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow('Test', vis)

        # 按键 - 关键修复：用0等待
        key = cv2.waitKey(100) & 0xFF

        if key == ord('q') or key == 27:
            break
        elif key == ord('a') or key == 81:  # 左箭头
            idx = max(0, idx - 1)
        elif key == ord('d') or key == 83:  # 右箭头
            idx = min(total - 1, idx + 1)
        elif key == ord('f'):
            proc.use_frangi = not proc.use_frangi
            print(f"Frangi: {'ON' if proc.use_frangi else 'OFF'}")
        elif key == ord('v'):
            view_mode = (view_mode + 1) % 4
            print(f"View: {['Overlay', 'GreenMask', 'Binary', 'Skeleton'][view_mode]}")
        elif key == ord('1'):
            proc.green_threshold = 10; print("阈值: 10")
        elif key == ord('2'):
            proc.green_threshold = 15; print("阈值: 15")
        elif key == ord('3'):
            proc.green_threshold = 20; print("阈值: 20")
        elif key == ord('4'):
            proc.green_threshold = 25; print("阈值: 25")
        elif key == ord('5'):
            proc.green_threshold = 30; print("阈值: 30")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    VIDEO_PATH = r"/neuron_growth_50.mp4"
    main(VIDEO_PATH)
