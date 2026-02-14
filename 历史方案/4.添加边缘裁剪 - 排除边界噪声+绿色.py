import cv2
import numpy as np
from skimage.morphology import skeletonize, remove_small_objects
import os
from tqdm import tqdm

try:
    import torch
    import torch.nn.functional as F

    HAS_CUDA = torch.cuda.is_available()
    if HAS_CUDA:
        DEVICE = torch.device('cuda')
        print(f"✓ GPU加速启用: {torch.cuda.get_device_name(0)}")
    else:
        DEVICE = torch.device('cpu')
except ImportError:
    HAS_CUDA = False
    DEVICE = None


class NeuronTrackerGreenOnly:
    """只追踪绿色神经元，排除白色区域"""

    def __init__(self,
                 clahe_clip_limit=3.0,
                 median_kernel=5,
                 adaptive_block_size=15,
                 adaptive_c=3,
                 morph_kernel_size=3,
                 morph_iterations=2,
                 min_object_size=100,
                 margin_ratio=0.05,
                 green_threshold=30):  # 绿色必须比红蓝高这么多

        self.clahe_clip_limit = clahe_clip_limit
        self.median_kernel = median_kernel
        self.adaptive_block_size = adaptive_block_size
        self.adaptive_c = adaptive_c
        self.morph_kernel_size = morph_kernel_size
        self.morph_iterations = morph_iterations
        self.min_object_size = min_object_size
        self.margin_ratio = margin_ratio
        self.green_threshold = green_threshold

        self.clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=(8, 8))

    def get_roi(self, h, w):
        margin_y = int(h * self.margin_ratio)
        margin_x = int(w * self.margin_ratio)
        return margin_y, h - margin_y, margin_x, w - margin_x

    def extract_green_mask(self, frame):
        """
        提取绿色区域掩码
        绿色神经元: G通道明显高于R和B
        白色区域: R≈G≈B，会被排除
        """
        b, g, r = cv2.split(frame)

        # 条件1: G通道足够亮
        bright_enough = g > 50

        # 条件2: G比R高 (排除白色和红色)
        g_over_r = g.astype(np.int16) - r.astype(np.int16) > self.green_threshold

        # 条件3: G比B高 (排除白色和蓝色)
        g_over_b = g.astype(np.int16) - b.astype(np.int16) > self.green_threshold

        # 综合条件
        green_mask = bright_enough & g_over_r & g_over_b

        return green_mask.astype(np.uint8) * 255

    def preprocess_frame(self, frame):
        """预处理 - 只保留绿色神经元"""
        # 1. 提取绿色掩码
        green_mask = self.extract_green_mask(frame)

        # 2. 提取绿色通道并增强
        green = frame[:, :, 1]
        enhanced = self.clahe.apply(green)

        # 3. 中值滤波
        if HAS_CUDA:
            img_tensor = torch.from_numpy(enhanced).float().to(DEVICE)
            img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)
            pad = self.median_kernel // 2
            img_padded = F.pad(img_tensor, (pad, pad, pad, pad), mode='reflect')
            patches = img_padded.unfold(2, self.median_kernel, 1).unfold(3, self.median_kernel, 1)
            patches = patches.contiguous().view(*patches.shape[:4], -1)
            denoised = patches.median(dim=-1)[0].squeeze().cpu().numpy().astype(np.uint8)
        else:
            denoised = cv2.medianBlur(enhanced, self.median_kernel)

        # 4. 自适应阈值
        binary = cv2.adaptiveThreshold(
            denoised, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            self.adaptive_block_size,
            -self.adaptive_c
        )

        # 5. ★关键★ 与绿色掩码相交，只保留绿色区域
        binary = cv2.bitwise_and(binary, green_mask)

        # 6. 形态学
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (self.morph_kernel_size, self.morph_kernel_size))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel,
                                  iterations=self.morph_iterations)

        # 7. 去小物体
        bool_mask = closed > 0
        cleaned = remove_small_objects(bool_mask, min_size=self.min_object_size, connectivity=1)

        # 8. 骨架化
        skeleton = skeletonize(cleaned)

        return skeleton, green_mask

    def find_rightmost_point_in_roi(self, skeleton, roi):
        y1, y2, x1, x2 = roi
        roi_mask = np.zeros_like(skeleton, dtype=bool)
        roi_mask[y1:y2, x1:x2] = True
        skeleton_roi = skeleton & roi_mask

        ys, xs = np.where(skeleton_roi)
        if len(xs) == 0:
            return None

        max_x_idx = np.argmax(xs)
        return (ys[max_x_idx], xs[max_x_idx])

    def trace_single_path(self, skeleton, start_point, roi):
        if start_point is None:
            return []

        y1, y2, x1, x2 = roi
        h, w = skeleton.shape
        visited = np.zeros_like(skeleton, dtype=bool)
        trajectory = [start_point]
        visited[start_point[0], start_point[1]] = True

        directions = [(-1, -1), (-1, 0), (-1, 1),
                      (0, -1), (0, 1),
                      (1, -1), (1, 0), (1, 1)]

        current = start_point

        while True:
            neighbors = []
            for dy, dx in directions:
                ny, nx = current[0] + dy, current[1] + dx
                if 0 <= ny < h and 0 <= nx < w:
                    if y1 <= ny < y2:
                        if skeleton[ny, nx] and not visited[ny, nx]:
                            neighbors.append((ny, nx, dy, dx))

            if len(neighbors) == 0:
                break

            def score_neighbor(n):
                ny, nx, dy, dx = n
                return -dx * 10 - abs(dy) * 5

            best = max(neighbors, key=score_neighbor)
            next_point = (best[0], best[1])
            visited[next_point[0], next_point[1]] = True
            trajectory.append(next_point)
            current = next_point

        return trajectory

    def smooth_trajectory(self, trajectory, window=5):
        if len(trajectory) < window:
            return np.array(trajectory)

        coords = np.array(trajectory, dtype=float)
        smoothed = coords.copy()

        for i in range(len(coords)):
            start = max(0, i - window // 2)
            end = min(len(coords), i + window // 2 + 1)
            smoothed[i] = coords[start:end].mean(axis=0)

        return smoothed.astype(int)

    def draw_trajectory(self, image, trajectory, roi, thickness=2):
        result = image.copy()
        y1, y2, x1, x2 = roi

        cv2.rectangle(result, (x1, y1), (x2, y2), (80, 80, 80), 1)

        if len(trajectory) < 2:
            cv2.putText(result, "No GREEN neuron found", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            return result

        coords = np.array(trajectory)
        sorted_indices = np.argsort(coords[:, 1])
        sorted_coords = coords[sorted_indices]
        smoothed = self.smooth_trajectory(sorted_coords.tolist(), window=7)

        # 绿色轨迹
        green_color = (0, 255, 0)

        for j in range(len(smoothed) - 1):
            pt1 = (int(smoothed[j, 1]), int(smoothed[j, 0]))
            pt2 = (int(smoothed[j + 1, 1]), int(smoothed[j + 1, 0]))
            cv2.line(result, pt1, pt2, green_color, thickness)

        # 起点
        leftmost = smoothed[0]
        cv2.circle(result, (int(leftmost[1]), int(leftmost[0])), 6, (0, 255, 128), -1)
        cv2.circle(result, (int(leftmost[1]), int(leftmost[0])), 8, (255, 255, 255), 1)

        # 终点
        rightmost = smoothed[-1]
        cv2.circle(result, (int(rightmost[1]), int(rightmost[0])), 6, (0, 0, 255), -1)
        cv2.circle(result, (int(rightmost[1]), int(rightmost[0])), 8, (255, 255, 255), 1)

        x_span = smoothed[-1, 1] - smoothed[0, 1]
        info = f"GREEN Neuron | Length: {len(smoothed)} | X-span: {x_span:.0f}"
        cv2.putText(result, info, (11, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(result, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return result

    def process_video(self, video_path, output_path, show_preview=True):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        roi = self.get_roi(height, width)

        print(f"视频: {width}x{height}, {fps}fps, {total_frames}帧")
        print(f"ROI: 去除{self.margin_ratio * 100:.0f}%边缘")
        print(f"绿色阈值: G必须比R和B高{self.green_threshold}以上")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        can_show = False
        if show_preview:
            try:
                cv2.namedWindow('test', cv2.WINDOW_NORMAL)
                cv2.destroyWindow('test')
                can_show = True
            except:
                print("⚠ GUI不可用")

        pbar = tqdm(total=total_frames, desc="处理中")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            skeleton, _ = self.preprocess_frame(frame)
            rightmost = self.find_rightmost_point_in_roi(skeleton, roi)
            trajectory = self.trace_single_path(skeleton, rightmost, roi)
            vis_frame = self.draw_trajectory(frame, trajectory, roi, thickness=2)

            writer.write(vis_frame)

            if can_show:
                scale = min(1.0, 1280 / width, 720 / height)
                display = cv2.resize(vis_frame, None, fx=scale, fy=scale)
                cv2.imshow('Green Only (Q quit)', display)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            pbar.update(1)

        pbar.close()
        cap.release()
        writer.release()
        if can_show:
            cv2.destroyAllWindows()

        print(f"✓ 完成! 保存到: {output_path}")


if __name__ == "__main__":
    VIDEO_PATH = r"/cde7198975117a224bb23963f96cbd3d.mp4"
    OUTPUT_PATH = r"/output/green_only_neuron.mp4"

    tracker = NeuronTrackerGreenOnly(
        clahe_clip_limit=3.0,
        adaptive_block_size=15,
        adaptive_c=3,
        morph_iterations=2,
        min_object_size=100,
        margin_ratio=0.05,
        green_threshold=60  # ← 调大可排除更多白色，调小可检测更暗的绿色
    )

    tracker.process_video(
        video_path=VIDEO_PATH,
        output_path=OUTPUT_PATH,
        show_preview=True
    )
