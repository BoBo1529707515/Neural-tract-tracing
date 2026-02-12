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


class NeuronTrackerLocked:
    """锁定追踪单个神经元 - 全程轨迹"""

    def __init__(self,
                 clahe_clip_limit=3.0,
                 median_kernel=5,
                 adaptive_block_size=15,
                 adaptive_c=3,
                 morph_kernel_size=3,
                 morph_iterations=2,
                 min_object_size=100,
                 margin_ratio=0.05,  # 边缘裁剪5%
                 green_threshold=40,  # 绿色阈值
                 max_gap=15,
                 search_radius=40):  # 跨帧搜索半径

        self.clahe_clip_limit = clahe_clip_limit
        self.median_kernel = median_kernel
        self.adaptive_block_size = adaptive_block_size
        self.adaptive_c = adaptive_c
        self.morph_kernel_size = morph_kernel_size
        self.morph_iterations = morph_iterations
        self.min_object_size = min_object_size
        self.margin_ratio = margin_ratio
        self.green_threshold = green_threshold
        self.max_gap = max_gap
        self.search_radius = search_radius

        self.clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=(8, 8))

        # ===== 锁定状态 =====
        self.locked = False
        self.locked_end_point = None  # 锁定的终点（最右）
        self.locked_start_point = None  # 锁定的起点（最左）
        self.locked_y_mean = None  # 锁定的Y位置（用于识别同一神经元）
        self.all_trajectories = []  # 全程轨迹点

    def get_roi(self, h, w):
        """计算感受野 - 去除边缘5%"""
        margin_y = int(h * self.margin_ratio)
        margin_x = int(w * self.margin_ratio)
        y1 = margin_y
        y2 = h - margin_y
        x1 = margin_x
        x2 = w - margin_x
        return y1, y2, x1, x2

    def is_green_pixel(self, frame, y, x):
        """严格检查绿色像素 - 排除白色"""
        h, w = frame.shape[:2]
        if y < 0 or y >= h or x < 0 or x >= w:
            return False

        b, g, r = int(frame[y, x, 0]), int(frame[y, x, 1]), int(frame[y, x, 2])

        # 条件1: G通道足够亮
        if g < 50:
            return False

        # 条件2: G比R高出阈值
        if g - r < self.green_threshold:
            return False

        # 条件3: G比B高出阈值
        if g - b < self.green_threshold:
            return False

        # 条件4: 排除白色 (R、G、B都高且接近)
        if r > 150 and b > 150 and g > 150:
            if abs(r - g) < 50 and abs(b - g) < 50:
                return False

        return True

    def extract_green_mask(self, frame):
        """提取绿色掩码 - 严格排除白色"""
        b, g, r = cv2.split(frame)
        b, g, r = b.astype(np.int16), g.astype(np.int16), r.astype(np.int16)

        bright_enough = g > 50
        g_over_r = (g - r) > self.green_threshold
        g_over_b = (g - b) > self.green_threshold
        not_white = ~((r > 150) & (b > 150) & (g > 150) &
                      (np.abs(r - g) < 50) & (np.abs(b - g) < 50))

        green_mask = bright_enough & g_over_r & g_over_b & not_white
        return green_mask.astype(np.uint8) * 255

    def preprocess_frame(self, frame, roi):
        """预处理 - 只在ROI内处理"""
        y1, y2, x1, x2 = roi

        # 1. 提取绿色掩码
        green_mask = self.extract_green_mask(frame)

        # 2. 应用ROI掩码
        roi_mask = np.zeros_like(green_mask)
        roi_mask[y1:y2, x1:x2] = 255
        green_mask = cv2.bitwise_and(green_mask, roi_mask)

        # 3. 绿色通道增强
        green = frame[:, :, 1]
        enhanced = self.clahe.apply(green)

        # 4. 中值滤波
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

        # 5. 自适应阈值
        binary = cv2.adaptiveThreshold(
            denoised, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            self.adaptive_block_size,
            -self.adaptive_c
        )

        # 6. 与绿色掩码相交
        binary = cv2.bitwise_and(binary, green_mask)

        # 7. 形态学
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (self.morph_kernel_size, self.morph_kernel_size))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel,
                                  iterations=self.morph_iterations)

        # 8. 去小物体
        bool_mask = closed > 0
        cleaned = remove_small_objects(bool_mask, min_size=self.min_object_size, connectivity=1)

        # 9. 骨架化
        skeleton = skeletonize(cleaned)

        return skeleton

    def find_rightmost_green_in_roi(self, skeleton, frame, roi):
        """在ROI内找最靠右的绿色骨架点"""
        y1, y2, x1, x2 = roi

        ys, xs = np.where(skeleton)
        if len(xs) == 0:
            return None

        # 按x从大到小排序
        sorted_indices = np.argsort(xs)[::-1]

        for idx in sorted_indices:
            y, x = ys[idx], xs[idx]
            # 检查是否在ROI内
            if y1 <= y < y2 and x1 <= x < x2:
                # 检查是否为绿色
                if self.is_green_pixel(frame, y, x):
                    return (y, x)

        return None

    def find_point_near_locked(self, skeleton, frame, roi):
        """在锁定位置附近找点（基于Y位置匹配同一神经元）"""
        if self.locked_end_point is None:
            return None

        y1, y2, x1, x2 = roi
        h, w = skeleton.shape
        prev_y, prev_x = self.locked_end_point

        best_point = None
        best_score = float('-inf')

        # 在搜索半径内找
        for radius in [self.search_radius, self.search_radius * 2]:
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    ny, nx = prev_y + dy, prev_x + dx

                    if 0 <= ny < h and 0 <= nx < w:
                        # 必须在ROI内
                        if y1 <= ny < y2 and x1 <= nx < x2:
                            if skeleton[ny, nx]:
                                # 必须是绿色
                                if self.is_green_pixel(frame, ny, nx):
                                    # 评分：Y位置接近 + X尽量靠右
                                    y_diff = abs(ny - self.locked_y_mean) if self.locked_y_mean else abs(ny - prev_y)

                                    # Y位置偏差太大的排除（不是同一根神经元）
                                    if y_diff > 50:
                                        continue

                                    score = nx * 10 - y_diff * 5  # X大加分，Y偏差扣分

                                    if score > best_score:
                                        best_score = score
                                        best_point = (ny, nx)

            if best_point is not None:
                return best_point

        return None

    def trace_from_point(self, skeleton, frame, start_point, roi):
        """从起始点向左追踪"""
        if start_point is None:
            return []

        y1, y2, x1, x2 = roi
        h, w = skeleton.shape
        visited = np.zeros_like(skeleton, dtype=bool)
        trajectory = [start_point]
        visited[start_point[0], start_point[1]] = True

        current = start_point

        while True:
            # 找下一个点
            next_point = None
            best_score = float('-inf')

            for search_r in [1, self.max_gap]:
                for dy in range(-search_r, search_r + 1):
                    for dx in range(-search_r, search_r + 1):
                        if dy == 0 and dx == 0:
                            continue

                        ny, nx = current[0] + dy, current[1] + dx

                        if 0 <= ny < h and 0 <= nx < w:
                            if y1 <= ny < y2:  # ROI限制
                                if skeleton[ny, nx] and not visited[ny, nx]:
                                    if self.is_green_pixel(frame, ny, nx):
                                        dist = np.sqrt(dy ** 2 + dx ** 2)
                                        left_bonus = -dx * 10
                                        horizontal_bonus = -abs(dy) * 3
                                        dist_penalty = -dist * 2

                                        score = left_bonus + horizontal_bonus + dist_penalty

                                        if score > best_score:
                                            best_score = score
                                            next_point = (ny, nx)

                if next_point is not None:
                    break

            if next_point is None:
                break

            visited[next_point[0], next_point[1]] = True
            trajectory.append(next_point)
            current = next_point

            if current[1] <= x1 + 5:
                break

        return trajectory

    def update_locked_state(self, trajectory):
        """更新锁定状态"""
        if len(trajectory) < 2:
            return

        coords = np.array(trajectory)
        sorted_indices = np.argsort(coords[:, 1])
        sorted_coords = coords[sorted_indices]

        self.locked_start_point = tuple(sorted_coords[0])
        self.locked_end_point = tuple(sorted_coords[-1])
        self.locked_y_mean = sorted_coords[:, 0].mean()

        # 累积全程轨迹
        for pt in trajectory:
            self.all_trajectories.append(pt)

    def draw_result(self, image, curr_trajectory, roi, frame_idx):
        """绘制结果"""
        result = image.copy()
        y1, y2, x1, x2 = roi

        # 绘制ROI边界
        cv2.rectangle(result, (x1, y1), (x2, y2), (60, 60, 60), 1)
        cv2.putText(result, "ROI", (x1 + 5, y1 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

        # 绘制全程轨迹（淡绿色）
        if len(self.all_trajectories) > 1:
            all_coords = np.array(self.all_trajectories)
            for i in range(len(all_coords) - 1):
                pt1 = (int(all_coords[i, 1]), int(all_coords[i, 0]))
                pt2 = (int(all_coords[i + 1, 1]), int(all_coords[i + 1, 0]))
                dist = np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)
                if dist < 20:
                    cv2.line(result, pt1, pt2, (0, 128, 0), 1)  # 淡绿色

        # 绘制当前帧轨迹（亮绿色）
        if len(curr_trajectory) >= 2:
            coords = np.array(curr_trajectory)
            sorted_indices = np.argsort(coords[:, 1])
            sorted_coords = coords[sorted_indices]

            for j in range(len(sorted_coords) - 1):
                pt1 = (int(sorted_coords[j, 1]), int(sorted_coords[j, 0]))
                pt2 = (int(sorted_coords[j + 1, 1]), int(sorted_coords[j + 1, 0]))
                dist = np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)
                if dist < self.max_gap + 10:
                    cv2.line(result, pt1, pt2, (0, 255, 0), 2)

            # 标记起点和终点
            leftmost = sorted_coords[0]
            cv2.circle(result, (int(leftmost[1]), int(leftmost[0])), 6, (0, 255, 0), -1)

            rightmost = sorted_coords[-1]
            cv2.circle(result, (int(rightmost[1]), int(rightmost[0])), 6, (0, 0, 255), -1)

        # 信息
        status = "LOCKED" if self.locked else "SEARCHING"
        info = f"Frame {frame_idx} | Status: {status} | Total pts: {len(self.all_trajectories)}"
        cv2.putText(result, info, (11, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(result, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

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
        y1, y2, x1, x2 = roi

        print(f"视频: {width}x{height}, {fps}fps, {total_frames}帧")
        print(f"ROI (去除{self.margin_ratio * 100:.0f}%边缘): y=[{y1},{y2}], x=[{x1},{x2}]")
        print(f"绿色阈值: G必须比R和B高{self.green_threshold}")

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

        # 重置状态
        self.locked = False
        self.locked_end_point = None
        self.locked_start_point = None
        self.locked_y_mean = None
        self.all_trajectories = []

        pbar = tqdm(total=total_frames, desc="锁定追踪")
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            skeleton = self.preprocess_frame(frame, roi)

            if not self.locked:
                # 第一帧：找最靠右的绿色点并锁定
                end_point = self.find_rightmost_green_in_roi(skeleton, frame, roi)
                if end_point is not None:
                    trajectory = self.trace_from_point(skeleton, frame, end_point, roi)
                    if len(trajectory) > 10:
                        self.locked = True
                        self.update_locked_state(trajectory)
                        print(f"✓ 帧{frame_idx}: 锁定神经元, Y≈{self.locked_y_mean:.0f}")
                else:
                    trajectory = []
            else:
                # 后续帧：在锁定位置附近搜索
                end_point = self.find_point_near_locked(skeleton, frame, roi)
                if end_point is not None:
                    trajectory = self.trace_from_point(skeleton, frame, end_point, roi)
                    self.update_locked_state(trajectory)
                else:
                    trajectory = []

            vis_frame = self.draw_result(frame, trajectory, roi, frame_idx)
            writer.write(vis_frame)

            if can_show:
                scale = min(1.0, 1280 / width, 720 / height)
                display = cv2.resize(vis_frame, None, fx=scale, fy=scale)
                cv2.imshow('Locked Tracking (Q quit)', display)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            pbar.update(1)

        pbar.close()
        cap.release()
        writer.release()
        if can_show:
            cv2.destroyAllWindows()

        print(f"\n✓ 完成! 保存到: {output_path}")
        print(f"  全程轨迹点数: {len(self.all_trajectories)}")


if __name__ == "__main__":
    VIDEO_PATH = r"F:\工作文件\RA\python\项目汇总\神经图像\neuron_growth_50.mp4"
    OUTPUT_PATH = r"F:\工作文件\RA\python\项目汇总\神经图像\output\locked_tracking_50.mp4"

    tracker = NeuronTrackerLocked(
        clahe_clip_limit=3.0,
        adaptive_block_size=15,
        adaptive_c=3,
        morph_iterations=2,
        min_object_size=100,
        margin_ratio=0.05,  # 边缘裁剪5%
        green_threshold=40,  # 绿色阈值
        max_gap=15,
        search_radius=40  # 跨帧搜索半径
    )

    tracker.process_video(
        video_path=VIDEO_PATH,
        output_path=OUTPUT_PATH,
        show_preview=True
    )
