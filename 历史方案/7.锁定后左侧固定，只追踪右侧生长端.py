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


class NeuronTrackerGrowth:
    """神经元生长追踪 - 左侧固定，只追踪右侧生长"""

    def __init__(self,
                 clahe_clip_limit=3.0,
                 median_kernel=5,
                 adaptive_block_size=15,
                 adaptive_c=3,
                 morph_kernel_size=3,
                 morph_iterations=2,
                 min_object_size=100,
                 margin_ratio=0.05,
                 green_threshold=40,
                 max_gap=15,
                 search_radius=40):

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

        # ===== 固定轨迹状态 =====
        self.locked = False
        self.fixed_trajectory = []  # 已固定的轨迹（不再改变）
        self.current_tip = None  # 当前生长端位置（最右点）
        self.locked_y_mean = None  # 锁定的Y位置

    def get_roi(self, h, w):
        margin_y = int(h * self.margin_ratio)
        margin_x = int(w * self.margin_ratio)
        return margin_y, h - margin_y, margin_x, w - margin_x

    def is_green_pixel(self, frame, y, x):
        h, w = frame.shape[:2]
        if y < 0 or y >= h or x < 0 or x >= w:
            return False

        b, g, r = int(frame[y, x, 0]), int(frame[y, x, 1]), int(frame[y, x, 2])

        if g < 50:
            return False
        if g - r < self.green_threshold:
            return False
        if g - b < self.green_threshold:
            return False
        if r > 150 and b > 150 and g > 150:
            if abs(r - g) < 50 and abs(b - g) < 50:
                return False
        return True

    def extract_green_mask(self, frame):
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
        y1, y2, x1, x2 = roi

        green_mask = self.extract_green_mask(frame)
        roi_mask = np.zeros_like(green_mask)
        roi_mask[y1:y2, x1:x2] = 255
        green_mask = cv2.bitwise_and(green_mask, roi_mask)

        green = frame[:, :, 1]
        enhanced = self.clahe.apply(green)

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

        binary = cv2.adaptiveThreshold(
            denoised, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            self.adaptive_block_size,
            -self.adaptive_c
        )

        binary = cv2.bitwise_and(binary, green_mask)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (self.morph_kernel_size, self.morph_kernel_size))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel,
                                  iterations=self.morph_iterations)

        bool_mask = closed > 0
        cleaned = remove_small_objects(bool_mask, min_size=self.min_object_size, connectivity=1)
        skeleton = skeletonize(cleaned)

        return skeleton

    def find_rightmost_green_in_roi(self, skeleton, frame, roi):
        y1, y2, x1, x2 = roi
        ys, xs = np.where(skeleton)
        if len(xs) == 0:
            return None

        sorted_indices = np.argsort(xs)[::-1]
        for idx in sorted_indices:
            y, x = ys[idx], xs[idx]
            if y1 <= y < y2 and x1 <= x < x2:
                if self.is_green_pixel(frame, y, x):
                    return (y, x)
        return None

    def trace_full_path(self, skeleton, frame, start_point, roi):
        """完整追踪（仅第一帧使用）"""
        if start_point is None:
            return []

        y1, y2, x1, x2 = roi
        h, w = skeleton.shape
        visited = np.zeros_like(skeleton, dtype=bool)
        trajectory = [start_point]
        visited[start_point[0], start_point[1]] = True

        current = start_point

        while True:
            next_point = None
            best_score = float('-inf')

            for search_r in [1, self.max_gap]:
                for dy in range(-search_r, search_r + 1):
                    for dx in range(-search_r, search_r + 1):
                        if dy == 0 and dx == 0:
                            continue

                        ny, nx = current[0] + dy, current[1] + dx

                        if 0 <= ny < h and 0 <= nx < w:
                            if y1 <= ny < y2:
                                if skeleton[ny, nx] and not visited[ny, nx]:
                                    if self.is_green_pixel(frame, ny, nx):
                                        dist = np.sqrt(dy ** 2 + dx ** 2)
                                        score = -dx * 10 - abs(dy) * 3 - dist * 2
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

    def find_new_growth(self, skeleton, frame, roi):
        """
        只在当前生长端附近搜索新的延伸
        不重新计算左侧已固定的轨迹
        """
        if self.current_tip is None:
            return []

        y1, y2, x1, x2 = roi
        h, w = skeleton.shape
        tip_y, tip_x = self.current_tip

        # 在生长端附近搜索新的骨架点（向右延伸）
        new_points = []
        visited = set()
        visited.add(self.current_tip)

        # 也标记已固定轨迹的点为已访问
        for pt in self.fixed_trajectory:
            visited.add(tuple(pt))

        # BFS从生长端向右搜索
        queue = [self.current_tip]

        while queue:
            current = queue.pop(0)
            cy, cx = current

            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    if dy == 0 and dx == 0:
                        continue

                    ny, nx = cy + dy, cx + dx

                    if 0 <= ny < h and 0 <= nx < w:
                        if y1 <= ny < y2 and x1 <= nx < x2:
                            if (ny, nx) not in visited:
                                if skeleton[ny, nx]:
                                    if self.is_green_pixel(frame, ny, nx):
                                        # Y位置不能偏离太多
                                        if abs(ny - self.locked_y_mean) < 50:
                                            visited.add((ny, nx))
                                            new_points.append((ny, nx))
                                            queue.append((ny, nx))

        return new_points

    def update_trajectory(self, new_points):
        """将新生长点追加到固定轨迹"""
        if len(new_points) == 0:
            return

        # 按x排序新点
        new_points_sorted = sorted(new_points, key=lambda p: p[1])

        # 追加到固定轨迹
        for pt in new_points_sorted:
            # 避免重复
            if len(self.fixed_trajectory) == 0 or pt != tuple(self.fixed_trajectory[-1]):
                self.fixed_trajectory.append(list(pt))

        # 更新生长端（最右点）
        if len(self.fixed_trajectory) > 0:
            all_coords = np.array(self.fixed_trajectory)
            rightmost_idx = np.argmax(all_coords[:, 1])
            self.current_tip = tuple(all_coords[rightmost_idx])

    def draw_result(self, image, roi, frame_idx):
        result = image.copy()
        y1, y2, x1, x2 = roi

        # ROI边界
        cv2.rectangle(result, (x1, y1), (x2, y2), (60, 60, 60), 1)

        if len(self.fixed_trajectory) < 2:
            cv2.putText(result, "Searching...", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            return result

        coords = np.array(self.fixed_trajectory)
        sorted_indices = np.argsort(coords[:, 1])
        sorted_coords = coords[sorted_indices]

        # 绘制固定轨迹（绿色）
        for j in range(len(sorted_coords) - 1):
            pt1 = (int(sorted_coords[j, 1]), int(sorted_coords[j, 0]))
            pt2 = (int(sorted_coords[j + 1, 1]), int(sorted_coords[j + 1, 0]))
            dist = np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)
            if dist < 25:
                cv2.line(result, pt1, pt2, (0, 255, 0), 2)

        # 起点（最左）- 绿色
        leftmost = sorted_coords[0]
        cv2.circle(result, (int(leftmost[1]), int(leftmost[0])), 6, (0, 255, 0), -1)
        cv2.circle(result, (int(leftmost[1]), int(leftmost[0])), 8, (255, 255, 255), 2)

        # 生长端（最右）- 红色
        rightmost = sorted_coords[-1]
        cv2.circle(result, (int(rightmost[1]), int(rightmost[0])), 6, (0, 0, 255), -1)
        cv2.circle(result, (int(rightmost[1]), int(rightmost[0])), 8, (255, 255, 255), 2)

        # 信息
        x_span = sorted_coords[-1, 1] - sorted_coords[0, 1]
        info = f"Frame {frame_idx} | LOCKED | Length: {len(self.fixed_trajectory)} | Span: {x_span:.0f}px"
        cv2.putText(result, info, (11, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(result, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 显示生长端位置
        tip_info = f"Tip: ({int(rightmost[1])}, {int(rightmost[0])})"
        cv2.putText(result, tip_info, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

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
        print(f"ROI: y=[{y1},{y2}], x=[{x1},{x2}]")
        print(f"模式: 左侧固定，只追踪右侧生长端")

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
        self.fixed_trajectory = []
        self.current_tip = None
        self.locked_y_mean = None

        pbar = tqdm(total=total_frames, desc="生长追踪")
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            skeleton = self.preprocess_frame(frame, roi)

            if not self.locked:
                # ===== 第一帧：完整追踪并锁定 =====
                end_point = self.find_rightmost_green_in_roi(skeleton, frame, roi)
                if end_point is not None:
                    trajectory = self.trace_full_path(skeleton, frame, end_point, roi)
                    if len(trajectory) > 10:
                        self.locked = True
                        self.fixed_trajectory = [list(pt) for pt in trajectory]

                        # 按x排序
                        coords = np.array(self.fixed_trajectory)
                        sorted_indices = np.argsort(coords[:, 1])
                        self.fixed_trajectory = coords[sorted_indices].tolist()

                        # 记录生长端和Y位置
                        self.current_tip = tuple(self.fixed_trajectory[-1])
                        self.locked_y_mean = coords[:, 0].mean()

                        print(
                            f"✓ 帧{frame_idx}: 锁定神经元, Y≈{self.locked_y_mean:.0f}, 初始长度={len(self.fixed_trajectory)}")
            else:
                # ===== 后续帧：只在生长端搜索新增部分 =====
                new_points = self.find_new_growth(skeleton, frame, roi)
                if len(new_points) > 0:
                    self.update_trajectory(new_points)

            vis_frame = self.draw_result(frame, roi, frame_idx)
            writer.write(vis_frame)

            if can_show:
                scale = min(1.0, 1280 / width, 720 / height)
                display = cv2.resize(vis_frame, None, fx=scale, fy=scale)
                cv2.imshow('Growth Tracking (Q quit)', display)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            pbar.update(1)

        pbar.close()
        cap.release()
        writer.release()
        if can_show:
            cv2.destroyAllWindows()

        print(f"\n✓ 完成! 保存到: {output_path}")
        print(f"  最终轨迹长度: {len(self.fixed_trajectory)} 点")
        if len(self.fixed_trajectory) > 0:
            coords = np.array(self.fixed_trajectory)
            print(f"  X范围: [{coords[:, 1].min():.0f}, {coords[:, 1].max():.0f}]")


if __name__ == "__main__":
    VIDEO_PATH = r"/neuron_growth_50.mp4"
    OUTPUT_PATH = r"/output/growth_tracking.mp4"

    tracker = NeuronTrackerGrowth(
        clahe_clip_limit=3.0,
        adaptive_block_size=15,
        adaptive_c=3,
        morph_iterations=2,
        min_object_size=100,
        margin_ratio=0.05,
        green_threshold=40,
        max_gap=15,
        search_radius=40
    )

    tracker.process_video(
        video_path=VIDEO_PATH,
        output_path=OUTPUT_PATH,
        show_preview=True
    )
