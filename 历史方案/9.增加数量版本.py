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


def generate_colors(n):
    """生成n种不同的颜色"""
    colors = []
    for i in range(n):
        hue = int(180 * i / n)
        hsv = np.uint8([[[hue, 255, 255]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
        colors.append((int(bgr[0]), int(bgr[1]), int(bgr[2])))
    return colors


class NeuronState:
    """单个神经元状态"""

    def __init__(self, neuron_id, color):
        self.id = neuron_id
        self.color = color
        self.fixed_trajectory = []
        self.current_tip = None
        self.locked_y_mean = None
        self.active = True


class MultiNeuronTracker:
    """多神经元追踪 - 按Y分段检测"""

    def __init__(self,
                 num_neurons=15,
                 clahe_clip_limit=3.0,
                 median_kernel=5,
                 adaptive_block_size=15,
                 adaptive_c=3,
                 morph_kernel_size=3,
                 morph_iterations=2,
                 min_object_size=50,
                 margin_ratio=0.05,
                 green_threshold=40,
                 max_gap=15,
                 search_radius=40,
                 min_neuron_length=15,
                 y_tolerance=20):  # 同一神经元的Y容差

        self.num_neurons = num_neurons
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
        self.min_neuron_length = min_neuron_length
        self.y_tolerance = y_tolerance

        self.clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=(8, 8))
        self.colors = generate_colors(num_neurons)
        self.neurons = []
        self.locked = False

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

    def find_rightmost_points_by_row(self, skeleton, frame, roi):
        """
        按行扫描，找每个Y位置段的最右点
        返回: [(y, x), ...] 候选起始点列表
        """
        y1, y2, x1, x2 = roi
        h, w = skeleton.shape

        # 收集所有骨架点
        ys, xs = np.where(skeleton)
        if len(xs) == 0:
            return []

        # 按Y分段，每段找最右的绿色点
        y_min, y_max = ys.min(), ys.max()
        y_range = y_max - y_min

        if y_range < 10:
            return []

        # 分成多个Y段
        num_segments = max(self.num_neurons * 2, 30)  # 多分一些段
        segment_height = y_range / num_segments

        rightmost_per_segment = {}

        for i in range(len(ys)):
            y, x = ys[i], xs[i]

            # 必须在ROI内
            if not (y1 <= y < y2 and x1 <= x < x2):
                continue

            # 必须是绿色
            if not self.is_green_pixel(frame, y, x):
                continue

            # 确定所属段
            segment_idx = int((y - y_min) / segment_height) if segment_height > 0 else 0

            # 保留该段最右的点
            if segment_idx not in rightmost_per_segment:
                rightmost_per_segment[segment_idx] = (y, x)
            else:
                if x > rightmost_per_segment[segment_idx][1]:
                    rightmost_per_segment[segment_idx] = (y, x)

        # 按Y排序
        candidates = list(rightmost_per_segment.values())
        candidates.sort(key=lambda p: p[0])

        return candidates

    def trace_from_point_limited(self, skeleton, frame, start_point, roi, used_points):
        """
        从起始点向左追踪，避开已使用的点
        """
        if start_point is None:
            return []

        y1, y2, x1, x2 = roi
        h, w = skeleton.shape
        visited = np.zeros_like(skeleton, dtype=bool)
        trajectory = [start_point]
        visited[start_point[0], start_point[1]] = True

        # 标记已使用的点
        for pt in used_points:
            if 0 <= pt[0] < h and 0 <= pt[1] < w:
                visited[pt[0], pt[1]] = True

        current = start_point
        start_y = start_point[0]

        while True:
            next_point = None
            best_score = float('-inf')

            for search_r in [1, 2, self.max_gap]:
                for dy in range(-search_r, search_r + 1):
                    for dx in range(-search_r, search_r + 1):
                        if dy == 0 and dx == 0:
                            continue

                        ny, nx = current[0] + dy, current[1] + dx

                        if 0 <= ny < h and 0 <= nx < w:
                            if y1 <= ny < y2:
                                if skeleton[ny, nx] and not visited[ny, nx]:
                                    if self.is_green_pixel(frame, ny, nx):
                                        # Y不能偏离太多
                                        if abs(ny - start_y) > self.y_tolerance * 2:
                                            continue

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

    def initialize_neurons(self, skeleton, frame, roi):
        """第一帧：检测多根神经元"""
        # 找所有候选起始点
        candidates = self.find_rightmost_points_by_row(skeleton, frame, roi)

        print(f"  找到 {len(candidates)} 个候选起始点")

        if len(candidates) == 0:
            return 0

        # 已使用的轨迹点
        used_points = set()

        # 已确定的神经元Y位置（避免重复）
        confirmed_y_means = []

        self.neurons = []
        neuron_id = 0

        for candidate in candidates:
            if neuron_id >= self.num_neurons:
                break

            # 检查是否与已有神经元Y位置太近
            too_close = False
            for y_mean in confirmed_y_means:
                if abs(candidate[0] - y_mean) < self.y_tolerance:
                    too_close = True
                    break

            if too_close:
                continue

            # 追踪
            trajectory = self.trace_from_point_limited(skeleton, frame, candidate, roi, used_points)

            if len(trajectory) < self.min_neuron_length:
                continue

            # 创建神经元
            coords = np.array(trajectory)
            sorted_indices = np.argsort(coords[:, 1])
            sorted_traj = coords[sorted_indices].tolist()

            y_mean = coords[:, 0].mean()

            # 再次检查Y位置
            too_close = False
            for existing_y in confirmed_y_means:
                if abs(y_mean - existing_y) < self.y_tolerance:
                    too_close = True
                    break

            if too_close:
                continue

            neuron = NeuronState(neuron_id, self.colors[neuron_id])
            neuron.fixed_trajectory = sorted_traj
            neuron.current_tip = tuple(sorted_traj[-1])
            neuron.locked_y_mean = y_mean

            self.neurons.append(neuron)
            confirmed_y_means.append(y_mean)

            # 标记已使用的点
            for pt in trajectory:
                used_points.add(tuple(pt))

            neuron_id += 1

        return len(self.neurons)

    def find_growth_for_neuron(self, skeleton, frame, neuron, roi):
        """为单个神经元找生长点"""
        if neuron.current_tip is None or not neuron.active:
            return []

        y1, y2, x1, x2 = roi
        h, w = skeleton.shape

        new_points = []
        visited = set()
        visited.add(neuron.current_tip)

        for pt in neuron.fixed_trajectory:
            visited.add(tuple(pt))

        queue = [neuron.current_tip]

        while queue:
            current = queue.pop(0)
            cy, cx = current

            for dy in range(-3, 4):
                for dx in range(-3, 4):
                    if dy == 0 and dx == 0:
                        continue

                    ny, nx = cy + dy, cx + dx

                    if 0 <= ny < h and 0 <= nx < w:
                        if y1 <= ny < y2 and x1 <= nx < x2:
                            if (ny, nx) not in visited:
                                if skeleton[ny, nx]:
                                    if self.is_green_pixel(frame, ny, nx):
                                        if abs(ny - neuron.locked_y_mean) < self.y_tolerance * 2:
                                            visited.add((ny, nx))
                                            new_points.append((ny, nx))
                                            queue.append((ny, nx))

        return new_points

    def update_neuron(self, neuron, new_points):
        if len(new_points) == 0:
            return

        new_points_sorted = sorted(new_points, key=lambda p: p[1])

        for pt in new_points_sorted:
            if len(neuron.fixed_trajectory) == 0 or pt != tuple(neuron.fixed_trajectory[-1]):
                neuron.fixed_trajectory.append(list(pt))

        if len(neuron.fixed_trajectory) > 0:
            all_coords = np.array(neuron.fixed_trajectory)
            rightmost_idx = np.argmax(all_coords[:, 1])
            neuron.current_tip = tuple(all_coords[rightmost_idx])

    def draw_result(self, image, roi, frame_idx):
        result = image.copy()
        y1, y2, x1, x2 = roi

        cv2.rectangle(result, (x1, y1), (x2, y2), (60, 60, 60), 1)

        active_count = 0

        for neuron in self.neurons:
            if len(neuron.fixed_trajectory) < 2:
                continue

            active_count += 1
            coords = np.array(neuron.fixed_trajectory)
            sorted_indices = np.argsort(coords[:, 1])
            sorted_coords = coords[sorted_indices]

            # 轨迹
            for j in range(len(sorted_coords) - 1):
                pt1 = (int(sorted_coords[j, 1]), int(sorted_coords[j, 0]))
                pt2 = (int(sorted_coords[j + 1, 1]), int(sorted_coords[j + 1, 0]))
                dist = np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)
                if dist < 30:
                    cv2.line(result, pt1, pt2, neuron.color, 2)

            # 起点
            leftmost = sorted_coords[0]
            cv2.circle(result, (int(leftmost[1]), int(leftmost[0])), 4, neuron.color, -1)

            # 生长端
            rightmost = sorted_coords[-1]
            cv2.circle(result, (int(rightmost[1]), int(rightmost[0])), 5, neuron.color, -1)
            cv2.circle(result, (int(rightmost[1]), int(rightmost[0])), 7, (255, 255, 255), 1)

            # 编号
            cv2.putText(result, str(neuron.id + 1),
                        (int(rightmost[1]) + 8, int(rightmost[0]) + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, neuron.color, 1)

        # 信息
        info = f"Frame {frame_idx} | Neurons: {active_count}/{len(self.neurons)}"
        cv2.putText(result, info, (11, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(result, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # 图例
        legend_y = 50
        for i, neuron in enumerate(self.neurons):
            cv2.rectangle(result, (10, legend_y + i * 14), (18, legend_y + i * 14 + 10), neuron.color, -1)
            info_text = f"N{neuron.id + 1}: {len(neuron.fixed_trajectory)}pts, Y={neuron.locked_y_mean:.0f}"
            cv2.putText(result, info_text, (22, legend_y + i * 14 + 9),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)

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
        print(f"目标追踪: {self.num_neurons} 根神经元")
        print(f"Y容差: {self.y_tolerance}px (区分不同神经元)")

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

        self.neurons = []
        self.locked = False

        pbar = tqdm(total=total_frames, desc="多神经元追踪")
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            skeleton = self.preprocess_frame(frame, roi)

            if not self.locked:
                num_found = self.initialize_neurons(skeleton, frame, roi)
                if num_found > 0:
                    self.locked = True
                    print(f"✓ 帧{frame_idx}: 锁定 {num_found} 根神经元")
                    for n in self.neurons:
                        print(f"   N{n.id + 1}: Y≈{n.locked_y_mean:.0f}, 初始长度={len(n.fixed_trajectory)}")
            else:
                for neuron in self.neurons:
                    new_points = self.find_growth_for_neuron(skeleton, frame, neuron, roi)
                    self.update_neuron(neuron, new_points)

            vis_frame = self.draw_result(frame, roi, frame_idx)
            writer.write(vis_frame)

            if can_show:
                scale = min(1.0, 1280 / width, 720 / height)
                display = cv2.resize(vis_frame, None, fx=scale, fy=scale)
                cv2.imshow('Multi-Neuron (Q quit)', display)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            pbar.update(1)

        pbar.close()
        cap.release()
        writer.release()
        if can_show:
            cv2.destroyAllWindows()

        print(f"\n✓ 完成! {output_path}")
        for n in self.neurons:
            print(f"  N{n.id + 1}: {len(n.fixed_trajectory)} pts")


if __name__ == "__main__":
    VIDEO_PATH = r"/neuron_growth_50.mp4"
    OUTPUT_PATH = r"/output/multi_neuron_15.mp4"

    tracker = MultiNeuronTracker(
        num_neurons=15,
        clahe_clip_limit=3.0,
        adaptive_block_size=15,
        adaptive_c=3,
        morph_iterations=2,
        min_object_size=50,
        margin_ratio=0.05,
        green_threshold=40,
        max_gap=15,
        search_radius=40,
        min_neuron_length=15,
        y_tolerance=20  # ← Y相差20px以上才算不同神经元
    )

    tracker.process_video(
        video_path=VIDEO_PATH,
        output_path=OUTPUT_PATH,
        show_preview=True
    )
