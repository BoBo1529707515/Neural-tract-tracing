import cv2
import numpy as np
from skimage.morphology import skeletonize, remove_small_objects
from skimage.measure import label, regionprops
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
    """生成n种不同的颜色（HSV色环）"""
    colors = []
    for i in range(n):
        hue = int(180 * i / n)  # 0-180 (OpenCV HSV)
        hsv = np.uint8([[[hue, 255, 255]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
        colors.append((int(bgr[0]), int(bgr[1]), int(bgr[2])))
    return colors


class NeuronState:
    """单个神经元的追踪状态"""

    def __init__(self, neuron_id, color):
        self.id = neuron_id
        self.color = color
        self.fixed_trajectory = []
        self.current_tip = None
        self.locked_y_mean = None
        self.active = True


class MultiNeuronTracker:
    """多神经元生长追踪"""

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
                 min_neuron_length=20):

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

        self.clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=(8, 8))

        # 生成颜色
        self.colors = generate_colors(num_neurons)

        # 神经元状态列表
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

    def find_connected_components(self, skeleton):
        """找到骨架中的所有连通组件"""
        labeled = label(skeleton.astype(np.uint8), connectivity=2)
        regions = regionprops(labeled)
        return labeled, regions

    def trace_component(self, skeleton, frame, start_point, roi):
        """从起始点追踪一条完整路径"""
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

    def initialize_neurons(self, skeleton, frame, roi):
        """第一帧：检测并初始化多根神经元"""
        y1, y2, x1, x2 = roi
        h, w = skeleton.shape

        # 找所有连通组件
        labeled, regions = self.find_connected_components(skeleton)

        # 收集所有候选神经元
        candidates = []

        for region in regions:
            # 获取该组件的像素
            component_mask = (labeled == region.label)
            ys, xs = np.where(component_mask)

            if len(xs) < self.min_neuron_length:
                continue

            # 找最右点
            max_x_idx = np.argmax(xs)
            rightmost = (ys[max_x_idx], xs[max_x_idx])

            # 检查是否在ROI内且是绿色
            if y1 <= rightmost[0] < y2 and x1 <= rightmost[1] < x2:
                if self.is_green_pixel(frame, rightmost[0], rightmost[1]):
                    # 追踪完整路径
                    trajectory = self.trace_component(skeleton, frame, rightmost, roi)

                    if len(trajectory) >= self.min_neuron_length:
                        coords = np.array(trajectory)
                        y_mean = coords[:, 0].mean()
                        x_span = coords[:, 1].max() - coords[:, 1].min()

                        candidates.append({
                            'trajectory': trajectory,
                            'y_mean': y_mean,
                            'x_span': x_span,
                            'rightmost': rightmost
                        })

        # 按Y位置排序，选取前num_neurons根
        candidates.sort(key=lambda c: c['y_mean'])

        # 去除Y位置太接近的（可能是同一根）
        filtered = []
        for c in candidates:
            too_close = False
            for f in filtered:
                if abs(c['y_mean'] - f['y_mean']) < 15:
                    too_close = True
                    break
            if not too_close:
                filtered.append(c)

        # 选取前num_neurons根
        selected = filtered[:self.num_neurons]

        # 创建神经元状态
        self.neurons = []
        for i, c in enumerate(selected):
            neuron = NeuronState(i, self.colors[i])

            # 按X排序轨迹
            coords = np.array(c['trajectory'])
            sorted_indices = np.argsort(coords[:, 1])
            neuron.fixed_trajectory = coords[sorted_indices].tolist()
            neuron.current_tip = tuple(neuron.fixed_trajectory[-1])
            neuron.locked_y_mean = c['y_mean']

            self.neurons.append(neuron)

        return len(self.neurons)

    def find_growth_for_neuron(self, skeleton, frame, neuron, roi):
        """为单个神经元找生长点"""
        if neuron.current_tip is None or not neuron.active:
            return []

        y1, y2, x1, x2 = roi
        h, w = skeleton.shape
        tip_y, tip_x = neuron.current_tip

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
                                        if abs(ny - neuron.locked_y_mean) < 40:
                                            visited.add((ny, nx))
                                            new_points.append((ny, nx))
                                            queue.append((ny, nx))

        return new_points

    def update_neuron(self, neuron, new_points):
        """更新神经元轨迹"""
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

        # ROI边界
        cv2.rectangle(result, (x1, y1), (x2, y2), (60, 60, 60), 1)

        active_count = 0

        for neuron in self.neurons:
            if len(neuron.fixed_trajectory) < 2:
                continue

            active_count += 1
            coords = np.array(neuron.fixed_trajectory)
            sorted_indices = np.argsort(coords[:, 1])
            sorted_coords = coords[sorted_indices]

            # 绘制轨迹
            for j in range(len(sorted_coords) - 1):
                pt1 = (int(sorted_coords[j, 1]), int(sorted_coords[j, 0]))
                pt2 = (int(sorted_coords[j + 1, 1]), int(sorted_coords[j + 1, 0]))
                dist = np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)
                if dist < 25:
                    cv2.line(result, pt1, pt2, neuron.color, 2)

            # 起点（小圆）
            leftmost = sorted_coords[0]
            cv2.circle(result, (int(leftmost[1]), int(leftmost[0])), 4, neuron.color, -1)

            # 生长端（大圆+白边）
            rightmost = sorted_coords[-1]
            cv2.circle(result, (int(rightmost[1]), int(rightmost[0])), 5, neuron.color, -1)
            cv2.circle(result, (int(rightmost[1]), int(rightmost[0])), 7, (255, 255, 255), 1)

            # 神经元编号
            cv2.putText(result, str(neuron.id + 1),
                        (int(rightmost[1]) + 8, int(rightmost[0]) + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, neuron.color, 1)

        # 信息
        total_pts = sum(len(n.fixed_trajectory) for n in self.neurons)
        info = f"Frame {frame_idx} | Neurons: {active_count}/{self.num_neurons} | Total pts: {total_pts}"
        cv2.putText(result, info, (11, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(result, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # 图例
        legend_y = 50
        for i, neuron in enumerate(self.neurons[:8]):  # 最多显示8个
            cv2.rectangle(result, (10, legend_y + i * 15), (20, legend_y + i * 15 + 10), neuron.color, -1)
            cv2.putText(result, f"N{neuron.id + 1}: {len(neuron.fixed_trajectory)}pts",
                        (25, legend_y + i * 15 + 9), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)

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
        print(f"追踪数量: {self.num_neurons} 根神经元")

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

        # 重置
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
                # 第一帧：初始化所有神经元
                num_found = self.initialize_neurons(skeleton, frame, roi)
                if num_found > 0:
                    self.locked = True
                    print(f"✓ 帧{frame_idx}: 锁定 {num_found} 根神经元")
                    for n in self.neurons:
                        print(f"   N{n.id + 1}: Y≈{n.locked_y_mean:.0f}, 长度={len(n.fixed_trajectory)}")
            else:
                # 后续帧：更新每根神经元
                for neuron in self.neurons:
                    new_points = self.find_growth_for_neuron(skeleton, frame, neuron, roi)
                    self.update_neuron(neuron, new_points)

            vis_frame = self.draw_result(frame, roi, frame_idx)
            writer.write(vis_frame)

            if can_show:
                scale = min(1.0, 1280 / width, 720 / height)
                display = cv2.resize(vis_frame, None, fx=scale, fy=scale)
                cv2.imshow('Multi-Neuron Tracking (Q quit)', display)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            pbar.update(1)

        pbar.close()
        cap.release()
        writer.release()
        if can_show:
            cv2.destroyAllWindows()

        print(f"\n✓ 完成! 保存到: {output_path}")
        print(f"追踪结果:")
        for n in self.neurons:
            coords = np.array(n.fixed_trajectory) if len(n.fixed_trajectory) > 0 else np.array([[0, 0]])
            x_span = coords[:, 1].max() - coords[:, 1].min() if len(coords) > 1 else 0
            print(f"  N{n.id + 1}: {len(n.fixed_trajectory)} pts, X跨度={x_span:.0f}px")


if __name__ == "__main__":
    VIDEO_PATH = r"F:\工作文件\RA\python\项目汇总\神经图像\neuron_growth_50.mp4"
    OUTPUT_PATH = r"F:\工作文件\RA\python\项目汇总\神经图像\output\multi_neuron_15.mp4"

    tracker = MultiNeuronTracker(
        num_neurons=15,  # ← 追踪15根
        clahe_clip_limit=3.0,
        adaptive_block_size=15,
        adaptive_c=3,
        morph_iterations=2,
        min_object_size=50,
        margin_ratio=0.05,
        green_threshold=40,
        max_gap=15,
        search_radius=40,
        min_neuron_length=20  # ← 最小长度阈值
    )

    tracker.process_video(
        video_path=VIDEO_PATH,
        output_path=OUTPUT_PATH,
        show_preview=True
    )
