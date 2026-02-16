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

    def __init__(self, neuron_id, color, init_point):
        self.id = neuron_id
        self.color = color
        self.init_point = init_point  # 用户标记的初始点
        self.fixed_trajectory = []
        self.current_tip = None
        self.locked_y_mean = None
        self.active = True


class ManualNeuronTracker:
    """手动标记初始点的神经元追踪"""

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
                 y_tolerance=30):

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
        self.y_tolerance = y_tolerance

        self.clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=(8, 8))
        self.colors = generate_colors(num_neurons)
        self.neurons = []

        # 鼠标点击状态
        self.clicked_points = []
        self.current_frame = None
        self.display_scale = 1.0

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

    def mouse_callback(self, event, x, y, flags, param):
        """鼠标点击回调"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # 转换回原始坐标
            orig_x = int(x / self.display_scale)
            orig_y = int(y / self.display_scale)

            if len(self.clicked_points) < self.num_neurons:
                self.clicked_points.append((orig_y, orig_x))  # (y, x)格式
                print(f"  标记点 {len(self.clicked_points)}: ({orig_x}, {orig_y})")

        elif event == cv2.EVENT_RBUTTONDOWN:
            # 右键撤销最后一个点
            if len(self.clicked_points) > 0:
                removed = self.clicked_points.pop()
                print(f"  撤销点: ({removed[1]}, {removed[0]})")

    def manual_select_points(self, frame, roi):
        """让用户手动标记初始点"""
        y1, y2, x1, x2 = roi
        h, w = frame.shape[:2]

        self.clicked_points = []
        self.current_frame = frame.copy()

        # 计算显示缩放
        max_display_width = 1400
        max_display_height = 800
        self.display_scale = min(1.0, max_display_width / w, max_display_height / h)

        window_name = 'Mark Neurons - Left Click to Add, Right Click to Undo, ENTER to Confirm, Q to Cancel'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, self.mouse_callback)

        print("\n" + "=" * 60)
        print("手动标记模式")
        print("=" * 60)
        print(f"请在第一帧上点击标记 {self.num_neurons} 个神经元的起始点")
        print("  - 左键: 添加标记点")
        print("  - 右键: 撤销上一个点")
        print("  - Enter: 确认完成")
        print("  - Q: 取消")
        print("=" * 60 + "\n")

        while True:
            # 绘制当前帧和标记点
            display = self.current_frame.copy()

            # 绘制ROI边界
            cv2.rectangle(display, (x1, y1), (x2, y2), (60, 60, 60), 1)

            # 绘制已标记的点
            for i, (py, px) in enumerate(self.clicked_points):
                color = self.colors[i]
                cv2.circle(display, (px, py), 8, color, -1)
                cv2.circle(display, (px, py), 10, (255, 255, 255), 2)
                cv2.putText(display, str(i + 1), (px + 12, py + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # 提示信息
            info = f"Marked: {len(self.clicked_points)}/{self.num_neurons} | ENTER=Confirm, Q=Cancel"
            cv2.putText(display, info, (11, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
            cv2.putText(display, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # 缩放显示
            display_resized = cv2.resize(display, None, fx=self.display_scale, fy=self.display_scale)
            cv2.imshow(window_name, display_resized)

            key = cv2.waitKey(30) & 0xFF

            if key == 13 or key == 10:  # Enter
                if len(self.clicked_points) > 0:
                    print(f"\n✓ 确认 {len(self.clicked_points)} 个标记点")
                    break
                else:
                    print("⚠ 请至少标记1个点")

            elif key == ord('q') or key == ord('Q'):
                print("✗ 取消标记")
                self.clicked_points = []
                break

        cv2.destroyWindow(window_name)
        return self.clicked_points

    def find_nearest_skeleton_point(self, skeleton, frame, click_point, roi, radius=30):
        """在点击位置附近找最近的绿色骨架点"""
        y1, y2, x1, x2 = roi
        h, w = skeleton.shape
        cy, cx = click_point

        best_point = None
        best_dist = float('inf')

        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                ny, nx = cy + dy, cx + dx

                if 0 <= ny < h and 0 <= nx < w:
                    if y1 <= ny < y2 and x1 <= nx < x2:
                        if skeleton[ny, nx]:
                            if self.is_green_pixel(frame, ny, nx):
                                dist = dy ** 2 + dx ** 2
                                if dist < best_dist:
                                    best_dist = dist
                                    best_point = (ny, nx)

        return best_point

    def trace_bidirectional(self, skeleton, frame, start_point, roi):
        """从起始点双向追踪（向左和向右）"""
        if start_point is None:
            return []

        y1, y2, x1, x2 = roi
        h, w = skeleton.shape

        # 追踪函数（可指定方向优先）
        def trace_direction(start, prefer_left=True):
            visited = set()
            trajectory = [start]
            visited.add(start)
            current = start
            start_y = start[0]

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
                                    if skeleton[ny, nx] and (ny, nx) not in visited:
                                        if self.is_green_pixel(frame, ny, nx):
                                            if abs(ny - start_y) > self.y_tolerance * 2:
                                                continue

                                            dist = np.sqrt(dy ** 2 + dx ** 2)
                                            if prefer_left:
                                                score = -dx * 10 - abs(dy) * 3 - dist * 2
                                            else:
                                                score = dx * 10 - abs(dy) * 3 - dist * 2

                                            if score > best_score:
                                                best_score = score
                                                next_point = (ny, nx)

                    if next_point is not None:
                        break

                if next_point is None:
                    break

                visited.add(next_point)
                trajectory.append(next_point)
                current = next_point

            return trajectory, visited

        # 向左追踪
        left_traj, left_visited = trace_direction(start_point, prefer_left=True)

        # 向右追踪（避开左边已访问的点）
        right_traj = [start_point]
        visited = left_visited.copy()
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
                                if skeleton[ny, nx] and (ny, nx) not in visited:
                                    if self.is_green_pixel(frame, ny, nx):
                                        if abs(ny - start_y) > self.y_tolerance * 2:
                                            continue

                                        dist = np.sqrt(dy ** 2 + dx ** 2)
                                        score = dx * 10 - abs(dy) * 3 - dist * 2

                                        if score > best_score:
                                            best_score = score
                                            next_point = (ny, nx)

                if next_point is not None:
                    break

            if next_point is None:
                break

            visited.add(next_point)
            right_traj.append(next_point)
            current = next_point

        # 合并轨迹：左边（倒序） + 右边
        full_trajectory = left_traj[::-1] + right_traj[1:]

        return full_trajectory

    def initialize_from_clicks(self, skeleton, frame, roi, clicked_points):
        """根据用户点击初始化神经元"""
        self.neurons = []

        for i, click_pt in enumerate(clicked_points):
            # 找最近的骨架点
            skeleton_pt = self.find_nearest_skeleton_point(skeleton, frame, click_pt, roi)

            if skeleton_pt is None:
                print(f"  ⚠ 点{i + 1}附近未找到骨架，跳过")
                continue

            # 双向追踪
            trajectory = self.trace_bidirectional(skeleton, frame, skeleton_pt, roi)

            if len(trajectory) < 5:
                print(f"  ⚠ 点{i + 1}轨迹太短({len(trajectory)}点)，跳过")
                continue

            # 按X排序
            coords = np.array(trajectory)
            sorted_indices = np.argsort(coords[:, 1])
            sorted_traj = coords[sorted_indices].tolist()

            y_mean = coords[:, 0].mean()

            neuron = NeuronState(len(self.neurons), self.colors[len(self.neurons)], click_pt)
            neuron.fixed_trajectory = sorted_traj
            neuron.current_tip = tuple(sorted_traj[-1])  # 最右点
            neuron.locked_y_mean = y_mean

            self.neurons.append(neuron)
            print(f"  ✓ N{neuron.id + 1}: 初始长度={len(sorted_traj)}, Y≈{y_mean:.0f}")

        return len(self.neurons)

    def find_growth_for_neuron(self, skeleton, frame, neuron, roi):
        """为神经元找生长点（只向右生长）"""
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

        for neuron in self.neurons:
            if len(neuron.fixed_trajectory) < 2:
                continue

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

            # 起点（最左）
            leftmost = sorted_coords[0]
            cv2.circle(result, (int(leftmost[1]), int(leftmost[0])), 4, neuron.color, -1)

            # 初始标记点
            cv2.circle(result, (neuron.init_point[1], neuron.init_point[0]), 3, (255, 255, 255), -1)

            # 生长端（最右）
            rightmost = sorted_coords[-1]
            cv2.circle(result, (int(rightmost[1]), int(rightmost[0])), 5, neuron.color, -1)
            cv2.circle(result, (int(rightmost[1]), int(rightmost[0])), 7, (255, 255, 255), 1)

            # 编号
            cv2.putText(result, str(neuron.id + 1),
                        (int(rightmost[1]) + 8, int(rightmost[0]) + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, neuron.color, 1)

        # 信息
        info = f"Frame {frame_idx} | Neurons: {len(self.neurons)}"
        cv2.putText(result, info, (11, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(result, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # 图例
        legend_y = 50
        for i, neuron in enumerate(self.neurons):
            cv2.rectangle(result, (10, legend_y + i * 14), (18, legend_y + i * 14 + 10), neuron.color, -1)
            info_text = f"N{neuron.id + 1}: {len(neuron.fixed_trajectory)}pts"
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

        # 读取第一帧
        ret, first_frame = cap.read()
        if not ret:
            raise ValueError("无法读取第一帧")

        # 手动标记
        clicked_points = self.manual_select_points(first_frame, roi)

        if len(clicked_points) == 0:
            print("未标记任何点，退出")
            cap.release()
            return

        # 预处理第一帧
        skeleton = self.preprocess_frame(first_frame, roi)

        # 初始化神经元
        num_found = self.initialize_from_clicks(skeleton, first_frame, roi, clicked_points)
        print(f"\n✓ 初始化 {num_found} 根神经元")

        if num_found == 0:
            print("未能初始化任何神经元，退出")
            cap.release()
            return

        # 准备输出
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # 写入第一帧
        vis_frame = self.draw_result(first_frame, roi, 1)
        writer.write(vis_frame)

        can_show = show_preview
        if can_show:
            try:
                cv2.namedWindow('Tracking', cv2.WINDOW_NORMAL)
            except:
                can_show = False

        pbar = tqdm(total=total_frames - 1, desc="追踪中")
        frame_idx = 1

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            skeleton = self.preprocess_frame(frame, roi)

            # 更新每根神经元
            for neuron in self.neurons:
                new_points = self.find_growth_for_neuron(skeleton, frame, neuron, roi)
                self.update_neuron(neuron, new_points)

            vis_frame = self.draw_result(frame, roi, frame_idx)
            writer.write(vis_frame)

            if can_show:
                scale = min(1.0, 1280 / width, 720 / height)
                display = cv2.resize(vis_frame, None, fx=scale, fy=scale)
                cv2.imshow('Tracking', display)
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
    OUTPUT_PATH = r"/历史方案/output\manual_tracking.mp4"

    tracker = ManualNeuronTracker(
        num_neurons=15,  # 最多标记15根
        clahe_clip_limit=3.0,
        adaptive_block_size=15,
        adaptive_c=3,
        morph_iterations=2,
        min_object_size=50,
        margin_ratio=0.05,
        green_threshold=40,
        max_gap=15,
        search_radius=40,
        y_tolerance=30
    )

    tracker.process_video(
        video_path=VIDEO_PATH,
        output_path=OUTPUT_PATH,
        show_preview=True
    )
