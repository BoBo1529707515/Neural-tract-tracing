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


class NeuronTrackerRightmost:
    """从最靠右的点回溯整条神经元轨迹"""

    def __init__(self,
                 clahe_clip_limit=3.0,
                 median_kernel=5,
                 adaptive_block_size=15,
                 adaptive_c=3,
                 morph_kernel_size=3,
                 morph_iterations=2,
                 min_object_size=100):

        self.clahe_clip_limit = clahe_clip_limit
        self.median_kernel = median_kernel
        self.adaptive_block_size = adaptive_block_size
        self.adaptive_c = adaptive_c
        self.morph_kernel_size = morph_kernel_size
        self.morph_iterations = morph_iterations
        self.min_object_size = min_object_size

        self.clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=(8, 8))

    def preprocess_frame(self, frame):
        """预处理"""
        green = frame[:, :, 1] if len(frame.shape) == 3 else frame
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

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (self.morph_kernel_size, self.morph_kernel_size))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel,
                                  iterations=self.morph_iterations)

        bool_mask = closed > 0
        cleaned = remove_small_objects(bool_mask, min_size=self.min_object_size, connectivity=1)
        skeleton = skeletonize(cleaned)

        return skeleton

    def find_rightmost_point(self, skeleton):
        """找到骨架中最靠右的点"""
        ys, xs = np.where(skeleton)
        if len(xs) == 0:
            return None

        # 找x最大的点
        max_x_idx = np.argmax(xs)
        return (ys[max_x_idx], xs[max_x_idx])  # (row, col) = (y, x)

    def trace_back_from_point(self, skeleton, start_point):
        """
        从起始点沿骨架回溯，找到整条连通的轨迹
        使用BFS/DFS遍历连通区域
        """
        if start_point is None:
            return []

        h, w = skeleton.shape
        visited = np.zeros_like(skeleton, dtype=bool)
        trajectory = []

        # 8邻域方向
        directions = [(-1, -1), (-1, 0), (-1, 1),
                      (0, -1), (0, 1),
                      (1, -1), (1, 0), (1, 1)]

        # BFS遍历整条连通轨迹
        queue = [start_point]
        visited[start_point[0], start_point[1]] = True

        while queue:
            current = queue.pop(0)
            trajectory.append(current)

            for dy, dx in directions:
                ny, nx = current[0] + dy, current[1] + dx

                if 0 <= ny < h and 0 <= nx < w:
                    if skeleton[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        queue.append((ny, nx))

        return trajectory

    def sort_trajectory(self, trajectory):
        """按x坐标排序轨迹点，从左到右"""
        if len(trajectory) == 0:
            return np.array([])

        coords = np.array(trajectory)
        sorted_indices = np.argsort(coords[:, 1])  # 按x(列)排序
        return coords[sorted_indices]

    def draw_trajectory(self, image, trajectory, color=(0, 255, 255), thickness=2):
        """绘制轨迹"""
        result = image.copy()

        if len(trajectory) == 0:
            return result

        # 排序后的轨迹
        sorted_traj = self.sort_trajectory(trajectory)

        # 绘制轨迹线
        for j in range(len(sorted_traj) - 1):
            pt1 = (int(sorted_traj[j, 1]), int(sorted_traj[j, 0]))  # (x, y)
            pt2 = (int(sorted_traj[j + 1, 1]), int(sorted_traj[j + 1, 0]))
            dist = np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)
            if dist < 15:  # 只连接近的点
                cv2.line(result, pt1, pt2, color, thickness)

        # 标记最靠右的点（终点）
        rightmost = sorted_traj[-1]
        cv2.circle(result, (int(rightmost[1]), int(rightmost[0])), 6, (0, 0, 255), -1)
        cv2.putText(result, "END", (int(rightmost[1]) - 30, int(rightmost[0]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # 标记最靠左的点（起点）
        leftmost = sorted_traj[0]
        cv2.circle(result, (int(leftmost[1]), int(leftmost[0])), 6, (0, 255, 0), -1)
        cv2.putText(result, "START", (int(leftmost[1]) + 5, int(leftmost[0]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # 显示轨迹信息
        x_span = sorted_traj[-1, 1] - sorted_traj[0, 1]
        info = f"Length: {len(sorted_traj)} px | Span: {x_span:.0f} px"
        cv2.putText(result, info, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return result

    def process_video(self, video_path, output_path, show_preview=True):
        """处理视频"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"视频: {width}x{height}, {fps}fps, {total_frames}帧")

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
                print("⚠ GUI不可用，禁用预览")

        pbar = tqdm(total=total_frames, desc="处理中")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 1. 预处理得到骨架
            skeleton = self.preprocess_frame(frame)

            # 2. 找最靠右的点
            rightmost_point = self.find_rightmost_point(skeleton)

            # 3. 从最靠右的点回溯整条轨迹
            trajectory = self.trace_back_from_point(skeleton, rightmost_point)

            # 4. 绘制
            vis_frame = self.draw_trajectory(frame, trajectory, color=(0, 255, 255), thickness=2)

            writer.write(vis_frame)

            if can_show:
                scale = min(1.0, 1280 / width, 720 / height)
                display = cv2.resize(vis_frame, None, fx=scale, fy=scale)
                cv2.imshow('Rightmost Neuron Trace (Q to quit)', display)
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
    VIDEO_PATH = r"F:\工作文件\RA\python\项目汇总\神经图像\cde7198975117a224bb23963f96cbd3d.mp4"
    OUTPUT_PATH = r"F:\工作文件\RA\python\项目汇总\神经图像\output\rightmost_traceback.mp4"

    tracker = NeuronTrackerRightmost(
        clahe_clip_limit=3.0,
        adaptive_block_size=15,
        adaptive_c=3,
        morph_iterations=2,
        min_object_size=100
    )

    tracker.process_video(
        video_path=VIDEO_PATH,
        output_path=OUTPUT_PATH,
        show_preview=True
    )
