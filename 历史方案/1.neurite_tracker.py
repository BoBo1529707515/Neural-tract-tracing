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
        print("✗ CUDA不可用，使用CPU")
except ImportError:
    HAS_CUDA = False
    print("✗ PyTorch未安装，使用CPU模式")


class NeuronTrackerGPU:
    def __init__(self,
                 clahe_clip_limit=3.0,
                 clahe_grid_size=(8, 8),
                 median_kernel=5,
                 adaptive_block_size=15,
                 adaptive_c=3,
                 morph_kernel_size=3,
                 morph_iterations=2,
                 min_object_size=100):

        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_grid_size = clahe_grid_size
        self.median_kernel = median_kernel
        self.adaptive_block_size = adaptive_block_size
        self.adaptive_c = adaptive_c
        self.morph_kernel_size = morph_kernel_size
        self.morph_iterations = morph_iterations
        self.min_object_size = min_object_size

        self.clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit,
            tileGridSize=self.clahe_grid_size
        )

    def gpu_median_filter(self, image, kernel_size=5):
        if HAS_CUDA:
            img_tensor = torch.from_numpy(image).float().to(DEVICE)
            img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)
            pad = kernel_size // 2
            img_padded = F.pad(img_tensor, (pad, pad, pad, pad), mode='reflect')
            patches = img_padded.unfold(2, kernel_size, 1).unfold(3, kernel_size, 1)
            patches = patches.contiguous().view(*patches.shape[:4], -1)
            median = patches.median(dim=-1)[0]
            return median.squeeze().cpu().numpy().astype(np.uint8)
        else:
            return cv2.medianBlur(image, kernel_size)

    def gpu_morphology(self, binary, kernel_size=3, iterations=2):
        if HAS_CUDA:
            img_tensor = torch.from_numpy(binary.astype(np.float32)).to(DEVICE)
            img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)
            pad = kernel_size // 2
            for _ in range(iterations):
                dilated = F.max_pool2d(
                    F.pad(img_tensor, (pad, pad, pad, pad), mode='constant', value=0),
                    kernel_size, stride=1
                )
                img_tensor = -F.max_pool2d(
                    F.pad(-dilated, (pad, pad, pad, pad), mode='constant', value=-255),
                    kernel_size, stride=1
                )
            return img_tensor.squeeze().cpu().numpy().astype(np.uint8)
        else:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=iterations)

    def process_frame(self, frame):
        green = frame[:, :, 1] if len(frame.shape) == 3 else frame
        enhanced = self.clahe.apply(green)
        denoised = self.gpu_median_filter(enhanced, self.median_kernel)

        binary = cv2.adaptiveThreshold(
            denoised, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            self.adaptive_block_size,
            -self.adaptive_c
        )

        closed = self.gpu_morphology(binary, self.morph_kernel_size, self.morph_iterations)

        # 修复: 使用新的API避免警告
        bool_mask = closed > 0
        cleaned = remove_small_objects(bool_mask, min_size=self.min_object_size, connectivity=1)
        optimized = (cleaned * 255).astype(np.uint8)

        skeleton = skeletonize(optimized > 0)
        return skeleton, optimized

    def visualize(self, original, skeleton, thickness=2):
        result = original.copy()
        skeleton_uint8 = (skeleton * 255).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (thickness, thickness))
        skeleton_thick = cv2.dilate(skeleton_uint8, kernel)
        result[skeleton_thick > 0] = (0, 0, 255)
        return result

    def process_video(self, video_path, output_path, show_preview=True):
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

        # 检测是否支持GUI显示
        can_show = False
        if show_preview:
            try:
                cv2.namedWindow('test', cv2.WINDOW_NORMAL)
                cv2.destroyWindow('test')
                can_show = True
            except:
                print("⚠ GUI不可用，禁用预览，只保存视频")
                can_show = False

        pbar = tqdm(total=total_frames, desc="处理中")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            skeleton, _ = self.process_frame(frame)
            vis_frame = self.visualize(frame, skeleton)

            writer.write(vis_frame)

            if can_show:
                scale = min(1.0, 1280 / width, 720 / height)
                display = cv2.resize(vis_frame, None, fx=scale, fy=scale)
                cv2.imshow('Neuron Tracking (Q to quit)', display)
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
    OUTPUT_PATH = r"/output/result_gpu.mp4"

    tracker = NeuronTrackerGPU(
        clahe_clip_limit=3.0,
        adaptive_block_size=15,
        adaptive_c=3,
        morph_iterations=2,
        min_object_size=100
    )

    tracker.process_video(
        video_path=VIDEO_PATH,
        output_path=OUTPUT_PATH,
        show_preview=True  # 会自动检测是否可用
    )
