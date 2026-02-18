import cv2
import numpy as np
import os
import time
import torch
import torch.nn.functional as F

# ╔══════════════════════════════════════════════════════════════╗
# ║  步骤7: 管状增强 · PyTorch GPU加速版                           ║
# ╚══════════════════════════════════════════════════════════════╝

# === 设备选择 ===
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️ 使用设备: {DEVICE}")

# === 输入输出路径 ===
INPUT_DIR = r"/05_二次空间去噪/frames_final"
OUTPUT_DIR = r"/07_管状增强"

# === 🎛️ Frangi参数 ===
SIGMAS = [1, 2, 3, 4, 5]
ALPHA = 0.5
BETA = 0.5
GAMMA = 15
BLACK_RIDGES = False

# === 二值化参数 ===
BINARIZE_OUTPUT = True
BINARY_THRESHOLD = 0.1

# === 批处理参数 ===
BATCH_SIZE = 8  # 一次处理多少帧（根据显存调整）

# === 视频参数 ===
FPS = 10


# =====================================================
# PyTorch Frangi滤波器
# =====================================================

def create_gaussian_kernel_2d(sigma, order, device):
    """
    创建高斯导数核
    order: (dy, dx) 导数阶数
    """
    radius = int(4 * sigma + 0.5)
    size = 2 * radius + 1

    x = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
    y = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)

    # 1D高斯及其导数
    gauss = torch.exp(-x ** 2 / (2 * sigma ** 2))
    gauss = gauss / gauss.sum()

    # 一阶导数
    gauss_d1 = -x / (sigma ** 2) * torch.exp(-x ** 2 / (2 * sigma ** 2))
    gauss_d1 = gauss_d1 / (gauss_d1.abs().sum() + 1e-10) * 2

    # 二阶导数
    gauss_d2 = (x ** 2 / sigma ** 4 - 1 / sigma ** 2) * torch.exp(-x ** 2 / (2 * sigma ** 2))
    gauss_d2 = gauss_d2 - gauss_d2.mean()  # 零均值

    # 根据order构建2D核
    dy, dx = order

    if dx == 0:
        kx = gauss
    elif dx == 1:
        kx = gauss_d1
    else:  # dx == 2
        kx = gauss_d2

    if dy == 0:
        ky = gauss
    elif dy == 1:
        ky = gauss_d1
    else:  # dy == 2
        ky = gauss_d2

    kernel = ky.unsqueeze(1) * kx.unsqueeze(0)
    return kernel.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]


def gaussian_filter_2d(img, sigma, order, device):
    """
    2D高斯滤波（支持导数）
    img: [B, 1, H, W]
    order: (dy, dx)
    """
    kernel = create_gaussian_kernel_2d(sigma, order, device)
    pad = kernel.shape[-1] // 2

    return F.conv2d(img, kernel, padding=pad)


def frangi_torch(img_batch, sigmas, alpha, beta, gamma, black_ridges, device):
    """
    PyTorch GPU加速的Frangi滤波器

    img_batch: numpy array [B, H, W] 或 [H, W]
    返回: numpy array 相同shape
    """
    # 转tensor
    if img_batch.ndim == 2:
        img_batch = img_batch[np.newaxis, ...]

    B, H, W = img_batch.shape
    img_t = torch.from_numpy(img_batch.astype(np.float32)).to(device)
    img_t = img_t.unsqueeze(1)  # [B, 1, H, W]

    response = torch.zeros_like(img_t)

    for sigma in sigmas:
        # Hessian矩阵元素
        Ixx = gaussian_filter_2d(img_t, sigma, (0, 2), device) * (sigma ** 2)
        Iyy = gaussian_filter_2d(img_t, sigma, (2, 0), device) * (sigma ** 2)
        Ixy = gaussian_filter_2d(img_t, sigma, (1, 1), device) * (sigma ** 2)

        # 特征值
        tmp = torch.sqrt((Ixx - Iyy) ** 2 + 4 * Ixy ** 2 + 1e-10)
        lambda1 = 0.5 * (Ixx + Iyy + tmp)
        lambda2 = 0.5 * (Ixx + Iyy - tmp)

        # 排序 |lambda1| <= |lambda2|
        idx = torch.abs(lambda1) > torch.abs(lambda2)
        l1_new = torch.where(idx, lambda2, lambda1)
        l2_new = torch.where(idx, lambda1, lambda2)
        lambda1, lambda2 = l1_new, l2_new

        # Frangi响应
        Rb = lambda1 / (lambda2 + 1e-10)
        S = torch.sqrt(lambda1 ** 2 + lambda2 ** 2)

        vesselness = torch.exp(-Rb ** 2 / (2 * alpha ** 2)) * \
                     (1 - torch.exp(-S ** 2 / (2 * gamma ** 2)))

        # 条件过滤
        if black_ridges:
            vesselness = torch.where(lambda2 > 0, torch.zeros_like(vesselness), vesselness)
        else:
            vesselness = torch.where(lambda2 < 0, torch.zeros_like(vesselness), vesselness)

        response = torch.maximum(response, vesselness)

    # 转回numpy
    result = response.squeeze(1).cpu().numpy()

    if B == 1:
        return result[0]
    return result


# =====================================================
# 工具函数
# =====================================================

def save_image(path, img):
    success, encoded = cv2.imencode('.png', img)
    if success:
        with open(path, 'wb') as f:
            f.write(encoded.tobytes())


def load_image(path):
    with open(path, 'rb') as f:
        file_bytes = np.frombuffer(f.read(), dtype=np.uint8)
        return cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)


def make_even(x):
    return x if x % 2 == 0 else x - 1


# =====================================================
# 主函数
# =====================================================

def main():
    print("╔" + "═" * 58 + "╗")
    print("║" + "  步骤7: 管状增强 · PyTorch GPU加速".center(44) + "║")
    print("╚" + "═" * 58 + "╝")

    t_start = time.time()

    # === 检查输入 ===
    if not os.path.exists(INPUT_DIR):
        print(f"❌ 目录不存在: {INPUT_DIR}")
        return

    frame_files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith('.png')])
    num_frames = len(frame_files)

    if num_frames == 0:
        print("❌ 没有找到PNG文件!")
        return

    print(f"\n📂 输入: {INPUT_DIR}")
    print(f"   找到 {num_frames} 帧")
    print(f"   设备: {DEVICE}")
    if DEVICE.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")

    # === 创建输出目录 ===
    enhanced_dir = os.path.join(OUTPUT_DIR, "frames_enhanced")
    binary_dir = os.path.join(OUTPUT_DIR, "frames_enhanced_binary") if BINARIZE_OUTPUT else None

    os.makedirs(enhanced_dir, exist_ok=True)
    if binary_dir:
        os.makedirs(binary_dir, exist_ok=True)

    # === 获取图像尺寸 ===
    sample = load_image(os.path.join(INPUT_DIR, frame_files[0]))
    height, width = sample.shape
    print(f"   尺寸: {width}×{height}")
    print(f"   批大小: {BATCH_SIZE}")

    # === 打印参数 ===
    print(f"\n🔧 Frangi参数: sigmas={SIGMAS}, gamma={GAMMA}")

    # === 视频设置 ===
    video_w, video_h = make_even(width), make_even(height)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    video_enhanced_path = os.path.join(OUTPUT_DIR, "enhanced_all_frames.mp4")
    out_enhanced = cv2.VideoWriter(video_enhanced_path, fourcc, FPS, (video_w, video_h), isColor=False)

    # === 批量处理 ===
    print(f"\n🚀 处理中...")

    stats = []
    num_batches = (num_frames + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, num_frames)
        batch_files = frame_files[start_idx:end_idx]

        # 加载批次图像
        batch_images = []
        for filename in batch_files:
            img = load_image(os.path.join(INPUT_DIR, filename))
            if img is not None:
                batch_images.append(img.astype(np.float64) / 255.0)

        if not batch_images:
            continue

        batch_array = np.stack(batch_images, axis=0)  # [B, H, W]

        # GPU批量Frangi
        t_batch = time.time()
        enhanced_batch = frangi_torch(batch_array, SIGMAS, ALPHA, BETA, GAMMA, BLACK_RIDGES, DEVICE)

        # 处理结果
        for i, (filename, enhanced) in enumerate(zip(batch_files, enhanced_batch)):
            idx = start_idx + i

            # 归一化
            enhanced_norm = (enhanced / (enhanced.max() + 1e-8) * 255).astype(np.uint8)

            # 统计
            white_before = np.sum(batch_images[i] > 0)

            stat = {'frame': idx, 'white_before': white_before}

            # 保存增强结果
            save_image(os.path.join(enhanced_dir, f"frame_{idx:04d}.png"), enhanced_norm)

            # 二值化
            if BINARIZE_OUTPUT:
                enhanced_binary = (enhanced > BINARY_THRESHOLD).astype(np.uint8) * 255
                save_image(os.path.join(binary_dir, f"frame_{idx:04d}.png"), enhanced_binary)
                stat['white_after'] = np.sum(enhanced_binary > 0)

            stats.append(stat)

            # 视频
            enhanced_resized = cv2.resize(enhanced_norm, (video_w, video_h))
            out_enhanced.write(enhanced_resized)

        # 进度
        elapsed = time.time() - t_start
        frames_done = end_idx
        fps_rate = frames_done / elapsed
        eta = (num_frames - frames_done) / fps_rate if fps_rate > 0 else 0

        print(f"   [{frames_done:3d}/{num_frames}] {fps_rate:.1f} 帧/秒 | 剩余: {eta:.1f}s")

        # 清理GPU缓存
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()

    out_enhanced.release()

    # === 保存统计 ===
    stats_path = os.path.join(OUTPUT_DIR, "enhancement_statistics.csv")
    with open(stats_path, 'w', encoding='utf-8') as f:
        header = "frame,white_before" + (",white_after" if BINARIZE_OUTPUT else "") + "\n"
        f.write(header)
        for s in stats:
            line = f"{s['frame']},{s['white_before']}"
            if BINARIZE_OUTPUT:
                line += f",{s.get('white_after', 0)}"
            f.write(line + "\n")

    # === 摘要 ===
    total_time = time.time() - t_start

    print("\n" + "=" * 60)
    print("📊 统计")
    print("=" * 60)
    print(f"   设备:     {DEVICE} {'(' + torch.cuda.get_device_name(0) + ')' if DEVICE.type == 'cuda' else ''}")
    print(f"   总耗时:   {total_time:.1f} 秒")
    print(f"   平均速度: {num_frames / total_time:.1f} 帧/秒")
    print(f"   每帧耗时: {total_time / num_frames * 1000:.0f} 毫秒")

    print("\n📁 输出")
    print(f"""
    {OUTPUT_DIR}/
    ├── frames_enhanced/         # Frangi增强结果
    ├── frames_enhanced_binary/  # 二值化结果
    ├── enhanced_all_frames.mp4
    └── enhancement_statistics.csv
    """)

    print("✅ 完成！")


if __name__ == "__main__":
    main()
