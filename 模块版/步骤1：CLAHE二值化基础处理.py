import cv2
import numpy as np
import tifffile
import os
import time

# ╔══════════════════════════════════════════════════════════════╗
# ║  步骤1: CLAHE 二值化 · 生成所有帧的基础数据                    ║
# ╚══════════════════════════════════════════════════════════════╝

# === 输入输出路径 ===
INPUT_FILE = r"F:\工作文件\RA\python\项目汇总\神经图像\recover_0-72h_XY2.tif"
OUTPUT_DIR = r"F:\工作文件\RA\python\项目汇总\神经图像\01_二值化基础"

# === CLAHE 参数 ===
CLIP_LIMIT = 4.0
TILE_SIZE = (8, 8)

# === 自适应阈值参数 ===
BLOCK_SIZE = 21
THRESHOLD_C = -3

# === 视频参数 ===
FPS = 10


# =====================================================
# 工具函数
# =====================================================

def save_image(path, img):
    """保存图像（避免中文路径问题）"""
    success, encoded = cv2.imencode('.png', img)
    if success:
        with open(path, 'wb') as f:
            f.write(encoded.tobytes())


def to_8bit(img):
    """转换为8位图像"""
    return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def make_even(x):
    """确保尺寸为偶数（视频编码要求）"""
    return x if x % 2 == 0 else x - 1


# =====================================================
# 核心处理函数
# =====================================================

def clahe_binarize(img_8bit):
    """
    CLAHE增强 + 自适应阈值二值化 + 形态学清理

    参数:
        img_8bit: 8位灰度图像

    返回:
        enhanced: CLAHE增强后的图像
        binary: 二值化结果 (0/255)
    """
    # Step 1: CLAHE 增强
    clahe = cv2.createCLAHE(clipLimit=CLIP_LIMIT, tileGridSize=TILE_SIZE)
    enhanced = clahe.apply(img_8bit)

    # Step 2: 轻微模糊去噪
    blur = cv2.GaussianBlur(enhanced, (3, 3), 1)

    # Step 3: 自适应阈值二值化
    binary = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        BLOCK_SIZE, THRESHOLD_C
    )

    # Step 4: 形态学操作（开运算去噪 + 闭运算填洞）
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    return enhanced, binary


# =====================================================
# 主函数
# =====================================================

def main():
    print("╔" + "═" * 58 + "╗")
    print("║" + "  步骤1: CLAHE 二值化 · 生成基础数据".center(50) + "║")
    print("╚" + "═" * 58 + "╝")

    # === 创建输出目录结构 ===
    frames_dir = os.path.join(OUTPUT_DIR, "frames_binary")
    original_dir = os.path.join(OUTPUT_DIR, "frames_original")

    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(original_dir, exist_ok=True)

    # === 读取原始数据 ===
    print(f"\n📂 读取: {INPUT_FILE}")
    t_start = time.time()

    with open(INPUT_FILE, 'rb') as f:
        stack = tifffile.imread(f)

    num_frames, height, width = stack.shape
    print(f"✅ 加载完成: {num_frames} 帧, {width}×{height} 像素")

    # === 视频设置 ===
    full_w = make_even(width)
    full_h = make_even(height)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # 视频1: 二值化结果
    video_binary_path = os.path.join(OUTPUT_DIR, "binary_all_frames.mp4")
    out_binary = cv2.VideoWriter(video_binary_path, fourcc, FPS, (full_w, full_h), isColor=False)

    # 视频2: 对比视频（原图 | 二值化）
    compare_w = make_even(int(width * 0.4)) * 2  # 两栏
    compare_h = make_even(int(height * 0.4))
    video_compare_path = os.path.join(OUTPUT_DIR, "compare_original_binary.mp4")
    out_compare = cv2.VideoWriter(video_compare_path, fourcc, FPS, (compare_w, compare_h), isColor=False)

    if not out_binary.isOpened() or not out_compare.isOpened():
        print("❌ 视频创建失败!")
        return

    # === 处理所有帧 ===
    print(f"\n🔄 处理中...")
    print(f"   CLAHE: clipLimit={CLIP_LIMIT}, tileSize={TILE_SIZE}")
    print(f"   阈值:  blockSize={BLOCK_SIZE}, C={THRESHOLD_C}")
    print()

    stats = []

    for idx in range(num_frames):
        # 转换为8位
        raw_8bit = to_8bit(stack[idx])

        # CLAHE + 二值化
        enhanced, binary = clahe_binarize(raw_8bit)

        # 统计白色像素
        white_pixels = np.sum(binary > 0)
        stats.append(white_pixels)

        # === 保存PNG图片 ===
        # 原图
        save_image(os.path.join(original_dir, f"frame_{idx:04d}.png"), raw_8bit)
        # 二值图
        save_image(os.path.join(frames_dir, f"frame_{idx:04d}.png"), binary)

        # === 写入视频 ===
        # 二值化视频
        binary_resized = cv2.resize(binary, (full_w, full_h))
        out_binary.write(binary_resized)

        # 对比视频
        small_w = compare_w // 2
        small_h = compare_h

        orig_small = cv2.resize(raw_8bit, (small_w, small_h))
        bin_small = cv2.resize(binary, (small_w, small_h))

        # 添加标签
        cv2.putText(orig_small, f"Original #{idx}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)
        cv2.putText(bin_small, f"Binary ({white_pixels // 1000}k)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)

        compare_frame = np.hstack([orig_small, bin_small])
        out_compare.write(compare_frame)

        # 进度显示
        if (idx + 1) % 20 == 0 or idx == num_frames - 1:
            elapsed = time.time() - t_start
            eta = elapsed / (idx + 1) * (num_frames - idx - 1)
            print(f"   [{idx + 1:3d}/{num_frames}] 白像素: {white_pixels:,} | 耗时: {elapsed:.1f}s | 剩余: {eta:.1f}s")

    out_binary.release()
    out_compare.release()

    # === 统计摘要 ===
    total_time = time.time() - t_start
    stats_arr = np.array(stats)

    print("\n" + "=" * 60)
    print("📊 统计摘要")
    print("=" * 60)
    print(f"   总帧数:     {num_frames}")
    print(f"   图像尺寸:   {width}×{height}")
    print(f"   处理耗时:   {total_time:.1f} 秒 ({total_time / num_frames:.2f} 秒/帧)")
    print(f"\n   白像素统计:")
    print(f"     平均:     {np.mean(stats_arr):,.0f}")
    print(f"     标准差:   {np.std(stats_arr):,.0f}")
    print(f"     最小:     {np.min(stats_arr):,} (帧 {np.argmin(stats_arr)})")
    print(f"     最大:     {np.max(stats_arr):,} (帧 {np.argmax(stats_arr)})")

    # === 保存统计数据 ===
    stats_path = os.path.join(OUTPUT_DIR, "frame_statistics.csv")
    with open(stats_path, 'w', encoding='utf-8') as f:
        f.write("frame,white_pixels\n")
        for i, wp in enumerate(stats):
            f.write(f"{i},{wp}\n")

    # === 保存处理参数 ===
    params_path = os.path.join(OUTPUT_DIR, "processing_params.txt")
    with open(params_path, 'w', encoding='utf-8') as f:
        f.write("CLAHE 二值化参数\n")
        f.write("=" * 40 + "\n")
        f.write(f"输入文件: {INPUT_FILE}\n")
        f.write(f"帧数: {num_frames}\n")
        f.write(f"尺寸: {width}×{height}\n")
        f.write(f"\nCLAHE参数:\n")
        f.write(f"  clipLimit: {CLIP_LIMIT}\n")
        f.write(f"  tileGridSize: {TILE_SIZE}\n")
        f.write(f"\n自适应阈值参数:\n")
        f.write(f"  blockSize: {BLOCK_SIZE}\n")
        f.write(f"  C: {THRESHOLD_C}\n")
        f.write(f"\n处理时间: {total_time:.1f} 秒\n")

    # === 输出文件列表 ===
    print("\n" + "=" * 60)
    print("📁 输出文件")
    print("=" * 60)
    print(f"""
    {OUTPUT_DIR}/
    │
    ├── frames_binary/           # 二值化图片 (后续处理用)
    │   ├── frame_0000.png
    │   ├── frame_0001.png
    │   └── ... ({num_frames} 张)
    │
    ├── frames_original/         # 原始8位图片 (参考用)
    │   ├── frame_0000.png
    │   ├── frame_0001.png
    │   └── ... ({num_frames} 张)
    │
    ├── binary_all_frames.mp4    # 二值化视频
    ├── compare_original_binary.mp4  # 对比视频
    ├── frame_statistics.csv     # 每帧白像素统计
    └── processing_params.txt    # 处理参数记录
    """)

    print("✅ 步骤1完成！后续处理直接读取 frames_binary/ 中的PNG文件。")


if __name__ == "__main__":
    main()
