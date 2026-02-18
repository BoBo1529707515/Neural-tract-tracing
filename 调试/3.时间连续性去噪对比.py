import cv2
import numpy as np
import tifffile
import os

# === 参数设置 ===
FILE_PATH = r"F:\工作文件\RA\python\项目汇总\神经图像\recover_0-72h_XY2.tif"
OUTPUT_DIR = r"F:\工作文件\RA\python\项目汇总\神经图像\时间去噪测试"

# === 🎛️ 可调参数 ===
WINDOW_SIZE_LIST = [3, 5, 7]  # 连续帧窗口大小
MIN_FRAMES_LIST = [2, 3, 4, 5]  # 最少出现帧数（在窗口内）

# === CLAHE 参数 ===
CLIP_LIMIT = 4.0
TILE_SIZE = (8, 8)
BLOCK_SIZE = 21
THRESHOLD_C = -3


def save_image(path, img):
    success, encoded = cv2.imencode('.png', img)
    if success:
        with open(path, 'wb') as f:
            f.write(encoded.tobytes())


def to_8bit(img):
    return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def preprocess_clahe(img):
    """CLAHE + 形态学"""
    clahe = cv2.createCLAHE(clipLimit=CLIP_LIMIT, tileGridSize=TILE_SIZE)
    enhanced = clahe.apply(img)
    blur = cv2.GaussianBlur(enhanced, (3, 3), 1)

    binary = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        BLOCK_SIZE, THRESHOLD_C
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel)

    return clean


def temporal_denoise(binary_stack, window_size, min_frames):
    """
    时间连续性去噪

    参数:
        binary_stack: 二值图像序列 (N, H, W)，值为 0/255
        window_size: 滑动窗口大小（连续帧数）
        min_frames: 窗口内至少出现的帧数

    返回:
        去噪后的图像序列
    """
    n_frames, height, width = binary_stack.shape
    output = np.zeros_like(binary_stack)

    half_win = window_size // 2

    for i in range(n_frames):
        # 确定窗口范围
        start = max(0, i - half_win)
        end = min(n_frames, i + half_win + 1)

        # 取窗口内的帧
        window = binary_stack[start:end]  # (win_size, H, W)

        # 统计每个像素在窗口内出现的次数
        count = np.sum(window > 0, axis=0)  # (H, W)

        # 只保留出现次数 >= min_frames 的像素
        output[i] = np.where(count >= min_frames, 255, 0).astype(np.uint8)

    return output


def main():
    print("=" * 50)
    print("时间连续性去噪 · 参数测试")
    print("=" * 50)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(FILE_PATH, 'rb') as f:
        stack = tifffile.imread(f)

    num_frames, height, width = stack.shape
    print(f"✅ 加载: {num_frames} 帧, {width}x{height}")

    # === 预处理所有帧 ===
    print("\n📐 预处理所有帧...")
    binary_stack = np.zeros((num_frames, height, width), dtype=np.uint8)

    for idx in range(num_frames):
        raw = to_8bit(stack[idx])
        binary_stack[idx] = preprocess_clahe(raw)
        if (idx + 1) % 30 == 0:
            print(f"  {idx + 1}/{num_frames}")

    print("✅ 预处理完成")

    # === 测试不同参数组合 ===
    test_frame = 70  # 选一个中间帧测试

    print(f"\n🧪 测试帧 {test_frame} 的不同参数组合...")

    # 原始二值图
    original_binary = binary_stack[test_frame]
    original_white = np.sum(original_binary > 0)

    results = []
    results.append(('Original', original_binary, original_white))

    print(f"\n{'窗口':<8} {'最少帧':<8} {'剩余像素':<12} {'保留率':<10}")
    print("-" * 45)

    for win_size in WINDOW_SIZE_LIST:
        for min_f in MIN_FRAMES_LIST:
            if min_f > win_size:
                continue  # 跳过不合理的组合

            # 只处理测试帧附近的窗口（快速测试）
            half = win_size // 2
            start = max(0, test_frame - half)
            end = min(num_frames, test_frame + half + 1)

            window = binary_stack[start:end]
            count = np.sum(window > 0, axis=0)
            denoised = np.where(count >= min_f, 255, 0).astype(np.uint8)

            remaining = np.sum(denoised > 0)
            ratio = remaining / original_white * 100

            label = f"W{win_size}_M{min_f}"
            results.append((label, denoised, remaining))

            print(f"{win_size:<8} {min_f:<8} {remaining:,} ({ratio:.1f}%)")

    # === 生成对比图 ===
    print("\n📊 生成对比图...")

    scale = 0.2
    small_h = int(height * scale)
    small_w = int(width * scale)

    def resize_label(img, label, pixels):
        small = cv2.resize(img, (small_w, small_h))
        cv2.rectangle(small, (0, 0), (small_w, 40), 0, -1)
        cv2.putText(small, label, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)
        cv2.putText(small, f"{pixels // 1000}k px", (5, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 200, 1)
        return small

    # 排列成网格
    imgs = [resize_label(img, label, px) for label, img, px in results]

    n_cols = 4
    while len(imgs) % n_cols != 0:
        imgs.append(np.zeros((small_h, small_w), dtype=np.uint8))

    rows = []
    for i in range(0, len(imgs), n_cols):
        rows.append(np.hstack(imgs[i:i + n_cols]))

    comparison = np.vstack(rows)
    save_image(os.path.join(OUTPUT_DIR, f"temporal_compare_frame{test_frame}.png"), comparison)

    # === 额外测试：与面积去噪对比 ===
    print("\n📊 与面积去噪对比...")

    # 面积去噪 (Area>=20)
    from cv2 import connectedComponentsWithStats
    num_labels, labels, stats, _ = connectedComponentsWithStats(original_binary, connectivity=8)
    area_denoised = np.zeros_like(original_binary)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= 20:
            area_denoised[labels == i] = 255

    # 时间去噪 (推荐参数: W5_M3)
    half = 2
    start = max(0, test_frame - half)
    end = min(num_frames, test_frame + half + 1)
    window = binary_stack[start:end]
    count = np.sum(window > 0, axis=0)
    temporal_denoised = np.where(count >= 3, 255, 0).astype(np.uint8)

    # 组合去噪（两种方法都通过）
    combined = cv2.bitwise_and(area_denoised, temporal_denoised)

    # 对比图
    raw = to_8bit(stack[test_frame])

    compare_imgs = [
        resize_label(raw, "Original", np.sum(raw > 0)),
        resize_label(original_binary, "Binary", original_white),
        resize_label(area_denoised, "Area>=20", np.sum(area_denoised > 0)),
        resize_label(temporal_denoised, "W5_M3", np.sum(temporal_denoised > 0)),
        resize_label(combined, "Combined", np.sum(combined > 0)),
    ]

    while len(compare_imgs) < 6:
        compare_imgs.append(np.zeros((small_h, small_w), dtype=np.uint8))

    method_compare = np.hstack(compare_imgs[:3])
    method_compare2 = np.hstack(compare_imgs[3:6])
    method_full = np.vstack([method_compare, method_compare2])

    save_image(os.path.join(OUTPUT_DIR, "method_comparison.png"), method_full)

    print("\n" + "=" * 50)
    print(f"📁 输出目录: {OUTPUT_DIR}")
    print("=" * 50)
    print("""
文件说明:
  - temporal_compare_frame70.png : 不同参数组合对比
  - method_comparison.png : 面积去噪 vs 时间去噪 vs 组合

参数说明:
┌────────────┬─────────────────────────────────────┐
│ 参数       │ 含义                                │
├────────────┼─────────────────────────────────────┤
│ W (窗口)   │ 连续帧数，如 W5 = 前后各2帧共5帧    │
│ M (最少)   │ 窗口内至少出现M帧才保留             │
├────────────┼─────────────────────────────────────┤
│ W3_M2      │ 连续3帧中至少2帧出现                │
│ W5_M3      │ 连续5帧中至少3帧出现（推荐）        │
│ W5_M5      │ 连续5帧全部出现（最严格）           │
└────────────┴─────────────────────────────────────┘

查看对比图后告诉我选哪个参数！
    """)


if __name__ == "__main__":
    main()
