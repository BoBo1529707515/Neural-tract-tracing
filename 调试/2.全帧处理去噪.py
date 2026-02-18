import cv2
import numpy as np
import tifffile
import os
import csv

# === 参数设置 ===
FILE_PATH = r"F:\工作文件\RA\python\项目汇总\神经图像\recover_0-72h_XY2.tif"
OUTPUT_DIR = r"/CLAHE去噪结果"

FPS = 10

# === 选定的阈值 ===
MIN_AREA = 25  # ✅ 你选定的去噪阈值

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


def make_even(x):
    return x if x % 2 == 0 else x - 1


# =====================================================
# 📐 CLAHE 预处理
# =====================================================

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

    return enhanced, clean


# =====================================================
# 🧹 去噪函数
# =====================================================

def remove_small_objects(binary, min_area):
    """去除面积小于 min_area 的连通域"""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    output = np.zeros_like(binary)
    removed = 0
    kept = 0

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            output[labels == i] = 255
            kept += 1
        else:
            removed += 1

    return output, removed, kept


# =====================================================
# 🎬 主函数
# =====================================================

def main():
    print("=" * 50)
    print("神经元图像处理 · CLAHE + 去噪 (Area>=20)")
    print("=" * 50)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(FILE_PATH, 'rb') as f:
        stack = tifffile.imread(f)

    num_frames, height, width = stack.shape
    print(f"✅ 加载: {num_frames} 帧, {width}x{height}")
    print(f"🎛️ 去噪阈值: MIN_AREA = {MIN_AREA}")

    # === 视频尺寸 ===
    scale = 0.3
    small_w = make_even(int(width * scale))
    small_h = make_even(int(height * scale))

    # 三栏对比: 原图 | 二值化 | 去噪后
    compare_w = small_w * 3
    compare_h = small_h

    # 全尺寸
    full_w = make_even(width)
    full_h = make_even(height)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # 视频1: 三栏对比
    out_compare = cv2.VideoWriter(
        os.path.join(OUTPUT_DIR, 'pipeline_compare.mp4'),
        fourcc, FPS, (compare_w, compare_h), isColor=False
    )

    # 视频2: 去噪后二值图
    out_binary = cv2.VideoWriter(
        os.path.join(OUTPUT_DIR, 'denoised_binary.mp4'),
        fourcc, FPS, (full_w, full_h), isColor=False
    )

    # 视频3: 叠加显示
    out_overlay = cv2.VideoWriter(
        os.path.join(OUTPUT_DIR, 'overlay.mp4'),
        fourcc, FPS, (full_w, full_h), isColor=True
    )

    if not all([out_compare.isOpened(), out_binary.isOpened(), out_overlay.isOpened()]):
        print("❌ 视频创建失败")
        return

    print("\n🎬 处理中...")

    stats = []

    for idx in range(num_frames):
        raw = to_8bit(stack[idx])
        enhanced, binary = preprocess_clahe(raw)
        denoised, removed, kept = remove_small_objects(binary, MIN_AREA)

        # 统计
        white_before = np.sum(binary > 0)
        white_after = np.sum(denoised > 0)

        stats.append({
            'frame': idx,
            'white_before': white_before,
            'white_after': white_after,
            'removed_objects': removed,
            'kept_objects': kept
        })

        # === 视频1: 三栏对比 ===
        raw_small = cv2.resize(raw, (small_w, small_h))
        binary_small = cv2.resize(binary, (small_w, small_h))
        denoised_small = cv2.resize(denoised, (small_w, small_h))

        cv2.putText(raw_small, f"Original #{idx}", (5, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 1)
        cv2.putText(binary_small, f"Binary ({white_before // 1000}k)", (5, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 1)
        cv2.putText(denoised_small, f"Denoised ({white_after // 1000}k)", (5, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 1)

        compare_frame = np.hstack([raw_small, binary_small, denoised_small])
        out_compare.write(compare_frame)

        # === 视频2: 去噪后全尺寸 ===
        denoised_full = cv2.resize(denoised, (full_w, full_h))
        out_binary.write(denoised_full)

        # === 视频3: 叠加 ===
        raw_full = cv2.resize(raw, (full_w, full_h))
        overlay = cv2.cvtColor(raw_full, cv2.COLOR_GRAY2BGR)
        denoised_mask = cv2.resize(denoised, (full_w, full_h))
        overlay[denoised_mask > 0] = [0, 255, 0]  # 绿色

        cv2.putText(overlay, f"Frame {idx} | Objects: {kept} | Pixels: {white_after:,}",
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

        out_overlay.write(overlay)

        # 保存关键帧
        if idx in [0, 30, 70, 100, 141]:
            save_image(os.path.join(OUTPUT_DIR, f"frame_{idx:03d}_original.png"), raw)
            save_image(os.path.join(OUTPUT_DIR, f"frame_{idx:03d}_binary.png"), binary)
            save_image(os.path.join(OUTPUT_DIR, f"frame_{idx:03d}_denoised.png"), denoised)
            save_image(os.path.join(OUTPUT_DIR, f"frame_{idx:03d}_overlay.png"), overlay)

        if (idx + 1) % 20 == 0 or idx == num_frames - 1:
            print(f"  {idx + 1}/{num_frames} | 去噪: {removed} 对象, 保留: {kept} 对象")

    out_compare.release()
    out_binary.release()
    out_overlay.release()

    # === 保存 CSV ===
    csv_path = os.path.join(OUTPUT_DIR, 'statistics.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['frame', 'white_before', 'white_after',
                                               'removed_objects', 'kept_objects'])
        writer.writeheader()
        writer.writerows(stats)

    # === 统计摘要 ===
    print("\n" + "=" * 50)
    print("📊 统计摘要")
    print("=" * 50)

    before_arr = np.array([s['white_before'] for s in stats])
    after_arr = np.array([s['white_after'] for s in stats])
    removed_arr = np.array([s['removed_objects'] for s in stats])
    kept_arr = np.array([s['kept_objects'] for s in stats])

    print(f"\n像素统计:")
    print(f"  去噪前: {np.mean(before_arr):,.0f} ± {np.std(before_arr):,.0f}")
    print(f"  去噪后: {np.mean(after_arr):,.0f} ± {np.std(after_arr):,.0f}")
    print(
        f"  减少:   {np.mean(before_arr - after_arr):,.0f} ({100 * (1 - np.mean(after_arr) / np.mean(before_arr)):.1f}%)")

    print(f"\n对象统计:")
    print(f"  平均去除: {np.mean(removed_arr):.0f} 个噪点/帧")
    print(f"  平均保留: {np.mean(kept_arr):.0f} 个对象/帧")

    print(f"\n✅ 输出文件:")
    print(f"  - pipeline_compare.mp4 (原图|二值|去噪)")
    print(f"  - denoised_binary.mp4 (去噪后二值)")
    print(f"  - overlay.mp4 (绿色叠加)")
    print(f"  - statistics.csv")
    print(f"  - 关键帧图片")
    print(f"\n📁 {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
