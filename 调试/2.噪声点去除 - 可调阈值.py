import cv2
import numpy as np
import tifffile
import os

# === 参数设置 ===
FILE_PATH = r"/recover_0-72h_XY2.tif"
OUTPUT_DIR = r"/去噪测试"

# === 🎛️ 测试的面积阈值列表 ===
MIN_AREA_LIST = [5, 10, 20, 50, 100, 200, 500]

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


def remove_small_objects(binary, min_area):
    """去除面积小于 min_area 的连通域"""
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    output = np.zeros_like(binary)
    removed_count = 0
    kept_count = 0

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            output[labels == i] = 255
            kept_count += 1
        else:
            removed_count += 1

    return output, removed_count, kept_count


def main():
    print("=" * 50)
    print("噪声去除 · 阈值对比图生成")
    print("=" * 50)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(FILE_PATH, 'rb') as f:
        stack = tifffile.imread(f)

    num_frames, height, width = stack.shape
    print(f"✅ 加载: {num_frames} 帧, {width}x{height}")

    # 选几个代表帧
    test_frames = [0, 50, 100]

    for frame_idx in test_frames:
        print(f"\n处理帧 {frame_idx}...")

        raw = to_8bit(stack[frame_idx])
        binary = preprocess_clahe(raw)

        # === 生成大对比图 ===
        # 布局: 4列 x 2行
        # 第1行: 原图, Binary原图, Area>=5, Area>=10
        # 第2行: Area>=20, Area>=50, Area>=100, Area>=200

        scale = 0.25
        small_h = int(height * scale)
        small_w = int(width * scale)

        def resize_and_label(img, label):
            """缩放并添加标签"""
            small = cv2.resize(img, (small_w, small_h))
            # 添加黑色背景条
            cv2.rectangle(small, (0, 0), (small_w, 35), 0, -1)
            cv2.putText(small, label, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)
            return small

        # 准备所有图像
        imgs = []

        # 原图
        imgs.append(resize_and_label(raw, f"Original #{frame_idx}"))

        # 原始二值化
        white_count = np.sum(binary > 0)
        imgs.append(resize_and_label(binary, f"Binary ({white_count // 1000}k)"))

        # 不同阈值
        for min_area in MIN_AREA_LIST:
            denoised, removed, kept = remove_small_objects(binary, min_area)
            remaining = np.sum(denoised > 0)
            label = f"Area>={min_area} ({remaining // 1000}k)"
            imgs.append(resize_and_label(denoised, label))

        # 排列成网格 (2行 x 多列)
        n_cols = 4
        n_rows = (len(imgs) + n_cols - 1) // n_cols

        # 补齐空白
        while len(imgs) < n_rows * n_cols:
            imgs.append(np.zeros((small_h, small_w), dtype=np.uint8))

        rows = []
        for r in range(n_rows):
            row_imgs = imgs[r * n_cols: (r + 1) * n_cols]
            rows.append(np.hstack(row_imgs))

        comparison = np.vstack(rows)

        # 保存
        output_path = os.path.join(OUTPUT_DIR, f"threshold_compare_frame_{frame_idx:03d}.png")
        save_image(output_path, comparison)
        print(f"  ✅ 保存: {output_path}")

        # === 额外：保存单独的去噪结果 ===
        for min_area in [20, 50, 100]:
            denoised, _, _ = remove_small_objects(binary, min_area)
            path = os.path.join(OUTPUT_DIR, f"frame_{frame_idx:03d}_area{min_area}.png")
            save_image(path, denoised)

    print("\n" + "=" * 50)
    print("📁 输出目录:", OUTPUT_DIR)
    print("=" * 50)
    print("""
请查看以下文件:
  - threshold_compare_frame_000.png
  - threshold_compare_frame_050.png  
  - threshold_compare_frame_100.png

每张图显示不同阈值的效果:
┌─────────────┬─────────────┬─────────────┬─────────────┐
│ Original    │ Binary      │ Area>=5     │ Area>=10    │
├─────────────┼─────────────┼─────────────┼─────────────┤
│ Area>=20    │ Area>=50    │ Area>=100   │ Area>=200   │
├─────────────┼─────────────┼─────────────┼─────────────┤
│ Area>=500   │             │             │             │
└─────────────┴─────────────┴─────────────┴─────────────┘

括号里是剩余白像素数量 (k = 千)

选好阈值后告诉我数值！
    """)


if __name__ == "__main__":
    main()
