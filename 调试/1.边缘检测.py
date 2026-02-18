import cv2
import numpy as np
import tifffile
import os
from skimage.morphology import skeletonize
from skan import Skeleton, summarize
import csv

# === 参数设置 ===
FILE_PATH = r"/recover_0-72h_XY2.tif"
OUTPUT_DIR = r"/预处理输出"

FPS = 10


def save_image(path, img):
    success, encoded = cv2.imencode('.png', img)
    if success:
        with open(path, 'wb') as f:
            f.write(encoded.tobytes())


def to_8bit(img):
    return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def make_even(x):
    """确保尺寸是偶数（视频编码要求）"""
    return x if x % 2 == 0 else x - 1


# =====================================================
# 📐 两种预处理方法
# =====================================================

def preprocess_clahe_morph(img):
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img)
    blur = cv2.GaussianBlur(enhanced, (3, 3), 1)

    binary = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 21, -3
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel)

    return enhanced, clean


def preprocess_tophat(img):
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    tophat = cv2.morphologyEx(enhanced, cv2.MORPH_TOPHAT, kernel)

    _, binary = cv2.threshold(tophat, 15, 255, cv2.THRESH_BINARY)

    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel2)

    return enhanced, clean


def extract_skeleton(binary):
    skeleton = skeletonize(binary > 0)
    return (skeleton * 255).astype(np.uint8)


def analyze_skeleton(skeleton):
    try:
        skel_obj = Skeleton(skeleton > 0)
        summary = summarize(skel_obj, separator='-')
        return {
            'n_branches': len(summary),
            'total_length': summary['branch-distance'].sum(),
        }
    except:
        return {'n_branches': 0, 'total_length': 0}


# =====================================================
# 🎬 主函数
# =====================================================

def main():
    print("=" * 50)
    print("神经元预处理 · 全帧处理")
    print("=" * 50)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(FILE_PATH, 'rb') as f:
        stack = tifffile.imread(f)

    num_frames, height, width = stack.shape
    print(f"✅ 加载: {num_frames} 帧, {width}x{height}")

    # 计算视频尺寸（确保是偶数）
    scale = 0.3
    small_w = make_even(int(width * scale))
    small_h = make_even(int(height * scale))
    compare_w = small_w * 3
    compare_h = small_h

    full_w = make_even(width)
    full_h = make_even(height)

    print(f"📐 对比视频尺寸: {compare_w}x{compare_h}")
    print(f"📐 骨架视频尺寸: {full_w}x{full_h}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # 对比视频 (灰度)
    out_compare = cv2.VideoWriter(
        os.path.join(OUTPUT_DIR, 'compare_all.mp4'),
        fourcc, FPS, (compare_w, compare_h), isColor=False
    )

    # 骨架视频 (彩色)
    out_skel_clahe = cv2.VideoWriter(
        os.path.join(OUTPUT_DIR, 'skeleton_clahe.mp4'),
        fourcc, FPS, (full_w, full_h), isColor=True
    )
    out_skel_tophat = cv2.VideoWriter(
        os.path.join(OUTPUT_DIR, 'skeleton_tophat.mp4'),
        fourcc, FPS, (full_w, full_h), isColor=True
    )

    if not out_compare.isOpened():
        print("❌ 对比视频创建失败")
        return
    if not out_skel_clahe.isOpened():
        print("❌ CLAHE骨架视频创建失败")
        return

    print("✅ 视频写入器初始化成功")

    stats_clahe = []
    stats_tophat = []

    print(f"\n🎬 处理中...")

    for idx in range(num_frames):
        raw = to_8bit(stack[idx])

        # CLAHE + 形态学
        enhanced1, binary1 = preprocess_clahe_morph(raw)
        skeleton1 = extract_skeleton(binary1)
        stat1 = analyze_skeleton(skeleton1)
        stats_clahe.append(stat1)

        # TopHat
        enhanced2, binary2 = preprocess_tophat(raw)
        skeleton2 = extract_skeleton(binary2)
        stat2 = analyze_skeleton(skeleton2)
        stats_tophat.append(stat2)

        # === 对比视频帧 ===
        raw_small = cv2.resize(raw, (small_w, small_h))
        bin1_small = cv2.resize(binary1, (small_w, small_h))
        bin2_small = cv2.resize(binary2, (small_w, small_h))

        cv2.putText(raw_small, f"Original #{idx}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)
        cv2.putText(bin1_small, f"CLAHE ({stat1['n_branches']})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)
        cv2.putText(bin2_small, f"TopHat ({stat2['n_branches']})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)

        compare_frame = np.hstack([raw_small, bin1_small, bin2_small])
        out_compare.write(compare_frame)

        # === 骨架叠加视频 ===
        raw_full = cv2.resize(raw, (full_w, full_h))
        skeleton1_full = cv2.resize(skeleton1, (full_w, full_h))
        skeleton2_full = cv2.resize(skeleton2, (full_w, full_h))

        overlay1 = cv2.cvtColor(raw_full, cv2.COLOR_GRAY2BGR)
        overlay1[skeleton1_full > 0] = [0, 0, 255]
        cv2.putText(overlay1, f"CLAHE #{idx} | {stat1['n_branches']} branches",
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
        out_skel_clahe.write(overlay1)

        overlay2 = cv2.cvtColor(raw_full, cv2.COLOR_GRAY2BGR)
        overlay2[skeleton2_full > 0] = [0, 255, 0]
        cv2.putText(overlay2, f"TopHat #{idx} | {stat2['n_branches']} branches",
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
        out_skel_tophat.write(overlay2)

        if (idx + 1) % 20 == 0 or idx == num_frames - 1:
            print(f"  {idx + 1}/{num_frames} ({100 * (idx + 1) // num_frames}%)")

    out_compare.release()
    out_skel_clahe.release()
    out_skel_tophat.release()

    # === 保存 CSV ===
    csv_path = os.path.join(OUTPUT_DIR, 'statistics.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['frame', 'clahe_branches', 'clahe_length',
                         'tophat_branches', 'tophat_length'])
        for i in range(num_frames):
            writer.writerow([
                i,
                stats_clahe[i]['n_branches'],
                f"{stats_clahe[i]['total_length']:.1f}",
                stats_tophat[i]['n_branches'],
                f"{stats_tophat[i]['total_length']:.1f}"
            ])

    # === 关键帧图片 ===
    key_frames = [0, 30, 70, 100, 141]
    for idx in key_frames:
        if idx >= num_frames:
            continue
        raw = to_8bit(stack[idx])
        _, binary1 = preprocess_clahe_morph(raw)
        _, binary2 = preprocess_tophat(raw)
        skeleton1 = extract_skeleton(binary1)
        skeleton2 = extract_skeleton(binary2)

        save_image(os.path.join(OUTPUT_DIR, f"frame_{idx:03d}_original.png"), raw)
        save_image(os.path.join(OUTPUT_DIR, f"frame_{idx:03d}_clahe_binary.png"), binary1)
        save_image(os.path.join(OUTPUT_DIR, f"frame_{idx:03d}_clahe_skeleton.png"), skeleton1)
        save_image(os.path.join(OUTPUT_DIR, f"frame_{idx:03d}_tophat_binary.png"), binary2)
        save_image(os.path.join(OUTPUT_DIR, f"frame_{idx:03d}_tophat_skeleton.png"), skeleton2)

    # === 统计 ===
    print("\n" + "=" * 50)
    print("📊 统计摘要")
    print("=" * 50)

    clahe_branches = [s['n_branches'] for s in stats_clahe]
    tophat_branches = [s['n_branches'] for s in stats_tophat]

    print(f"\nCLAHE+形态学:")
    print(f"  分支数: {np.mean(clahe_branches):.0f} ± {np.std(clahe_branches):.0f}")
    print(f"  CV: {np.std(clahe_branches) / np.mean(clahe_branches):.3f}")

    print(f"\nTopHat:")
    print(f"  分支数: {np.mean(tophat_branches):.0f} ± {np.std(tophat_branches):.0f}")
    print(f"  CV: {np.std(tophat_branches) / np.mean(tophat_branches):.3f}")

    print(f"\n✅ 输出: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
