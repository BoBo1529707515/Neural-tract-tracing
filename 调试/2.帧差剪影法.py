import cv2
import numpy as np
import tifffile
import os
from skimage.morphology import skeletonize, dilation, disk
import csv

# === 参数设置 ===
FILE_PATH = r"/recover_0-72h_XY2.tif"
OUTPUT_DIR = r"/帧差分析"

FPS = 10
FRAME_SKIP = 1  # 帧间隔，1=相邻帧，5=隔5帧比较


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
# 📐 预处理 (沿用之前的CLAHE+形态学)
# =====================================================

def preprocess(img):
    """CLAHE + 形态学"""
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


def extract_skeleton(binary):
    skeleton = skeletonize(binary > 0)
    return (skeleton * 255).astype(np.uint8)


# =====================================================
# 📊 帧差法
# =====================================================

def frame_difference(img1, img2):
    """
    计算帧差
    返回：新增区域、消失区域、变化幅度
    """
    # 确保是同类型
    img1 = img1.astype(np.int16)
    img2 = img2.astype(np.int16)

    diff = img2 - img1

    # 新增区域（当前帧有，前一帧没有）
    growth = np.clip(diff, 0, 255).astype(np.uint8)

    # 消失区域（前一帧有，当前帧没有）
    retract = np.clip(-diff, 0, 255).astype(np.uint8)

    return growth, retract


def analyze_growth(skeleton_prev, skeleton_curr):
    """
    分析骨架生长
    """
    growth, retract = frame_difference(skeleton_prev, skeleton_curr)

    # 去小噪点
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    growth_clean = cv2.morphologyEx(growth, cv2.MORPH_OPEN, kernel)
    retract_clean = cv2.morphologyEx(retract, cv2.MORPH_OPEN, kernel)

    # 统计
    growth_pixels = np.sum(growth_clean > 0)
    retract_pixels = np.sum(retract_clean > 0)
    net_growth = growth_pixels - retract_pixels

    return {
        'growth_img': growth_clean,
        'retract_img': retract_clean,
        'growth_pixels': growth_pixels,
        'retract_pixels': retract_pixels,
        'net_growth': net_growth
    }


def create_color_overlay(raw, skeleton, growth, retract):
    """
    创建彩色叠加图
    - 灰色: 原图
    - 白色: 当前骨架
    - 绿色: 生长区域
    - 红色: 回缩区域
    """
    overlay = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)

    # 骨架用白色
    overlay[skeleton > 0] = [255, 255, 255]

    # 膨胀生长/回缩区域使其更可见
    growth_dilated = dilation(growth > 0, disk(2))
    retract_dilated = dilation(retract > 0, disk(2))

    # 生长区域用绿色
    overlay[growth_dilated] = [0, 255, 0]

    # 回缩区域用红色
    overlay[retract_dilated] = [0, 0, 255]

    return overlay


# =====================================================
# 🎬 主函数
# =====================================================

def main():
    print("=" * 50)
    print("帧差法 · 生长区域检测")
    print("=" * 50)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(FILE_PATH, 'rb') as f:
        stack = tifffile.imread(f)

    num_frames, height, width = stack.shape
    print(f"✅ 加载: {num_frames} 帧, {width}x{height}")
    print(f"📊 帧间隔: {FRAME_SKIP}")

    # 视频尺寸
    full_w = make_even(width)
    full_h = make_even(height)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # 输出视频：生长检测叠加
    out_growth = cv2.VideoWriter(
        os.path.join(OUTPUT_DIR, f'growth_detection_skip{FRAME_SKIP}.mp4'),
        fourcc, FPS, (full_w, full_h), isColor=True
    )

    if not out_growth.isOpened():
        print("❌ 视频创建失败")
        return

    # 统计数据
    stats = []

    # 预处理第一帧
    print("\n🎬 处理中...")

    raw_prev = to_8bit(stack[0])
    _, binary_prev = preprocess(raw_prev)
    skeleton_prev = extract_skeleton(binary_prev)

    for idx in range(FRAME_SKIP, num_frames, 1):
        # 当前帧
        raw_curr = to_8bit(stack[idx])
        _, binary_curr = preprocess(raw_curr)
        skeleton_curr = extract_skeleton(binary_curr)

        # 比较帧
        compare_idx = idx - FRAME_SKIP
        raw_compare = to_8bit(stack[compare_idx])
        _, binary_compare = preprocess(raw_compare)
        skeleton_compare = extract_skeleton(binary_compare)

        # 帧差分析
        result = analyze_growth(skeleton_compare, skeleton_curr)

        stats.append({
            'frame': idx,
            'compare_frame': compare_idx,
            'growth_pixels': result['growth_pixels'],
            'retract_pixels': result['retract_pixels'],
            'net_growth': result['net_growth']
        })

        # 生成叠加图
        overlay = create_color_overlay(
            raw_curr, skeleton_curr,
            result['growth_img'], result['retract_img']
        )

        # 添加信息
        cv2.putText(overlay, f"Frame {idx} vs {compare_idx}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
        cv2.putText(overlay, f"Growth: {result['growth_pixels']} px (green)", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(overlay, f"Retract: {result['retract_pixels']} px (red)", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(overlay, f"Net: {result['net_growth']:+d} px", (20, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # resize 并写入
        overlay_resized = cv2.resize(overlay, (full_w, full_h))
        out_growth.write(overlay_resized)

        # 保存关键帧
        if idx in [10, 30, 50, 70, 100, 140]:
            save_image(os.path.join(OUTPUT_DIR, f"growth_frame_{idx:03d}.png"), overlay)

        if (idx + 1) % 20 == 0 or idx == num_frames - 1:
            print(f"  {idx + 1}/{num_frames} | 净生长: {result['net_growth']:+d} px")

    out_growth.release()

    # === 保存 CSV ===
    csv_path = os.path.join(OUTPUT_DIR, f'growth_stats_skip{FRAME_SKIP}.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['frame', 'compare_frame',
                                               'growth_pixels', 'retract_pixels', 'net_growth'])
        writer.writeheader()
        writer.writerows(stats)

    # === 累积生长曲线 ===
    cumulative_growth = np.cumsum([s['net_growth'] for s in stats])

    # 简单文本图表
    print("\n" + "=" * 50)
    print("📈 累积生长趋势")
    print("=" * 50)

    sample_points = [0, len(stats) // 4, len(stats) // 2, 3 * len(stats) // 4, len(stats) - 1]
    for i in sample_points:
        if i < len(stats):
            frame = stats[i]['frame']
            cum = cumulative_growth[i]
            bar = '█' * max(0, int(cum / 500))
            print(f"Frame {frame:3d}: {cum:+8.0f} px {bar}")

    # === 统计摘要 ===
    print("\n" + "=" * 50)
    print("📊 统计摘要")
    print("=" * 50)

    growth_arr = np.array([s['growth_pixels'] for s in stats])
    retract_arr = np.array([s['retract_pixels'] for s in stats])
    net_arr = np.array([s['net_growth'] for s in stats])

    print(f"\n每帧平均:")
    print(f"  生长: {np.mean(growth_arr):.0f} ± {np.std(growth_arr):.0f} px")
    print(f"  回缩: {np.mean(retract_arr):.0f} ± {np.std(retract_arr):.0f} px")
    print(f"  净增: {np.mean(net_arr):.0f} ± {np.std(net_arr):.0f} px")

    print(f"\n总计 ({num_frames} 帧):")
    print(f"  总生长: {np.sum(growth_arr):,} px")
    print(f"  总回缩: {np.sum(retract_arr):,} px")
    print(f"  净增长: {cumulative_growth[-1]:+,.0f} px")

    print(f"\n✅ 输出文件:")
    print(f"  - growth_detection_skip{FRAME_SKIP}.mp4")
    print(f"  - growth_stats_skip{FRAME_SKIP}.csv")
    print(f"  - growth_frame_xxx.png (关键帧)")
    print(f"\n📁 {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
