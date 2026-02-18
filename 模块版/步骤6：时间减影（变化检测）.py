import cv2
import numpy as np
import os
import time

# ╔══════════════════════════════════════════════════════════════╗
# ║  步骤6: 时间减影 · 仅显示变化部分                              ║
# ╚══════════════════════════════════════════════════════════════╝

# === 输入输出路径 ===
INPUT_DIR = r"F:\工作文件\RA\python\项目汇总\神经图像\05_二次空间去噪\frames_final"
OUTPUT_DIR = r"F:\工作文件\RA\python\项目汇总\神经图像\06_时间减影"

# === 视频参数 ===
FPS = 10


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
        img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    return img


def make_even(x):
    return x if x % 2 == 0 else x - 1


# =====================================================
# 主函数
# =====================================================

def main():
    print("╔" + "═" * 58 + "╗")
    print("║" + "  步骤6: 时间减影 · 仅显示变化部分".center(44) + "║")
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

    # === 创建输出目录 ===
    change_dir = os.path.join(OUTPUT_DIR, "frames_change")
    os.makedirs(change_dir, exist_ok=True)

    # === 预扫描找最小尺寸 ===
    print(f"\n🔍 预扫描尺寸...")
    min_h, min_w = float('inf'), float('inf')

    for filename in frame_files:
        img = load_image(os.path.join(INPUT_DIR, filename))
        if img is not None:
            h, w = img.shape
            min_h, min_w = min(min_h, h), min(min_w, w)

    min_h, min_w = make_even(min_h), make_even(min_w)
    print(f"   统一尺寸: {min_w}×{min_h}")

    # === 视频设置 ===
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    video_change_path = os.path.join(OUTPUT_DIR, "change_only.mp4")
    out_change = cv2.VideoWriter(video_change_path, fourcc, FPS, (min_w, min_h), isColor=False)

    # 对比视频
    compare_scale = 0.4
    single_w = int(min_w * compare_scale)
    compare_w = make_even(single_w * 3)
    compare_h = make_even(int(min_h * compare_scale))

    video_compare_path = os.path.join(OUTPUT_DIR, "compare_change.mp4")
    out_compare = cv2.VideoWriter(video_compare_path, fourcc, FPS, (compare_w, compare_h), isColor=False)

    # === 加载第一帧 ===
    prev_frame = load_image(os.path.join(INPUT_DIR, frame_files[0]))
    prev_frame = prev_frame[:min_h, :min_w]

    # === 处理 ===
    print(f"\n🔄 处理中...")

    stats = []

    for idx, filename in enumerate(frame_files):
        current = load_image(os.path.join(INPUT_DIR, filename))
        if current is None:
            continue

        # 裁剪到统一尺寸
        current = current[:min_h, :min_w]

        # 计算变化部分 = 异或 (XOR)
        # 当前有而前帧无 OR 当前无而前帧有
        change = cv2.bitwise_xor(current, prev_frame)

        # 统计
        change_pixels = np.sum(change > 0)
        current_pixels = np.sum(current > 0)

        stats.append({
            'frame': idx,
            'current_pixels': current_pixels,
            'change_pixels': change_pixels
        })

        # 保存变化图
        save_image(os.path.join(change_dir, f"frame_{idx:04d}.png"), change)

        # 写入变化视频
        out_change.write(change)

        # 对比视频: 前帧 | 当前 | 变化
        prev_small = cv2.resize(prev_frame, (single_w, compare_h))
        curr_small = cv2.resize(current, (single_w, compare_h))
        change_small = cv2.resize(change, (single_w, compare_h))

        # 添加标签
        cv2.putText(prev_small, f"Prev", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 200, 1)
        cv2.putText(curr_small, f"#{idx}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 200, 1)
        cv2.putText(change_small, f"Chg:{change_pixels // 1000}k", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, 200, 1)

        compare_frame = np.hstack([prev_small, curr_small, change_small])
        if compare_frame.shape[1] < compare_w:
            pad = compare_w - compare_frame.shape[1]
            compare_frame = np.pad(compare_frame, ((0, 0), (0, pad)), mode='constant')
        compare_frame = compare_frame[:, :compare_w]
        out_compare.write(compare_frame)

        # 更新前一帧
        prev_frame = current.copy()

        if (idx + 1) % 20 == 0 or idx == num_frames - 1:
            print(f"   [{idx + 1:3d}/{num_frames}] 变化: {change_pixels // 1000}k 像素")

    out_change.release()
    out_compare.release()

    # === 保存统计 ===
    stats_path = os.path.join(OUTPUT_DIR, "change_statistics.csv")
    with open(stats_path, 'w', encoding='utf-8') as f:
        f.write("frame,current_pixels,change_pixels\n")
        for s in stats:
            f.write(f"{s['frame']},{s['current_pixels']},{s['change_pixels']}\n")

    # === 摘要 ===
    total_time = time.time() - t_start
    change_arr = np.array([s['change_pixels'] for s in stats[1:]])

    print("\n" + "=" * 60)
    print("📊 统计")
    print("=" * 60)
    print(f"   变化像素 (平均): {np.mean(change_arr):,.0f} /帧")
    print(f"   变化像素 (总计): {np.sum(change_arr):,}")
    print(f"   最大变化帧: #{np.argmax(change_arr) + 1} ({np.max(change_arr):,} 像素)")
    print(f"   耗时: {total_time:.1f} 秒")

    print("\n📁 输出")
    print(f"""
    {OUTPUT_DIR}/
    ├── frames_change/       # 变化部分图像 (白色=变化区域)
    ├── change_only.mp4      # 变化视频
    ├── compare_change.mp4   # 对比视频 (前帧|当前|变化)
    └── change_statistics.csv
    """)

    print("✅ 完成！")


if __name__ == "__main__":
    main()
