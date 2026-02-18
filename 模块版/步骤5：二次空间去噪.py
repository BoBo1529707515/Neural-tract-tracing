import cv2
import numpy as np
import os
import time

# ╔══════════════════════════════════════════════════════════════╗
# ║  步骤5: 二次空间去噪 · 去除时空去噪后的残留小噪点                ║
# ╚══════════════════════════════════════════════════════════════╝

# === 输入输出路径 ===
INPUT_DIR = r"F:\工作文件\RA\python\项目汇总\神经图像\04_时空去噪\frames_temporal_denoised"
OUTPUT_DIR = r"F:\工作文件\RA\python\项目汇总\神经图像\05_二次空间去噪"

# === 🎛️ 去噪参数 ===
MIN_AREA = 40  # 最小保留面积（像素数）

# === 视频参数 ===
FPS = 10


# =====================================================
# 工具函数
# =====================================================

def save_image(path, img):
    """保存图像"""
    success, encoded = cv2.imencode('.png', img)
    if success:
        with open(path, 'wb') as f:
            f.write(encoded.tobytes())


def load_image(path):
    """加载图像"""
    with open(path, 'rb') as f:
        file_bytes = np.frombuffer(f.read(), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    return img


def make_even(x):
    """确保尺寸为偶数"""
    return x if x % 2 == 0 else x - 1


# =====================================================
# 核心处理函数
# =====================================================

def remove_small_objects(binary, min_area):
    """
    去除面积小于 min_area 的连通域
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

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


# =====================================================
# 主函数
# =====================================================

def main():
    print("╔" + "═" * 58 + "╗")
    print("║" + f"  步骤5: 二次空间去噪 · 面积过滤 (Area>={MIN_AREA})".center(44) + "║")
    print("╚" + "═" * 58 + "╝")

    t_start = time.time()

    # === 检查输入目录 ===
    print(f"\n📂 输入目录: {INPUT_DIR}")

    if not os.path.exists(INPUT_DIR):
        print(f"❌ 目录不存在! 请先运行步骤4（时空去噪）")
        return

    # === 获取文件列表 ===
    frame_files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith('.png')])
    num_frames = len(frame_files)

    if num_frames == 0:
        print("❌ 没有找到PNG文件!")
        return

    print(f"   找到 {num_frames} 帧")

    # === 创建输出目录 ===
    frames_final_dir = os.path.join(OUTPUT_DIR, "frames_final")
    os.makedirs(frames_final_dir, exist_ok=True)

    # === 获取图像尺寸 ===
    sample = load_image(os.path.join(INPUT_DIR, frame_files[0]))
    height, width = sample.shape
    print(f"   图像尺寸: {width}×{height}")
    print(f"   去噪阈值: MIN_AREA = {MIN_AREA}")

    # === 视频设置 ===
    video_w = make_even(width)
    video_h = make_even(height)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # 最终结果视频
    video_final_path = os.path.join(OUTPUT_DIR, "final_result.mp4")
    out_final = cv2.VideoWriter(video_final_path, fourcc, FPS, (video_w, video_h), isColor=False)

    # 对比视频
    compare_scale = 0.4
    compare_w = make_even(int(width * compare_scale) * 2)
    compare_h = make_even(int(height * compare_scale))

    video_compare_path = os.path.join(OUTPUT_DIR, "compare_before_after.mp4")
    out_compare = cv2.VideoWriter(video_compare_path, fourcc, FPS, (compare_w, compare_h), isColor=False)

    # === 处理所有帧 ===
    print(f"\n🔄 处理中...")

    stats = []
    total_removed = 0
    total_kept = 0

    for idx, filename in enumerate(frame_files):
        # 读取时空去噪后的图像
        filepath = os.path.join(INPUT_DIR, filename)
        binary = load_image(filepath)

        if binary is None:
            print(f"   ⚠️ 无法读取: {filename}")
            continue

        # 二次空间去噪
        final, removed, kept = remove_small_objects(binary, MIN_AREA)

        total_removed += removed
        total_kept += kept

        # 统计
        white_before = np.sum(binary > 0)
        white_after = np.sum(final > 0)

        stats.append({
            'frame': idx,
            'white_before': white_before,
            'white_after': white_after,
            'removed_objects': removed,
            'kept_objects': kept
        })

        # 保存最终结果
        output_filename = f"frame_{idx:04d}.png"
        save_image(os.path.join(frames_final_dir, output_filename), final)

        # 写入最终视频
        final_resized = cv2.resize(final, (video_w, video_h))
        out_final.write(final_resized)

        # 写入对比视频
        small_w = compare_w // 2

        before_small = cv2.resize(binary, (small_w, compare_h))
        after_small = cv2.resize(final, (small_w, compare_h))

        # 添加标签
        cv2.putText(before_small, f"Before #{idx} ({white_before // 1000}k)", (5, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 2)
        cv2.putText(after_small, f"Final ({white_after // 1000}k) -{removed}obj", (5, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 2)

        compare_frame = np.hstack([before_small, after_small])
        out_compare.write(compare_frame)

        # 进度显示
        if (idx + 1) % 20 == 0 or idx == num_frames - 1:
            elapsed = time.time() - t_start
            eta = elapsed / (idx + 1) * (num_frames - idx - 1)
            print(f"   [{idx + 1:3d}/{num_frames}] 去除: {removed} 对象 | 耗时: {elapsed:.1f}s | 剩余: {eta:.1f}s")

    out_final.release()
    out_compare.release()

    # === 保存统计数据 ===
    stats_path = os.path.join(OUTPUT_DIR, "final_statistics.csv")
    with open(stats_path, 'w', encoding='utf-8') as f:
        f.write("frame,white_before,white_after,removed_objects,kept_objects\n")
        for s in stats:
            f.write(f"{s['frame']},{s['white_before']},{s['white_after']},{s['removed_objects']},{s['kept_objects']}\n")

    # === 统计摘要 ===
    total_time = time.time() - t_start

    before_arr = np.array([s['white_before'] for s in stats])
    after_arr = np.array([s['white_after'] for s in stats])
    removed_arr = np.array([s['removed_objects'] for s in stats])

    print("\n" + "=" * 60)
    print("📊 二次空间去噪统计")
    print("=" * 60)

    print(f"\n   像素统计:")
    print(f"     去噪前平均: {np.mean(before_arr):,.0f} 像素/帧")
    print(f"     去噪后平均: {np.mean(after_arr):,.0f} 像素/帧")
    print(f"     减少比例:   {100 * (1 - np.mean(after_arr) / np.mean(before_arr)):.1f}%")

    print(f"\n   对象统计:")
    print(f"     总共去除:   {total_removed:,} 个小对象")
    print(f"     总共保留:   {total_kept:,} 个对象")
    print(f"     平均每帧:   去除 {np.mean(removed_arr):.0f} 个")

    print(f"\n   处理耗时: {total_time:.1f} 秒")

    # === 保存参数 ===
    params_path = os.path.join(OUTPUT_DIR, "final_params.txt")
    with open(params_path, 'w', encoding='utf-8') as f:
        f.write("二次空间去噪参数\n")
        f.write("=" * 40 + "\n")
        f.write(f"输入目录: {INPUT_DIR}\n")
        f.write(f"帧数: {num_frames}\n")
        f.write(f"尺寸: {width}×{height}\n")
        f.write(f"\n去噪参数:\n")
        f.write(f"  MIN_AREA: {MIN_AREA}\n")
        f.write(f"\n统计:\n")
        f.write(f"  去噪前平均像素: {np.mean(before_arr):,.0f}\n")
        f.write(f"  去噪后平均像素: {np.mean(after_arr):,.0f}\n")
        f.write(f"  减少比例: {100 * (1 - np.mean(after_arr) / np.mean(before_arr)):.1f}%\n")
        f.write(f"  总去除对象: {total_removed:,}\n")
        f.write(f"  总保留对象: {total_kept:,}\n")

    # === 输出文件列表 ===
    print("\n" + "=" * 60)
    print("📁 输出文件")
    print("=" * 60)
    print(f"""
    {OUTPUT_DIR}/
    │
    ├── frames_final/            # 🔑 最终处理结果
    │   ├── frame_0000.png
    │   ├── frame_0001.png
    │   └── ... ({num_frames} 张)
    │
    ├── final_result.mp4         # 最终结果视频
    ├── compare_before_after.mp4 # 对比视频
    ├── final_statistics.csv     # 统计数据
    └── final_params.txt         # 参数记录
    """)

    print("✅ 步骤5完成！frames_final/ 是最终处理结果。")


if __name__ == "__main__":
    main()
