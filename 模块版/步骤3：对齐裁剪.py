import cv2
import numpy as np
import os
import time

# ╔══════════════════════════════════════════════════════════════╗
# ║  步骤3: 对齐裁剪 · 基于白色占比检测左边界                       ║
# ╚══════════════════════════════════════════════════════════════╝

# === 输入输出路径 ===
INPUT_DIR = r"F:\工作文件\RA\python\项目汇总\神经图像\02_空间去噪\frames_denoised"
OUTPUT_DIR = r"F:\工作文件\RA\python\项目汇总\神经图像\03_对齐裁剪"

# === 🎛️ 对齐参数 ===
WHITE_RATIO_THRESHOLD = 0.20  # 白色占比阈值 (50%)
PADDING_LEFT = 10  # 左边界向左多保留的像素（安全边距）

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

def find_left_edge(binary, threshold=0.50):
    """
    找到第一个白色占比 > threshold 的列位置
    """
    height, width = binary.shape
    white_per_column = np.sum(binary > 0, axis=0) / height

    for col in range(width):
        if white_per_column[col] > threshold:
            return col

    return 0


def crop_and_align(binary, crop_start, target_width):
    """
    从 crop_start 开始裁剪到统一宽度
    """
    height, width = binary.shape

    # 裁剪
    cropped = binary[:, crop_start:]

    # 调整到目标宽度
    current_width = cropped.shape[1]

    if current_width >= target_width:
        cropped = cropped[:, :target_width]
    else:
        pad_width = target_width - current_width
        cropped = np.pad(cropped, ((0, 0), (0, pad_width)),
                         mode='constant', constant_values=0)

    return cropped


# =====================================================
# 主函数
# =====================================================

def main():
    print("╔" + "═" * 58 + "╗")
    print("║" + "  步骤3: 对齐裁剪 · 基于白色占比检测".center(48) + "║")
    print("╚" + "═" * 58 + "╝")

    t_start = time.time()

    # === 检查输入目录 ===
    print(f"\n📂 输入目录: {INPUT_DIR}")

    if not os.path.exists(INPUT_DIR):
        print(f"❌ 目录不存在! 请先运行步骤2（空间去噪）")
        return

    # === 获取文件列表 ===
    frame_files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith('.png')])
    num_frames = len(frame_files)

    if num_frames == 0:
        print("❌ 没有找到PNG文件!")
        return

    print(f"   找到 {num_frames} 帧")

    # === 创建输出目录 ===
    frames_aligned_dir = os.path.join(OUTPUT_DIR, "frames_aligned")
    os.makedirs(frames_aligned_dir, exist_ok=True)

    # === Step 1: 分析所有帧的边界 ===
    print(f"\n🔍 Step 1: 分析边界位置")
    print(f"   白色占比阈值: {WHITE_RATIO_THRESHOLD * 100:.0f}%")
    print(f"   左侧安全边距: {PADDING_LEFT} 像素")

    left_edges = []
    sample_shape = None

    for i, filename in enumerate(frame_files):
        filepath = os.path.join(INPUT_DIR, filename)
        binary = load_image(filepath)

        if binary is None:
            left_edges.append(0)
            continue

        if sample_shape is None:
            sample_shape = binary.shape

        edge = find_left_edge(binary, WHITE_RATIO_THRESHOLD)
        left_edges.append(edge)

        if (i + 1) % 30 == 0:
            print(f"   [{i + 1}/{num_frames}] 已分析")

    height, width = sample_shape
    print(f"\n   ✅ 分析完成")
    print(f"   原始尺寸: {width}×{height}")

    # === 边界统计 ===
    edges_arr = np.array(left_edges)
    print(f"\n   边界位置统计:")
    print(f"     最小: {np.min(edges_arr)} (帧 {np.argmin(edges_arr)})")
    print(f"     最大: {np.max(edges_arr)} (帧 {np.argmax(edges_arr)})")
    print(f"     平均: {np.mean(edges_arr):.1f}")
    print(f"     标准差: {np.std(edges_arr):.1f}")

    # === 计算统一裁剪参数 ===
    max_left_edge = np.max(edges_arr)
    crop_start_unified = max(0, max_left_edge - PADDING_LEFT)
    target_width = make_even(width - crop_start_unified)

    print(f"\n   统一裁剪参数:")
    print(f"     裁剪起点: 列 {crop_start_unified}")
    print(f"     目标宽度: {target_width}")
    print(f"     最终尺寸: {target_width}×{height}")

    # === Step 2: 裁剪对齐所有帧 ===
    print(f"\n🔄 Step 2: 裁剪对齐所有帧...")

    # 视频设置
    video_w = make_even(target_width)
    video_h = make_even(height)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # 对齐后视频
    video_aligned_path = os.path.join(OUTPUT_DIR, "aligned_all_frames.mp4")
    out_aligned = cv2.VideoWriter(video_aligned_path, fourcc, FPS, (video_w, video_h), isColor=False)

    # 对比视频
    compare_scale = 0.35
    orig_w_scaled = int(width * compare_scale)
    aligned_w_scaled = int(target_width * compare_scale)
    compare_w = make_even(orig_w_scaled + aligned_w_scaled)
    compare_h = make_even(int(height * compare_scale))

    video_compare_path = os.path.join(OUTPUT_DIR, "compare_before_after.mp4")
    out_compare = cv2.VideoWriter(video_compare_path, fourcc, FPS, (compare_w, compare_h), isColor=True)

    stats = []

    for idx, filename in enumerate(frame_files):
        # 读取去噪后的二值图
        filepath = os.path.join(INPUT_DIR, filename)
        binary = load_image(filepath)

        if binary is None:
            continue

        # 统一裁剪（所有帧用相同的裁剪起点）
        aligned = crop_and_align(binary, crop_start_unified, target_width)

        # 统计
        white_before = np.sum(binary > 0)
        white_after = np.sum(aligned > 0)

        stats.append({
            'frame': idx,
            'left_edge': left_edges[idx],
            'white_before': white_before,
            'white_after': white_after
        })

        # 保存对齐后的帧
        output_filename = f"frame_{idx:04d}.png"
        save_image(os.path.join(frames_aligned_dir, output_filename), aligned)

        # 写入对齐视频
        aligned_resized = cv2.resize(aligned, (video_w, video_h))
        out_aligned.write(aligned_resized)

        # 写入对比视频
        orig_small = cv2.resize(binary, (orig_w_scaled, compare_h))
        aligned_small = cv2.resize(aligned, (aligned_w_scaled, compare_h))

        # 转彩色
        orig_color = cv2.cvtColor(orig_small, cv2.COLOR_GRAY2BGR)
        aligned_color = cv2.cvtColor(aligned_small, cv2.COLOR_GRAY2BGR)

        # 在原图上画红色边界线（检测到的边界）
        edge_scaled = int(left_edges[idx] * compare_scale)
        cv2.line(orig_color, (edge_scaled, 0), (edge_scaled, compare_h), (0, 0, 255), 2)

        # 画绿色线（统一裁剪位置）
        crop_scaled = int(crop_start_unified * compare_scale)
        cv2.line(orig_color, (crop_scaled, 0), (crop_scaled, compare_h), (0, 255, 0), 2)

        # 添加标签
        cv2.putText(orig_color, f"Original #{idx}", (5, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(orig_color, f"Edge:{left_edges[idx]} Crop:{crop_start_unified}", (5, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        cv2.putText(aligned_color, f"Aligned {target_width}x{height}", (5, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # 拼接
        compare_frame = np.hstack([orig_color, aligned_color])
        # 确保宽度正确
        if compare_frame.shape[1] < compare_w:
            pad = compare_w - compare_frame.shape[1]
            compare_frame = np.pad(compare_frame, ((0, 0), (0, pad), (0, 0)),
                                   mode='constant', constant_values=0)
        compare_frame = compare_frame[:, :compare_w]

        out_compare.write(compare_frame)

        # 进度显示
        if (idx + 1) % 20 == 0 or idx == num_frames - 1:
            elapsed = time.time() - t_start
            print(f"   [{idx + 1:3d}/{num_frames}] 耗时: {elapsed:.1f}s")

    out_aligned.release()
    out_compare.release()

    # === 保存统计数据 ===
    stats_path = os.path.join(OUTPUT_DIR, "alignment_statistics.csv")
    with open(stats_path, 'w', encoding='utf-8') as f:
        f.write("frame,left_edge,white_before,white_after\n")
        for s in stats:
            f.write(f"{s['frame']},{s['left_edge']},{s['white_before']},{s['white_after']}\n")

    # === 保存参数 ===
    params_path = os.path.join(OUTPUT_DIR, "alignment_params.txt")
    with open(params_path, 'w', encoding='utf-8') as f:
        f.write("对齐裁剪参数\n")
        f.write("=" * 40 + "\n")
        f.write(f"输入目录: {INPUT_DIR}\n")
        f.write(f"帧数: {num_frames}\n")
        f.write(f"\n原始尺寸: {width}×{height}\n")
        f.write(f"对齐后尺寸: {target_width}×{height}\n")
        f.write(f"\n检测参数:\n")
        f.write(f"  白色占比阈值: {WHITE_RATIO_THRESHOLD * 100:.0f}%\n")
        f.write(f"  左侧安全边距: {PADDING_LEFT} 像素\n")
        f.write(f"  统一裁剪起点: 列 {crop_start_unified}\n")
        f.write(f"\n边界统计:\n")
        f.write(f"  最小: {np.min(edges_arr)}\n")
        f.write(f"  最大: {np.max(edges_arr)}\n")
        f.write(f"  平均: {np.mean(edges_arr):.1f}\n")

    # === 输出摘要 ===
    total_time = time.time() - t_start

    print("\n" + "=" * 60)
    print("📊 处理摘要")
    print("=" * 60)
    print(f"   原始尺寸:   {width}×{height}")
    print(f"   对齐后尺寸: {target_width}×{height}")
    print(f"   裁剪掉:     {crop_start_unified} 列 ({crop_start_unified / width * 100:.1f}%)")
    print(f"   处理耗时:   {total_time:.1f} 秒")

    print("\n" + "=" * 60)
    print("📁 输出文件")
    print("=" * 60)
    print(f"""
    {OUTPUT_DIR}/
    │
    ├── frames_aligned/          # 对齐后的二值图
    │   ├── frame_0000.png
    │   ├── frame_0001.png
    │   └── ... ({num_frames} 张)
    │
    ├── aligned_all_frames.mp4   # 对齐后视频
    ├── compare_before_after.mp4 # 对比视频
    │   (红线=检测边界, 绿线=统一裁剪位置)
    ├── alignment_statistics.csv # 各帧边界统计
    └── alignment_params.txt     # 参数记录
    """)

    print("✅ 步骤3完成！")


if __name__ == "__main__":
    main()
