import cv2
import numpy as np
import os
import time

# ╔══════════════════════════════════════════════════════════════╗
# ║  步骤4: 时空去噪 · 基于时间连续性过滤                          ║
# ╚══════════════════════════════════════════════════════════════╝

# === 输入输出路径 ===
INPUT_DIR = r"F:\工作文件\RA\python\项目汇总\神经图像\03_对齐裁剪\frames_aligned"
OUTPUT_DIR = r"F:\工作文件\RA\python\项目汇总\神经图像\04_时空去噪"

# === 🎛️ 时空去噪参数 ===
WINDOW_SIZE = 5  # 时间窗口大小（连续帧数）
MIN_FRAMES = 3  # 窗口内至少出现的帧数

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

def temporal_denoise(binary_stack, window_size, min_frames):
    """
    时间连续性去噪

    原理: 真实神经元在连续帧中持续存在，噪声点随机闪烁

    参数:
        binary_stack: 二值图像序列 (N, H, W)，值为 0/255
        window_size: 滑动窗口大小（连续帧数）
        min_frames: 窗口内至少出现的帧数

    返回:
        output: 去噪后的图像序列
    """
    n_frames, height, width = binary_stack.shape

    # 转为 0/1 便于计算
    stack_01 = (binary_stack > 0).astype(np.float32)

    # 使用 cumsum 实现快速滑动窗口求和
    cumsum = np.cumsum(stack_01, axis=0)

    output = np.zeros((n_frames, height, width), dtype=np.uint8)
    half_win = window_size // 2

    for i in range(n_frames):
        # 确定窗口范围
        start = max(0, i - half_win)
        end = min(n_frames - 1, i + half_win)

        # 计算窗口内的像素出现次数
        if start == 0:
            window_sum = cumsum[end]
        else:
            window_sum = cumsum[end] - cumsum[start - 1]

        # 只保留出现次数 >= min_frames 的像素
        output[i] = np.where(window_sum >= min_frames, 255, 0).astype(np.uint8)

    return output


# =====================================================
# 主函数
# =====================================================

def main():
    print("╔" + "═" * 58 + "╗")
    print("║" + f"  步骤4: 时空去噪 · W{WINDOW_SIZE}_M{MIN_FRAMES} (窗口{WINDOW_SIZE}帧,至少{MIN_FRAMES}帧)".center(
        44) + "║")
    print("╚" + "═" * 58 + "╝")

    t_start = time.time()

    # === 检查输入目录 ===
    print(f"\n📂 输入目录: {INPUT_DIR}")

    if not os.path.exists(INPUT_DIR):
        print(f"❌ 目录不存在! 请先运行步骤3（对齐裁剪）")
        return

    # === 获取文件列表 ===
    frame_files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith('.png')])
    num_frames = len(frame_files)

    if num_frames == 0:
        print("❌ 没有找到PNG文件!")
        return

    print(f"   找到 {num_frames} 帧")

    # === 创建输出目录 ===
    frames_output_dir = os.path.join(OUTPUT_DIR, "frames_temporal_denoised")
    os.makedirs(frames_output_dir, exist_ok=True)

    # === Step 1: 加载所有帧到内存 ===
    print(f"\n📥 Step 1: 加载所有帧到内存...")

    # 获取图像尺寸
    sample = load_image(os.path.join(INPUT_DIR, frame_files[0]))
    height, width = sample.shape
    print(f"   图像尺寸: {width}×{height}")

    # 加载所有帧
    binary_stack = np.zeros((num_frames, height, width), dtype=np.uint8)

    for idx, filename in enumerate(frame_files):
        filepath = os.path.join(INPUT_DIR, filename)
        binary_stack[idx] = load_image(filepath)

        if (idx + 1) % 30 == 0:
            print(f"   [{idx + 1}/{num_frames}] 已加载")

    print(f"   ✅ 加载完成，内存占用: {binary_stack.nbytes / 1024 ** 2:.1f} MB")

    # === Step 2: 时空去噪 ===
    print(f"\n🕐 Step 2: 时空去噪 (W{WINDOW_SIZE}_M{MIN_FRAMES})...")
    print(f"   原理: 连续{WINDOW_SIZE}帧中至少出现{MIN_FRAMES}帧才保留")

    t_denoise_start = time.time()

    denoised_stack = temporal_denoise(binary_stack, WINDOW_SIZE, MIN_FRAMES)

    t_denoise_end = time.time()
    print(f"   ✅ 去噪完成: {t_denoise_end - t_denoise_start:.2f} 秒")

    # === Step 3: 保存结果 ===
    print(f"\n💾 Step 3: 保存结果...")

    # 视频设置
    video_w = make_even(width)
    video_h = make_even(height)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # 去噪后视频
    video_denoised_path = os.path.join(OUTPUT_DIR, "temporal_denoised.mp4")
    out_denoised = cv2.VideoWriter(video_denoised_path, fourcc, FPS, (video_w, video_h), isColor=False)

    # 对比视频（去噪前 | 去噪后 | 差异）
    compare_scale = 0.35
    single_w = int(width * compare_scale)
    compare_w = make_even(single_w * 3)
    compare_h = make_even(int(height * compare_scale))

    video_compare_path = os.path.join(OUTPUT_DIR, "compare_temporal.mp4")
    out_compare = cv2.VideoWriter(video_compare_path, fourcc, FPS, (compare_w, compare_h), isColor=True)

    stats = []

    for idx in range(num_frames):
        before = binary_stack[idx]
        after = denoised_stack[idx]

        # 计算差异（被去除的像素）
        removed = cv2.bitwise_and(before, cv2.bitwise_not(after))

        # 统计
        white_before = np.sum(before > 0)
        white_after = np.sum(after > 0)
        white_removed = np.sum(removed > 0)

        stats.append({
            'frame': idx,
            'white_before': white_before,
            'white_after': white_after,
            'white_removed': white_removed
        })

        # 保存去噪后的帧
        output_filename = f"frame_{idx:04d}.png"
        save_image(os.path.join(frames_output_dir, output_filename), after)

        # 写入去噪视频
        after_resized = cv2.resize(after, (video_w, video_h))
        out_denoised.write(after_resized)

        # 写入对比视频
        before_small = cv2.resize(before, (single_w, compare_h))
        after_small = cv2.resize(after, (single_w, compare_h))
        removed_small = cv2.resize(removed, (single_w, compare_h))

        # 转彩色
        before_color = cv2.cvtColor(before_small, cv2.COLOR_GRAY2BGR)
        after_color = cv2.cvtColor(after_small, cv2.COLOR_GRAY2BGR)

        # 差异图：红色显示被去除的像素
        removed_color = cv2.cvtColor(before_small, cv2.COLOR_GRAY2BGR)
        removed_color[removed_small > 0] = [0, 0, 255]  # 红色标记被去除的
        removed_color[after_small > 0] = [0, 255, 0]  # 绿色标记保留的

        # 添加标签
        cv2.putText(before_color, f"Before #{idx}", (5, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(before_color, f"{white_before // 1000}k px", (5, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        cv2.putText(after_color, f"After W{WINDOW_SIZE}M{MIN_FRAMES}", (5, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(after_color, f"{white_after // 1000}k px", (5, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        cv2.putText(removed_color, f"Diff", (5, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(removed_color, f"R:-{white_removed // 1000}k G:kept", (5, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        # 拼接
        compare_frame = np.hstack([before_color, after_color, removed_color])
        # 确保宽度正确
        if compare_frame.shape[1] < compare_w:
            pad = compare_w - compare_frame.shape[1]
            compare_frame = np.pad(compare_frame, ((0, 0), (0, pad), (0, 0)),
                                   mode='constant', constant_values=0)
        compare_frame = compare_frame[:, :compare_w]

        out_compare.write(compare_frame)

        # 进度显示
        if (idx + 1) % 30 == 0 or idx == num_frames - 1:
            elapsed = time.time() - t_start
            print(f"   [{idx + 1:3d}/{num_frames}] 保留: {white_after // 1000}k | 去除: {white_removed // 1000}k")

    out_denoised.release()
    out_compare.release()

    # === 保存统计数据 ===
    stats_path = os.path.join(OUTPUT_DIR, "temporal_denoise_statistics.csv")
    with open(stats_path, 'w', encoding='utf-8') as f:
        f.write("frame,white_before,white_after,white_removed,retention_rate\n")
        for s in stats:
            rate = s['white_after'] / s['white_before'] * 100 if s['white_before'] > 0 else 0
            f.write(f"{s['frame']},{s['white_before']},{s['white_after']},{s['white_removed']},{rate:.1f}\n")

    # === 统计摘要 ===
    total_time = time.time() - t_start

    before_arr = np.array([s['white_before'] for s in stats])
    after_arr = np.array([s['white_after'] for s in stats])
    removed_arr = np.array([s['white_removed'] for s in stats])

    avg_retention = np.mean(after_arr) / np.mean(before_arr) * 100

    print("\n" + "=" * 60)
    print("📊 时空去噪统计")
    print("=" * 60)

    print(f"\n   参数: W{WINDOW_SIZE}_M{MIN_FRAMES}")
    print(f"   含义: 连续{WINDOW_SIZE}帧窗口，像素至少在{MIN_FRAMES}帧中出现")

    print(f"\n   像素统计:")
    print(f"     去噪前平均: {np.mean(before_arr):,.0f} 像素/帧")
    print(f"     去噪后平均: {np.mean(after_arr):,.0f} 像素/帧")
    print(f"     平均去除:   {np.mean(removed_arr):,.0f} 像素/帧")
    print(f"     保留率:     {avg_retention:.1f}%")

    print(f"\n   处理耗时: {total_time:.1f} 秒")

    # === 保存参数 ===
    params_path = os.path.join(OUTPUT_DIR, "temporal_denoise_params.txt")
    with open(params_path, 'w', encoding='utf-8') as f:
        f.write("时空去噪参数\n")
        f.write("=" * 40 + "\n")
        f.write(f"输入目录: {INPUT_DIR}\n")
        f.write(f"帧数: {num_frames}\n")
        f.write(f"尺寸: {width}×{height}\n")
        f.write(f"\n时空去噪参数:\n")
        f.write(f"  WINDOW_SIZE: {WINDOW_SIZE} (连续帧数)\n")
        f.write(f"  MIN_FRAMES: {MIN_FRAMES} (最少出现帧数)\n")
        f.write(f"\n统计:\n")
        f.write(f"  去噪前平均像素: {np.mean(before_arr):,.0f}\n")
        f.write(f"  去噪后平均像素: {np.mean(after_arr):,.0f}\n")
        f.write(f"  保留率: {avg_retention:.1f}%\n")
        f.write(f"\n原理说明:\n")
        f.write(f"  真实神经元信号在连续帧中持续存在\n")
        f.write(f"  随机噪点只会在少数帧中闪烁出现\n")
        f.write(
            f"  W{WINDOW_SIZE}_M{MIN_FRAMES}: 前后各{WINDOW_SIZE // 2}帧共{WINDOW_SIZE}帧，至少{MIN_FRAMES}帧出现才保留\n")

    # === 输出文件列表 ===
    print("\n" + "=" * 60)
    print("📁 输出文件")
    print("=" * 60)
    print(f"""
    {OUTPUT_DIR}/
    │
    ├── frames_temporal_denoised/    # 时空去噪后的图像
    │   ├── frame_0000.png
    │   ├── frame_0001.png
    │   └── ... ({num_frames} 张)
    │
    ├── temporal_denoised.mp4        # 去噪后视频
    ├── compare_temporal.mp4         # 对比视频
    │   (左:去噪前 | 中:去噪后 | 右:差异图)
    │   (红色=被去除 | 绿色=保留)
    │
    ├── temporal_denoise_statistics.csv  # 统计数据
    └── temporal_denoise_params.txt      # 参数记录
    """)

    print("✅ 步骤4完成！")


if __name__ == "__main__":
    main()
