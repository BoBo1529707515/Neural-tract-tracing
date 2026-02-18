import cv2
import numpy as np
import os
import time
from skimage.morphology import skeletonize, remove_small_objects
from scipy import ndimage

# ╔══════════════════════════════════════════════════════════════╗
# ║  步骤8: 骨架提取 · 提取管状结构中心线                          ║
# ╚══════════════════════════════════════════════════════════════╝

# === 输入输出路径 ===
INPUT_DIR = r"/07_管状增强/frames_enhanced_binary"
OUTPUT_DIR = r"/08_骨架提取"

# === 🎛️ 骨架化参数 ===
PRUNE_LENGTH = 5  # 修剪短分支的最小长度（像素），0=不修剪
MIN_SKELETON_SIZE = 10  # 最小骨架连通域大小

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
        return cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)


def make_even(x):
    return x if x % 2 == 0 else x - 1


# =====================================================
# 骨架处理函数
# =====================================================

def extract_skeleton(binary):
    """
    提取骨架

    参数:
        binary: 二值图 (0/255)

    返回:
        skeleton: 骨架图 (0/255)
    """
    # 转为布尔图
    binary_bool = binary > 0

    # 骨架化 (Zhang-Suen算法)
    skeleton = skeletonize(binary_bool)

    return (skeleton * 255).astype(np.uint8)


def find_endpoints_and_branches(skeleton):
    """
    找到骨架的端点和分支点

    端点: 只有1个邻居
    分支点: 有3个或更多邻居
    """
    # 卷积核计算邻居数
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)

    skel_bool = skeleton > 0
    neighbor_count = cv2.filter2D(skel_bool.astype(np.uint8), -1, kernel)

    # 端点: 邻居数=1
    endpoints = np.logical_and(skel_bool, neighbor_count == 1)

    # 分支点: 邻居数>=3
    branch_points = np.logical_and(skel_bool, neighbor_count >= 3)

    return endpoints, branch_points


def prune_short_branches(skeleton, min_length):
    """
    修剪短分支（去除骨架上的毛刺）

    参数:
        skeleton: 骨架图
        min_length: 最小分支长度

    返回:
        pruned: 修剪后的骨架
    """
    if min_length <= 0:
        return skeleton

    skel = skeleton.copy()

    for _ in range(min_length):
        endpoints, _ = find_endpoints_and_branches(skel)

        # 移除端点
        skel[endpoints] = 0

    # 重新骨架化确保连通性
    if np.any(skel > 0):
        skel = extract_skeleton(skel)

    return skel


def remove_small_skeletons(skeleton, min_size):
    """
    移除过小的骨架连通域
    """
    if min_size <= 0:
        return skeleton

    skel_bool = skeleton > 0

    # 连通域分析
    labeled, num_features = ndimage.label(skel_bool)

    # 计算每个连通域大小
    sizes = ndimage.sum(skel_bool, labeled, range(1, num_features + 1))

    # 移除小连通域
    mask = np.zeros_like(skel_bool)
    for i, size in enumerate(sizes):
        if size >= min_size:
            mask[labeled == (i + 1)] = True

    return (mask * 255).astype(np.uint8)


def create_overlay(original, skeleton):
    """
    创建叠加可视化图

    原图: 灰色
    骨架: 红色
    端点: 绿色
    分支点: 蓝色
    """
    h, w = original.shape
    overlay = np.zeros((h, w, 3), dtype=np.uint8)

    # 原图作为灰色背景
    overlay[:, :, 0] = original // 3
    overlay[:, :, 1] = original // 3
    overlay[:, :, 2] = original // 3

    # 骨架 - 红色
    overlay[skeleton > 0] = [0, 0, 255]

    # 找端点和分支点
    endpoints, branch_points = find_endpoints_and_branches(skeleton)

    # 膨胀端点和分支点以便可视化
    kernel = np.ones((3, 3), np.uint8)
    endpoints_dilated = cv2.dilate(endpoints.astype(np.uint8), kernel)
    branch_dilated = cv2.dilate(branch_points.astype(np.uint8), kernel)

    # 端点 - 绿色
    overlay[endpoints_dilated > 0] = [0, 255, 0]

    # 分支点 - 蓝色
    overlay[branch_dilated > 0] = [255, 0, 0]

    return overlay


# =====================================================
# 主函数
# =====================================================

def main():
    print("╔" + "═" * 58 + "╗")
    print("║" + "  步骤8: 骨架提取 · 中心线提取".center(46) + "║")
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
    skeleton_dir = os.path.join(OUTPUT_DIR, "frames_skeleton")
    overlay_dir = os.path.join(OUTPUT_DIR, "frames_overlay")

    os.makedirs(skeleton_dir, exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)

    # === 获取图像尺寸 ===
    sample = load_image(os.path.join(INPUT_DIR, frame_files[0]))
    height, width = sample.shape
    print(f"   尺寸: {width}×{height}")

    # === 打印参数 ===
    print(f"\n🔧 参数:")
    print(f"   修剪长度: {PRUNE_LENGTH} (0=不修剪)")
    print(f"   最小骨架: {MIN_SKELETON_SIZE} 像素")

    # === 视频设置 ===
    video_w, video_h = make_even(width), make_even(height)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # 骨架视频
    video_skeleton_path = os.path.join(OUTPUT_DIR, "skeleton_all_frames.mp4")
    out_skeleton = cv2.VideoWriter(video_skeleton_path, fourcc, FPS, (video_w, video_h), isColor=False)

    # 叠加视频
    video_overlay_path = os.path.join(OUTPUT_DIR, "overlay_all_frames.mp4")
    out_overlay = cv2.VideoWriter(video_overlay_path, fourcc, FPS, (video_w, video_h), isColor=True)

    # 对比视频
    compare_scale = 0.4
    single_w = int(width * compare_scale)
    compare_w = make_even(single_w * 3)
    compare_h = make_even(int(height * compare_scale))

    video_compare_path = os.path.join(OUTPUT_DIR, "compare_skeleton.mp4")
    out_compare = cv2.VideoWriter(video_compare_path, fourcc, FPS, (compare_w, compare_h), isColor=True)

    # === 处理 ===
    print(f"\n🔄 处理中...")

    stats = []

    for idx, filename in enumerate(frame_files):
        binary = load_image(os.path.join(INPUT_DIR, filename))
        if binary is None:
            continue

        # 1. 提取骨架
        skeleton = extract_skeleton(binary)

        # 2. 修剪短分支
        if PRUNE_LENGTH > 0:
            skeleton = prune_short_branches(skeleton, PRUNE_LENGTH)

        # 3. 移除小骨架
        if MIN_SKELETON_SIZE > 0:
            skeleton = remove_small_skeletons(skeleton, MIN_SKELETON_SIZE)

        # 4. 创建叠加图
        overlay = create_overlay(binary, skeleton)

        # 统计
        endpoints, branch_points = find_endpoints_and_branches(skeleton)

        stat = {
            'frame': idx,
            'binary_pixels': np.sum(binary > 0),
            'skeleton_pixels': np.sum(skeleton > 0),
            'endpoints': np.sum(endpoints),
            'branch_points': np.sum(branch_points)
        }
        stats.append(stat)

        # 保存
        save_image(os.path.join(skeleton_dir, f"frame_{idx:04d}.png"), skeleton)
        save_image(os.path.join(overlay_dir, f"frame_{idx:04d}.png"), overlay)

        # 视频
        skeleton_resized = cv2.resize(skeleton, (video_w, video_h))
        out_skeleton.write(skeleton_resized)

        overlay_resized = cv2.resize(overlay, (video_w, video_h))
        out_overlay.write(overlay_resized)

        # 对比视频: 原图 | 骨架 | 叠加
        binary_small = cv2.resize(binary, (single_w, compare_h))
        skeleton_small = cv2.resize(skeleton, (single_w, compare_h))
        overlay_small = cv2.resize(overlay, (single_w, compare_h))

        binary_color = cv2.cvtColor(binary_small, cv2.COLOR_GRAY2BGR)
        skeleton_color = cv2.cvtColor(skeleton_small, cv2.COLOR_GRAY2BGR)

        cv2.putText(binary_color, f"Binary #{idx}", (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.putText(skeleton_color, f"Skel:{stat['skeleton_pixels'] // 100}00", (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(overlay_small, f"E:{stat['endpoints']} B:{stat['branch_points']}", (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        compare_frame = np.hstack([binary_color, skeleton_color, overlay_small])
        if compare_frame.shape[1] < compare_w:
            pad = compare_w - compare_frame.shape[1]
            compare_frame = np.pad(compare_frame, ((0, 0), (0, pad), (0, 0)), mode='constant')
        compare_frame = compare_frame[:, :compare_w]
        out_compare.write(compare_frame)

        if (idx + 1) % 20 == 0 or idx == num_frames - 1:
            elapsed = time.time() - t_start
            fps_rate = (idx + 1) / elapsed
            print(
                f"   [{idx + 1:3d}/{num_frames}] {fps_rate:.1f} 帧/秒 | 端点:{stat['endpoints']} 分支:{stat['branch_points']}")

    out_skeleton.release()
    out_overlay.release()
    out_compare.release()

    # === 保存统计 ===
    stats_path = os.path.join(OUTPUT_DIR, "skeleton_statistics.csv")
    with open(stats_path, 'w', encoding='utf-8') as f:
        f.write("frame,binary_pixels,skeleton_pixels,endpoints,branch_points\n")
        for s in stats:
            f.write(f"{s['frame']},{s['binary_pixels']},{s['skeleton_pixels']},{s['endpoints']},{s['branch_points']}\n")

    # === 摘要 ===
    total_time = time.time() - t_start

    binary_arr = np.array([s['binary_pixels'] for s in stats])
    skeleton_arr = np.array([s['skeleton_pixels'] for s in stats])
    endpoints_arr = np.array([s['endpoints'] for s in stats])
    branch_arr = np.array([s['branch_points'] for s in stats])

    print("\n" + "=" * 60)
    print("📊 统计")
    print("=" * 60)
    print(f"\n   骨架压缩率: {np.mean(skeleton_arr) / np.mean(binary_arr) * 100:.1f}%")
    print(f"   (原始 {np.mean(binary_arr):,.0f} → 骨架 {np.mean(skeleton_arr):,.0f} 像素/帧)")
    print(f"\n   拓扑特征 (平均):")
    print(f"     端点:   {np.mean(endpoints_arr):.0f} 个/帧")
    print(f"     分支点: {np.mean(branch_arr):.0f} 个/帧")
    print(f"\n   耗时: {total_time:.1f} 秒 ({num_frames / total_time:.1f} 帧/秒)")

    print("\n📁 输出")
    print(f"""
    {OUTPUT_DIR}/
    ├── frames_skeleton/         # 纯骨架图 (白线)
    ├── frames_overlay/          # 叠加可视化
    │   (红=骨架, 绿=端点, 蓝=分支点)
    ├── skeleton_all_frames.mp4
    ├── overlay_all_frames.mp4   # 推荐查看
    ├── compare_skeleton.mp4
    └── skeleton_statistics.csv
    """)

    print("""
📌 颜色含义 (overlay图):
   🔴 红色 = 骨架线
   🟢 绿色 = 端点 (神经末梢)
   🔵 蓝色 = 分支点 (分叉处)
    """)

    print("✅ 完成！")


if __name__ == "__main__":
    main()
