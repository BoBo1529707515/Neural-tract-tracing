import cv2
import numpy as np
import os
import glob

# === 参数设置 ===
INPUT_DIR = r"F:\工作文件\RA\python\项目汇总\神经图像\CLAHE二值化"
OUTPUT_DIR = r"F:\工作文件\RA\python\项目汇总\神经图像\对齐测试"

# === 🎛️ 可调参数 ===
WHITE_RATIO_THRESHOLD = 0.50  # 白色占比阈值 (50%)


def save_image(path, img):
    success, encoded = cv2.imencode('.png', img)
    if success:
        with open(path, 'wb') as f:
            f.write(encoded.tobytes())


def find_left_edge(binary, threshold=0.50):
    """找到第一个白色占比 > threshold 的列位置"""
    height, width = binary.shape
    white_per_column = np.sum(binary > 0, axis=0) / height

    for col in range(width):
        if white_per_column[col] > threshold:
            return col

    return 0


def align_and_crop(binary, left_edge, target_width=None):
    """从 left_edge 开始裁剪图像"""
    height, width = binary.shape
    cropped = binary[:, left_edge:]

    if target_width is not None:
        current_width = cropped.shape[1]
        if current_width >= target_width:
            cropped = cropped[:, :target_width]
        else:
            pad_width = target_width - current_width
            cropped = np.pad(cropped, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)

    return cropped


def main():
    print("=" * 50)
    print("图像对齐测试 · 按白色占比裁剪")
    print("=" * 50)

    # === 检查输入目录 ===
    print(f"\n📂 检查输入目录: {INPUT_DIR}")

    if not os.path.exists(INPUT_DIR):
        print(f"❌ 目录不存在!")
        print(f"\n请确认正确的目录路径，或者先运行二值化代码生成图片。")
        return

    # 列出目录中的所有文件
    all_files = os.listdir(INPUT_DIR)
    print(f"📋 目录中共有 {len(all_files)} 个文件:")
    for f in all_files[:10]:  # 只显示前10个
        print(f"   - {f}")
    if len(all_files) > 10:
        print(f"   ... 还有 {len(all_files) - 10} 个文件")

    # 查找 binary 图片
    binary_files = [f for f in all_files if 'binary' in f.lower() and f.endswith('.png')]
    print(f"\n🔍 找到 {len(binary_files)} 个 binary 图片:")
    for f in binary_files:
        print(f"   - {f}")

    if len(binary_files) == 0:
        print("\n❌ 没有找到 binary 图片!")
        print("请先运行 CLAHE 二值化代码生成图片。")
        return

    # === 创建输出目录 ===
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # === 处理找到的文件 ===
    results = []

    print(f"\n🎛️ 白色占比阈值: {WHITE_RATIO_THRESHOLD * 100:.0f}%")
    print(f"\n{'文件名':<35} {'左边界':<10} {'原宽度':<10}")
    print("-" * 60)

    for filename in binary_files:
        filepath = os.path.join(INPUT_DIR, filename)

        # 使用二进制方式读取，避免路径编码问题
        with open(filepath, 'rb') as f:
            file_bytes = np.frombuffer(f.read(), dtype=np.uint8)
            binary = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

        if binary is None:
            print(f"⚠️ 无法读取: {filename}")
            continue

        height, width = binary.shape

        # 找左边界
        left_edge = find_left_edge(binary, WHITE_RATIO_THRESHOLD)

        # 裁剪
        cropped = align_and_crop(binary, left_edge)

        results.append({
            'filename': filename,
            'left_edge': left_edge,
            'original': binary,
            'cropped': cropped
        })

        print(f"{filename:<35} {left_edge:<10} {width:<10}")

        # 保存裁剪后的图像
        output_name = filename.replace('binary', 'cropped')
        save_image(os.path.join(OUTPUT_DIR, output_name), cropped)

    if len(results) == 0:
        print("\n❌ 没有成功处理任何图片!")
        return

    # === 生成边界可视化 ===
    print("\n📊 生成边界可视化...")

    for r in results:
        binary = r['original'].copy()
        left_edge = r['left_edge']
        filename = r['filename']

        # 转为彩色
        vis = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

        # 画红色竖线
        cv2.line(vis, (left_edge, 0), (left_edge, binary.shape[0]), (0, 0, 255), 3)

        # 添加文字
        cv2.putText(vis, f"Edge at col {left_edge}",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

        # 缩小保存
        scale = 0.3
        vis_small = cv2.resize(vis, None, fx=scale, fy=scale)
        vis_name = filename.replace('binary', 'edge_vis')
        save_image(os.path.join(OUTPUT_DIR, vis_name), vis_small)

    # === 统一宽度对比 ===
    min_width = min(r['cropped'].shape[1] for r in results)
    print(f"\n📏 裁剪后最小宽度: {min_width}")

    # 生成对比图
    scale = 0.2
    aligned_imgs = []

    for r in results:
        unified = align_and_crop(r['original'], r['left_edge'], target_width=min_width)
        small = cv2.resize(unified, None, fx=scale, fy=scale)
        cv2.putText(small, r['filename'][:10], (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, 255, 1)
        aligned_imgs.append(small)

    if len(aligned_imgs) > 1:
        comparison = np.hstack(aligned_imgs)
        save_image(os.path.join(OUTPUT_DIR, "aligned_comparison.png"), comparison)

    print("\n" + "=" * 50)
    print(f"✅ 完成!")
    print(f"📁 输出目录: {OUTPUT_DIR}")
    print("=" * 50)


if __name__ == "__main__":
    main()
