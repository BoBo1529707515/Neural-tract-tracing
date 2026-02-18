import cv2
import numpy as np
import tifffile
import os
from skimage.filters import frangi

# === 参数设置 ===
FILE_PATH = r"F:\工作文件\RA\python\项目汇总\神经图像\recover_0-72h_XY2.tif"
OUTPUT_VIDEO = r"F:\工作文件\RA\python\项目汇总\神经图像\enhanced_silhouette.mp4"

FPS = 10

# 增强参数
ENHANCE_METHOD = 2  # 1=CLAHE, 2=Frangi, 3=CLAHE+Frangi

# Frangi 参数
FRANGI_SIGMAS = [1, 2, 3]

# 差分参数
DIFF_THRESHOLD = 5  # 低于此值视为噪声


def to_8bit(img):
    return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


# =====================================================
# 📐 图像增强方法
# =====================================================

def enhance_clahe(img):
    """CLAHE 对比度增强"""
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(img)


def enhance_frangi(img):
    """Frangi 血管/神经元滤波"""
    img_f = img.astype(np.float32) / 255.0
    vessel = frangi(img_f, sigmas=FRANGI_SIGMAS, black_ridges=False)
    # 归一化
    if vessel.max() > 0:
        return (vessel / vessel.max() * 255).astype(np.uint8)
    return img


def enhance_combined(img):
    """CLAHE + Frangi 组合"""
    clahe_result = enhance_clahe(img)
    return enhance_frangi(clahe_result)


def enhance_frame(img, method):
    """根据选择的方法增强图像"""
    if method == 1:
        return enhance_clahe(img)
    elif method == 2:
        return enhance_frangi(img)
    elif method == 3:
        return enhance_combined(img)
    return img


# =====================================================
# 🎬 主函数
# =====================================================

def main():
    print("=" * 50)
    print("神经元追踪 · 增强 + 剪影")
    print("=" * 50)

    method_names = {1: "CLAHE", 2: "Frangi", 3: "CLAHE+Frangi"}
    print(f"📐 增强方法: {method_names[ENHANCE_METHOD]}")

    # 读取数据
    print("读取数据中...")
    with open(FILE_PATH, 'rb') as f:
        stack = tifffile.imread(f)

    num_frames, height, width = stack.shape
    print(f"✅ {num_frames} 帧, {width}x{height}")

    # 视频写入器 (输出三栏对比: 原图 | 增强 | 剪影)
    out_width = width * 3
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, FPS, (out_width, height), isColor=False)

    # 单独的剪影视频
    silhouette_video = OUTPUT_VIDEO.replace('.mp4', '_silhouette_only.mp4')
    out_sil = cv2.VideoWriter(silhouette_video, fourcc, FPS, (width, height), isColor=False)

    previous_enhanced = None

    print(f"🎥 生成视频中...")

    for idx in range(num_frames):
        # 1. 原始帧 → 8bit
        raw = to_8bit(stack[idx])

        # 2. 图像增强
        enhanced = enhance_frame(raw, ENHANCE_METHOD)

        # 3. 帧差剪影
        if idx == 0 or previous_enhanced is None:
            silhouette = np.zeros_like(enhanced)
        else:
            diff = enhanced.astype(np.float32) - previous_enhanced.astype(np.float32)
            diff_relu = np.maximum(diff, 0)

            # 阈值过滤噪声
            diff_relu[diff_relu < DIFF_THRESHOLD] = 0

            # 归一化
            if diff_relu.max() > 0:
                silhouette = (diff_relu / diff_relu.max() * 255).astype(np.uint8)
            else:
                silhouette = np.zeros_like(enhanced)

        # 更新前一帧
        previous_enhanced = enhanced.copy()

        # 4. 拼接三栏对比
        # 添加标签
        raw_labeled = raw.copy()
        enhanced_labeled = enhanced.copy()
        silhouette_labeled = silhouette.copy()

        cv2.putText(raw_labeled, f"Original #{idx}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
        cv2.putText(enhanced_labeled, f"Enhanced ({method_names[ENHANCE_METHOD]})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
        cv2.putText(silhouette_labeled, "Silhouette", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)

        combined = np.hstack([raw_labeled, enhanced_labeled, silhouette_labeled])

        out.write(combined)
        out_sil.write(silhouette)

        if (idx + 1) % 20 == 0 or idx == num_frames - 1:
            print(f"  {idx + 1}/{num_frames} ({100 * (idx + 1) // num_frames}%)")

    out.release()
    out_sil.release()

    print(f"\n✅ 对比视频: {OUTPUT_VIDEO}")
    print(f"✅ 剪影视频: {silhouette_video}")

    size1 = os.path.getsize(OUTPUT_VIDEO) / 1024 / 1024
    size2 = os.path.getsize(silhouette_video) / 1024 / 1024
    print(f"   文件大小: {size1:.1f} MB / {size2:.1f} MB")


if __name__ == "__main__":
    main()
