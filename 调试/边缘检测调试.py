import cv2
import numpy as np
import tifffile
import os

# === 参数设置 ===
FILE_PATH = r"/recover_0-72h_XY2.tif"
OUTPUT_DIR = r"/边缘检测对比"

CHECK_FRAMES = [0, 30, 70]


def to_8bit(img):
    return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def save_image(path, img):
    """解决中文路径问题的保存函数"""
    success, encoded = cv2.imencode('.png', img)
    if success:
        with open(path, 'wb') as f:
            f.write(encoded.tobytes())
        return True
    return False


def main():
    print("=" * 50)
    print("神经元边缘检测")
    print("=" * 50)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"✅ 输出目录: {OUTPUT_DIR}")

    # 读取数据
    print("读取数据...")
    with open(FILE_PATH, 'rb') as f:
        stack = tifffile.imread(f)
    print(f"✅ 数据形状: {stack.shape}")

    for idx in CHECK_FRAMES:
        print(f"\n处理帧 {idx}...")
        raw = to_8bit(stack[idx])

        # 方法1: 原图
        path1 = os.path.join(OUTPUT_DIR, f"frame_{idx:03d}_1_original.png")
        save_image(path1, raw)
        print(f"  原图: {os.path.exists(path1)}")

        # 方法2: Canny
        blur = cv2.GaussianBlur(raw, (5, 5), 1.5)
        canny = cv2.Canny(blur, 30, 100)
        path2 = os.path.join(OUTPUT_DIR, f"frame_{idx:03d}_2_canny.png")
        save_image(path2, canny)
        print(f"  Canny: {os.path.exists(path2)}")

        # 方法3: CLAHE + 自适应阈值
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(raw)
        binary = cv2.adaptiveThreshold(enhanced, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 21, -3)
        path3 = os.path.join(OUTPUT_DIR, f"frame_{idx:03d}_3_clahe_thresh.png")
        save_image(path3, binary)
        print(f"  CLAHE: {os.path.exists(path3)}")

        # 方法4: 白帽变换
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        tophat = cv2.morphologyEx(raw, cv2.MORPH_TOPHAT, kernel)
        tophat_norm = cv2.normalize(tophat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, tophat_bin = cv2.threshold(tophat_norm, 20, 255, cv2.THRESH_BINARY)
        path4 = os.path.join(OUTPUT_DIR, f"frame_{idx:03d}_4_tophat.png")
        save_image(path4, tophat_bin)
        print(f"  TopHat: {os.path.exists(path4)}")

        # 对比图
        scale = 0.25
        imgs = [
            cv2.resize(raw, None, fx=scale, fy=scale),
            cv2.resize(canny, None, fx=scale, fy=scale),
            cv2.resize(binary, None, fx=scale, fy=scale),
            cv2.resize(tophat_bin, None, fx=scale, fy=scale)
        ]
        labels = ['Original', 'Canny', 'CLAHE+Thresh', 'TopHat']

        for img, label in zip(imgs, labels):
            cv2.putText(img, label, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)

        combined = np.hstack(imgs)
        compare_path = os.path.join(OUTPUT_DIR, f"compare_frame_{idx:03d}.png")
        save_image(compare_path, combined)
        print(f"  对比图: {os.path.exists(compare_path)}")

    print("\n" + "=" * 50)
    files = os.listdir(OUTPUT_DIR)
    print(f"✅ 生成了 {len(files)} 个文件")
    for f in files:
        print(f"  - {f}")


if __name__ == "__main__":
    main()
