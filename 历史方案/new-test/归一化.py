import tifffile
import cv2
import os
import numpy as np
from tqdm import tqdm

# ================= 配置区域 =================
TIFF_PATH = r"F:\工作文件\RA\python\项目汇总\神经图像\recover_0-72h_XY2.tif"
OUTPUT_DIR = r"F:\工作文件\RA\python\项目汇总\神经图像\dataset_yolo"
IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
FRAME_INTERVAL = 1


# ===========================================

def normalize_image(img):
    """
    智能归一化：把任意范围的图像拉伸到 0-255
    """
    # 转为 float 计算
    img = img.astype(np.float32)

    # 获取实际的最小值和最大值
    min_val = np.percentile(img, 1)  # 用 1% 分位数，避免极端噪点影响
    max_val = np.percentile(img, 99)  # 用 99% 分位数

    print(f"   📊 像素范围: min={min_val:.1f}, max={max_val:.1f}")

    # 避免除零
    if max_val - min_val < 1e-6:
        return np.zeros_like(img, dtype=np.uint8)

    # 归一化到 0-255
    img = (img - min_val) / (max_val - min_val) * 255.0
    img = np.clip(img, 0, 255)  # 限制范围

    return img.astype(np.uint8)


def convert_tif_to_dataset():
    print(f"🔄 正在读取 TIFF 文件...")

    try:
        img_stack = tifffile.imread(TIFF_PATH)
        print(f"✅ 读取成功！")
        print(f"   形状: {img_stack.shape}")
        print(f"   数据类型: {img_stack.dtype}")
        print(f"   原始像素范围: [{img_stack.min()}, {img_stack.max()}]")
    except Exception as e:
        print(f"❌ 读取失败: {e}")
        return

    os.makedirs(IMAGES_DIR, exist_ok=True)

    total_frames = img_stack.shape[0]

    # === 先预览第一帧，确认效果 ===
    print("\n🔍 预览第一帧...")
    first_frame = img_stack[0]
    first_normalized = normalize_image(first_frame)

    if len(first_normalized.shape) == 2:
        first_normalized = cv2.cvtColor(first_normalized, cv2.COLOR_GRAY2BGR)

    cv2.imshow("Preview - Press any key to continue (ESC to abort)", first_normalized)
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()

    if key == 27:  # ESC
        print("❌ 用户取消")
        return

    # === 开始批量导出 ===
    print(f"\n📂 开始导出到: {IMAGES_DIR}")
    saved_count = 0

    for i in tqdm(range(0, total_frames, FRAME_INTERVAL)):
        img = img_stack[i]

        # 智能归一化
        img = normalize_image(first_frame) if i == 0 else normalize_image(img)

        # 转 3 通道
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        save_name = f"frame_{i:06d}.jpg"
        save_path = os.path.join(IMAGES_DIR, save_name)

        is_success, buffer = cv2.imencode(".jpg", img)
        if is_success:
            with open(save_path, "wb") as f:
                f.write(buffer)
            saved_count += 1

    print(f"\n✅ 完成！共保存 {saved_count} 张图片")


if __name__ == "__main__":
    # 先关掉 tqdm 的逐帧打印，只在预览时显示
    convert_tif_to_dataset()
