import tifffile
import cv2
import os
import numpy as np
from tqdm import tqdm

# ================= 配置区域 =================
# 🔴 请修改为你的 .tif 文件路径
TIFF_PATH = r"F:\工作文件\RA\python\项目汇总\神经图像\recover_0-72h_XY2.tif"

# 输出文件夹 (会自动创建)
OUTPUT_DIR = r"F:\工作文件\RA\python\项目汇总\神经图像\dataset_yolo"
IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")

# 采样间隔 (每隔多少帧保存一张？)
# 如果视频变化很慢，建议设为 5 或 10，减少标注重复劳动
FRAME_INTERVAL = 1


# ===========================================

def convert_tif_to_dataset():
    print(f"🔄 正在读取 TIFF 文件: {TIFF_PATH}")

    try:
        # 使用 tifffile 读取，它对中文路径支持较好，且支持多页 TIFF
        # 返回格式通常是 (Frames, Height, Width) 或 (Frames, H, W, Channels)
        img_stack = tifffile.imread(TIFF_PATH)
        print(f"✅ 读取成功！数据形状: {img_stack.shape}")
        print(f"   数据类型: {img_stack.dtype}")
    except Exception as e:
        print(f"❌ 读取失败: {e}")
        return

    # 创建输出目录
    os.makedirs(IMAGES_DIR, exist_ok=True)

    total_frames = img_stack.shape[0]
    saved_count = 0

    print(f"📂 开始导出图片到: {IMAGES_DIR}")

    for i in tqdm(range(0, total_frames, FRAME_INTERVAL)):
        # 获取单帧
        img = img_stack[i]

        # --- 数据标准化 (非常重要) ---
        # 很多显微镜图像是 16-bit (0-65535) 的，OpenCV/YOLO 需要 8-bit (0-255)
        if img.dtype == np.uint16:
            # 简单的归一化：映射到 0-255
            img = (img / 256).astype('uint8')
        elif img.dtype == np.float32:
            img = (img * 255).astype('uint8')

        # 如果是单通道(灰度)，转为 3通道 (BGR)，因为 YOLO 默认吃彩色图
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif len(img.shape) == 3 and img.shape[2] == 3:
            # 如果本身是 RGB，tifffile 读出来通常是 RGB，OpenCV 需要 BGR
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # 保存文件名： frame_000001.jpg
        save_name = f"frame_{i:06d}.jpg"
        save_path = os.path.join(IMAGES_DIR, save_name)

        # 使用 cv2.imencode 保存以支持中文路径
        is_success, buffer = cv2.imencode(".jpg", img)
        if is_success:
            with open(save_path, "wb") as f:
                f.write(buffer)
            saved_count += 1

    print(f"✅ 处理完成！共保存了 {saved_count} 张图片。")
    print(f"👉 下一步：请使用 labelImg 对这些图片进行标注。")


if __name__ == "__main__":
    convert_tif_to_dataset()
