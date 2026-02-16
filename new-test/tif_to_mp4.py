import tifffile
import cv2
import numpy as np
from tqdm import tqdm

# ================= 配置区域 =================
TIFF_PATH = r"/recover_0-72h_XY2.tif"
OUTPUT_VIDEO = r"F:\工作文件\RA\python\项目汇总\神经图像\preview_video.mp4"
FPS = 10  # 帧率，可以调整（越大播放越快）


# ===========================================

def normalize_image(img):
    """智能归一化：拉伸到 0-255"""
    img = img.astype(np.float32)
    min_val = np.percentile(img, 1)
    max_val = np.percentile(img, 99)

    if max_val - min_val < 1e-6:
        return np.zeros_like(img, dtype=np.uint8)

    img = (img - min_val) / (max_val - min_val) * 255.0
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)


def convert_tif_to_mp4():
    print(f"🔄 读取 TIFF: {TIFF_PATH}")

    img_stack = tifffile.imread(TIFF_PATH)
    print(f"✅ 成功！形状: {img_stack.shape}, 类型: {img_stack.dtype}")
    print(f"   像素范围: [{img_stack.min()}, {img_stack.max()}]")

    total_frames = img_stack.shape[0]

    # 获取尺寸
    first_frame = normalize_image(img_stack[0])
    if len(first_frame.shape) == 2:
        h, w = first_frame.shape
    else:
        h, w = first_frame.shape[:2]

    print(f"📐 视频尺寸: {w}x{h}, 共 {total_frames} 帧, FPS={FPS}")
    print(f"   预计时长: {total_frames / FPS:.1f} 秒")

    # 创建视频写入器 (H264 编码)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, FPS, (w, h))

    if not out.isOpened():
        print("❌ 无法创建视频文件！尝试使用英文路径。")
        return

    print(f"🎬 开始转换...")

    for i in tqdm(range(total_frames)):
        img = normalize_image(img_stack[i])

        # 转为 3 通道 BGR
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        out.write(img)

    out.release()
    print(f"\n✅ 完成！视频已保存到:\n   {OUTPUT_VIDEO}")
    print(f"💡 可以用 VLC 或 PotPlayer 打开查看")


if __name__ == "__main__":
    convert_tif_to_mp4()
