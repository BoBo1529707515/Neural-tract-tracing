import cv2
import os

# ================= 配置 =================
VIDEO_PATH = r"F:\工作文件\RA\python\项目汇总\神经图像\preview_video.mp4"
OUTPUT_DIR = r"F:\工作文件\RA\python\项目汇总\神经图像\frames_to_label"
FRAME_INTERVAL = 10  # 每隔10帧抽1张（可根据视频长度调整）
# ========================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

cap = cv2.VideoCapture(VIDEO_PATH)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"📹 视频共 {total_frames} 帧，每隔 {FRAME_INTERVAL} 帧抽取")

saved = 0
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx % FRAME_INTERVAL == 0:
        save_path = os.path.join(OUTPUT_DIR, f"frame_{frame_idx:06d}.jpg")
        cv2.imwrite(save_path, frame)
        saved += 1

    frame_idx += 1

cap.release()
print(f"✅ 完成！共抽取 {saved} 张图片到:\n   {OUTPUT_DIR}")

