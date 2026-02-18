import os
import cv2
import numpy as np
import csv

from ui_02 import select_neuron_point
from tracker_03 import PointTracker

标识_04_流程 = "04_流程"


def read_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 10
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames, fps


def run_tracking(config):
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)

    frames, fps = read_video_frames(config.video_path)
    if len(frames) == 0:
        print("03_未读取到视频帧")
        return

    # 1. 交互式选择标记（支持缩放、多帧）
    marks_by_frame = select_neuron_point(frames, config.window_name)
    if not marks_by_frame:
        print("04_用户取消")
        return

    # 2. 图像预处理（适配灰白图像）
    # 对所有帧进行预处理：转灰度 -> 增强 -> 二值化（可选）
    processed_frames = []
    print("04_正在预处理图像...")
    for f in frames:
        if len(f.shape) == 3:
            gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        else:
            gray = f
        # CLAHE增强对比度
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        processed_frames.append(enhanced)

    # 3. 初始化跟踪器
    print("05_开始跟踪...")
    tracker = PointTracker(marks_by_frame)
    # 这里 start_frame_idx 参数其实在 tracker 内部逻辑中已经用不到了，因为会为每个 neuron 自动找起始帧
    # 但为了兼容接口保持传递，或者干脆去掉
    all_trajectories = tracker.track(processed_frames, start_frame_idx=0)

    # 4. 结果可视化与输出
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(config.output_video_path, fourcc, fps, (width, height))

    # 颜色生成器
    colors = {}
    def get_color(nid):
        if nid not in colors:
            np.random.seed(nid)
            colors[nid] = tuple(map(int, np.random.randint(50, 255, 3)))
        return colors[nid]

    for i, frame in enumerate(frames):
        overlay = frame.copy()
        
        # 绘制每条轨迹
        for nid, trajectory in all_trajectories.items():
            color = get_color(nid)
            
            # 筛选出当前帧之前的点
            pts = [(x, y) for (x, y, fidx) in trajectory if fidx <= i]
            
            if len(pts) > 1:
                cv2.polylines(overlay, [np.array(pts, np.int32)], False, color, 2)
                
            # 绘制当前点
            # 找到当前帧的点
            curr_pt = next((pt for pt in trajectory if pt[2] == i), None)
            
            if curr_pt:
                x, y, _ = curr_pt
                cv2.circle(overlay, (int(x), int(y)), 6, color, -1)
                cv2.putText(overlay, f"ID:{nid}", (int(x)+10, int(y)-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.putText(overlay, "05_跟踪中", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        out.write(overlay)

        cv2.namedWindow(config.track_window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(config.track_window_name, overlay)
        if cv2.waitKey(config.show_delay_ms) & 0xFF in [ord("q"), 27]:
            break

    out.release()
    cv2.destroyAllWindows()

    with open(config.output_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["06_神经元ID", "07_帧号", "08_X", "09_Y"])
        for nid, trajectory in all_trajectories.items():
            for x, y, fidx in trajectory:
                writer.writerow([nid, fidx, x, y])

    print(f"09_完成 输出视频: {config.output_video_path}")
    print(f"10_完成 输出数据: {config.output_csv_path}")
