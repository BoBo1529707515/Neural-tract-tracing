import cv2
import numpy as np
import tifffile
from tracker import DirectionalSort
from skimage import filters, feature

def preprocess_frame(frame):
    """预处理：Frangi Vesselness Filter 增强管状结构"""
    # 归一化到0-1
    frame_norm = (frame - frame.min()) / (frame.max() - frame.min() + 1e-8)
    
    # Frangi滤波增强血管/轴突结构
    vesselness = filters.frangi(frame_norm, sigmas=range(1, 4), beta=0.5)
    
    # 再次归一化
    vesselness = (vesselness - vesselness.min()) / (vesselness.max() - vesselness.min() + 1e-8)
    return vesselness

def detect_tips(vesselness_map, threshold=0.1):
    """检测：LoG 检测轴突尖端"""
    blobs = feature.blob_log(vesselness_map, 
                           min_sigma=1, 
                           max_sigma=4, 
                           num_sigma=10, 
                           threshold=threshold)
    
    if len(blobs) == 0:
        return []
        
    # 返回 (x, y) 坐标，注意 blob_log 返回 (y, x, sigma)
    return blobs[:, [1, 0]]

def main():
    # 真实数据路径
    tif_path = r"/recover_0-72h_XY2.tif"
    print(f"正在加载图像: {tif_path}")
    
    try:
        stack = tifffile.imread(tif_path)
        print(f"加载完成，共 {stack.shape[0]} 帧，尺寸 {stack.shape[1]}x{stack.shape[2]}")
    except Exception as e:
        print(f"无法加载图像: {e}")
        return

    # 初始化跟踪器
    # 真实数据通常比较嘈杂，参数需要调整
    # max_age: 允许遮挡/丢失的帧数
    # min_hits: 确认为有效轨迹所需的最小连续匹配数
    # dist_threshold: 允许匹配的最大距离 (像素)
    tracker = DirectionalSort(max_age=10, min_hits=5, dist_threshold=60, direction_weight=0.6)
    
    # 输出视频设置
    height, width = stack.shape[1], stack.shape[2]
    output_video_path = "real_tracking_result.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, 10.0, (width, height))
    
    track_history = {} # {id: [(x,y), ...]}
    
    print("开始处理与跟踪...")
    
    for i, frame in enumerate(stack):
        print(f"Processing frame {i+1}/{len(stack)}...", end='\r')
        
        # 1. 检测 (Detection)
        # 这里使用简单的图像处理模拟检测器
        # 实际应用中建议替换为 DeepLabCut/YOLO 的推理结果
        vesselness = preprocess_frame(frame)
        detections = detect_tips(vesselness, threshold=0.05) # 阈值可能需要根据图像亮度调整
        
        # 2. 跟踪 (Tracking)
        trackers = tracker.update(detections)
        
        # 3. 可视化
        # 转为RGB
        frame_disp = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        frame_color = cv2.cvtColor(frame_disp, cv2.COLOR_GRAY2BGR)
        
        # 绘制检测点 (绿色小圆点)
        for det in detections:
            cv2.circle(frame_color, (int(det[0]), int(det[1])), 3, (0, 255, 0), 1)
        
        # 绘制跟踪结果
        for trk in trackers:
            x, y, tid = trk
            tid = int(tid)
            
            # 记录历史
            if tid not in track_history:
                track_history[tid] = []
            track_history[tid].append((x, y))
            
            # 颜色生成
            np.random.seed(tid)
            color = np.random.randint(0, 255, 3).tolist()
            
            # 绘制当前位置
            cv2.circle(frame_color, (int(x), int(y)), 6, color, -1)
            cv2.putText(frame_color, f"{tid}", (int(x)+8, int(y)-8), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # 绘制轨迹
            if len(track_history[tid]) > 1:
                pts = np.array(track_history[tid], np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(frame_color, [pts], False, color, 2)
        
        out.write(frame_color)
        
        # 实时显示
        cv2.namedWindow("Real Tracking", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Real Tracking", 1024, 768) # 缩放窗口以免太大
        cv2.imshow("Real Tracking", frame_color)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    out.release()
    cv2.destroyAllWindows()
    print(f"\n处理完成，视频保存至: {output_video_path}")

if __name__ == "__main__":
    main()
