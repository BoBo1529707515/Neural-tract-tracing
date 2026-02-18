import cv2
import numpy as np
import os
import time
from collections import defaultdict

# ╔══════════════════════════════════════════════════════════════╗
# ║  步骤7: 神经元生长路径追踪 · 排除背景结构                       ║
# ╚══════════════════════════════════════════════════════════════╝

# === 输入输出路径 ===
INPUT_DIR = r"F:\工作文件\RA\python\项目汇总\神经图像\05_二次空间去噪\frames_final"
OUTPUT_DIR = r"F:\工作文件\RA\python\项目汇总\神经图像\07_生长追踪"

# === 🎛️ 追踪参数 ===
REFERENCE_MODE = "first"  # "first" = 与第一帧比, "interval" = 与N帧前比
REFERENCE_INTERVAL = 60  # 当mode="interval"时，参考帧间隔（如10分钟=60帧@0.1fps）

BACKGROUND_THRESHOLD = 50  # 参考帧亮度阈值：超过此值认为是背景老结构
CURRENT_THRESHOLD = 30  # 当前帧亮度阈值：信号检测

# === 追踪器参数 ===
SEARCH_RADIUS = 15  # 追踪搜索半径（像素）
MIN_GROWTH_LENGTH = 5  # 最小生长长度（像素），过短不记录
MAX_TRACKERS = 500  # 最大同时追踪数量
MERGE_DISTANCE = 10  # 追踪器合并距离

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
# 神经元生长追踪器
# =====================================================

class GrowthTracker:
    """单个生长追踪器"""

    def __init__(self, start_pos, start_frame, tracker_id):
        self.id = tracker_id
        self.start_frame = start_frame
        self.path = [start_pos]  # [(x, y), ...]
        self.active = True
        self.stall_count = 0  # 停滞计数
        self.total_length = 0

    def get_tip(self):
        """获取当前末端位置"""
        return self.path[-1]

    def add_point(self, pos):
        """添加新点"""
        if len(self.path) > 0:
            last = self.path[-1]
            dist = np.sqrt((pos[0] - last[0]) ** 2 + (pos[1] - last[1]) ** 2)
            self.total_length += dist
        self.path.append(pos)
        self.stall_count = 0

    def stall(self):
        """停滞一帧"""
        self.stall_count += 1
        if self.stall_count > 10:  # 连续10帧无生长则停用
            self.active = False


class GrowthTrackingSystem:
    """生长追踪系统"""

    def __init__(self, reference_frame, bg_threshold, curr_threshold, search_radius):
        self.reference_frame = reference_frame.astype(np.float32)
        self.bg_threshold = bg_threshold
        self.curr_threshold = curr_threshold
        self.search_radius = search_radius
        self.trackers = []
        self.next_id = 0
        self.all_paths = []  # 存储所有完成的路径

    def update_reference(self, new_reference):
        """更新参考帧"""
        self.reference_frame = new_reference.astype(np.float32)

    def is_new_growth(self, x, y):
        """
        判断该点是否是新生长

        条件：参考帧中该位置较暗（不是老结构）
        """
        h, w = self.reference_frame.shape
        if x < 0 or x >= w or y < 0 or y >= h:
            return False

        ref_brightness = self.reference_frame[y, x]
        return ref_brightness < self.bg_threshold

    def find_growth_candidates(self, current_frame, tip_x, tip_y):
        """
        从末端位置寻找生长候选点

        条件：
        1. 当前帧有信号
        2. 参考帧无信号（是新生长）
        3. 在搜索半径内
        """
        h, w = current_frame.shape
        candidates = []

        r = self.search_radius
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dx == 0 and dy == 0:
                    continue

                nx, ny = tip_x + dx, tip_y + dy

                # 边界检查
                if nx < 0 or nx >= w or ny < 0 or ny >= h:
                    continue

                # 当前帧有信号
                if current_frame[ny, nx] < self.curr_threshold:
                    continue

                # 关键：检查是否是新生长（参考帧暗）
                if not self.is_new_growth(nx, ny):
                    continue

                # 计算距离和方向得分
                dist = np.sqrt(dx ** 2 + dy ** 2)
                if dist > r:
                    continue

                brightness = current_frame[ny, nx]
                score = brightness / (dist + 1)  # 亮度高、距离近得分高

                candidates.append((nx, ny, score, dist))

        # 按得分排序
        candidates.sort(key=lambda x: -x[2])
        return candidates

    def trace_growth(self, current_frame, start_x, start_y, max_steps=100):
        """
        从起点开始追踪生长路径

        返回路径点列表
        """
        path = [(start_x, start_y)]
        visited = set()
        visited.add((start_x, start_y))

        cx, cy = start_x, start_y

        for _ in range(max_steps):
            # 寻找下一个点（小范围，逐步追踪）
            candidates = []

            for dy in range(-3, 4):
                for dx in range(-3, 4):
                    if dx == 0 and dy == 0:
                        continue

                    nx, ny = cx + dx, cy + dy

                    if (nx, ny) in visited:
                        continue

                    h, w = current_frame.shape
                    if nx < 0 or nx >= w or ny < 0 or ny >= h:
                        continue

                    # 当前帧有信号
                    if current_frame[ny, nx] < self.curr_threshold:
                        continue

                    # 必须是新生长
                    if not self.is_new_growth(nx, ny):
                        continue

                    dist = np.sqrt(dx ** 2 + dy ** 2)
                    brightness = current_frame[ny, nx]
                    score = brightness / (dist + 0.5)

                    candidates.append((nx, ny, score))

            if not candidates:
                break

            # 选择最佳候选
            candidates.sort(key=lambda x: -x[2])
            nx, ny, _ = candidates[0]

            path.append((nx, ny))
            visited.add((nx, ny))
            cx, cy = nx, ny

        return path

    def find_new_growth_seeds(self, current_frame, existing_tips):
        """
        在当前帧中找到新的生长起点

        条件：
        1. 当前帧有信号
        2. 参考帧无信号
        3. 远离现有追踪器
        """
        h, w = current_frame.shape

        # 计算新生长区域
        new_growth_mask = np.logical_and(
            current_frame > self.curr_threshold,
            self.reference_frame < self.bg_threshold
        )

        # 找到连通域
        new_growth_uint8 = (new_growth_mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(new_growth_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        seeds = []
        for contour in contours:
            if cv2.contourArea(contour) < 5:  # 过滤太小的区域
                continue

            # 找到轮廓的一个端点作为种子
            M = cv2.moments(contour)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                # 检查是否远离现有追踪器
                too_close = False
                for tip in existing_tips:
                    dist = np.sqrt((cx - tip[0]) ** 2 + (cy - tip[1]) ** 2)
                    if dist < MERGE_DISTANCE * 2:
                        too_close = True
                        break

                if not too_close:
                    seeds.append((cx, cy))

        return seeds[:MAX_TRACKERS - len(self.trackers)]  # 限制数量

    def update(self, current_frame, frame_idx):
        """
        更新所有追踪器
        """
        # 收集现有末端位置
        existing_tips = [t.get_tip() for t in self.trackers if t.active]

        # 更新现有追踪器
        for tracker in self.trackers:
            if not tracker.active:
                continue

            tip_x, tip_y = tracker.get_tip()

            # 寻找生长候选
            candidates = self.find_growth_candidates(current_frame, tip_x, tip_y)

            if candidates:
                # 追踪到新点
                nx, ny, score, dist = candidates[0]

                # 如果距离较远，尝试追踪中间路径
                if dist > 3:
                    path = self.trace_growth(current_frame, tip_x, tip_y, max_steps=int(dist * 2))
                    for px, py in path[1:]:
                        tracker.add_point((px, py))
                else:
                    tracker.add_point((nx, ny))
            else:
                tracker.stall()

        # 保存并移除停用的追踪器
        for tracker in self.trackers:
            if not tracker.active and tracker.total_length >= MIN_GROWTH_LENGTH:
                self.all_paths.append({
                    'id': tracker.id,
                    'start_frame': tracker.start_frame,
                    'end_frame': frame_idx,
                    'path': tracker.path.copy(),
                    'length': tracker.total_length
                })

        self.trackers = [t for t in self.trackers if t.active]

        # 寻找新的生长起点
        existing_tips = [t.get_tip() for t in self.trackers if t.active]
        new_seeds = self.find_new_growth_seeds(current_frame, existing_tips)

        for sx, sy in new_seeds:
            new_tracker = GrowthTracker((sx, sy), frame_idx, self.next_id)
            self.next_id += 1
            self.trackers.append(new_tracker)

        return len(self.trackers), len(new_seeds)


# =====================================================
# 可视化函数
# =====================================================

def draw_tracking_result(current_frame, reference_frame, tracking_system, frame_idx):
    """
    绘制追踪结果可视化
    """
    h, w = current_frame.shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)

    # 背景：当前帧灰度
    vis[:, :, 0] = current_frame // 2
    vis[:, :, 1] = current_frame // 2
    vis[:, :, 2] = current_frame // 2

    # 标记背景结构（参考帧中已存在）- 暗红色
    old_structure = reference_frame > BACKGROUND_THRESHOLD
    vis[old_structure, 2] = np.minimum(vis[old_structure, 2].astype(int) + 50, 255).astype(np.uint8)

    # 标记新生长区域 - 绿色
    new_growth = np.logical_and(
        current_frame > CURRENT_THRESHOLD,
        reference_frame < BACKGROUND_THRESHOLD
    )
    vis[new_growth, 1] = 255

    # 绘制所有追踪路径
    colors = [
        (255, 100, 100),  # 浅蓝
        (100, 255, 100),  # 浅绿
        (100, 100, 255),  # 浅红
        (255, 255, 100),  # 青色
        (255, 100, 255),  # 紫色
        (100, 255, 255),  # 黄色
    ]

    for tracker in tracking_system.trackers:
        if len(tracker.path) < 2:
            continue

        color = colors[tracker.id % len(colors)]

        # 画路径
        for i in range(1, len(tracker.path)):
            p1 = tracker.path[i - 1]
            p2 = tracker.path[i]
            cv2.line(vis, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), color, 2)

        # 画末端（大点）
        tip = tracker.get_tip()
        cv2.circle(vis, (int(tip[0]), int(tip[1])), 5, color, -1)
        cv2.circle(vis, (int(tip[0]), int(tip[1])), 7, (255, 255, 255), 1)

        # 标注追踪器ID
        cv2.putText(vis, f"#{tracker.id}", (int(tip[0]) + 8, int(tip[1]) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    # 信息面板
    cv2.putText(vis, f"Frame {frame_idx}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(vis, f"Trackers: {len(tracking_system.trackers)}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)

    return vis


# =====================================================
# 主函数
# =====================================================

def main():
    print("╔" + "═" * 58 + "╗")
    print("║" + "  步骤7: 神经元生长路径追踪".center(46) + "║")
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
    tracking_dir = os.path.join(OUTPUT_DIR, "frames_tracking")
    os.makedirs(tracking_dir, exist_ok=True)

    # === 预扫描尺寸 ===
    print(f"\n🔍 预扫描...")
    min_h, min_w = float('inf'), float('inf')
    for filename in frame_files:
        img = load_image(os.path.join(INPUT_DIR, filename))
        if img is not None:
            h, w = img.shape
            min_h, min_w = min(min_h, h), min(min_w, w)
    min_h, min_w = make_even(min_h), make_even(min_w)
    print(f"   统一尺寸: {min_w}×{min_h}")

    # === 打印参数 ===
    print(f"\n🔧 参数:")
    print(f"   参考模式: {REFERENCE_MODE}")
    if REFERENCE_MODE == "interval":
        print(f"   参考间隔: {REFERENCE_INTERVAL} 帧")
    print(f"   背景阈值: {BACKGROUND_THRESHOLD}")
    print(f"   搜索半径: {SEARCH_RADIUS} px")

    # === 加载参考帧 ===
    first_frame = load_image(os.path.join(INPUT_DIR, frame_files[0]))
    first_frame = first_frame[:min_h, :min_w]

    # 存储所有帧用于interval模式
    all_frames = [first_frame.copy()]

    # === 初始化追踪系统 ===
    tracking_system = GrowthTrackingSystem(
        reference_frame=first_frame,
        bg_threshold=BACKGROUND_THRESHOLD,
        curr_threshold=CURRENT_THRESHOLD,
        search_radius=SEARCH_RADIUS
    )

    # === 视频设置 ===
    video_w, video_h = min_w, min_h
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    video_tracking_path = os.path.join(OUTPUT_DIR, "growth_tracking.mp4")
    out_tracking = cv2.VideoWriter(video_tracking_path, fourcc, FPS, (video_w, video_h), isColor=True)

    # === 处理 ===
    print(f"\n🔄 追踪中...")

    stats = []

    for idx, filename in enumerate(frame_files):
        current = load_image(os.path.join(INPUT_DIR, filename))
        if current is None:
            continue

        current = current[:min_h, :min_w]

        # 存储帧（用于interval模式）
        if REFERENCE_MODE == "interval":
            all_frames.append(current.copy())

            # 更新参考帧
            ref_idx = max(0, idx - REFERENCE_INTERVAL)
            if ref_idx < len(all_frames):
                tracking_system.update_reference(all_frames[ref_idx])

        # 更新追踪
        num_trackers, num_new = tracking_system.update(current, idx)

        # 可视化
        vis = draw_tracking_result(current, tracking_system.reference_frame, tracking_system, idx)

        # 保存
        save_image(os.path.join(tracking_dir, f"frame_{idx:04d}.png"), vis)
        out_tracking.write(vis)

        # 统计
        total_path_length = sum(t.total_length for t in tracking_system.trackers)
        stats.append({
            'frame': idx,
            'num_trackers': num_trackers,
            'new_trackers': num_new,
            'total_path_length': total_path_length
        })

        if (idx + 1) % 20 == 0 or idx == num_frames - 1:
            elapsed = time.time() - t_start
            print(
                f"   [{idx + 1:3d}/{num_frames}] 追踪器: {num_trackers} | 新增: {num_new} | 总长: {total_path_length:.0f}px")

    out_tracking.release()

    # === 保存所有路径 ===
    # 添加仍在活动的追踪器
    for tracker in tracking_system.trackers:
        if tracker.total_length >= MIN_GROWTH_LENGTH:
            tracking_system.all_paths.append({
                'id': tracker.id,
                'start_frame': tracker.start_frame,
                'end_frame': num_frames - 1,
                'path': tracker.path.copy(),
                'length': tracker.total_length
            })

    # 保存路径数据
    paths_path = os.path.join(OUTPUT_DIR, "growth_paths.csv")
    with open(paths_path, 'w', encoding='utf-8') as f:
        f.write("tracker_id,start_frame,end_frame,length,num_points,path\n")
        for p in tracking_system.all_paths:
            path_str = ";".join([f"{x},{y}" for x, y in p['path']])
            f.write(
                f"{p['id']},{p['start_frame']},{p['end_frame']},{p['length']:.1f},{len(p['path'])},\"{path_str}\"\n")

    # 保存统计
    stats_path = os.path.join(OUTPUT_DIR, "tracking_statistics.csv")
    with open(stats_path, 'w', encoding='utf-8') as f:
        f.write("frame,num_trackers,new_trackers,total_path_length\n")
        for s in stats:
            f.write(f"{s['frame']},{s['num_trackers']},{s['new_trackers']},{s['total_path_length']:.1f}\n")

    # === 生成最终路径汇总图 ===
    print(f"\n🎨 生成路径汇总图...")

    final_vis = np.zeros((min_h, min_w, 3), dtype=np.uint8)

    # 背景：最后一帧
    last_frame = load_image(os.path.join(INPUT_DIR, frame_files[-1]))[:min_h, :min_w]
    final_vis[:, :, 0] = last_frame // 3
    final_vis[:, :, 1] = last_frame // 3
    final_vis[:, :, 2] = last_frame // 3

    # 绘制所有路径
    colors = [(255, 100, 100), (100, 255, 100), (100, 100, 255),
              (255, 255, 100), (255, 100, 255), (100, 255, 255)]

    for p in tracking_system.all_paths:
        color = colors[p['id'] % len(colors)]
        path = p['path']

        for i in range(1, len(path)):
            p1, p2 = path[i - 1], path[i]
            cv2.line(final_vis, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), color, 2)

        # 起点
        cv2.circle(final_vis, (int(path[0][0]), int(path[0][1])), 5, (0, 255, 0), -1)
        # 终点
        cv2.circle(final_vis, (int(path[-1][0]), int(path[-1][1])), 5, (0, 0, 255), -1)

    save_image(os.path.join(OUTPUT_DIR, "all_growth_paths.png"), final_vis)

    # === 摘要 ===
    total_time = time.time() - t_start

    print("\n" + "=" * 60)
    print("📊 追踪统计")
    print("=" * 60)
    print(f"\n   检测到的生长路径: {len(tracking_system.all_paths)} 条")

    if tracking_system.all_paths:
        lengths = [p['length'] for p in tracking_system.all_paths]
        print(f"   路径长度:")
        print(f"     总计:  {sum(lengths):.0f} 像素")
        print(f"     平均:  {np.mean(lengths):.1f} 像素/条")
        print(f"     最长:  {max(lengths):.1f} 像素")

    print(f"\n   耗时: {total_time:.1f} 秒")

    print("\n📁 输出")
    print(f"""
    {OUTPUT_DIR}/
    ├── frames_tracking/         # 逐帧追踪可视化
    ├── growth_tracking.mp4      # 追踪视频
    ├── all_growth_paths.png     # 所有路径汇总图
    ├── growth_paths.csv         # 路径数据
    └── tracking_statistics.csv
    """)

    print("""
📌 颜色含义:
   🟢 绿色区域 = 新生长部分 (当前亮，参考暗)
   🔴 暗红区域 = 背景老结构 (参考帧已存在)
   彩色线条   = 追踪路径
   ⚪ 白圈    = 追踪末端
    """)

    print("✅ 完成！")


if __name__ == "__main__":
    main()
