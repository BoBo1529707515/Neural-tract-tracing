import numpy as np
import cv2
import random


def generate_neuron_video_limited(output_file='neuron_growth_50.mp4', width=1280, height=720, duration_sec=5, fps=30):
    # 初始化
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    num_frames = duration_sec * fps
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    tips = []

    # 【参数设置】
    max_neurons = 50  # <--- 核心修改：最大神经元数量限制
    initial_count = 15  # 初始种子数量
    current_count = 0  # 当前计数器

    # 1. 初始生成种子
    for _ in range(initial_count):
        y = np.random.randint(height // 4, 3 * height // 4)  # 让初始点集中在中间高度
        tips.append({
            'x': 0.0,
            'y': float(y),
            'angle': np.random.normal(0, 0.2),
            'speed': np.random.uniform(5, 12),  # 稍微快一点，以便5秒内跑完屏幕
            'thickness': np.random.uniform(2, 4),
            'intensity': np.random.randint(120, 255),
            'active': True
        })
        current_count += 1

    print(f"开始生成视频，限制最大数量为: {max_neurons}")

    for frame in range(num_frames):
        # 2. 随机添加新种子 (仅当总数未达标时)
        # 在前2秒内，如果数量很少，偶尔在左侧补充新的
        if frame < num_frames // 2.5 and current_count < max_neurons and np.random.random() < 0.05:
            y = np.random.randint(0, height)
            tips.append({
                'x': 0.0,
                'y': float(y),
                'angle': np.random.normal(0, 0.3),
                'speed': np.random.uniform(5, 10),
                'thickness': np.random.uniform(2, 3),
                'intensity': np.random.randint(150, 255),
                'active': True
            })
            current_count += 1

        # 3. 生长循环
        for tip in tips[:]:  # 遍历当前所有生长点
            if not tip['active']:
                continue

            prev_x, prev_y = tip['x'], tip['y']

            # 运动逻辑：随机摆动 + 向右趋势
            noise = np.random.normal(0, 0.1)
            tip['angle'] += noise
            tip['angle'] = np.clip(tip['angle'], -1.2, 1.2)  # 限制角度防止倒车

            tip['x'] += tip['speed'] * np.cos(tip['angle'])
            tip['y'] += tip['speed'] * np.sin(tip['angle'])

            # 绘制轨迹
            color = (0, int(tip['intensity']), 0)
            thickness = max(1, int(tip['thickness']))
            cv2.line(canvas, (int(prev_x), int(prev_y)), (int(tip['x']), int(tip['y'])), color, thickness)

            # 4. 分支逻辑 (核心限制)
            # 只有当 current_count < max_neurons 时才允许分叉
            if current_count < max_neurons and tip['thickness'] > 1.5 and np.random.random() < 0.02:
                branch = tip.copy()
                branch['angle'] += np.random.uniform(-0.5, 0.5)  # 分支角度偏转
                branch['thickness'] *= 0.7
                branch['speed'] *= 0.9
                tips.append(branch)
                current_count += 1  # 计数器+1

            # 边界检查
            if tip['x'] > width or tip['y'] < 0 or tip['y'] > height:
                tip['active'] = False

            # 随机停止
            if np.random.random() < 0.005:
                tip['active'] = False

        # 写入视频帧 (可选：绘制临时的生长锥高亮)
        frame_display = canvas.copy()
        for tip in tips:
            if tip['active']:
                cv2.circle(frame_display, (int(tip['x']), int(tip['y'])), 2, (100, 255, 100), -1)

        out.write(frame_display)

    out.release()
    print(f"视频生成完毕，最终神经元数量: {current_count}")


# 运行
generate_neuron_video_limited()
