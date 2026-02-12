import cv2
import numpy as np
import json
import os
from datetime import datetime


def generate_colors(n):
    colors = []
    for i in range(n):
        hue = int(180 * i / n)
        hsv = np.uint8([[[hue, 255, 255]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
        colors.append((int(bgr[0]), int(bgr[1]), int(bgr[2])))
    return colors


class NeuronMarkerTool:
    """
    神经元标记工具
    - 支持在任意帧标记
    - 同一神经元可在多帧标记多个点
    - 保存/加载标记数据
    """

    def __init__(self, max_neurons=15):
        self.max_neurons = max_neurons
        self.colors = generate_colors(max_neurons)

        # 标记数据结构
        # {
        #   neuron_id: {
        #     "color": [b, g, r],
        #     "marks": [
        #       {"frame": 帧号, "x": x坐标, "y": y坐标},
        #       ...
        #     ]
        #   }
        # }
        self.neuron_marks = {}

        # 当前状态
        self.current_neuron_id = 0
        self.current_frame_idx = 0
        self.total_frames = 0
        self.video_path = ""
        self.display_scale = 1.0
        self.cap = None
        self.current_frame = None

        # 视频信息
        self.video_info = {}

    def load_video(self, video_path):
        """加载视频"""
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))

        self.video_info = {
            "path": video_path,
            "width": width,
            "height": height,
            "fps": fps,
            "total_frames": self.total_frames
        }

        # 计算显示缩放
        max_display_width = 1400
        max_display_height = 800
        self.display_scale = min(1.0, max_display_width / width, max_display_height / height)

        print(f"✓ 加载视频: {width}x{height}, {fps}fps, {self.total_frames}帧")

        return True

    def read_frame(self, frame_idx):
        """读取指定帧"""
        if self.cap is None:
            return None

        frame_idx = max(0, min(frame_idx, self.total_frames - 1))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()

        if ret:
            self.current_frame_idx = frame_idx
            self.current_frame = frame
            return frame
        return None

    def add_mark(self, neuron_id, frame_idx, x, y):
        """添加标记点"""
        if neuron_id not in self.neuron_marks:
            color_idx = len(self.neuron_marks) % self.max_neurons
            self.neuron_marks[neuron_id] = {
                "color": list(self.colors[color_idx]),
                "marks": []
            }

        # 检查是否已存在相同帧的标记
        for mark in self.neuron_marks[neuron_id]["marks"]:
            if mark["frame"] == frame_idx:
                # 更新已有标记
                mark["x"] = x
                mark["y"] = y
                return

        # 添加新标记
        self.neuron_marks[neuron_id]["marks"].append({
            "frame": frame_idx,
            "x": x,
            "y": y
        })

    def remove_last_mark(self, neuron_id):
        """删除神经元的最后一个标记"""
        if neuron_id in self.neuron_marks:
            if len(self.neuron_marks[neuron_id]["marks"]) > 0:
                removed = self.neuron_marks[neuron_id]["marks"].pop()
                return removed
        return None

    def remove_mark_at_frame(self, neuron_id, frame_idx):
        """删除指定帧的标记"""
        if neuron_id in self.neuron_marks:
            marks = self.neuron_marks[neuron_id]["marks"]
            for i, mark in enumerate(marks):
                if mark["frame"] == frame_idx:
                    return marks.pop(i)
        return None

    def delete_neuron(self, neuron_id):
        """删除整个神经元"""
        if neuron_id in self.neuron_marks:
            del self.neuron_marks[neuron_id]
            return True
        return False

    def get_marks_at_frame(self, frame_idx):
        """获取指定帧的所有标记"""
        marks = []
        for neuron_id, data in self.neuron_marks.items():
            for mark in data["marks"]:
                if mark["frame"] == frame_idx:
                    marks.append({
                        "neuron_id": neuron_id,
                        "x": mark["x"],
                        "y": mark["y"],
                        "color": data["color"]
                    })
        return marks

    def get_all_neuron_ids(self):
        """获取所有神经元ID"""
        return list(self.neuron_marks.keys())

    def save_marks(self, save_path):
        """保存标记数据到JSON"""
        data = {
            "version": "1.0",
            "created": datetime.now().isoformat(),
            "video_info": self.video_info,
            "neurons": self.neuron_marks
        }

        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"✓ 保存标记到: {save_path}")
        return True

    def load_marks(self, load_path):
        """从JSON加载标记数据"""
        if not os.path.exists(load_path):
            print(f"⚠ 文件不存在: {load_path}")
            return False

        with open(load_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.neuron_marks = data.get("neurons", {})
        # 转换key为int（JSON会把int key转成str）
        self.neuron_marks = {int(k): v for k, v in self.neuron_marks.items()}

        print(f"✓ 加载标记: {len(self.neuron_marks)} 个神经元")
        for nid, ndata in self.neuron_marks.items():
            print(f"   N{nid}: {len(ndata['marks'])} 个标记点")

        return True

    def mouse_callback(self, event, x, y, flags, param):
        """鼠标回调"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # 左键添加标记
            orig_x = int(x / self.display_scale)
            orig_y = int(y / self.display_scale)

            self.add_mark(self.current_neuron_id, self.current_frame_idx, orig_x, orig_y)
            print(f"  + N{self.current_neuron_id} @ 帧{self.current_frame_idx + 1}: ({orig_x}, {orig_y})")

        elif event == cv2.EVENT_RBUTTONDOWN:
            # 右键删除当前帧的标记
            removed = self.remove_mark_at_frame(self.current_neuron_id, self.current_frame_idx)
            if removed:
                print(f"  - N{self.current_neuron_id} @ 帧{self.current_frame_idx + 1}: 已删除")

    def draw_frame(self):
        """绘制当前帧及标记"""
        if self.current_frame is None:
            return None

        display = self.current_frame.copy()
        h, w = display.shape[:2]

        # 绘制当前帧所有标记
        marks_here = self.get_marks_at_frame(self.current_frame_idx)
        for mark in marks_here:
            color = tuple(mark["color"])
            x, y = mark["x"], mark["y"]
            nid = mark["neuron_id"]

            # 当前选中的神经元用大圆
            if nid == self.current_neuron_id:
                cv2.circle(display, (x, y), 12, color, 2)
                cv2.circle(display, (x, y), 8, color, -1)
            else:
                cv2.circle(display, (x, y), 6, color, -1)

            cv2.putText(display, str(nid), (x + 10, y + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 绘制其他帧的标记（半透明）
        for nid, ndata in self.neuron_marks.items():
            color = tuple(ndata["color"])
            for mark in ndata["marks"]:
                if mark["frame"] != self.current_frame_idx:
                    x, y = mark["x"], mark["y"]
                    # 用小十字表示其他帧的标记
                    cv2.drawMarker(display, (x, y), color, cv2.MARKER_CROSS, 8, 1)

        # 信息栏
        info1 = f"Frame: {self.current_frame_idx + 1}/{self.total_frames}"
        info2 = f"Current Neuron: N{self.current_neuron_id} (1-9 switch, 0=N10)"
        info3 = f"Neurons: {len(self.neuron_marks)} | Marks here: {len(marks_here)}"

        cv2.putText(display, info1, (11, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(display, info1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.putText(display, info2, (11, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(display, info2, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    tuple(self.colors[self.current_neuron_id % len(self.colors)]), 2)

        cv2.putText(display, info3, (11, 81), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(display, info3, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 2)

        # 进度条
        bar_x, bar_y = 20, h - 25
        bar_w = w - 40
        progress = self.current_frame_idx / max(1, self.total_frames - 1)
        cv2.rectangle(display, (bar_x, bar_y), (bar_x + bar_w, bar_y + 10), (50, 50, 50), -1)
        cv2.rectangle(display, (bar_x, bar_y), (bar_x + int(bar_w * progress), bar_y + 10), (0, 200, 0), -1)

        # 在进度条上标记有标记点的帧
        for nid, ndata in self.neuron_marks.items():
            color = tuple(ndata["color"])
            for mark in ndata["marks"]:
                mark_x = bar_x + int(bar_w * mark["frame"] / max(1, self.total_frames - 1))
                cv2.line(display, (mark_x, bar_y - 3), (mark_x, bar_y + 13), color, 2)

        # 图例
        legend_y = 100
        cv2.putText(display, "Neurons:", (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        for i, (nid, ndata) in enumerate(self.neuron_marks.items()):
            color = tuple(ndata["color"])
            y_pos = legend_y + 15 + i * 16
            cv2.rectangle(display, (10, y_pos), (20, y_pos + 12), color, -1)
            mark_count = len(ndata["marks"])
            frames = sorted(set(m["frame"] + 1 for m in ndata["marks"]))
            frames_str = ",".join(map(str, frames[:5])) + ("..." if len(frames) > 5 else "")
            cv2.putText(display, f"N{nid}: {mark_count}pts @ frames [{frames_str}]",
                        (25, y_pos + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)

        return display

    def run(self, video_path, marks_path=None):
        """运行标记工具"""
        self.load_video(video_path)

        # 如果有已存在的标记，加载
        if marks_path and os.path.exists(marks_path):
            self.load_marks(marks_path)

        # 默认保存路径
        if marks_path is None:
            base = os.path.splitext(video_path)[0]
            marks_path = base + "_marks.json"

        self.read_frame(0)

        window_name = 'Neuron Marker Tool'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, self.mouse_callback)

        # 创建滑动条
        def on_trackbar(val):
            self.read_frame(val)

        cv2.createTrackbar('Frame', window_name, 0, self.total_frames - 1, on_trackbar)

        print("\n" + "=" * 70)
        print("神经元标记工具")
        print("=" * 70)
        print("帧导航:")
        print("  A/← : 上一帧       D/→ : 下一帧")
        print("  W/↑ : 前进10帧     S/↓ : 后退10帧")
        print("  滑动条 : 快速跳转")
        print("")
        print("标记操作:")
        print("  左键点击 : 在当前位置添加标记")
        print("  右键点击 : 删除当前帧的标记")
        print("  1-9, 0 : 切换到神经元 N1-N9, N10")
        print("  N : 创建新神经元")
        print("  Delete : 删除当前神经元所有标记")
        print("")
        print("保存/加载:")
        print("  Ctrl+S : 保存标记")
        print("  Ctrl+L : 加载标记")
        print("  Enter : 保存并退出")
        print("  Q/Esc : 退出(会询问保存)")
        print("=" * 70 + "\n")

        while True:
            display = self.draw_frame()
            if display is not None:
                display_resized = cv2.resize(display, None,
                                             fx=self.display_scale, fy=self.display_scale)
                cv2.imshow(window_name, display_resized)

            cv2.setTrackbarPos('Frame', window_name, self.current_frame_idx)

            key = cv2.waitKey(30) & 0xFF

            # 获取滑动条位置
            trackbar_pos = cv2.getTrackbarPos('Frame', window_name)
            if trackbar_pos != self.current_frame_idx:
                self.read_frame(trackbar_pos)

            # 帧导航
            if key == ord('a') or key == 81:  # A / Left
                self.read_frame(self.current_frame_idx - 1)
            elif key == ord('d') or key == 83:  # D / Right
                self.read_frame(self.current_frame_idx + 1)
            elif key == ord('w') or key == 82:  # W / Up
                self.read_frame(self.current_frame_idx + 10)
            elif key == ord('s') or key == 84:  # S / Down
                self.read_frame(self.current_frame_idx - 10)

            # 切换神经元 (1-9, 0)
            elif ord('1') <= key <= ord('9'):
                self.current_neuron_id = key - ord('1')
                print(f"  切换到 N{self.current_neuron_id}")
            elif key == ord('0'):
                self.current_neuron_id = 9
                print(f"  切换到 N{self.current_neuron_id}")

            # N: 创建新神经元
            elif key == ord('n') or key == ord('N'):
                existing_ids = set(self.neuron_marks.keys())
                new_id = 0
                while new_id in existing_ids:
                    new_id += 1
                self.current_neuron_id = new_id
                print(f"  创建新神经元 N{self.current_neuron_id}")

            # Delete: 删除当前神经元
            elif key == 127 or key == 8:  # Delete or Backspace
                if self.delete_neuron(self.current_neuron_id):
                    print(f"  删除 N{self.current_neuron_id}")

            # Ctrl+S: 保存
            elif key == 19:  # Ctrl+S
                self.save_marks(marks_path)

            # Enter: 保存并退出
            elif key == 13 or key == 10:
                self.save_marks(marks_path)
                print("✓ 保存并退出")
                break

            # Q/Esc: 退出
            elif key == ord('q') or key == ord('Q') or key == 27:
                if len(self.neuron_marks) > 0:
                    print("是否保存? (Y/N)")
                    save_key = cv2.waitKey(0) & 0xFF
                    if save_key == ord('y') or save_key == ord('Y'):
                        self.save_marks(marks_path)
                break

        cv2.destroyAllWindows()
        if self.cap:
            self.cap.release()

        return marks_path


if __name__ == "__main__":
    VIDEO_PATH = r"F:\工作文件\RA\python\项目汇总\神经图像\neuron_growth_50.mp4"
    MARKS_PATH = r"F:\工作文件\RA\python\项目汇总\神经图像\neuron_marks.json"

    tool = NeuronMarkerTool(max_neurons=15)
    tool.run(VIDEO_PATH, MARKS_PATH)
