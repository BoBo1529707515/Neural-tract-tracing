"""
数据管理模块
负责标记数据和轨迹数据的存储、加载、增删改查
"""

import json
import os
from datetime import datetime
from .config import Config, generate_neuron_colors


class DataManager:
    """
    数据管理类

    功能:
        - 标记数据的增删改查
        - 轨迹数据的管理
        - JSON格式的保存和加载
    """

    def __init__(self):
        """初始化数据管理器"""
        self.neuron_marks = {}  # 神经元标记 {nid: {"color": [], "marks": [...]}}
        self.neuron_trajectories = {}  # 神经元轨迹 {nid: [(y, x), ...]}
        self.video_info = {}  # 视频信息
        self.colors = generate_neuron_colors(Config.MAX_NEURONS)

    def set_video_info(self, info):
        """
        设置视频信息

        参数:
            info: 视频信息字典
        """
        self.video_info = info

    # ==================== 标记管理 ====================

    def add_mark(self, neuron_id, frame_idx, x, y):
        """
        添加或更新标记点

        参数:
            neuron_id: 神经元ID
            frame_idx: 帧索引
            x, y: 像素坐标
        """
        # 确保神经元存在
        if neuron_id not in self.neuron_marks:
            color = list(self.colors[neuron_id % len(self.colors)])
            self.neuron_marks[neuron_id] = {"color": color, "marks": []}

        # 检查是否已有该帧的标记
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

    def remove_mark_at_frame(self, neuron_id, frame_idx):
        """
        删除指定帧的标记

        参数:
            neuron_id: 神经元ID
            frame_idx: 帧索引

        返回:
            dict: 被删除的标记，未找到返回None
        """
        if neuron_id not in self.neuron_marks:
            return None

        marks = self.neuron_marks[neuron_id]["marks"]
        for i, mark in enumerate(marks):
            if mark["frame"] == frame_idx:
                return marks.pop(i)
        return None

    def remove_last_mark(self, neuron_id):
        """
        删除神经元的最后一个标记

        参数:
            neuron_id: 神经元ID

        返回:
            dict: 被删除的标记，未找到返回None
        """
        if neuron_id in self.neuron_marks:
            marks = self.neuron_marks[neuron_id]["marks"]
            if marks:
                return marks.pop()
        return None

    def get_marks_at_frame(self, frame_idx):
        """
        获取指定帧的所有标记

        参数:
            frame_idx: 帧索引

        返回:
            list: 标记列表 [{"neuron_id", "x", "y", "color"}, ...]
        """
        result = []
        for nid, data in self.neuron_marks.items():
            for mark in data["marks"]:
                if mark["frame"] == frame_idx:
                    result.append({
                        "neuron_id": nid,
                        "x": mark["x"],
                        "y": mark["y"],
                        "color": data["color"]
                    })
        return result

    def get_neuron_marks(self, neuron_id):
        """
        获取指定神经元的所有标记

        参数:
            neuron_id: 神经元ID

        返回:
            list: 标记列表
        """
        if neuron_id in self.neuron_marks:
            return self.neuron_marks[neuron_id]["marks"]
        return []

    # ==================== 神经元管理 ====================

    def delete_neuron(self, neuron_id):
        """
        删除指定神经元的所有数据（标记和轨迹）

        参数:
            neuron_id: 神经元ID

        返回:
            bool: 是否删除了数据
        """
        deleted = False

        if neuron_id in self.neuron_marks:
            del self.neuron_marks[neuron_id]
            deleted = True
            print(f"  ✓ 删除 N{neuron_id} 的标记数据")

        if neuron_id in self.neuron_trajectories:
            del self.neuron_trajectories[neuron_id]
            deleted = True
            print(f"  ✓ 删除 N{neuron_id} 的轨迹数据")

        return deleted

    def get_all_neuron_ids(self):
        """
        获取所有神经元ID

        返回:
            set: 神经元ID集合
        """
        return set(self.neuron_marks.keys()) | set(self.neuron_trajectories.keys())

    def get_new_neuron_id(self):
        """
        获取新的神经元ID（未使用的最小ID）

        返回:
            int: 新的神经元ID
        """
        existing = self.get_all_neuron_ids()
        nid = 0
        while nid in existing:
            nid += 1
        return nid

    def get_neuron_color(self, neuron_id):
        """
        获取神经元颜色

        参数:
            neuron_id: 神经元ID

        返回:
            tuple: BGR颜色
        """
        if neuron_id in self.neuron_marks:
            return tuple(self.neuron_marks[neuron_id]["color"])
        return tuple(self.colors[neuron_id % len(self.colors)])

    # ==================== 轨迹管理 ====================

    def set_trajectory(self, neuron_id, trajectory):
        """
        设置神经元轨迹

        参数:
            neuron_id: 神经元ID
            trajectory: 轨迹点列表 [(y, x), ...]
        """
        self.neuron_trajectories[neuron_id] = trajectory

    def get_trajectory(self, neuron_id):
        """
        获取神经元轨迹

        参数:
            neuron_id: 神经元ID

        返回:
            list: 轨迹点列表
        """
        return self.neuron_trajectories.get(neuron_id, [])

    def clear_all_trajectories(self):
        """清空所有轨迹"""
        count = len(self.neuron_trajectories)
        self.neuron_trajectories = {}
        print(f"  ✓ 清空 {count} 条轨迹")

    def clear_all_data(self):
        """清空所有数据（标记和轨迹）"""
        mark_count = len(self.neuron_marks)
        traj_count = len(self.neuron_trajectories)
        self.neuron_marks = {}
        self.neuron_trajectories = {}
        print(f"  ✓ 清空所有数据: {mark_count}个标记, {traj_count}条轨迹")

    # ==================== 文件操作 ====================

    def save(self, file_path):
        """
        保存数据到JSON文件

        参数:
            file_path: 保存路径
        """
        data = {
            "version": "7.0",
            "created": datetime.now().isoformat(),
            "video_info": self.video_info,
            "neurons": self.neuron_marks,
            "trajectories": {
                str(k): [list(p) for p in v]
                for k, v in self.neuron_trajectories.items()
            }
        }

        # 确保目录存在
        dir_path = os.path.dirname(file_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"✓ 保存数据: {file_path}")

    def load(self, file_path):
        """
        从JSON文件加载数据

        参数:
            file_path: 文件路径

        返回:
            bool: 是否成功加载
        """
        if not os.path.exists(file_path):
            print(f"⚠ 文件不存在: {file_path}")
            return False

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 加载标记（key转为int）
            self.neuron_marks = {
                int(k): v for k, v in data.get("neurons", {}).items()
            }

            # 加载轨迹
            trajs = data.get("trajectories", {})
            self.neuron_trajectories = {
                int(k): [tuple(p) for p in v] for k, v in trajs.items()
            }

            # 加载视频信息
            self.video_info = data.get("video_info", {})

            print(f"✓ 加载数据: {len(self.neuron_marks)}个标记, "
                  f"{len(self.neuron_trajectories)}条轨迹")
            return True

        except Exception as e:
            print(f"⚠ 加载失败: {e}")
            return False

    # ==================== 统计信息 ====================

    def get_statistics(self):
        """
        获取数据统计信息

        返回:
            dict: 统计信息
        """
        total_marks = sum(len(d["marks"]) for d in self.neuron_marks.values())
        total_traj_points = sum(len(t) for t in self.neuron_trajectories.values())

        return {
            "neuron_count": len(self.get_all_neuron_ids()),
            "mark_count": total_marks,
            "trajectory_count": len(self.neuron_trajectories),
            "total_trajectory_points": total_traj_points
        }
