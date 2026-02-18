"""
数据管理模块
"""
import json
import os
import numpy as np
import cv2
from .config import Config

def gen_colors(n):
    colors = []
    for i in range(n):
        h = int(180 * i / n)
        hsv = np.uint8([[[h, 255, 255]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
        colors.append((int(bgr[0]), int(bgr[1]), int(bgr[2])))
    return colors

class Data:
    def __init__(self):
        self.marks = {}  # {nid: {"color": [...], "marks": [...]}}
        self.trajs = {}  # {nid: [(y,x), ...]}
        self.colors = gen_colors(Config.MAX_NEURONS)

    def add_mark(self, nid, fidx, x, y):
        if nid not in self.marks:
            self.marks[nid] = {"color": list(self.colors[nid % len(self.colors)]), "marks": []}
        for m in self.marks[nid]["marks"]:
            if m["frame"] == fidx and np.sqrt((m["x"]-x)**2 + (m["y"]-y)**2) < 5:
                m["x"], m["y"] = x, y
                return
        self.marks[nid]["marks"].append({"frame": fidx, "x": x, "y": y})

    def del_mark_near(self, nid, fidx, x, y, r=20):
        if nid not in self.marks: return
        ms = self.marks[nid]["marks"]
        best, bd = None, 1e9
        for i, m in enumerate(ms):
            if m["frame"] == fidx:
                d = np.sqrt((m["x"]-x)**2 + (m["y"]-y)**2)
                if d < r and d < bd: bd, best = d, i
        if best is not None: ms.pop(best)

    def get_marks(self, nid):
        return self.marks[nid]["marks"] if nid in self.marks else []

    def color(self, nid):
        return tuple(self.marks[nid]["color"]) if nid in self.marks else tuple(self.colors[nid % len(self.colors)])

    def del_neuron(self, nid):
        self.marks.pop(nid, None)
        self.trajs.pop(nid, None)

    def new_id(self):
        ids = set(self.marks) | set(self.trajs)
        i = 0
        while i in ids: i += 1
        return i

    def save(self, path):
        d = {"version": "9.0", "marks": self.marks,
             "trajs": {str(k): [list(p) for p in v] for k, v in self.trajs.items()}}
        with open(path, 'w') as f: json.dump(d, f, indent=2)
        print(f"✓ 数据已保存: {path}")

    def load(self, path):
        if not os.path.exists(path): return False
        with open(path) as f: d = json.load(f)
        self.marks = {int(k): v for k, v in d.get("marks", {}).items()}
        self.trajs = {int(k): [tuple(p) for p in v] for k, v in d.get("trajs", {}).items()}
        print(f"✓ 数据已加载: {len(self.marks)}个神经元")
        return True
