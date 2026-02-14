"""
神经元追踪工具 v9.0
彻底重构版本
"""

import cv2
import numpy as np
import json
import os
import heapq
from datetime import datetime
from skimage.morphology import skeletonize, remove_small_objects

try:
    import tkinter as tk
    from tkinter import filedialog
    HAS_TK = True
except ImportError:
    HAS_TK = False


# ============================================================================
#                              配置
# ============================================================================

class Config:
    MAX_NEURONS = 50
    DISPLAY_WIDTH = 1600
    DISPLAY_HEIGHT = 1000
    PANEL_HEIGHT = 130

    MIN_ZOOM, MAX_ZOOM, ZOOM_STEP = 0.2, 10.0, 1.4
    DEFAULT_MARK_RADIUS = 1

    # 图像处理
    GREEN_THRESHOLD = 40
    CLAHE_CLIP_LIMIT = 3.0
    MEDIAN_KERNEL = 5
    ADAPTIVE_BLOCK_SIZE = 15
    MIN_OBJECT_SIZE = 50

    # 追踪
    MAX_GAP = 15
    Y_TOLERANCE = 35
    DIRECTION_WEIGHT = 15

    # 颜色
    COLOR_BG = (30, 30, 30)
    COLOR_PANEL = (25, 25, 25)
    COLOR_BTN = (65, 65, 65)
    COLOR_BTN_HOVER = (85, 85, 85)
    COLOR_BTN_ACTIVE = (70, 130, 70)


def gen_colors(n):
    colors = []
    for i in range(n):
        h = int(180 * i / n)
        hsv = np.uint8([[[h, 255, 255]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
        colors.append((int(bgr[0]), int(bgr[1]), int(bgr[2])))
    return colors


# ============================================================================
#                              UI
# ============================================================================

class Button:
    def __init__(self, x, y, w, h, text, color=None):
        self.x, self.y, self.w, self.h = x, y, w, h
        self.text = text
        self.color = color or Config.COLOR_BTN
        self.hover = self.active = False

    def contains(self, px, py):
        return self.x <= px < self.x + self.w and self.y <= py < self.y + self.h

    def draw(self, img):
        bg = Config.COLOR_BTN_ACTIVE if self.active else (Config.COLOR_BTN_HOVER if self.hover else self.color)
        cv2.rectangle(img, (self.x, self.y), (self.x+self.w, self.y+self.h), bg, -1)
        cv2.rectangle(img, (self.x, self.y), (self.x+self.w, self.y+self.h), (100,100,100), 1)
        (tw, th), _ = cv2.getTextSize(self.text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.putText(img, self.text, (self.x+(self.w-tw)//2, self.y+(self.h+th)//2),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)


class InputBox:
    def __init__(self, x, y, w, h, label):
        self.x, self.y, self.w, self.h = x, y, w, h
        self.label = label
        self.text = ""
        self.active = False

    def contains(self, px, py):
        return self.x <= px < self.x + self.w and self.y <= py < self.y + self.h

    def draw(self, img, highlight=None):
        cv2.putText(img, self.label, (self.x, self.y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180,180,180), 1)
        bg = (55, 55, 75) if self.active else (45, 45, 45)
        cv2.rectangle(img, (self.x, self.y), (self.x+self.w, self.y+self.h), bg, -1)
        border = (150, 180, 255) if self.active else (100, 100, 100)
        cv2.rectangle(img, (self.x, self.y), (self.x+self.w, self.y+self.h), border, 1)
        tx = self.x + 6
        if highlight:
            cv2.rectangle(img, (self.x+4, self.y+4), (self.x+20, self.y+self.h-4), highlight, -1)
            tx = self.x + 26
        cv2.putText(img, self.text, (tx, self.y+self.h-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220,220,220), 1)

    def handle_key(self, key):
        if key in [13, 10]: return 'confirm'
        if key == 27: return 'cancel'
        if key in [8, 127]: self.text = self.text[:-1]
        elif ord('0') <= key <= ord('9'): self.text += chr(key)
        return None


# ============================================================================
#                              视频
# ============================================================================

class VideoHandler:
    def __init__(self):
        self.cap = self.writer = None
        self.path = ""
        self.total = self.w = self.h = 0
        self.fps = 30
        self.frame = None
        self.idx = 0

    def load(self, path):
        self.path = path
        self.cap = cv2.VideoCapture(path)
        self.total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        print(f"✓ 视频: {self.w}x{self.h}, {self.fps:.1f}fps, {self.total}帧")

    def read(self, idx):
        idx = max(0, min(idx, self.total - 1))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, self.frame = self.cap.read()
        if ret: self.idx = idx
        return self.frame

    def start_write(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), int(self.fps), (self.w, self.h))

    def write(self, f):
        if self.writer: self.writer.write(f)

    def stop_write(self):
        if self.writer: self.writer.release(); self.writer = None

    def release(self):
        if self.cap: self.cap.release()
        self.stop_write()


# ============================================================================
#                              图像处理
# ============================================================================

class ImageProc:
    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=Config.CLAHE_CLIP_LIMIT, tileGridSize=(8,8))
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

    def is_green(self, f, y, x):
        if not (0 <= y < f.shape[0] and 0 <= x < f.shape[1]): return False
        b, g, r = int(f[y,x,0]), int(f[y,x,1]), int(f[y,x,2])
        return g > 50 and g - r > Config.GREEN_THRESHOLD and g - b > Config.GREEN_THRESHOLD

    def skeleton(self, f):
        b, g, r = cv2.split(f)
        b, g, r = b.astype(np.int16), g.astype(np.int16), r.astype(np.int16)
        mask = (g > 50) & ((g - r) > Config.GREEN_THRESHOLD) & ((g - b) > Config.GREEN_THRESHOLD)
        green = (mask.astype(np.uint8) * 255)
        enh = self.clahe.apply(f[:,:,1])
        den = cv2.medianBlur(enh, Config.MEDIAN_KERNEL)
        bw = cv2.adaptiveThreshold(den, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, Config.ADAPTIVE_BLOCK_SIZE, -3)
        bw = cv2.bitwise_and(bw, green)
        cl = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, self.kernel, iterations=2)
        clean = remove_small_objects(cl > 0, min_size=Config.MIN_OBJECT_SIZE)
        return skeletonize(clean)

    def nearest_skel(self, skel, f, pt, r=30):
        h, w = skel.shape
        px, py = int(pt[0]), int(pt[1])
        best, bd = None, 1e9
        for dy in range(-r, r+1):
            for dx in range(-r, r+1):
                ny, nx = py+dy, px+dx
                if 0 <= ny < h and 0 <= nx < w and skel[ny, nx] and self.is_green(f, ny, nx):
                    d = dy*dy + dx*dx
                    if d < bd: bd, best = d, (ny, nx)
        return best


# ============================================================================
#                              追踪
# ============================================================================

class Tracker:
    def __init__(self, ip):
        self.ip = ip

    def _dir(self, t, n=8):
        if len(t) < 2: return (0, 1)
        r = t[-min(n, len(t)):]
        dy, dx = r[-1][0] - r[0][0], r[-1][1] - r[0][1]
        ln = np.sqrt(dy*dy + dx*dx)
        return (dy/ln, dx/ln) if ln > 0.001 else (0, 1)

    def trace(self, skel, f, markers):
        if not markers: return []
        pts = []
        for m in markers:
            p = self.ip.nearest_skel(skel, f, (m["x"], m["y"]))
            if p: pts.append(p)
        if not pts: return []
        # 去重
        uniq = []
        for p in pts:
            if not any(abs(p[0]-u[0])<5 and abs(p[1]-u[1])<5 for u in uniq):
                uniq.append(p)
        uniq.sort(key=lambda p: p[1])

        if len(uniq) == 1:
            return self._bidir(skel, f, uniq[0])

        # 多点A*连接
        traj = []
        for i in range(len(uniq)-1):
            seg = self._astar(skel, f, uniq[i], uniq[i+1])
            if seg:
                if traj and seg[0] == traj[-1]: seg = seg[1:]
                traj.extend(seg)

        # 两端延伸
        if uniq:
            left = self._trace1(skel, f, uniq[0], uniq[0][0], True, set(traj))
            traj = left[::-1] + traj
            right = self._trace1(skel, f, uniq[-1], uniq[-1][0], False, set(traj))
            traj = traj + right

        seen = set()
        return [p for p in traj if not (p in seen or seen.add(p))]

    def _astar(self, skel, f, start, end):
        h, w = skel.shape
        heur = lambda p: np.sqrt((p[0]-end[0])**2 + (p[1]-end[1])**2)
        cnt = 0
        heap = [(heur(start), cnt, start, [start])]
        vis = {start}
        for _ in range(5000):
            if not heap: break
            _, _, cur, path = heapq.heappop(heap)
            if cur == end or heur(cur) < 3:
                return path + [end] if cur != end else path
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    if dy == 0 and dx == 0: continue
                    ny, nx = cur[0]+dy, cur[1]+dx
                    if not (0 <= ny < h and 0 <= nx < w): continue
                    if (ny, nx) in vis: continue
                    if not skel[ny, nx] or not self.ip.is_green(f, ny, nx): continue
                    vis.add((ny, nx))
                    cnt += 1
                    heapq.heappush(heap, (len(path) + heur((ny, nx)), cnt, (ny, nx), path + [(ny, nx)]))
        # 直连
        path = [start]
        steps = int(heur(start)) + 1
        for i in range(1, steps+1):
            t = i / steps
            y, x = int(start[0] + t*(end[0]-start[0])), int(start[1] + t*(end[1]-start[1]))
            if (y, x) != path[-1]: path.append((y, x))
        if path[-1] != end: path.append(end)
        return path

    def _trace1(self, skel, f, start, yc, left, avoid=None):
        if not start: return []
        h, w = skel.shape
        vis = set(avoid) if avoid else set()
        t = [start]
        vis.add(start)
        cur = start
        while True:
            d = self._dir(t)
            cands = []
            for sr in [1, 2, Config.MAX_GAP]:
                for dy in range(-sr, sr+1):
                    for dx in range(-sr, sr+1):
                        if dy == 0 and dx == 0: continue
                        ny, nx = cur[0]+dy, cur[1]+dx
                        if not (0 <= ny < h and 0 <= nx < w): continue
                        if (ny, nx) in vis: continue
                        if not skel[ny, nx] or not self.ip.is_green(f, ny, nx): continue
                        if abs(ny - yc) > Config.Y_TOLERANCE * 2: continue
                        dist = np.sqrt(dy*dy + dx*dx)
                        sc = (-dx if left else dx) * 10 - abs(dy) * 3 - dist * 2
                        if dist > 0:
                            cd = (dy/dist, dx/dist)
                            sc += (d[0]*cd[0] + d[1]*cd[1]) * Config.DIRECTION_WEIGHT
                        cands.append(((ny, nx), sc))
                if cands: break
            if not cands: break
            cands.sort(key=lambda x: x[1], reverse=True)
            best = cands[0][0]
            vis.add(best)
            t.append(best)
            cur = best
        return t

    def _bidir(self, skel, f, start):
        if not start: return []
        yc = start[0]
        left = self._trace1(skel, f, start, yc, True)
        right = self._trace1(skel, f, start, yc, False)
        return left[::-1] + right[1:] if len(right) > 1 else left[::-1]

    def grow(self, skel, f, traj, yc):
        if not traj: return []
        return self._trace1(skel, f, traj[-1], yc, False, set(traj))


# ============================================================================
#                              数据
# ============================================================================

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
        print(f"✓ 保存: {path}")

    def load(self, path):
        if not os.path.exists(path): return False
        with open(path) as f: d = json.load(f)
        self.marks = {int(k): v for k, v in d.get("marks", {}).items()}
        self.trajs = {int(k): [tuple(p) for p in v] for k, v in d.get("trajs", {}).items()}
        print(f"✓ 加载: {len(self.marks)}神经元")
        return True


# ============================================================================
#                              主程序
# ============================================================================

class NeuronTool:
    def __init__(self):
        self.vid = VideoHandler()
        self.ip = ImageProc()
        self.data = Data()
        self.tracker = Tracker(self.ip)

        self.zoom = 1.0
        self.pan_x = self.pan_y = 0
        self.panning = False
        self.pan_start = (0, 0)

        self.cur_nid = 0
        self.mode = 'mark'
        self.buttons = {}
        self.frame_inp = self.neuron_inp = self.active_inp = None
        self.mx = self.my = 0
        self.action = None
        self.data_path = self.out_path = None

        if HAS_TK:
            self.root = tk.Tk()
            self.root.withdraw()
        else:
            self.root = None

    @property
    def img_h(self):
        return Config.DISPLAY_HEIGHT - Config.PANEL_HEIGHT

    def _init_ui(self):
        y1, y2, y3 = self.img_h + 12, self.img_h + 55, self.img_h + 95
        bh = 36
        x = 15
        self.buttons['prev'] = Button(x, y1, 55, bh, '<Prev'); x += 60
        self.buttons['next'] = Button(x, y1, 55, bh, 'Next>'); x += 60
        self.buttons['p10'] = Button(x, y1, 40, bh, '-10'); x += 45
        self.buttons['n10'] = Button(x, y1, 40, bh, '+10'); x += 55
        self.frame_inp = InputBox(x, y1+2, 80, 32, "Frame:"); x += 95
        self.buttons['zin'] = Button(x, y1, 35, bh, 'Z+'); x += 40
        self.buttons['zout'] = Button(x, y1, 35, bh, 'Z-'); x += 40
        self.buttons['zfit'] = Button(x, y1, 35, bh, 'Fit'); x += 50
        self.neuron_inp = InputBox(x, y1+2, 60, 32, "Neuron:"); self.neuron_inp.text = "0"; x += 75
        self.buttons['new'] = Button(x, y1, 45, bh, 'New'); x += 50
        self.buttons['deln'] = Button(x, y1, 50, bh, 'Del N'); x += 55

        x = 15
        self.buttons['mark'] = Button(x, y2, 60, bh, 'MARK'); x += 65
        self.buttons['track'] = Button(x, y2, 60, bh, 'TRACK'); x += 75
        self.buttons['run'] = Button(x, y2, 80, bh, '> RUN', (60,110,60)); x += 85
        self.buttons['save'] = Button(x, y2, 50, bh, 'Save', (60,110,60)); x += 55
        self.buttons['load'] = Button(x, y2, 50, bh, 'Load'); x += 55
        self.buttons['clr'] = Button(x, y2, 70, bh, 'ClearAll', (60,60,110)); x += 80
        self.buttons['exit'] = Button(x, y2, 50, bh, 'EXIT', (60,60,110))

        for i in range(15):
            self.buttons[f'n{i}'] = Button(15 + i*50, y3, 46, 28, f'N{i}')

    def s2i(self, sx, sy):
        dw, dh = int(self.vid.w * self.zoom), int(self.vid.h * self.zoom)
        ox = (Config.DISPLAY_WIDTH - dw) // 2 + self.pan_x
        oy = (self.img_h - dh) // 2 + self.pan_y
        return (sx - ox) / self.zoom, (sy - oy) / self.zoom

    def i2s(self, ix, iy):
        dw, dh = int(self.vid.w * self.zoom), int(self.vid.h * self.zoom)
        ox = (Config.DISPLAY_WIDTH - dw) // 2 + self.pan_x
        oy = (self.img_h - dh) // 2 + self.pan_y
        return int(ix * self.zoom + ox), int(iy * self.zoom + oy)

    def _fit(self):
        self.zoom = min((Config.DISPLAY_WIDTH - 40) / self.vid.w, (self.img_h - 40) / self.vid.h, 1.0)
        self.pan_x = self.pan_y = 0

    def _btn_click(self, name):
        if name == 'prev': self.vid.read(self.vid.idx - 1); self.frame_inp.text = str(self.vid.idx + 1)
        elif name == 'next': self.vid.read(self.vid.idx + 1); self.frame_inp.text = str(self.vid.idx + 1)
        elif name == 'p10': self.vid.read(self.vid.idx - 10); self.frame_inp.text = str(self.vid.idx + 1)
        elif name == 'n10': self.vid.read(self.vid.idx + 10); self.frame_inp.text = str(self.vid.idx + 1)
        elif name == 'zin': self.zoom = min(Config.MAX_ZOOM, self.zoom * Config.ZOOM_STEP)
        elif name == 'zout': self.zoom = max(Config.MIN_ZOOM, self.zoom / Config.ZOOM_STEP)
        elif name == 'zfit': self._fit()
        elif name == 'new': self.cur_nid = self.data.new_id(); self.neuron_inp.text = str(self.cur_nid)
        elif name == 'deln': self.data.del_neuron(self.cur_nid)
        elif name == 'mark': self.mode = 'mark'
        elif name == 'track': self.mode = 'track'
        elif name == 'run': return 'run'
        elif name == 'save': return 'save'
        elif name == 'load': return 'load'
        elif name == 'clr': self.data.marks.clear(); self.data.trajs.clear()
        elif name == 'exit': return 'exit'
        elif name.startswith('n') and name[1:].isdigit():
            self.cur_nid = int(name[1:]); self.neuron_inp.text = str(self.cur_nid)

    def _confirm_inp(self):
        if self.frame_inp.active:
            try: self.vid.read(int(self.frame_inp.text) - 1)
            except: pass
            self.frame_inp.active = False
        if self.neuron_inp.active:
            try: self.cur_nid = max(0, min(int(self.neuron_inp.text), Config.MAX_NEURONS - 1))
            except: pass
            self.neuron_inp.text = str(self.cur_nid)
            self.neuron_inp.active = False
        self.active_inp = None

    def _mouse(self, ev, x, y, flags, _):
        self.mx, self.my = x, y
        in_panel = y >= self.img_h

        for b in self.buttons.values(): b.hover = b.contains(x, y)
        for i in range(15): self.buttons[f'n{i}'].active = (i == self.cur_nid)
        self.buttons['mark'].active = (self.mode == 'mark')
        self.buttons['track'].active = (self.mode == 'track')

        if ev == cv2.EVENT_LBUTTONDOWN:
            if self.frame_inp.contains(x, y):
                self.active_inp = self.frame_inp; self.frame_inp.active = True; self.neuron_inp.active = False; return
            if self.neuron_inp.contains(x, y):
                self.active_inp = self.neuron_inp; self.neuron_inp.active = True; self.frame_inp.active = False; return
            self._confirm_inp()
            if in_panel:
                for nm, b in self.buttons.items():
                    if b.contains(x, y):
                        r = self._btn_click(nm)
                        if r: self.action = r
                        return
            elif self.mode == 'mark':
                ix, iy = self.s2i(x, y)
                if 0 <= ix < self.vid.w and 0 <= iy < self.vid.h:
                    self.data.add_mark(self.cur_nid, self.vid.idx, int(ix), int(iy))
                    print(f"  + N{self.cur_nid} @ F{self.vid.idx+1}: ({int(ix)},{int(iy)})")
        elif ev == cv2.EVENT_RBUTTONDOWN and not in_panel and self.mode == 'mark':
            ix, iy = self.s2i(x, y)
            self.data.del_mark_near(self.cur_nid, self.vid.idx, int(ix), int(iy))
        elif ev == cv2.EVENT_MBUTTONDOWN:
            self.panning = True; self.pan_start = (x, y)
        elif ev == cv2.EVENT_MBUTTONUP:
            self.panning = False
        elif ev == cv2.EVENT_MOUSEMOVE and self.panning:
            self.pan_x += x - self.pan_start[0]; self.pan_y += y - self.pan_start[1]; self.pan_start = (x, y)
        elif ev == cv2.EVENT_MOUSEWHEEL and not in_panel:
            self.zoom = min(Config.MAX_ZOOM, self.zoom * 1.2) if flags > 0 else max(Config.MIN_ZOOM, self.zoom / 1.2)

    def _draw(self):
        if self.vid.frame is None: return None
        c = np.full((Config.DISPLAY_HEIGHT, Config.DISPLAY_WIDTH, 3), Config.COLOR_BG, np.uint8)

        dw, dh = int(self.vid.w * self.zoom), int(self.vid.h * self.zoom)
        interp = cv2.INTER_LINEAR if self.zoom < 1 else cv2.INTER_NEAREST
        resized = cv2.resize(self.vid.frame, (dw, dh), interpolation=interp)
        ox = (Config.DISPLAY_WIDTH - dw) // 2 + self.pan_x
        oy = (self.img_h - dh) // 2 + self.pan_y

        sx1, sy1 = max(0, -ox), max(0, -oy)
        sx2, sy2 = min(dw, Config.DISPLAY_WIDTH - ox), min(dh, self.img_h - oy)
        dx1, dy1 = max(0, ox), max(0, oy)
        dx2, dy2 = min(Config.DISPLAY_WIDTH, ox + dw), min(self.img_h, oy + dh)
        if sx2 > sx1 and sy2 > sy1:
            c[dy1:dy2, dx1:dx2] = resized[sy1:sy2, sx1:sx2]

        # 画标记
        for nid, nd in self.data.marks.items():
            col = tuple(nd["color"])
            for m in nd["marks"]:
                sx, sy = self.i2s(m["x"], m["y"])
                if m["frame"] == self.vid.idx:
                    if nid == self.cur_nid: cv2.circle(c, (sx, sy), 10, (255,255,255), 2)
                    cv2.circle(c, (sx, sy), 6, col, -1)
                    cv2.putText(c, str(nid), (sx+8, sy+4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
                else:
                    cv2.drawMarker(c, (sx, sy), col, cv2.MARKER_TILTED_CROSS, 8, 1)

        # track模式画轨迹
        if self.mode == 'track':
            for nid, traj in self.data.trajs.items():
                col = self.data.color(nid)
                for i in range(len(traj)-1):
                    p1, p2 = self.i2s(traj[i][1], traj[i][0]), self.i2s(traj[i+1][1], traj[i+1][0])
                    if np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2) < 50:
                        cv2.line(c, p1, p2, col, 2, cv2.LINE_AA)
                if traj:
                    cv2.circle(c, self.i2s(traj[-1][1], traj[-1][0]), 5, (0,0,255), -1)

        # 面板
        cv2.rectangle(c, (0, self.img_h), (Config.DISPLAY_WIDTH, Config.DISPLAY_HEIGHT), Config.COLOR_PANEL, -1)
        for b in self.buttons.values(): b.draw(c)
        self.frame_inp.draw(c)
        self.neuron_inp.draw(c, self.data.color(self.cur_nid))

        # 信息
        info = f"Frame: {self.vid.idx+1}/{self.vid.total} | Zoom: {self.zoom:.2f} | Mode: {self.mode.upper()} | N{self.cur_nid}"
        cv2.putText(c, info, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,220,0), 2, cv2.LINE_AA)
        return c

    # ==========================================================================
    #                    追踪核心（彻底重写）
    # ==========================================================================

    def run_tracking(self):
        """执行追踪并播放"""
        if not self.data.marks:
            print("⚠ 没有标记")
            return

        print("\n" + "="*50)
        print("开始追踪...")
        print("="*50)

        self.data.trajs.clear()

        # 找初始帧
        init_f = min(m["frame"] for nd in self.data.marks.values() for m in nd["marks"])
        print(f"  初始帧: {init_f + 1}")

        # 每帧的轨迹快照 {帧号: {nid: [(y,x)...]}}
        history = {}

        # 追踪
        for fidx in range(self.vid.total):
            self.vid.read(fidx)

            if fidx < init_f:
                history[fidx] = {}
            elif fidx == init_f:
                skel = self.ip.skeleton(self.vid.frame)
                for nid in self.data.marks:
                    ms = self.data.get_marks(nid)
                    if not ms: continue
                    t = self.tracker.trace(skel, self.vid.frame, ms)
                    if len(t) >= 5:
                        t = sorted(t, key=lambda p: p[1])
                        self.data.trajs[nid] = t
                        print(f"  N{nid}: 初始化 {len(t)} 点")
                history[fidx] = {nid: list(t) for nid, t in self.data.trajs.items()}
            else:
                skel = self.ip.skeleton(self.vid.frame)
                for nid, t in list(self.data.trajs.items()):
                    if not t: continue
                    yc = np.mean([p[0] for p in t])
                    new = self.tracker.grow(skel, self.vid.frame, t, yc)
                    if new:
                        t.extend(new)
                        self.data.trajs[nid] = sorted(t, key=lambda p: p[1])
                history[fidx] = {nid: list(t) for nid, t in self.data.trajs.items()}

            if (fidx + 1) % 10 == 0:
                print(f"  进度: {fidx+1}/{self.vid.total}")

        print(f"\n✓ 追踪完成!")
        for nid, t in self.data.trajs.items():
            print(f"  N{nid}: {len(t)} 点")

        # =========================
        # 播放生长动画
        # =========================
        print("\n播放生长动画 (Q=退出 Space=暂停 A/D=逐帧)")

        win = 'Growth Animation'
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 1280, 720)

        self.vid.start_write(self.out_path)

        delay = max(1, int(1000 / self.vid.fps))
        fidx = 0
        paused = False

        while True:
            self.vid.read(fidx)
            vis = self.vid.frame.copy()

            # 获取该帧的轨迹
            ft = history.get(fidx, {})

            # 绘制轨迹
            for nid, traj in ft.items():
                if len(traj) < 2: continue
                col = self.data.color(nid)

                # 画线
                pts = np.array([(int(p[1]), int(p[0])) for p in traj], np.int32)
                cv2.polylines(vis, [pts], False, col, 2, cv2.LINE_AA)

                # 起点（绿色）
                cv2.circle(vis, (int(traj[0][1]), int(traj[0][0])), 5, (0, 255, 0), -1)
                # 末端（红色）
                cv2.circle(vis, (int(traj[-1][1]), int(traj[-1][0])), 7, (0, 0, 255), -1)
                # ID
                cv2.putText(vis, f"N{nid}", (int(traj[-1][1])+10, int(traj[-1][0])+5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)

            # 信息
            total_pts = sum(len(t) for t in ft.values())
            info = f"Frame {fidx+1}/{self.vid.total} | Neurons: {len(ft)} | Points: {total_pts}"
            cv2.putText(vis, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 4, cv2.LINE_AA)
            cv2.putText(vis, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)

            self.vid.write(vis)

            # 显示
            scale = min(1.0, 1280 / self.vid.w)
            disp = cv2.resize(vis, None, fx=scale, fy=scale)
            st = "PAUSED" if paused else "PLAYING"
            cv2.putText(disp, f"{st} | Q=exit Space=pause A/D=step", (10, disp.shape[0]-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
            cv2.imshow(win, disp)

            key = cv2.waitKey(delay if not paused else 30) & 0xFF
            if key == ord('q') or key == 27: break
            elif key == ord(' '): paused = not paused
            elif key == ord('a'): fidx = max(0, fidx - 1); paused = True
            elif key == ord('d'): fidx = min(self.vid.total - 1, fidx + 1); paused = True

            if not paused:
                fidx += 1
                if fidx >= self.vid.total: fidx = 0

        self.vid.stop_write()
        cv2.destroyWindow(win)
        print(f"\n✓ 视频已保存: {self.out_path}")

    # ==========================================================================
    #                    主循环
    # ==========================================================================

    def run(self, video_path, data_path=None):
        self.vid.load(video_path)
        self.data_path = data_path or os.path.splitext(video_path)[0] + "_data.json"
        self.out_path = os.path.splitext(video_path)[0] + "_tracked.mp4"

        if os.path.exists(self.data_path):
            self.data.load(self.data_path)

        self._init_ui()
        self.vid.read(0)
        self.frame_inp.text = "1"
        self._fit()

        win = 'Neuron Tool v9'
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, Config.DISPLAY_WIDTH, Config.DISPLAY_HEIGHT)
        cv2.setMouseCallback(win, self._mouse)

        print("\n" + "="*50)
        print("神经元追踪工具 v9.0")
        print("="*50)
        print("左键=标记 | 右键=删除 | 滚轮=缩放 | 中键=平移")
        print("A/D=帧 | 0-9=神经元 | RUN=追踪")
        print("="*50 + "\n")

        while True:
            disp = self._draw()
            if disp is not None: cv2.imshow(win, disp)

            if self.action == 'run':
                self.run_tracking()
                self.data.save(self.data_path)
                self.action = None
            elif self.action == 'save':
                self.data.save(self.data_path)
                self.action = None
            elif self.action == 'load':
                if HAS_TK:
                    p = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
                    if p: self.data_path = p; self.data.load(p)
                self.action = None
            elif self.action == 'exit':
                break

            key = cv2.waitKey(30) & 0xFF

            if self.active_inp:
                r = self.active_inp.handle_key(key)
                if r == 'confirm': self._confirm_inp()
                elif r == 'cancel': self.active_inp.active = False; self.active_inp = None
                continue

            if key == 27: break
            elif key == ord('a'): self.vid.read(self.vid.idx - 1); self.frame_inp.text = str(self.vid.idx + 1)
            elif key == ord('d'): self.vid.read(self.vid.idx + 1); self.frame_inp.text = str(self.vid.idx + 1)
            elif key == ord('w'): self.vid.read(self.vid.idx + 10); self.frame_inp.text = str(self.vid.idx + 1)
            elif key == ord('s'): self.vid.read(self.vid.idx - 10); self.frame_inp.text = str(self.vid.idx + 1)
            elif ord('0') <= key <= ord('9'):
                self.cur_nid = key - ord('0'); self.neuron_inp.text = str(self.cur_nid)

        cv2.destroyAllWindows()
        self.vid.release()
        if self.root: self.root.destroy()


# ============================================================================
#                              入口
# ============================================================================

if __name__ == "__main__":
    VIDEO_PATH = r"F:\工作文件\RA\python\项目汇总\神经图像\neuron_growth_50.mp4"

    tool = NeuronTool()
    tool.run(VIDEO_PATH)
