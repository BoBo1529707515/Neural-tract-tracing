"""
主界面模块
"""
import cv2
import numpy as np
import os
try:
    import tkinter as tk
    from tkinter import filedialog
    HAS_TK = True
except ImportError:
    HAS_TK = False

from .config import Config
from .ui_elements import Button, InputBox
from .video_handler import VideoHandler
from .image_processor import ImageProc
from .tracker import Tracker
from .data_manager import Data

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
        print(f"[UI] Button Clicked: {name}")
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

    def run_tracking(self):
        """执行追踪并实时播放（一边算一边画）"""
        if not self.data.marks:
            print("⚠ 没有标记")
            return

        print("\n" + "="*50)
        print("开始实时追踪...")
        print("="*50)

        self.data.trajs.clear()

        # 找初始帧
        init_f = min(m["frame"] for nd in self.data.marks.values() for m in nd["marks"])
        print(f"  初始帧: {init_f + 1}")

        # 准备输出视频
        self.vid.start_write(self.out_path)
        
        # 实时显示窗口
        win = 'Real-time Tracking'
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 1280, 720) # 预览窗口可以小一点

        # 追踪循环
        for fidx in range(self.vid.total):
            self.vid.read(fidx)
            frame_disp = self.vid.frame.copy()

            # 1. 追踪计算
            # 始终只基于前一帧的状态进行增量计算，而不是一次性算完
            if fidx == init_f:
                # 初始帧：只读取该帧的标记，不进行生长
                skel = self.ip.skeleton(self.vid.frame)
                for nid in self.data.marks:
                    ms = self.data.get_marks(nid)
                    if not ms: continue
                    # 仅处理当前帧的标记
                    curr_ms = [m for m in ms if m['frame'] == fidx]
                    if not curr_ms: continue
                    
                    # 在初始帧寻找骨架点
                    t = self.tracker.trace(skel, self.vid.frame, curr_ms)
                    if len(t) > 0:
                        t = sorted(t, key=lambda p: p[1])
                        self.data.trajs[nid] = t
                        print(f"  N{nid}: 初始化 {len(t)} 点")
            
            elif fidx > init_f:
                # 后续帧：在前一帧基础上生长
                skel = self.ip.skeleton(self.vid.frame)
                
                # 检查当前帧是否有人工标记（干预）
                for nid in self.data.marks:
                    ms = self.data.get_marks(nid)
                    curr_ms = [m for m in ms if m['frame'] == fidx]
                    
                    # 如果当前帧有人工标记，强制将轨迹对齐到人工标记
                    if curr_ms and nid in self.data.trajs:
                        print(f"  N{nid}: 人工干预修正 @ F{fidx+1}")
                        # 找到骨架上最近的点
                        p = self.ip.nearest_skel(skel, self.vid.frame, (curr_ms[0]['x'], curr_ms[0]['y']))
                        if p:
                            # 简单粗暴：直接连线到新点（实际应该用A*，这里简化演示）
                            last = self.data.trajs[nid][-1]
                            self.data.trajs[nid].append(p)
                
                # 自动生长
                for nid, t in list(self.data.trajs.items()):
                    if not t: continue
                    
                    # 计算生长方向
                    yc = np.mean([p[0] for p in t[-20:]]) if len(t) > 20 else t[-1][0]
                    new = self.tracker.grow(skel, self.vid.frame, t, yc)
                    
                    if new:
                        t.extend(new)
                        self.data.trajs[nid] = sorted(t, key=lambda p: p[1])

            # 2. 实时渲染
            # 绘制当前所有活跃轨迹
            active_count = 0
            for nid, traj in self.data.trajs.items():
                if len(traj) < 2: continue
                active_count += 1
                col = self.data.color(nid)

                # 画全量轨迹（如果太慢可以只画最近的线段）
                pts = np.array([(int(p[1]), int(p[0])) for p in traj], np.int32)
                cv2.polylines(frame_disp, [pts], False, col, 2, cv2.LINE_AA)

                # 绘制头部（追踪点）
                head = traj[-1]
                cv2.circle(frame_disp, (int(head[1]), int(head[0])), 5, (0, 0, 255), -1)
                cv2.putText(frame_disp, f"N{nid}", (int(head[1])+10, int(head[0])), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)

            # 状态信息
            info = f"Tracking: {fidx+1}/{self.vid.total} | Active: {active_count}"
            cv2.putText(frame_disp, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # 3. 输出与显示
            self.vid.write(frame_disp)
            
            cv2.imshow(win, frame_disp)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'): # ESC or Q to stop
                print("⚠ 用户终止追踪")
                break

        self.vid.stop_write()
        cv2.destroyWindow(win)
        print(f"\n✓ 追踪完成! 视频已保存: {self.out_path}")
        
        # 刷新主界面显示最后状态
        self.vid.read(self.vid.idx)

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

        win = 'Neuron Tool v9 (Modular)'
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, Config.DISPLAY_WIDTH, Config.DISPLAY_HEIGHT)
        cv2.setMouseCallback(win, self._mouse)

        print("\n" + "="*50)
        print("神经元追踪工具 v9.0 (Modular)")
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
