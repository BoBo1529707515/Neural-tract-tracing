
# NeuronTracker

多神经元轴突生长追踪工具，基于 Python / OpenCV / Tkinter。

---

## 依赖

```bash
pip install opencv-python numpy Pillow
```

Python ≥ 3.8，Tkinter 为标准库。

---

## 文件结构

```
├── gui.py        # 主界面
├── tracker.py    # 追踪算法
├── colors.py     # 神经元颜色（BGR列表）
└── config.py     # 默认路径
```

**config.py**
```python
DEFAULT_INPUT_DIR  = "./input_frames"
DEFAULT_OUTPUT_DIR = "./output"
```

---

## 使用流程

```bash
python gui.py
```

1. 加载 PNG 帧序列
2. 左键标记神经元必经点，右键删除，`Ctrl+Z` 撤销
3. `▶ 计算当前` 或 `▶▶ 计算全部`
4. `Enter` 确认，`Esc` 取消
5. 导出视频 / CSV / 速度数据

---

## 快捷键

| 键 | 功能 |
|----|------|
| `←` `→` | 移动 1 帧 |
| `↑` `↓` | 移动 10 帧 |
| `1`–`9` | 切换神经元 |
| `Enter` | 确认结果 |
| `Esc` | 取消预览 |
| `Ctrl+Z` | 撤销标记 |
| 中键拖拽 | 平移 |
| 滚轮 | 缩放 |

---

## 参数

| 参数 | 默认 | 说明 |
|------|------|------|
| 亮度阈值 | 30 | 背景剔除阈值 |
| 搜索半径 | 30 px | 候选点搜索范围 |
| 最大转角 | 60° | 相邻步最大偏转角 |
| 最大步距 | 15 px | 单步最大位移 |
| FPS | 10 | 帧率（速度换算用） |
| 像素/μm | 1.0 | 物理尺寸标定值 |

---
## 跟踪效果
<img width="1108" height="639" alt="42377f858e065a8c14745d88630b164" src="https://github.com/user-attachments/assets/717d47ce-34cc-487f-b7ed-0bae25bc9b11" />

## 算法

```
compute_neuron_trajectory
├── 选取亮度最优参考帧
├── 在参考帧向左追踪初始路径（经过全部必经点）
│     平滑约束：转角过滤 + 步距过滤 + 方向加权评分
├── 路径平滑（保留必经点）
└── 逐帧向右生长，输出 paths_by_frame
```

---

## 输出格式

### neuron_paths.csv

| 字段 | 说明 |
|------|------|
| `neuron_id` | 神经元编号 |
| `frame` | 帧编号 |
| `path_index` | 路径点序号 |
| `x`, `y` | 像素坐标 |

### neuron_speed.csv

| 字段 | 说明 |
|------|------|
| `neuron_id` | 神经元编号 |
| `frame` | 帧编号 |
| `tip_x`, `tip_y` | 末梢坐标 |
| `dx`, `dy` | 帧间位移分量 |
| `speed_px_per_frame` | 速度（px/帧） |
| `speed_um_per_sec` | 速度（μm/s） |

速度计算：

$$
v \ (\mu m/s) = \sqrt{\Delta x^2 + \Delta y^2} \times pixel\_um \times fps
$$

---


