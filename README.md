
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
目前在配置好存储路径后可以直接运行整合版_自动预处理里的run_pipeline.py文件对前几步统一处理。  
随后使用模块版/smooth_tracker_modular中的main.py启动终端
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

### 更新历史
# Changelog

## v1.3.0

### 新增
- `local_recompute_with_waypoints()` — 局部重算方法，编辑模式下不再整体重算
- 三种局部重算策略：
  - 新点 X 在已有 waypoints 之间 → 仅清除夹住它的两锚点间路径，重追该段（含段内所有旧 waypoints + 新点）
  - 新点 X 小于最左 waypoint → 清除路径左端，从新点向左重追至边界
  - 新点 X 大于最右 waypoint → 清除路径右端，从最右锚点追经过新点

### 修改
- `compute_neuron_trajectory()` 返回的 result 新增 `best_frame` 字段，供编辑模式锁定参考帧
- 局部段重追后仍走完整逐帧生长流程，保证 `paths_by_frame` 完整，以下这种情况不会再出现了。
<img width="439" height="326" alt="image" src="https://github.com/user-attachments/assets/d0f38304-f013-44fa-bd10-f802bc02cf8f" />

---

## v1.2.0

### 新增
- `compute_neuron_trajectory()` 新增 `locked_frame` 参数，编辑重算时锁定原参考帧，防止轨迹漂移
- result 字典新增 `best_frame` 字段记录实际使用的参考帧

### 修复
- 编辑模式下新加标记点后重算，`find_best_frame` 可能选中不同帧导致轨迹完全改变
- 新加点 X 坐标最大时成为追踪起点，导致整条路径偏移

---

## v1.1.0

### 新增
- `compute_neuron_speed()` — 逐帧计算末梢位移速度，支持 FPS 与 μm/px 换算
- `compute_all_speeds()` — 批量计算所有已追踪神经元速度
- 导出 `neuron_speed.csv`，包含 `tip_x/y`、`dx/dy`、`speed_px_per_frame`、`speed_um_per_sec`

### 速度公式

$$
v \ (\mu m/s) = \sqrt{\Delta x^2 + \Delta y^2} \times pixel\_um \times fps
$$

---

## v1.0.0

### 初始版本
- 向量化候选点搜索（`find_candidates_leftward_vectorized` / `find_candidates_rightward_vectorized`）
- 平滑约束：最大转角过滤 + 步距限制 + 方向加权评分
- `trace_segment_left()` — 带必经点目标的向左追踪
- `trace_left_unified()` — 多必经点串联追踪，追不到时自动直连
- `grow_rightward()` — 逐帧向右生长
- `smooth_path_preserve_waypoints()` — 路径平滑，保留必经点坐标不变
- `connect_points_directly()` — 追踪失败时的直线兜底连接
- `find_best_frame()` — 自动选取必经点亮度最高的帧作为参考帧
- 多神经元支持，最多 99 条，结果存入 `tracking_results`
- 导出 `neuron_paths.csv`

