# Neuron Tracking Algorithm Design Notes
(神经元轨迹追踪算法设计笔记)

## 1. Core Design Philosophy (核心设计思想)

### 1.1 Problem Definition (问题定义)

```text
Input:
  - Temporal Image Sequence {I₀, I₁, ..., Iₙ} (Grayscale)
  - User-defined Waypoints {P₁, P₂, ..., Pₖ} (Marked on any frame)

Output:
  - Complete Neuron Trajectory for each frame {T₀, T₁, ..., Tₙ}
```

### 1.2 Key Assumptions (核心假设)

| Assumption (假设) | Description (说明) |
| ----------------- | ------------------ |
| **Spatial Continuity** | Neuron trajectories are continuous curves without sudden jumps.<br>神经元轨迹是连续的曲线，不会突然跳跃。 |
| **Temporal Monotonicity** | Neurons only grow (extend rightward) and never retract.<br>神经元只会生长（向右延伸），不会收缩。 |
| **Intensity Indication** | Neuronal regions are brighter than the background.<br>神经元区域比背景更亮。 |
| **Directional Inertia** | Growth direction is continuous; no sharp turns.<br>神经元生长方向具有连续性，不会急转弯。 |

### 1.3 Divide and Conquer Strategy (分治策略)

```text
 ┌─────────────────────────────────────────────────────────────────┐
 │                 Why Two Stages? (为什么分两阶段？)               │
 ├─────────────────────────────────────────────────────────────────┤
 │                                                                 │
 │  Observation: Neuron trajectory consists of two parts           │
 │                                                                 │
 │  ┌──────────────┐    ┌──────────────┐                           │
 │  │ Stable Part  │ +  │ Growing Part │                           │
 │  │ (Near Soma)  │    │ (Growth Cone)│                           │
 │  └──────────────┘    └──────────────┘                           │
 │         ↓                    ↓                                  │
 │  Consistent across frames   Extends over time                   │
 │         ↓                    ↓                                  │
 │  Trace Left Unified       Trace Right Accumulatively            │
 │                                                                 │
 └─────────────────────────────────────────────────────────────────┘
```

---

## 2. Two-Stage Tracking Strategy (两阶段追踪策略)

### 2.1 Stage 1: Trace Left Unified (向左统一追踪)

**Goal**: Generate a unified base trajectory passing through all waypoints.
(生成一条经过所有必经点的统一基础轨迹)

```text
 Timeline:
   Frame 0   Frame 10   Frame 20   Frame 30   Frame 40
      |          |          |          |          |
                            ↑
                       Representative Frame
                       (Signal Strongest)

 Trajectory:
   |←━━━━━━━●━━━━━━━●━━━━━━━●
   Boundary Mark C   Mark B   Mark A (Start)
```

**Key Points**:
- Trace on the frame with the **strongest signal**. (在信号最强的帧进行追踪)
- Must pass through **all** user-defined waypoints. (必须经过所有用户标记的必经点)
- The generated path is applied as a **base** for all frames. (生成的路径应用于所有帧作为基础)

**Algorithm Flow**:
```python
def trace_left_unified(waypoints):
    # 1. Sort waypoints by x-coordinate descending
    waypoints = sorted(waypoints, key=lambda p: -p[0])
    
    # 2. Trace from each waypoint to the next
    for i in range(len(waypoints)):
        start = waypoints[i]
        target = waypoints[i+1] if i+1 < len(waypoints) else None
        
        # 3. Trace leftward, guided by target
        segment = trace_segment_left(start, target)
        path.extend(segment)
    
    # 4. Trace from the last waypoint to the boundary
    final_segment = trace_to_boundary()
    path.extend(final_segment)
    
    return path
```

### 2.2 Stage 2: Trace Right Accumulatively (向右累积生长)

**Goal**: Simulate the neuron growth process rightward over time.
(模拟神经元随时间向右生长的过程)

```text
 Accumulation Principle:
   
   Frame t:    |━━━━━━━━━━━━━━━━●|          Current Path
                           ↑
   Frame t+1:  |━━━━━━━━━━━━━━━━●━━━●|      Keep + Append
           └────Keep Previous───┘ └New┘
   
   Frame t+2:  |━━━━━━━━━━━━━━━━●━━━●━━●| 
           └─────Keep Frame t+1────┘ └New┘
```

**Key Rules**:
- **Monotonic Increase**: Previous path is fully preserved. (只增不减)
- **Rightward Only**: `dx ≥ 0`, no backtracking. (只能向右)
- **Frame-by-Frame**: Compute based on the tip of the previous frame. (逐帧计算)

**Algorithm Flow**:
```python
def grow_rightward(prev_path, frame_idx):
    # 1. Start from the tip of the previous path
    start_point = prev_path[-1]
    
    # 2. Search rightward in the current frame
    new_points = []
    while not reached_boundary:
        # Search candidates in the right semicircle
        candidates = search_rightward(current_point)
        
        # Select the best candidate
        best = select_best(candidates)
        new_points.append(best)
    
    # 3. Accumulate: New Path = Old Path + New Points
    return prev_path + new_points
```

---

## 3. Semicircular Search Strategy (半圆形搜索策略)

### 3.1 Search Region Definition (搜索区域定义)

```text
 ┌─────────────────────────────────────────────────────────────────┐
 │  Why Semicircle? (为什么是半圆形？)                              │
 ├─────────────────────────────────────────────────────────────────┤
 │                                                                 │
 │  Trace Left: Search Left Half Only                              │
 │                                                                 │
 │       ╭─────╮                                                   │
 │      ╱   ×   ╲    ← No Right Search (Backtracking)              │
 │     │    ●━━━→│                                                 │
 │      ╲   ✓   ╱    ← Search Left                                 │
 │       ╰─────╯                                                   │
 │                                                                 │
 │  Trace Right: Search Right Half Only                            │
 │                                                                 │
 │       ╭─────╮                                                   │
 │      ╱   ✓   ╲    ← Search Right                                │
 │     │←━━━●    │                                                 │
 │      ╲   ×   ╱    ← No Left Search (Backtracking)               │
 │       ╰─────╯                                                   │
 │                                                                 │
 └─────────────────────────────────────────────────────────────────┘
```

### 3.2 Search Parameters (搜索范围参数)

```python
# Coordinate Range
Trace Left:  dx ∈ [-radius, 0], dy ∈ [-radius, radius]
Trace Right: dx ∈ [0, radius],  dy ∈ [-radius, radius]

# Distance Constraint
dist = √(dx² + dy²) ≤ radius
dist ≤ max_step_distance  # Speed Limit
```

---

## 4. Candidate Scoring System (候选点评分系统)

### 4.1 Scoring Formula (完整评分公式)

```python
score = (brightness / 255)                    # Normalized Brightness
      × (linearity ^ linearity_weight)        # Linearity
      / (distance + 0.5)                      # Distance Penalty
      × (1 + smoothness_weight × smoothness)  # Smoothness Bonus
      × (1 + direction_bonus)                 # Direction Consistency
      × (1 + side_bonus)                      # Target Direction Bonus
      × target_bonus                          # Waypoint Guidance
```

### 4.2 Factors Breakdown (各因素详解)

#### 4.2.1 Brightness (亮度)
Principle: Neurons are brighter than background.
```python
brightness = I[y, x]  # Pixel value 0-255

# Threshold Filtering
if brightness < brightness_threshold:
    exclude(candidate)
```

#### 4.2.2 Linearity (线性度)
Principle: Trajectories should maintain directional continuity.
```python
# 1. Calculate average previous direction (N points)
prev_dir = Σ(P[i] - P[i-1]) / N

# 2. Calculate cosine similarity
new_dir = candidate - current
cos(θ) = (prev_dir · new_dir) / (|prev| × |new|)

# 3. Normalize to [0, 1]
linearity = (cos(θ) + 1) / 2

Weight: linearity ^ 2.0 (Amplify differences)
```

#### 4.2.3 Distance Penalty (距离惩罚)
Principle: Closer candidates are more reliable.
Formula: `1 / (distance + 0.5)`

#### 4.2.4 Smoothness (平滑度)
Principle: Minimize turning angles between adjacent steps.
Weight:
- Trace Left: `× (1 + smoothness)`
- Trace Right: `× (1 + 2.0 × smoothness)` (Stricter during growth)

#### 4.2.5 Direction Bonus (方向一致性奖励)
Principle: Reward candidates aligned with current momentum.

#### 4.2.6 Target Bonus (必经点引导)
Principle: Significant reward for moving towards user-marked waypoints.
```python
target_bonus = 1 + 3.0 × (cos_to_target + 1) / 2
# Effect: Direct approach score × 4, deviating score × 1
```

---

## 5. Constraints Detail (约束条件详解)

### 5.1 Angle Constraint (转角约束)

```text
 Parameter: max_turn_angle = 60°

 Current Dir →
           ╲  ✓ θ < 60° (Allowed)
            ╲
             ↘ θ = 60° (Limit)
              │
              ↓  ✗ θ > 60° (Forbidden)

 Math Condition:
   cos(θ) ≥ cos(60°) = 0.5
```

### 5.2 Step Distance Constraint (步距约束)

```text
 Parameter: max_step_distance = 15 pixels

 Current ●
         ╲
          ╲ ✓ dist ≤ 15 (Allowed)
           ╲
            ●
                         ✗ dist > 15 (Forbidden, prevent teleport)
                                 ●
 Effect:
   • Ensures trajectory continuity
   • Limits max speed to 15px/step
```

### 5.3 Brightness Threshold (亮度阈值约束)
`brightness_threshold = 30`: Filters out background noise effectively.

### 5.4 Boundary Constraint (边界约束)
Prevents tracking outside image area (`left_margin`, `right_margin`).

### 5.5 Temporal Monotonicity (时间单调性)
Guarantees `path[t+1]` is always a superset of `path[t]`.

---

## 6. Handling Special Cases (特殊情况处理)

### 6.1 Gap Handling (空缺处理)

```text
 Normal: ●━━●━━●━━●━━●  (Continuous Signal)
 Gap:    ●━━●━━●     ●━━●  (Signal Break)

 Strategy: Progressively expand search radius
   radius = 30  → No candidate
   radius = 50  → No candidate
   radius = 70  → Found!
        ...
   radius = 150 → Max Limit
```

### 6.2 Waypoint Connection (必经点连接)
If natural tracking fails to reach a waypoint, force a connection using a guided "best-effort" path.

### 6.3 Visited Set (重复点避免)
Maintains a `visited` set to prevent loops and backtracking.

---

## 7. Performance Optimization (性能优化技术)

### 7.1 NumPy Vectorization (向量化)
Batch calculation of coordinates, distances, and scores using NumPy arrays instead of Python loops.

### 7.2 Search Grid Caching (搜索网格预计算)
Pre-calculate `dx`, `dy`, and `dist` grids for different radii to avoid redundant computation.

### 7.3 3D Array Storage (3D数组存储)
Store frames as `(N, H, W)` NumPy array for contiguous memory access and faster indexing.

---

## 8. Parameter Cheat Sheet (参数速查表)

| Parameter | Default | Type | Function |
|-----------|---------|------|----------|
| `brightness_threshold` | 30 | Filter | Min pixel intensity |
| `search_radius` | 30 | Search | Base search radius (px) |
| `max_search_radius` | 150 | Search | Max radius for gaps |
| `max_turn_angle` | 60° | Constraint | Max allowed turn |
| `max_step_distance` | 15 | Constraint | Max step size |
| `linearity_weight` | 2.0 | Score | Weight for linearity |
| `waypoint_capture_radius`| 25 | Waypoint | Distance to capture target |

---

## 9. Complexity Analysis (算法复杂度)

```text
 Time Complexity:
   Trace Left:  O(L × R²)
     L = Path Length, R = Search Radius
   
   Trace Right: O(N × G × R²)
     N = Frames, G = Growth Steps/Frame
 
 Space Complexity:
   O(N × H × W) Frame Storage
   O(N × L) Path Storage
 
 Real-world Performance:
   ~1-3 seconds / neuron (100 frames, optimized)
```

---

## 10. Summary: Design Philosophy (总结：设计哲学)

```text
 ┌─────────────────────────────────────────────────────────────────┐
 │                      Core Principles                            │
 ├─────────────────────────────────────────────────────────────────┤
 │                                                                 │
 │  1️⃣ Greedy + Constraints (贪心 + 约束)                          │
 │     Local optimal at each step, bounded by global rules.        │
 │     → Local Optima + Global Rationality                         │
 │                                                                 │
 │  2️⃣ Human-in-the-Loop (人机协作)                                │
 │     Algorithm handles details; User provides guidance.          │
 │     → Semi-auto, High Controllability                           │
 │                                                                 │
 │  3️⃣ Physical Priors (物理先验)                                  │
 │     Smoothness → Biological Continuity                          │
 │     Monotonicity → Irreversible Growth                          │
 │     → Biologically plausible results                            │
 │                                                                 │
 │  4️⃣ Robustness (鲁棒性)                                         │
 │     Gap Handling → Works with imperfect signals                 │
 │     Multi-factor Scoring → Adapts to varying image quality      │
 │                                                                 │
 └─────────────────────────────────────────────────────────────────┘
```
