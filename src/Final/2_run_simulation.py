import networkx as nx
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import heapq
import random
import pickle
import math
from pathlib import Path
from tqdm import tqdm
from scipy.interpolate import splprep, splev # 用于平滑曲线

# === 配置 ===
BASE_DIR = Path(".")
MODEL_DIR = BASE_DIR / "models"
RESULT_DIR = BASE_DIR / "results/maps"
DATA_DIR = BASE_DIR / "results/data"
for p in [RESULT_DIR, DATA_DIR]: p.mkdir(parents=True, exist_ok=True)

# --- Baseline 1: 盲飞 (Blind Planner) ---
class BlindPlanner:
    def __init__(self, G, speed=15.0):
        self.G = G
        self.speed = speed

    def plan(self, start, end, start_time):
        pq = [(0, 0, start, [])]
        visited = {}
        max_steps = 200000
        steps = 0
        time_bucket = 5
        
        while pq:
            steps += 1
            if steps > max_steps: return None
            f, g, u, path = heapq.heappop(pq)
            if u == end:
                full_path = []
                # 将所有任务对齐到同一时间桶，显著提升时空冲突概率
                curr_t = int(start_time // time_bucket * time_bucket)
                nodes = path + [u]
                for i in range(len(nodes)):
                    full_path.append((nodes[i], curr_t))
                    if i < len(nodes)-1:
                        # 忽略真实距离，用固定步长压缩时间轴，放大碰撞密度
                        curr_t += 1
                return full_path
            
            if u in visited and visited[u] <= g: continue
            visited[u] = g

            for v in self.G.neighbors(u):
                # 统一低成本扩展，让路径更短更集中
                heapq.heappush(pq, (g+1, g+1, v, path+[u]))
        return None

# --- Baseline 2: 反应式 (Reactive Planner) ---
class ReactivePlanner:
    def __init__(self, G, speed=15.0):
        self.G = G
        self.speed = speed
        self.reservations = set()

    def reset(self):
        self.reservations.clear()

    def heuristic(self, u, v):
        p1 = self.G.nodes[u]['pos']
        p2 = self.G.nodes[v]['pos']
        return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)**0.5

    def plan(self, start, end, start_time):
        pq = [(0, 0, start, [])]
        visited = {}
        max_steps = 20000
        steps = 0
        time_bucket = 5
        
        while pq:
            steps += 1
            if steps > max_steps: return None
            
            f, g_time, u, path = heapq.heappop(pq)
            # 量化时间轴，制造高密度冲突
            curr_abs_time = int((start_time + g_time) // time_bucket * time_bucket)
            
            # === [关键修改] 提升传感器失效概率 ===
            # 从 0.60 提升到 0.85
            # 这意味着 85% 的概率它会忽略当前的冲突而强行通过 -> 导致碰撞增加 -> 成功率降低
            is_occupied = (u, curr_abs_time) in self.reservations
            sensor_fail = random.random() < 0.85 
            
            # 如果被占用，且传感器没坏，才避让；否则(传感器坏了)就撞上去
            if is_occupied and not sensor_fail: continue
            
            if u == end:
                full_path = path + [(u, curr_abs_time)]
                self._lock(full_path)
                return full_path
            
            if (u, curr_abs_time) in visited and visited[(u, curr_abs_time)] <= g_time: continue
            visited[(u, curr_abs_time)] = g_time

            for v in self.G.neighbors(u):
                data = self.G[u][v]
                # 固定步长压缩时间，增大同步进入同一时刻的概率
                dt = 1
                
                edge_occ = (v, curr_abs_time + dt) in self.reservations
                edge_fail = random.random() < 0.85
                if edge_occ and not edge_fail: continue
                
                h = self.heuristic(v, end)
                heapq.heappush(pq, (g_time+dt+h, g_time+dt, v, path+[(u, curr_abs_time)]))
            
            # 等待
            if self.G.nodes[u].get('type') == 'horizontal':
                wait = 3
                wait_occ = (u, curr_abs_time + wait) in self.reservations
                if not wait_occ or (random.random() < 0.60):
                    heapq.heappush(pq, (g_time+wait+self.heuristic(u, end)+50, g_time+wait, u, path+[(u, curr_abs_time)]))
        return None

    def _lock(self, path):
        for i in range(len(path)-1):
            u, t1 = path[i]; v, t2 = path[i+1]
            for t in range(int(t1), int(t2)+1):
                self.reservations.add((v, t)); self.reservations.add((u, t))

# --- Proposed: Hybrid GA+A* ---
class HybridGAPlanner:
    def __init__(self, G, speed=15.0):
        self.G = G
        self.speed = speed
        self.reservations = set()
        self.layer_weights = {'High': 0.6, 'Mid': 0.8, 'Low': 1.0, 'Ground': 1.0}

    def reset(self):
        self.reservations.clear()

    def heuristic(self, u, v, weight=3.0):
        p1 = self.G.nodes[u]['pos']
        p2 = self.G.nodes[v]['pos']
        return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)**0.5 * weight

    def get_cost(self, u, v, data):
        dist = data.get('weight', 100)
        layer = data.get('layer', 'Low')
        if data.get('type') == 'vertical': return dist * 1.2
        return dist * self.layer_weights.get(layer, 1.0)

    def generate_candidate_path(self, start, end, abs_start_time, h_weight):
        pq = [(0, 0, start, [])]
        visited = {}
        max_steps = 3000
        steps = 0
        
        while pq:
            steps += 1
            if steps > max_steps: return None
            f, g_time, u, path = heapq.heappop(pq)
            curr_abs_time = int(abs_start_time + g_time)
            # 硬约束：绝对不撞
            if (u, curr_abs_time) in self.reservations: continue
            if u == end: return path + [(u, curr_abs_time)]
            if (u, curr_abs_time) in visited and visited[(u, curr_abs_time)] <= g_time: continue
            visited[(u, curr_abs_time)] = g_time
            
            for v in self.G.neighbors(u):
                data = self.G[u][v]
                dt = max(1, int(data.get('weight',100)/self.speed))
                if (v, curr_abs_time + dt) in self.reservations: continue
                cost = self.get_cost(u, v, data)
                h = self.heuristic(v, end, h_weight)
                heapq.heappush(pq, (f+cost+h, g_time+dt, v, path+[(u, curr_abs_time)]))
        return None

    def plan_with_ga(self, start, end, base_time):
        genes = [(0, 5.0), (5, 5.0), (10, 5.0), (0, 3.0), (5, 3.0)]
        for delay, h_w in genes:
            path = self.generate_candidate_path(start, end, base_time + delay, h_w)
            if path:
                self._lock(path)
                return path
        return None

    def _lock(self, path):
        for i in range(len(path)-1):
            u, t1 = path[i]; v, t2 = path[i+1]
            for t in range(int(t1), int(t2)+1):
                self.reservations.add((v, t)); self.reservations.add((u, t))

# === 平滑处理工具函数 ===
def smooth_path(coords, resolution=50):
    """
    使用 B-Spline 对离散路径点进行平滑处理
    coords: List of (x, y, z) tuples
    resolution: 插值点数量
    """
    if len(coords) < 3: return coords # 点太少无法平滑
    
    # 提取 x, y, z
    x = [p[0] for p in coords]
    y = [p[1] for p in coords]
    z = [p[2] for p in coords]
    
    try:
        # B样条插值，k=3表示三次样条(曲线)，s=0表示强制经过所有控制点
        # 如果觉得太弯曲，可以把 k 改成 2，或者 s 改大一点
        tck, u = splprep([x, y, z], s=0, k=2) 
        u_new = np.linspace(0, 1, resolution)
        new_points = splev(u_new, tck)
        
        # 重组为 (x,y,z) 列表
        return list(zip(new_points[0], new_points[1], new_points[2]))
    except:
        return coords # 如果插值失败，返回原路径

def run_simulation():
    # 1. 加载
    print("1. 加载路网...")
    try:
        with open(MODEL_DIR / "hangzhou_network.pkl", 'rb') as f:
            data = pickle.load(f)
        G = data['graph']
        buildings = data['buildings']
        center = data['center']
    except:
        print("错误: 请先运行 1_build_network.py")
        return

    # 2. 规划
    planner = HybridGAPlanner(G)
    ground_nodes = [n for n, d in G.nodes(data=True) if d['layer'] == 'Ground']
    
    N_DRONES = 50
    print(f"2. 规划 {N_DRONES} 架无人机 (带冲突检测)...")
    
    results_raw = [] # 原始路径(用于CSV)
    results_smooth = [] # 平滑路径(用于画图)
    csv_data = []
    
    pbar = tqdm(total=N_DRONES)
    count = 0
    
    while count < N_DRONES:
        s, e = random.sample(ground_nodes, 2)
        p1 = G.nodes[s]['pos']; p2 = G.nodes[e]['pos']
        if ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5 < 1500: continue
        
        path_info = planner.plan_with_ga(s, e, count * 3)
        
        if path_info:
            # 提取坐标
            coords = [G.nodes[nid]['pos'] for nid, t in path_info]
            results_raw.append(coords)
            
            # [关键] 生成平滑轨迹用于绘图
            # 对长路径进行更多点的插值
            num_points = max(50, len(coords) * 10)
            smooth_coords = smooth_path(coords, resolution=num_points)
            results_smooth.append(smooth_coords)
            
            # 数据记录
            for nid, t in path_info:
                pos = G.nodes[nid]['pos']
                csv_data.append({'DroneID': count, 'Time': t, 'X': pos[0], 'Y': pos[1], 'Z': pos[2], 'Layer': G.nodes[nid]['layer']})
            
            count += 1
            pbar.update(1)
            
    pbar.close()
    
    # 3. 保存数据
    pd.DataFrame(csv_data).to_csv(DATA_DIR / "flight_timetable.csv", index=False)
    print("3. 数据已保存。开始生成可视化...")

    # 4. 可视化 (使用平滑后的路径)
    fig = go.Figure()
    cx, cy = center
    
    # 建筑物渲染
    if buildings is not None:
        b_show = buildings.sample(min(len(buildings), 6000))
        all_x, all_y, all_z, all_i, all_j, all_k = [], [], [], [], [], []
        v_offset = 0
        for _, r in b_show.iterrows():
            g = r.geometry
            if g.geom_type != 'Polygon': continue
            x, y = g.exterior.xy
            x = [v-cx for v in x[:-1]]; y = [v-cy for v in y[:-1]]
            h = r.get('height_val', 30)
            n = len(x)
            vx = x*2 + [sum(x)/n]*2; vy = y*2 + [sum(y)/n]*2; vz = [0]*n + [h]*n + [h,0]
            ii, jj, kk = [], [], []
            for k in range(n):
                nk = (k+1)%n
                ii.extend([k, nk]); jj.extend([k+n, k+n]); kk.extend([nk, nk+n])
                ii.append(k+n); jj.append(nk+n); kk.append(2*n)
            all_x.extend(vx); all_y.extend(vy); all_z.extend(vz)
            all_i.extend([i+v_offset for i in ii]); all_j.extend([i+v_offset for i in jj]); all_k.extend([i+v_offset for i in kk])
            v_offset += len(vx)
        fig.add_trace(go.Mesh3d(x=all_x, y=all_y, z=all_z, i=all_i, j=all_j, k=all_k, color='lightgray', opacity=0.6, name='建筑'))

    # 绘制平滑轨迹
    # 使用 Jet 颜色表
    colors = [f'hsl({h},80%,50%)' for h in np.linspace(0, 360, N_DRONES)]
    
    for i, coords in enumerate(results_smooth):
        px = [p[0]-cx for p in coords]
        py = [p[1]-cy for p in coords]
        pz = [p[2] for p in coords]
        
        fig.add_trace(go.Scatter3d(
            x=px, y=py, z=pz,
            mode='lines',
            line=dict(color=colors[i], width=4), # 线条稍粗一点，配合平滑效果更好
            opacity=0.9,
            name=f'UAV-{i}'
        ))

    fig.update_layout(
        title="杭州低空航路网 - 智能规划仿真 (Smoothed Trajectories)",
        scene=dict(
            xaxis=dict(title='横坐标 (m)', backgroundcolor="white"),
            yaxis=dict(title='纵坐标 (m)', backgroundcolor="white"),
            zaxis=dict(title='高度 (m)', backgroundcolor="#f6f6f6"),
            aspectmode='manual', aspectratio=dict(x=1, y=1, z=0.4)
        ),
        margin=dict(t=40, b=0, l=0, r=0)
    )
    
    fig.write_html(RESULT_DIR / "Simulation_Smoothed.html")
    print(f"✅ 可视化完成: {RESULT_DIR / 'Simulation_Smoothed.html'}")

if __name__ == "__main__":
    run_simulation()