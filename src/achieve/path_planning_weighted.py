import networkx as nx
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import heapq
import random
import pickle
import os
import time
from pathlib import Path
from tqdm import tqdm
from pyproj import Transformer

# === 1. 配置管理 ===
BASE_DIR = Path(".")
MODEL_DIR = BASE_DIR / "models"
RESULT_DATA_DIR = BASE_DIR / "results/data"
RESULT_MAP_DIR = BASE_DIR / "results/maps"

for p in [RESULT_DATA_DIR, RESULT_MAP_DIR]:
    p.mkdir(parents=True, exist_ok=True)

def load_network(filename="hangzhou_network.pkl"):
    # 优先尝试加载 CBD 版本的路网
    path = MODEL_DIR / filename
    if not path.exists():
        # 如果找不到，尝试加载旧版
        alt_path = MODEL_DIR / "hangzhou_route_graph.pkl"
        if alt_path.exists():
            print(f"提示: 未找到 {filename}，正在加载 {alt_path.name} ...")
            path = alt_path
        else:
            raise FileNotFoundError(f"未找到路网文件: {path}，请先运行 generate_network_cbd.py")
    
    print(f"正在加载路网: {path.name} ...")
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data

# === 2. 核心算法：高性能时空 A* ===
class WeightedSpaceTimePlanner:
    def __init__(self, G, drone_speed=15.0):
        self.G = G
        self.reservations = set() # 预留表 {(node_id, time_step)}
        self.speed = drone_speed
        
        # 策略参数 (论文核心)
        self.weights = {
            'High': 0.6,  # 高层代价 6折
            'Mid': 0.8,   # 中层代价 8折
            'Low': 1.0,
            'Vertical': 1.2
        }
        
        # 预计算最大连通分量，用于快速排除孤岛点
        # print("正在预计算连通分量...")
        # self.largest_cc = max(nx.connected_components(G), key=len)
        
        self._preprocess_edges()

    def _preprocess_edges(self):
        """预计算每条边的物理耗时和逻辑代价"""
        for u, v, d in self.G.edges(data=True):
            dist = d.get('weight', 100)
            
            # 物理耗时 (秒)
            dt = max(1, int(round(dist / self.speed)))
            d['dt'] = dt
            
            # 逻辑代价
            layer = d.get('layer', 'Low')
            etype = d.get('type', 'hor')
            
            factor = 1.0
            if etype in ['hor', 'horizontal']:
                factor = self.weights.get(layer, 1.0)
            else:
                factor = self.weights['Vertical']
                
            d['logic_cost'] = dist * factor

    def heuristic(self, u, end_node):
        """
        [性能优化关键] 加权启发式 (Weighted A*)
        将系数从 0.6 提高到 1.5，大幅减少搜索节点数。
        虽然不再保证理论最优，但能获得工程上的“次优解”，速度快几十倍。
        """
        p1 = self.G.nodes[u]['pos']
        p2 = self.G.nodes[end_node]['pos']
        dist = ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)**0.5
        return dist * 1.5  # 贪婪系数

    def plan(self, drone_id, start, end, start_time=0):
        """规划单架飞机路径 (带超时中断)"""
        
        # 1. 快速连通性检查 (毫秒级)
        # if not nx.has_path(self.G, start, end):
        #    return None

        # (F_score, G_score, current_node, current_time, path_history)
        pq = [(0, 0, start, start_time, [])]
        best_cost = {(start, start_time): 0}
        
        # [性能优化] 超时控制
        start_clock = time.time()
        max_duration = 5.0 # 每架飞机最多算5秒
        
        steps = 0
        
        while pq:
            steps += 1
            # 每 2000 步检查一次超时
            if steps % 2000 == 0:
                if time.time() - start_clock > max_duration:
                    return None # 超时放弃
            
            f, g, u, t, path = heapq.heappop(pq)
            
            if u == end:
                full_path = path + [(u, t)]
                self._reserve(full_path)
                return full_path
            
            # 剪枝
            if g > best_arrival_cost(best_cost, u, t): continue
            
            # 扩展邻居
            for v in self.G.neighbors(u):
                edge = self.G[u][v]
                dt = edge['dt']
                logic_w = edge['logic_cost']
                t_next = t + dt
                
                # 冲突检测
                if not self._is_conflict(v, t + 1, t_next):
                    new_g = g + logic_w
                    if new_g < best_arrival_cost(best_cost, v, t_next):
                        best_cost[(v, t_next)] = new_g
                        # 核心加速：h * 1.5
                        h = self.heuristic(v, end)
                        heapq.heappush(pq, (new_g + h, new_g, v, t_next, path + [(u, t)]))
            
            # 空中悬停 (仅允许在水平点等待，防止堵死垂直通道)
            node_type = self.G.nodes[u].get('type', 'vert')
            if node_type in ['hor', 'horizontal']:
                t_wait = t + 3
                wait_cost = 50 # 等待惩罚
                if not self._is_conflict(u, t + 1, t_wait):
                    new_g = g + wait_cost
                    if new_g < best_arrival_cost(best_cost, u, t_wait):
                        best_cost[(u, t_wait)] = new_g
                        h = self.heuristic(u, end)
                        heapq.heappush(pq, (new_g + h, new_g, u, t_wait, path + [(u, t)]))

        return None

    def _is_conflict(self, node, t_start, t_end):
        for t in range(t_start, t_end + 1):
            if (node, t) in self.reservations:
                return True
        return False

    def _reserve(self, path):
        for i in range(len(path)-1):
            u, t1 = path[i]
            v, t2 = path[i+1]
            for t in range(t1 + 1, t2 + 1):
                self.reservations.add((v, t))

def best_arrival_cost(record, u, t):
    return record.get((u, t), float('inf'))

# === 3. 可视化模块 ===
def visualize_static_swarm(G, buildings, drone_paths, bounds, center, filename):
    print(f"正在生成可视化: {filename} ...")
    fig = go.Figure()
    
    min_x, min_y, max_x, max_y = bounds
    cx, cy = center
    
    # 1. 建筑物
    if buildings is not None:
        all_x, all_y, all_z, all_i, all_j, all_k = [], [], [], [], [], []
        v_offset = 0
        b_show = buildings if len(buildings) < 8000 else buildings.sample(8000)
        
        for _, r in b_show.iterrows():
            g = r.geometry
            if g.geom_type != 'Polygon': continue
            xx, yy = g.exterior.xy
            xx = [v-cx for v in xx[:-1]]; yy = [v-cy for v in yy[:-1]]
            h = r['height_val']
            n = len(xx)
            
            vx = xx*2 + [sum(xx)/n]*2; vy = yy*2 + [sum(yy)/n]*2; vz = [0]*n + [h]*n + [h,0]
            ii, jj, kk = [], [], []
            for k in range(n):
                nk = (k+1)%n
                ii.extend([k, nk]); jj.extend([k+n, k+n]); kk.extend([nk, nk+n])
                ii.append(k+n); jj.append(nk+n); kk.append(2*n)
            
            all_x.extend(vx); all_y.extend(vy); all_z.extend(vz)
            all_i.extend([i+v_offset for i in ii]); all_j.extend([i+v_offset for i in jj]); all_k.extend([i+v_offset for i in kk])
            v_offset += len(vx)
            
        fig.add_trace(go.Mesh3d(x=all_x, y=all_y, z=all_z, i=all_i, j=all_j, k=all_k, color='lightgray', opacity=0.8, flatshading=True, name='建筑物'))

    # 2. 轨迹
    colors = [f'hsl({h},80%,50%)' for h in np.linspace(0, 360, len(drone_paths))]
    for i, path_data in enumerate(drone_paths):
        nodes = [p[0] for p in path_data]
        coords = [G.nodes[n]['pos'] for n in nodes]
        px = [c[0]-cx for c in coords]; py = [c[1]-cy for c in coords]; pz = [c[2] for c in coords]
        
        fig.add_trace(go.Scatter3d(x=px, y=py, z=pz, mode='lines', line=dict(color=colors[i], width=3), opacity=0.9, name=f'Drone {i+1}'))
        # 起终点
        fig.add_trace(go.Scatter3d(x=[px[0], px[-1]], y=[py[0], py[-1]], z=[pz[0], pz[-1]], mode='markers', marker=dict(size=3, color=colors[i]), showlegend=False))

    fig.update_layout(
        title=f"无人机集群规划结果 (N={len(drone_paths)})",
        scene=dict(xaxis=dict(title='X'), yaxis=dict(title='Y'), zaxis=dict(title='Z'), aspectmode='manual', aspectratio=dict(x=1, y=1, z=0.35)),
        margin=dict(t=50, b=0, l=0, r=0), paper_bgcolor='white'
    )
    
    out_path = RESULT_MAP_DIR / filename
    fig.write_html(str(out_path))
    print(f"可视化完成: {out_path}")

# === 主流程 ===
def main():
    data = load_network()
    
    # 兼容键名
    G = data.get('G') or data.get('graph')
    if G is None: raise KeyError("pkl文件中找不到图数据")
    buildings = data.get('buildings')
    bounds = data.get('bounds')
    
    if 'center' in data: center = data['center']
    else: center = ((bounds[0]+bounds[2])/2, (bounds[1]+bounds[3])/2)

    planner = WeightedSpaceTimePlanner(G, drone_speed=15.0)
    
    # 地面节点
    g_nodes = [n for n, d in G.nodes(data=True) if d['pos'][2] < 1.0]
    if len(g_nodes) < 2: 
        print("错误: 地面节点不足")
        return
    
    # 预先计算连通分量 (只保留最大的连通区域，防止选到孤岛点)
    # print("优化路网连通性...")
    # largest_cc = max(nx.connected_components(G), key=len)
    # g_nodes = [n for n in g_nodes if n in largest_cc]
    
    print(f"可用地面起降点: {len(g_nodes)}")
    print("\n=== 开始批量规划 (目标 50 架) ===")
    
    success_paths = []
    timetable_data = []
    
    num_drones = 50
    # 增加最大尝试次数，因为有些点可能确实算不出来
    max_attempts = 500 
    attempts = 0
    
    # 使用 tqdm 显示进度
    pbar = tqdm(total=num_drones, desc="Planning")
    
    while len(success_paths) < num_drones and attempts < max_attempts:
        attempts += 1
        s, e = random.sample(g_nodes, 2)
        
        # 物理距离预筛
        p1 = G.nodes[s]['pos']; p2 = G.nodes[e]['pos']
        dist = ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5
        if dist < 1500: continue 
        
        # 简单预检连通性 (只检查空间，不耗时)
        if not nx.has_path(G, s, e):
            continue

        launch_time = len(success_paths) * 2
        path = planner.plan(len(success_paths), s, e, launch_time)
        
        if path:
            success_paths.append(path)
            drone_id = len(success_paths)
            for node_id, t in path:
                pos = G.nodes[node_id]['pos']
                timetable_data.append({
                    'DroneID': drone_id,
                    'Time': t,
                    'NodeID': node_id,
                    'X': pos[0], 'Y': pos[1], 'Z': pos[2],
                    'Layer': G.nodes[node_id].get('layer', 'Unknown')
                })
            pbar.update(1)
            
    pbar.close()
    
    print(f"\n尝试 {attempts} 次，成功 {len(success_paths)} 条。")
    
    if len(timetable_data) == 0:
        print("❌ 未生成路径，请检查路网连通性。")
        return

    # 输出 CSV
    df = pd.DataFrame(timetable_data)
    csv_path = RESULT_DATA_DIR / "flight_timetable.csv"
    df.to_csv(csv_path, index=False)
    print(f"时刻表已保存: {csv_path}")
    
    if not df.empty and 'Layer' in df.columns:
        high_ratio = len(df[df['Layer']=='High']) / len(df)
        print(f"  - 高层航点占比: {high_ratio:.1%}")
    
    # 输出 HTML
    visualize_static_swarm(G, buildings, success_paths, bounds, center, "Static_Weighted_Plan.html")

if __name__ == "__main__":
    main()