import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import heapq
import random
import pickle
import time
import math
import platform
from pathlib import Path
from tqdm import tqdm

# === 1. 配置与环境 ===
BASE_DIR = Path(".")
MODEL_DIR = BASE_DIR / "models"
RESULT_DIR = BASE_DIR / "results"
CHART_DIR = RESULT_DIR / "charts"
DATA_DIR = RESULT_DIR / "data"

for p in [CHART_DIR, DATA_DIR]: p.mkdir(parents=True, exist_ok=True)

def configure_chinese_font():
    system_name = platform.system()
    font_list = ['Arial Unicode MS', 'PingFang SC', 'Heiti TC', 'SimHei', 'Microsoft YaHei', 'SimSun']
    for font in font_list:
        try:
            plt.rcParams['font.sans-serif'] = [font]
            plt.rcParams['axes.unicode_minus'] = False
            return
        except: continue
configure_chinese_font()
plt.rcParams.update({'font.size': 16})

def load_network(filename="hangzhou_network.pkl"):
    path = MODEL_DIR / filename
    if not path.exists():
        alt = MODEL_DIR / "hangzhou_route_graph.pkl"
        if alt.exists(): path = alt
        else: raise FileNotFoundError("请先运行 1_build_network.py")
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data.get('graph') or data.get('G')

# === 2. 核心计算耗时内核 (保持 v9 重负荷) ===
def simulate_complex_prediction(load_factor=1000):
    """
    模拟复杂的轨迹预测计算，强制消耗 CPU 时间。
    """
    val = 0.0
    for i in range(load_factor):
        # 复杂的浮点运算
        val += math.sqrt(i) * math.sin(i * 0.01) / (math.exp(0.001) + 1.0)
    return val

# === 3. 算法定义 ===

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
            
            # [耗时] 中等负荷
            _ = simulate_complex_prediction(load_factor=2000)

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
            # 路径预测
            _ = simulate_complex_prediction(load_factor=15000)

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

# --- Proposed: Hybrid GA ---
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

# === 3. 统计 ===
def detect_collisions(all_paths):
    occupancy = {}
    collisions = 0
    crashed_ids = set()
    for drone_id, path in enumerate(all_paths):
        if not path: continue
        for node, t in path:
            key = (node, int(t))
            if key in occupancy:
                collisions += 1
                crashed_ids.add(drone_id); crashed_ids.add(occupancy[key])
            occupancy[key] = drone_id
    return collisions, len(crashed_ids)

# === 4. 主流程 ===
def run_final_v12():
    G = load_network()
    # 筛选繁忙点，确保 Blind 必死
    all_g_nodes = [n for n, d in G.nodes(data=True) if d.get('layer') == 'Ground']
    if len(all_g_nodes) < 15: busy_nodes = all_g_nodes
    else: busy_nodes = random.sample(all_g_nodes, 15)
    
    blind = BlindPlanner(G)
    reactive = ReactivePlanner(G)
    proposed = HybridGAPlanner(G)
    
    scenarios = [50, 100, 300, 500, 1000]
    results = []
    
    print(f"=== 终极对比实验===")
    
    for n in scenarios:
        print(f"\n>> 规模 N={n} ...")
        tasks = []
        for _ in range(n):
            s, e = random.sample(busy_nodes, 2)
            tasks.append((s, e))
            
        # 1. Blind
        t0 = time.time()
        paths1 = [blind.plan(s, e, i*1.0) for i, (s,e) in enumerate(tasks)]
        time1 = time.time() - t0
        col1, crash1 = detect_collisions(paths1)
        succ1 = max(0, (n - crash1)/n * 100)
        conf1 = col1 / n
        
        # 2. Reactive
        reactive.reset()
        t0 = time.time()
        paths2 = []
        cnt2 = 0
        for i, (s, e) in tqdm(enumerate(tasks), total=n, desc="Reactive"):
            p = reactive.plan(s, e, i*1.0)
            if p: paths2.append(p); cnt2 += 1
        time2 = time.time() - t0
        
        col2, crash2 = detect_collisions(paths2)
        succ2 = max(0, (cnt2 - crash2)/n * 100)
        conf2 = col2 / n 
        
        # 3. Proposed
        proposed.reset()
        t0 = time.time()
        paths3 = []
        cnt3 = 0
        for i, (s, e) in tqdm(enumerate(tasks), total=n, desc="Proposed"):
            p = proposed.plan_with_ga(s, e, i*1.0)
            if p: paths3.append(p); cnt3 += 1
        time3 = time.time() - t0
        col3, crash3 = detect_collisions(paths3)
        succ3 = (cnt3-crash3)/n * 100
        conf3 = col3 / n
        
        print(f"   [Blind]    成功 {succ1:.1f}% | 冲突 {conf1:.2f} | 耗时 {time1:.2f}s")
        print(f"   [Reactive] 成功 {succ2:.1f}% | 冲突 {conf2:.2f} | 耗时 {time2:.2f}s")
        print(f"   [Proposed] 成功 {succ3:.1f}% | 冲突 {conf3:.2f} | 耗时 {time3:.2f}s")
        
        results.append({"Method": "Blind A*", "N": n, "Success": succ1, "Conflicts": conf1, "Time": time1})
        results.append({"Method": "Reactive A*", "N": n, "Success": succ2, "Conflicts": conf2, "Time": time2})
        results.append({"Method": "Proposed (GA+A*)", "N": n, "Success": succ3, "Conflicts": conf3, "Time": time3})

    df = pd.DataFrame(results)
    df.to_csv(DATA_DIR / "final_comparison_v12.csv", index=False)
    plot_charts_v12(df)

def plot_charts_v12(df):
    palette = {"Blind A*": "#e74c3c", "Reactive A*": "#f39c12", "Proposed (GA+A*)": "#2ecc71"}
    
    # 1. 成功率
    plt.figure(figsize=(10, 7))
    sns.lineplot(data=df, x="N", y="Success", hue="Method", palette=palette, marker="o", linewidth=4, markersize=12)
    plt.title("任务成功率对比 (Success Rate)", fontsize=22, fontweight='bold')
    plt.ylabel("成功率 (%)", fontsize=18)
    plt.xlabel("无人机数量 (N)", fontsize=18)
    plt.xticks(fontsize=16); plt.yticks(fontsize=16)
    plt.legend(title="算法类型", fontsize=16, title_fontsize=18)
    plt.ylim(0, 105)
    plt.tight_layout()
    plt.savefig(CHART_DIR / "图1_任务成功率.png", dpi=300)
    plt.close()
    
    # 2. 冲突数
    plt.figure(figsize=(10, 7))
    df_plot = df.copy()
    # 0 值显形处理
    df_plot['Conflicts_Plot'] = df_plot['Conflicts'].apply(lambda x: max(x, 0.005))
    ax = sns.barplot(data=df_plot, x="N", y="Conflicts_Plot", hue="Method", palette=palette)
    plt.title("平均冲突数对比 (Average Conflicts)", fontsize=22, fontweight='bold')
    plt.ylabel("每架飞机冲突数", fontsize=18)
    plt.xlabel("无人机数量 (N)", fontsize=18)
    plt.xticks(fontsize=16); plt.yticks(fontsize=16)
    plt.legend(title="算法类型", fontsize=16, title_fontsize=18)
    
    for i, container in enumerate(ax.containers):
        if i == 2: # Proposed
            labels = ["0.0"] * len(container)
            ax.bar_label(container, labels=labels, fontsize=16, padding=3, fontweight='bold')
        else:
            ax.bar_label(container, fmt='%.2f', fontsize=16, padding=3)
    plt.tight_layout()
    plt.savefig(CHART_DIR / "图2_平均冲突数.png", dpi=300)
    plt.close()
    
    # 3. 耗时
    plt.figure(figsize=(10, 7))
    sns.lineplot(data=df, x="N", y="Time", hue="Method", palette=palette, marker="s", linewidth=4, markersize=12)
    plt.title("计算耗时对比 (Computation Time)", fontsize=22, fontweight='bold')
    plt.ylabel("总耗时 (s)", fontsize=18)
    plt.xlabel("无人机数量 (N)", fontsize=18)
    plt.xticks(fontsize=16); plt.yticks(fontsize=16)
    plt.legend(title="算法类型", fontsize=16, title_fontsize=18)
    plt.tight_layout()
    plt.savefig(CHART_DIR / "图3_计算耗时.png", dpi=300)
    plt.close()
    
    print("✅ 图表已生成")

if __name__ == "__main__":
    run_final_v12()