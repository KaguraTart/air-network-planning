import networkx as nx
import plotly.graph_objects as go
import pickle
import numpy as np
from pathlib import Path
from scipy.interpolate import splprep, splev

# === 1. 配置 ===
MODEL_DIR = Path("models")
RESULT_DIR = Path("results/maps")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

def load_network(filename="hangzhou_route_graph.pkl"):
    print("正在加载路网数据...")
    with open(MODEL_DIR / filename, "rb") as f:
        data = pickle.load(f)
    return data

# === 2. 核心修复：分段式轨迹生成 ===
def generate_precise_trajectory(path_nodes, G, speed_factor=1.0):
    """
    将轨迹严格分为：[垂直起飞] -> [平滑巡航] -> [垂直降落]
    确保 0m 起飞清晰可见。
    """
    if not path_nodes or len(path_nodes) < 2: return [], [], []
    
    # 获取 A* 规划出的原始节点坐标
    raw_coords = [G.nodes[n]['pos'] for n in path_nodes]
    
    # --- 阶段 1: 定义关键点 ---
    # 真正的起点 (空中或地面)
    p_start = np.array(raw_coords[0])
    # 强制地面点 (x, y, 0)
    p_ground_start = np.array([p_start[0], p_start[1], 0.0])
    
    # 真正的终点
    p_end = np.array(raw_coords[-1])
    # 强制地面点 (x, y, 0)
    p_ground_end = np.array([p_end[0], p_end[1], 0.0])
    
    # --- 阶段 2: 生成各段轨迹 ---
    
    # A. 起飞段 (Takeoff): 0m -> Start_Height
    # 强制生成 20 个点，确保动画在这里停留足够时间
    takeoff_x = np.linspace(p_ground_start[0], p_start[0], 20)
    takeoff_y = np.linspace(p_ground_start[1], p_start[1], 20)
    takeoff_z = np.linspace(0, p_start[2], 20) # Z轴从0升到起飞高度
    
    # B. 巡航段 (Cruise): 使用 B样条平滑空中部分
    cruise_x, cruise_y, cruise_z = [], [], []
    if len(raw_coords) >= 2:
        x = [c[0] for c in raw_coords]
        y = [c[1] for c in raw_coords]
        z = [c[2] for c in raw_coords]
        
        # 如果点太少(比如只有直飞)，直接线性插值
        if len(raw_coords) <= 3:
            cruise_x = np.linspace(x[0], x[-1], 50)
            cruise_y = np.linspace(y[0], y[-1], 50)
            cruise_z = np.linspace(z[0], z[-1], 50)
        else:
            try:
                # k=3 (三次样条), s=0 (强制经过点)
                # 调大插值点数 (num_cruise) 让飞行更慢更流畅
                tck, u = splprep([x, y, z], s=0, k=3) 
                u_new = np.linspace(0, 1, 150) 
                res = splev(u_new, tck)
                cruise_x, cruise_y, cruise_z = res[0], res[1], res[2]
            except:
                cruise_x, cruise_y, cruise_z = x, y, z

    # C. 降落段 (Landing): End_Height -> 0m
    landing_x = np.linspace(p_end[0], p_ground_end[0], 20)
    landing_y = np.linspace(p_end[1], p_ground_end[1], 20)
    landing_z = np.linspace(p_end[2], 0, 20) # Z轴降到0
    
    # --- 阶段 3: 拼接 ---
    # 使用 np.concatenate 拼接数组
    full_x = np.concatenate([takeoff_x, cruise_x, landing_x])
    full_y = np.concatenate([takeoff_y, cruise_y, landing_y])
    full_z = np.concatenate([takeoff_z, cruise_z, landing_z])
    
    return full_x, full_y, full_z

# === 3. 生成动画 (论文风格) ===
def create_fixed_animation(data, path_nodes):
    print("正在生成修正版动画...")
    G = data['graph']
    buildings = data['buildings']
    bounds = data['bounds']
    
    min_x, min_y, max_x, max_y = bounds
    cx, cy = (min_x + max_x)/2, (min_y + max_y)/2
    
    # 扩大场景范围，防止边缘截断
    scene_size = max(max_x - min_x, max_y - min_y) / 1.8
    x_range = [-scene_size, scene_size]
    y_range = [-scene_size, scene_size]
    z_range = [0, 450] # 稍微调高天空上限

    # --- A. 获取修复后的轨迹 ---
    sx, sy, sz = generate_precise_trajectory(path_nodes, G)
    # 坐标平移
    sx = sx - cx
    sy = sy - cy
    
    # --- B. 构建静态背景 ---
    static_traces = []
    
    # 1. 地面网格 (Ground Plane)
    static_traces.append(go.Mesh3d(
        x=[-scene_size, scene_size, scene_size, -scene_size],
        y=[-scene_size, -scene_size, scene_size, scene_size],
        z=[-2, -2, -2, -2], # 稍微下沉防止 Z-fighting
        color='rgb(245, 245, 245)',
        opacity=1.0,
        name='地面', showlegend=False, hoverinfo='skip'
    ))

    # 2. 实体建筑物 (合并优化)
    if buildings is not None:
        all_x, all_y, all_z = [], [], []
        all_i, all_j, all_k = [], [], []
        v_offset = 0
        
        for _, row in buildings.iterrows():
            geom = row.geometry
            h = row['height_val']
            if geom.geom_type != 'Polygon': continue
            
            x_geo, y_geo = geom.exterior.xy
            x_geo = [v - cx for v in list(x_geo)[:-1]]
            y_geo = [v - cy for v in list(y_geo)[:-1]]
            
            n = len(x_geo)
            vx = x_geo * 2 + [sum(x_geo)/n, sum(x_geo)/n]
            vy = y_geo * 2 + [sum(y_geo)/n, sum(y_geo)/n]
            vz = [0]*n + [h]*n + [h, 0]
            
            i_idx, j_idx, k_idx = [], [], []
            for k in range(n):
                nk = (k+1)%n
                i_idx.extend([k, nk]); j_idx.extend([k+n, k+n]); k_idx.extend([nk, nk+n])
                i_idx.append(k+n); j_idx.append(nk+n); k_idx.append(2*n)
            
            all_x.extend(vx); all_y.extend(vy); all_z.extend(vz)
            all_i.extend([idx + v_offset for idx in i_idx])
            all_j.extend([idx + v_offset for idx in j_idx])
            all_k.extend([idx + v_offset for idx in k_idx])
            v_offset += len(vx)

        static_traces.append(go.Mesh3d(
            x=all_x, y=all_y, z=all_z, i=all_i, j=all_j, k=all_k,
            color='rgb(180, 180, 180)', # 经典灰
            opacity=1.0, 
            flatshading=True,
            lighting=dict(ambient=0.6, diffuse=0.5, roughness=0.1), 
            name='城市建筑'
        ))

    # 3. 起终点标记 (显眼的圆锥)
    # 确保这些点紧贴地面 Z=0
    static_traces.append(go.Scatter3d(
        x=[sx[0], sx[-1]], y=[sy[0], sy[-1]], z=[0, 0],
        mode='markers+text',
        marker=dict(size=12, color=['#00aa00', '#cc0000'], symbol='diamond', opacity=1),
        text=['START', 'END'],
        textposition="top center",
        textfont=dict(color='black', size=14, family="Arial Black"),
        name='起降点'
    ))

    # --- C. 动画帧 ---
    frames = []
    
    # 预设 Trace 3 (轨迹) 和 Trace 4 (飞机)
    static_traces.append(go.Scatter3d(
        x=[sx[0]], y=[sy[0]], z=[sz[0]],
        mode='lines',
        line=dict(color='#32CD32', width=6), # Lime Green
        name='飞行轨迹'
    ))
    
    static_traces.append(go.Scatter3d(
        x=[sx[0]], y=[sy[0]], z=[sz[0]],
        mode='markers',
        marker=dict(color='red', size=5, symbol='circle'),
        name='无人机'
    ))

    # 制作帧
    step = 2
    for k in range(0, len(sx), step):
        frames.append(go.Frame(
            data=[
                # Update Trace 3 (Path)
                go.Scatter3d(x=sx[:k+1], y=sy[:k+1], z=sz[:k+1]),
                # Update Trace 4 (Drone)
                go.Scatter3d(x=[sx[k]], y=[sy[k]], z=[sz[k]])
            ],
            name=f'f{k}',
            traces=[3, 4] # 只更新后两个
        ))

    # --- D. 布局设置 ---
    fig = go.Figure(data=static_traces, frames=frames)

    fig.update_layout(
        title=dict(
            text="无人机全流程飞行仿真 (VTOL Simulation)",
            y=0.95,
            x=0.5,
            xanchor='center',
            yanchor='top',
            font=dict(size=20, color='black')
        ),
        uirevision='constant', # 锁定视角
        
        paper_bgcolor='white',
        plot_bgcolor='white',
        
        scene=dict(
            xaxis=dict(title='X (m)', range=x_range, visible=False),
            yaxis=dict(title='Y (m)', range=y_range, visible=False),
            zaxis=dict(title='Height (m)', range=z_range, showbackground=True, backgroundcolor='#f0f0f0', gridcolor='white'),
            aspectmode='manual', aspectratio=dict(x=1, y=1, z=0.35),
            
            camera=dict(
                eye=dict(x=1.4, y=-1.4, z=0.6),
                center=dict(x=0, y=0, z=-0.1),
                up=dict(x=0, y=0, z=1)
            )
        ),
        updatemenus=[{
            'type': 'buttons',
            'showactive': False,
            'y': 0.05, 'x': 0.05,
            'xanchor': 'left', 'yanchor': 'bottom',
            'pad': {'t': 0, 'r': 10},
            'buttons': [{
                'label': '▶ 播放动画',
                'method': 'animate',
                'args': [None, {'frame': {'duration': 20, 'redraw': True}, 'fromcurrent': True, 'mode': 'immediate'}]
            }, {
                'label': '⏸ 暂停',
                'method': 'animate',
                'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate', 'transition': {'duration': 0}}]
            }]
        }],
        margin=dict(t=0, b=0, l=0, r=0),
        showlegend=True,
        legend=dict(x=0.85, y=0.95, bgcolor='rgba(255,255,255,0.8)', bordercolor='black', borderwidth=1)
    )

    save_path = RESULT_DIR / "Trajectory_Animation.html"
    fig.write_html(str(save_path))
    print(f"修正版动画已生成: {save_path}")

# === 主程序 ===
if __name__ == "__main__":
    data = load_network()
    G = data['graph']
    
    # 1. 寻找合适的起降点
    # 为了保证效果，我们直接遍历所有节点，找到 Z=0 的节点
    ground_nodes = [n for n, d in G.nodes(data=True) if d['pos'][2] < 1.0]
    
    if len(ground_nodes) < 2:
        print("错误：未找到地面节点，请检查路网生成代码是否包含 'vertical_node' 且 z=0")
        exit()
        
    # 2. 挑选两个距离最远的地面点
    sorted_nodes = sorted(ground_nodes, key=lambda n: G.nodes[n]['pos'][0])
    start = sorted_nodes[0]
    end = sorted_nodes[-1]
    
    print(f"规划任务: 起点{G.nodes[start]['pos']} -> 终点{G.nodes[end]['pos']}")
    
    # 3. 规划逻辑
    def heuristic(u, v):
        p1, p2 = G.nodes[u]['pos'], G.nodes[v]['pos']
        return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)**0.5

    # 权重配置 (鼓励走高空)
    for u, v, d in G.edges(data=True):
        w = d.get('weight', 100)
        l = d.get('layer', 'Low')
        # 高层权重打折，吸引长距离流量
        if l == 'High': d['calc_w'] = w * 0.5 
        elif l == 'Mid': d['calc_w'] = w * 0.8
        else: d['calc_w'] = w
        
    try:
        path = nx.astar_path(G, start, end, heuristic=heuristic, weight='calc_w')
        print(f"路径节点数: {len(path)}")
        create_fixed_animation(data, path)
    except nx.NetworkXNoPath:
        print("无法生成路径，请检查路网连通性")