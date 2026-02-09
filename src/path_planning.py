import networkx as nx
import plotly.graph_objects as go
import pickle
import numpy as np
from pathlib import Path

# === 1. 配置与加载 ===
MODEL_DIR = Path("models")
RESULT_DIR = Path("results/maps")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

def load_network(filename="hangzhou_route_graph.pkl"):
    print("正在加载路网数据...")
    with open(MODEL_DIR / filename, "rb") as f:
        data = pickle.load(f)
    return data

# === 2. 改进的寻径逻辑 (对应论文 3.4.1) ===
def find_optimal_path(G, start_node, end_node):
    """
    使用 A* 算法寻找最优路径。
    为了体现论文中“长途高飞”的策略，我们在权重上做了隐式处理：
    假设高层航路阻力更小（权重更低），引导算法优先选择高层。
    """
    print(f"正在规划路径: 节点 {start_node} -> 节点 {end_node}")
    
    # 定义启发式函数 (欧氏距离)
    def heuristic(u, v):
        pos_u = G.nodes[u]['pos']
        pos_v = G.nodes[v]['pos']
        return np.sqrt((pos_u[0]-pos_v[0])**2 + (pos_u[1]-pos_v[1])**2 + (pos_u[2]-pos_v[2])**2)

    # 动态调整权重 (模拟论文中的代价函数 F = G + H + LayerPenalty)
    # 我们通过修改图的属性来实现：让高层的 weight 变小，鼓励长距离走高层
    # 注意：这里只是临时修改内存中的图，不保存回文件
    for u, v, d in G.edges(data=True):
        base_weight = d.get('weight', 100)
        layer = d.get('layer', 'Low')
        
        # 策略：高层代价打折，吸引长距离任务
        if layer == 'High':
            d['calc_weight'] = base_weight * 0.6  # 高层阻力最小
        elif layer == 'Mid':
            d['calc_weight'] = base_weight * 0.8
        else:
            d['calc_weight'] = base_weight * 1.0
            
        # 垂直通道给予一定惩罚，防止频繁换层，但对于长距离是有益的
        if d.get('type') in ['vertical_shaft', 'access_link']:
             d['calc_weight'] = base_weight * 1.2

    try:
        path = nx.astar_path(G, start_node, end_node, heuristic=heuristic, weight='calc_weight')
        print(f"路径规划成功! 路径长度: {len(path)} 个节点")
        return path
    except nx.NetworkXNoPath:
        print("无可行路径!")
        return None

# === 3. 可视化绘制 (保持高性能优化) ===
def visualize_trajectory(data, path, filename="Trajectory_Plan.html"):
    G = data['graph']
    buildings = data['buildings']
    bounds = data['bounds']
    
    print("正在渲染轨迹演示图...")
    fig = go.Figure()

    # --- 3.1 绘制建筑物 (使用合并网格优化) ---
    # 计算中心点用于坐标平移，保持坐标轴数字简洁
    min_x, min_y, max_x, max_y = bounds
    cx, cy = (min_x + max_x)/2, (min_y + max_y)/2
    
    if buildings is not None:
        all_x, all_y, all_z = [], [], []
        all_i, all_j, all_k = [], [], []
        v_offset = 0
        
        # 简单的 Mesh 生成器
        def make_mesh(geom, h):
            if geom.geom_type != 'Polygon': return None, None, None, None, None, None
            x, y = geom.exterior.xy
            x = [v - cx for v in list(x)[:-1]]; y = [v - cy for v in list(y)[:-1]]
            n = len(x)
            vx = x * 2 + [sum(x)/n, sum(x)/n]
            vy = y * 2 + [sum(y)/n, sum(y)/n]
            vz = [0]*n + [h]*n + [h, 0]
            idx_i, idx_j, idx_k = [], [], []
            for k in range(n):
                nk = (k+1)%n
                idx_i.extend([k, nk]); idx_j.extend([k+n, k+n]); idx_k.extend([nk, nk+n])
                idx_i.append(k+n); idx_j.append(nk+n); idx_k.append(2*n)
            return vx, vy, vz, idx_i, idx_j, idx_k

        for _, row in buildings.iterrows():
            vx, vy, vz, i, j, k = make_mesh(row.geometry, row['height_val'])
            if vx:
                all_x.extend(vx); all_y.extend(vy); all_z.extend(vz)
                all_i.extend([x + v_offset for x in i])
                all_j.extend([x + v_offset for x in j])
                all_k.extend([x + v_offset for x in k])
                v_offset += len(vx)

        fig.add_trace(go.Mesh3d(
            x=all_x, y=all_y, z=all_z, i=all_i, j=all_j, k=all_k,
            color='lightgray', opacity=0.8, flatshading=True, name='城市建筑'
        ))

    # --- 3.2 绘制背景航路网 (半透明，作为背景) ---
    node_pos = nx.get_node_attributes(G, 'pos')
    edge_x, edge_y, edge_z = [], [], []
    # 随机抽样一部分边作为背景，全画太乱且卡
    for u, v in list(G.edges())[::5]: 
        x0, y0, z0 = node_pos[u]; x1, y1, z1 = node_pos[v]
        edge_x.extend([x0-cx, x1-cx, None])
        edge_y.extend([y0-cy, y1-cy, None])
        edge_z.extend([z0, z1, None])
    
    fig.add_trace(go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines', line=dict(color='rgba(200, 200, 200, 0.2)', width=1),
        name='航路网络(背景)', hoverinfo='skip'
    ))

    # --- 3.3 绘制规划出的路径 (高亮主角) ---
    if path:
        px = [node_pos[n][0]-cx for n in path]
        py = [node_pos[n][1]-cy for n in path]
        pz = [node_pos[n][2] for n in path]
        
        # 路径线 (霓虹光感)
        fig.add_trace(go.Scatter3d(
            x=px, y=py, z=pz,
            mode='lines',
            line=dict(color='#00ff00', width=8), # 亮绿色粗线
            name='规划轨迹 (A*)'
        ))
        
        # 关键点 (起终点)
        fig.add_trace(go.Scatter3d(
            x=[px[0], px[-1]], y=[py[0], py[-1]], z=[pz[0], pz[-1]],
            mode='markers+text',
            marker=dict(size=10, color='red', symbol='diamond'),
            text=['起点 (Start)', '终点 (End)'],
            textposition="top center",
            name='任务端点'
        ))

    # 设置漂亮的布局
    fig.update_layout(
        title="单机任务三维路径规划演示 (3D Trajectory Planning)",
        scene=dict(
            xaxis_title='相对 X (m)', yaxis_title='相对 Y (m)', zaxis_title='高度 (m)',
            xaxis=dict(showbackground=False, gridcolor='white'),
            yaxis=dict(showbackground=False, gridcolor='white'),
            zaxis=dict(showbackground=False, gridcolor='white', range=[0, 350]),
            aspectmode='manual', aspectratio=dict(x=1, y=1, z=0.4),
            camera=dict(eye=dict(x=1.5, y=-1.5, z=0.8))
        ),
        paper_bgcolor='white',
        margin=dict(t=40, b=0, l=0, r=0)
    )
    
    save_path = RESULT_DIR / filename
    fig.write_html(str(save_path))
    print(f"可视化已生成: {save_path}")

# === 主程序 ===
if __name__ == "__main__":
    # 1. 加载
    data = load_network()
    G = data['graph']
    
    # 2. 随机找两个地面起降点作为测试任务
    # 筛选所有地面节点 (z=0)
    ground_nodes = [n for n, d in G.nodes(data=True) if d['pos'][2] == 0]
    
    if len(ground_nodes) < 2:
        print("错误: 路网中找不到足够的地面起降点。请检查 generate_network 是否生成了垂直通道。")
    else:
        # 挑选两个距离较远的点，迫使算法选择高层航路
        # 这里简单随机选，你也可以指定坐标找最近节点
        start = ground_nodes[0]
        end = ground_nodes[-1] # 选列表两端，通常距离较远
        
        # 3. 规划
        path = find_optimal_path(G, start, end)
        
        # 4. 可视化
        visualize_trajectory(data, path)