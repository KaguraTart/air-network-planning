import numpy as np
import pandas as pd  # <--- 关键修复：添加 pandas 库导入
import geopandas as gpd
import networkx as nx
import plotly.graph_objects as go
from shapely.geometry import Point, Polygon
import random
import pickle
import os
from pathlib import Path

'''
生成的全杭州的三维航路网太大了,没办法渲染打开哦!!
'''



# === 1. 配置路径管理 ===
BASE_DIR = Path(".")
# 建议确保这里的路径指向正确，例如你的shp文件确实在 data 目录下
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
RESULT_MAP_DIR = BASE_DIR / "results" / "maps"

# 自动创建文件夹
for p in [DATA_DIR, MODEL_DIR, RESULT_MAP_DIR]:
    p.mkdir(parents=True, exist_ok=True)

class LowAltitudeRouteNetwork:
    def __init__(self, bounds=None, grid_step=200):
        # bounds 格式: [min_x, min_y, max_x, max_y]
        self.bounds = bounds if bounds else [0, 0, 5000, 5000]
        self.step = grid_step
        self.G = nx.Graph()
        self.buildings = None
        
        # --- 核心设计参数 ---
        self.layers = {
            'Low':  {'height': 120, 'width': 80,  'color': '#1f77b4', 'name': '低层 (120m, 宽80m)'},
            'Mid':  {'height': 200, 'width': 100, 'color': '#2ca02c', 'name': '中层 (200m, 宽100m)'},
            'High': {'height': 280, 'width': 120, 'color': '#d62728', 'name': '高层 (280m, 宽120m)'}
        }
        
        # 垂直通道参数
        self.vert_spacing = 1000  # 垂直通道水平间距
        self.vert_width = 20      # 垂直通道物理宽度
        self.safety_margin = 10   # 避障安全缓冲距离

    def load_buildings_from_file(self, shp_path):
        """
        读取真实的 Shapefile 文件并进行预处理
        """
        file_path = Path(shp_path)
        if not file_path.exists():
            raise FileNotFoundError(f"找不到文件: {file_path}")
            
        print(f"正在读取 GIS 数据: {file_path} ...")
        gdf = gpd.read_file(file_path)
        
        # 1. 坐标系检查
        print(f"原始坐标系: {gdf.crs}")
        # 如果是经纬度 (is_geographic 为 True)，则转为 UTM 投影
        if gdf.crs and gdf.crs.is_geographic:
            print("检测到地理坐标系 (经纬度)，正在转换为 UTM Zone 51N (杭州区域，单位:米)...")
            try:
                gdf = gdf.to_crs(epsg=32651)
            except:
                gdf = gdf.to_crs(epsg=3857)
            print(f"转换后坐标系: {gdf.crs}")
        else:
            print("坐标系已为投影坐标系 (单位:米)，无需转换。")
            
        # 2. 识别并标准化高度字段
        possible_height_cols = ['height', 'Height', 'HEIGHT', 'elevation', 'Elevation', 'ELEVATION']
        possible_floor_cols = ['floor', 'Floor', 'FLOOR', '层数', 'Story', 'story']
        
        target_col = None
        is_floor = False
        
        for col in gdf.columns:
            if col in possible_height_cols:
                target_col = col
                break
            if col in possible_floor_cols:
                target_col = col
                is_floor = True
                break
        
        if target_col:
            print(f"使用字段 '{target_col}' 作为高度依据 (is_floor={is_floor})")
            # --- 修复点：这里现在可以正确调用 pd.to_numeric 了 ---
            gdf[target_col] = pd.to_numeric(gdf[target_col], errors='coerce').fillna(10)
            
            if is_floor:
                gdf['height_val'] = gdf[target_col] * 3.0 # 假设层高3米
            else:
                gdf['height_val'] = gdf[target_col]
        else:
            print("未找到高度字段，使用随机高度 (20m-100m) 填充...")
            gdf['height_val'] = np.random.uniform(20, 100, size=len(gdf))

        # 3. 更新区域边界 (Bounds)
        self.buildings = gdf
        self.sindex = self.buildings.sindex
        self.bounds = list(gdf.total_bounds) # [minx, miny, maxx, maxy]
        print(f"已加载 {len(gdf)} 个建筑物。")
        print(f"更新规划区域范围: {self.bounds}")

    def create_mock_buildings(self, count=80):
        """生成虚拟建筑物数据"""
        print(f"正在生成 {count} 个虚拟建筑物...")
        buildings = []
        min_x, min_y, max_x, max_y = self.bounds
        for _ in range(count):
            cx = random.uniform(min_x + 200, max_x - 200)
            cy = random.uniform(min_y + 200, max_y - 200)
            w = random.uniform(80, 250)
            h_geo = random.uniform(80, 250)
            
            r = random.random()
            if r < 0.3: height = random.uniform(30, 80)
            elif r < 0.8: height = random.uniform(90, 180)
            else: height = random.uniform(220, 320)
            
            poly = Polygon([
                (cx-w/2, cy-h_geo/2), (cx+w/2, cy-h_geo/2), 
                (cx+w/2, cy+h_geo/2), (cx-w/2, cy+h_geo/2)
            ])
            buildings.append({'geometry': poly, 'height_val': height})
        
        self.buildings = gpd.GeoDataFrame(buildings)
        self.sindex = self.buildings.sindex

    def _check_collision(self, x, y, z, route_width):
        """基于航路宽度的碰撞检测"""
        if self.buildings is None: return False
        
        point = Point(x, y)
        safe_radius = (route_width / 2.0) + self.safety_margin
        query_area = point.buffer(safe_radius)
        
        possible_idx = list(self.sindex.intersection(query_area.bounds))
        possible_matches = self.buildings.iloc[possible_idx]
        
        for _, row in possible_matches.iterrows():
            if row.geometry.intersects(query_area):
                if row['height_val'] + self.safety_margin > z:
                    return True
        return False

    def generate_graph(self):
        print("正在构建高保真三维航路网...")
        min_x, min_y, max_x, max_y = self.bounds
        
        x_range = np.arange(min_x, max_x, self.step)
        y_range = np.arange(min_y, max_y, self.step)
        
        node_id = 0
        
        # 1. 生成水平航路
        print("  - 生成水平航路节点与连线...")
        for layer_name, config in self.layers.items():
            z = config['height']
            width = config['width']
            for x in x_range:
                for y in y_range:
                    if not self._check_collision(x, y, z, width):
                        self.G.add_node(node_id, pos=(x, y, z), layer=layer_name, 
                                      grid_idx=(x, y), type='horizontal')
                        node_id += 1
                        
        # 水平连线
        nodes = list(self.G.nodes(data=True))
        pos_map = {}
        for n, d in nodes:
            if d.get('type') == 'horizontal':
                pos_map[(d['grid_idx'][0], d['grid_idx'][1], d['layer'])] = n
        
        for n, d in nodes:
            if d.get('type') != 'horizontal': continue
            x, y = d['grid_idx']
            l = d['layer']
            
            neighbors = [(x + self.step, y), (x, y + self.step)]
            for nx_c, ny_c in neighbors:
                if (nx_c, ny_c, l) in pos_map:
                    target = pos_map[(nx_c, ny_c, l)]
                    mid_x, mid_y = (x + nx_c)/2, (y + ny_c)/2
                    if not self._check_collision(mid_x, mid_y, self.layers[l]['height'], self.layers[l]['width']):
                        self.G.add_edge(n, target, weight=self.step, type='horizontal', layer=l)

        # 2. 生成垂直通道
        print("  - 生成双向垂直起降通道...")
        layer_levels = [0] + [self.layers[k]['height'] for k in ['Low', 'Mid', 'High']]
        layer_names_ordered = ['Ground'] + ['Low', 'Mid', 'High']
        
        for x in x_range:
            for y in y_range:
                rel_x = x - min_x
                rel_y = y - min_y
                
                # 垂直通道生成逻辑
                if (abs(rel_x) % self.vert_spacing < 1.0) and (abs(rel_y) % self.vert_spacing < 1.0):
                    
                    if self._check_collision(x, y, 320, self.vert_width): 
                        continue

                    offset = 20 
                    for direction, offset_val in [('Up', offset), ('Down', -offset)]:
                        vx, vy = x + offset_val, y
                        prev_node = None
                        
                        for i, h in enumerate(layer_levels):
                            current_layer_name = layer_names_ordered[i]
                            v_node_id = node_id
                            node_id += 1
                            self.G.add_node(v_node_id, pos=(vx, vy, h), 
                                          layer=current_layer_name, 
                                          type='vertical_node', direction=direction)
                            
                            if prev_node is not None:
                                weight = h - layer_levels[i-1]
                                self.G.add_edge(prev_node, v_node_id, weight=weight, 
                                              type='vertical_shaft', direction=direction)
                            prev_node = v_node_id
                            
                            if current_layer_name in ['Low', 'Mid', 'High']:
                                if (x, y, current_layer_name) in pos_map:
                                    h_node = pos_map[(x, y, current_layer_name)]
                                    dist = np.sqrt((vx-x)**2 + (vy-y)**2)
                                    self.G.add_edge(v_node_id, h_node, weight=dist, 
                                                  type='access_link', direction=direction)

        print(f"路网构建完成: 总节点 {self.G.number_of_nodes()}, 总边数 {self.G.number_of_edges()}")

    def save_network(self, filename="route_network.pkl"):
        filepath = MODEL_DIR / filename
        try:
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'graph': self.G, 
                    'buildings': self.buildings, 
                    'layers': self.layers,
                    'bounds': self.bounds
                }, f)
            print(f"成功: 路网数据已保存至 {filepath}")
        except Exception as e:
            print(f"错误: 保存失败 - {e}")

    def _create_building_mesh(self, geom, height):
        if geom.geom_type != 'Polygon': return None, None, None, None, None, None
        x, y = geom.exterior.xy
        x, y = list(x)[:-1], list(y)[:-1]
        n = len(x)
        avg_x = sum(x) / n
        avg_y = sum(y) / n
        vx = x * 2 + [avg_x, avg_x] 
        vy = y * 2 + [avg_y, avg_y]
        vz = [0]*n + [height]*n + [height, 0]
        i, j, k = [], [], []
        for idx in range(n):
            next_idx = (idx + 1) % n
            i.append(idx); j.append(idx+n); k.append(next_idx)
            i.append(next_idx); j.append(idx+n); k.append(next_idx+n)
            center_top_idx = 2 * n
            i.append(idx+n); j.append(next_idx+n); k.append(center_top_idx)
        return vx, vy, vz, i, j, k

    def visualize(self, filename="Hangzhou_Route_Final.html"):
        print("正在渲染可视化图表...")
        fig = go.Figure()

        if self.buildings is not None:
            print("  - 渲染建筑物模型...")
            for _, row in self.buildings.iterrows():
                vx, vy, vz, i, j, k = self._create_building_mesh(row.geometry, row['height_val'])
                if vx:
                    fig.add_trace(go.Mesh3d(
                        x=vx, y=vy, z=vz, i=i, j=j, k=k,
                        color='lightgray', opacity=1.0, flatshading=True,
                        lighting=dict(ambient=0.6, diffuse=0.5, roughness=0.1),
                        hoverinfo='skip', showlegend=False
                    ))
            fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode='markers',
                                     marker=dict(size=10, color='lightgray', symbol='square'),
                                     name='建筑物'))

        node_pos = nx.get_node_attributes(self.G, 'pos')
        
        # 水平航路
        for layer_name, config in self.layers.items():
            edge_x, edge_y, edge_z = [], [], []
            for u, v, d in self.G.edges(data=True):
                if d.get('type') == 'horizontal' and d.get('layer') == layer_name:
                    x0, y0, z0 = node_pos[u]
                    x1, y1, z1 = node_pos[v]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])
                    edge_z.extend([z0, z1, None])
            
            width_px = (config['width'] - 60) / 15 + 2
            fig.add_trace(go.Scatter3d(
                x=edge_x, y=edge_y, z=edge_z, mode='lines',
                line=dict(color=config['color'], width=width_px),
                opacity=0.5, name=config['name'], hoverinfo='skip'
            ))

        # 垂直通道
        dirs = {'Up': {'color': 'cyan', 'name': '上行通道'}, 'Down': {'color': 'magenta', 'name': '下行通道'}}
        for direction, style in dirs.items():
            vx, vy, vz = [], [], []
            for u, v, d in self.G.edges(data=True):
                if d.get('type') in ['vertical_shaft', 'access_link'] and d.get('direction') == direction:
                    x0, y0, z0 = node_pos[u]
                    x1, y1, z1 = node_pos[v]
                    vx.extend([x0, x1, None])
                    vy.extend([y0, y1, None])
                    vz.extend([z0, z1, None])
            fig.add_trace(go.Scatter3d(
                x=vx, y=vy, z=vz, mode='lines',
                line=dict(color=style['color'], width=4), name=style['name']
            ))

        port_x, port_y = [], []
        for n, d in self.G.nodes(data=True):
            if d.get('type') == 'vertical_node' and d['pos'][2] == 0:
                port_x.append(d['pos'][0])
                port_y.append(d['pos'][1])
        
        fig.add_trace(go.Scatter3d(
            x=port_x, y=port_y, z=[0]*len(port_x), mode='markers',
            marker=dict(size=4, color='black', symbol='circle-open'), name='地面起降点'
        ))

        fig.update_layout(
            title="杭州城市低空三维立体航路网设计 (Generated)",
            scene=dict(
                xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Z (m)',
                xaxis=dict(backgroundcolor="rgb(245, 245, 245)"),
                yaxis=dict(backgroundcolor="rgb(245, 245, 245)"),
                zaxis=dict(backgroundcolor="rgb(240, 240, 240)"),
                aspectmode='manual', aspectratio=dict(x=1, y=1, z=0.35),
                camera=dict(eye=dict(x=1.4, y=-1.4, z=0.6))
            ),
            margin=dict(t=40, b=0, l=0, r=0)
        )
        
        save_path = RESULT_MAP_DIR / filename
        fig.write_html(str(save_path))
        print(f"可视化文件已生成: {save_path}")

# === 主程序入口 ===
if __name__ == "__main__":
    # 1. 初始化
    net = LowAltitudeRouteNetwork(bounds=None, grid_step=200)
    
    # 2. 尝试加载真实数据
    shp_path = DATA_DIR / "Hangzhou_Buildings_DWG-Polygon.shp"
    
    try:
        if shp_path.exists():
            net.load_buildings_from_file(shp_path)
        else:
            print(f"警告: 未找到 {shp_path}，使用虚拟数据演示。")
            net.bounds = [0, 0, 5000, 5000]
            net.create_mock_buildings(count=70)
    except Exception as e:
        print(f"读取数据出错: {e}，回退到虚拟数据。")
        # 即使出错也打印出来看看是什么错（已经在 load_buildings_from_file 里 print 了，但上面那个 try 捕获后可以 re-raise 调试，或者像这样降级处理）
        import traceback
        traceback.print_exc()
        net.bounds = [0, 0, 5000, 5000]
        net.create_mock_buildings(count=70)
    
    # 3. 生成拓扑网络
    net.generate_graph()
    
    # 4. 保存为文件
    net.save_network("hangzhou_route_graph.pkl")
    
    # 5. 生成可视化
    net.visualize("Route_Network_hangzhou_all.html")