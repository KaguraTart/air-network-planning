import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
import plotly.graph_objects as go
from shapely.geometry import Point, Polygon
import random
import pickle
import os
from pathlib import Path
from pyproj import Transformer # 用于经纬度转换标注

# === 1. 配置路径管理 ===
BASE_DIR = Path(".")
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
RESULT_MAP_DIR = BASE_DIR / "results" / "maps"

for p in [DATA_DIR, MODEL_DIR, RESULT_MAP_DIR]:
    p.mkdir(parents=True, exist_ok=True)

class LowAltitudeRouteNetwork:
    def __init__(self, bounds=None, grid_step=200):
        self.bounds = bounds if bounds else [0, 0, 5000, 5000]
        self.step = grid_step
        self.G = nx.Graph()
        self.buildings = None
        
        # 记录原始投影坐标系，用于后续转换
        self.crs = None 
        
        self.layers = {
            'Low':  {'height': 120, 'width': 80,  'color': '#1f77b4', 'name': '低层 (120m)'},
            'Mid':  {'height': 200, 'width': 100, 'color': '#2ca02c', 'name': '中层 (200m)'},
            'High': {'height': 280, 'width': 120, 'color': '#d62728', 'name': '高层 (280m)'}
        }
        self.vert_spacing = 1000
        self.vert_width = 20
        self.safety_margin = 10

    def select_area(self, gdf, area_size_m=5000, center_point=None):
        print(f"\n=== 智能选取核心区域 ({area_size_m/1000:.1f}km x {area_size_m/1000:.1f}km) ===")
        
        if gdf.crs and gdf.crs.is_geographic:
            try:
                gdf_proj = gdf.to_crs(epsg=32651)
            except:
                gdf_proj = gdf.to_crs(epsg=3857)
        else:
            gdf_proj = gdf.copy()
        
        # 保存投影信息
        self.crs = gdf_proj.crs
        
        if center_point is None:
            # 使用密度扫描
            try:
                from sklearn.neighbors import KernelDensity
                gdf_proj['centroid'] = gdf_proj.geometry.centroid
                points = np.array([[p.x, p.y] for p in gdf_proj['centroid']])
                
                # 降采样以加速
                sample_points = points[::10] if len(points) > 5000 else points
                
                kde = KernelDensity(bandwidth=500, metric='euclidean', kernel='gaussian')
                kde.fit(sample_points)
                
                xmin, ymin, xmax, ymax = gdf_proj.total_bounds
                xx, yy = np.meshgrid(np.linspace(xmin, xmax, 30), np.linspace(ymin, ymax, 30))
                grid_points = np.vstack([xx.ravel(), yy.ravel()]).T
                
                log_density = kde.score_samples(grid_points)
                max_idx = np.argmax(log_density)
                center_x, center_y = grid_points[max_idx]
                print(f"已锁定最密集区域中心 (UTM): ({center_x:.1f}, {center_y:.1f})")
            except:
                print("sklearn 密度扫描失败，回退到几何中心。")
                minx, miny, maxx, maxy = gdf_proj.total_bounds
                center_x, center_y = (minx+maxx)/2, (miny+maxy)/2
        else:
            p = gpd.GeoSeries([Point(center_point)], crs="EPSG:4326").to_crs(gdf_proj.crs)
            center_x, center_y = p[0].x, p[0].y

        half = area_size_m / 2
        bbox = gpd.GeoDataFrame(geometry=[Point(center_x, center_y).buffer(half, cap_style=3)], crs=gdf_proj.crs)
        selected = gpd.sjoin(gdf_proj, bbox, how='inner', predicate='intersects')
        
        self.bounds = list(selected.total_bounds)
        return selected

    def load_buildings_from_file(self, shp_path, use_small_area=True, area_size_m=5000, center_point=None):
        file_path = Path(shp_path)
        if not file_path.exists(): raise FileNotFoundError(f"找不到文件: {file_path}")
            
        print(f"读取 GIS 数据: {file_path}")
        gdf = gpd.read_file(file_path)
        
        if use_small_area:
            gdf = self.select_area(gdf, area_size_m, center_point)
        else:
            if gdf.crs and gdf.crs.is_geographic: gdf = gdf.to_crs(epsg=32651)
            self.bounds = list(gdf.total_bounds)
            self.crs = gdf.crs

        # 高度处理
        target_col = None
        is_floor = False
        for col in gdf.columns:
            if col in ['height', 'Height', 'HEIGHT', 'Elevation', 'floor', 'Floor', '层数']:
                target_col = col
                if 'floor' in col.lower() or '层' in col: is_floor = True
                break
        
        if target_col:
            gdf['height_val'] = pd.to_numeric(gdf[target_col], errors='coerce').fillna(15)
            if is_floor: gdf['height_val'] *= 3.0
        else:
            gdf['height_val'] = np.abs(np.random.normal(40, 20, size=len(gdf))) + 15

        self.buildings = gdf
        self.sindex = self.buildings.sindex
        print(f"范围: {self.bounds}")

    def create_mock_buildings(self, count=80):
        print(f"生成虚拟数据...")
        buildings = []
        min_x, min_y, max_x, max_y = self.bounds
        for _ in range(count):
            cx = random.uniform(min_x + 200, max_x - 200)
            cy = random.uniform(min_y + 200, max_y - 200)
            w = random.uniform(80, 250)
            h_geo = random.uniform(80, 250)
            height = random.uniform(30, 200)
            poly = Polygon([(cx-w/2, cy-h_geo/2), (cx+w/2, cy-h_geo/2), (cx+w/2, cy+h_geo/2), (cx-w/2, cy+h_geo/2)])
            buildings.append({'geometry': poly, 'height_val': height})
        self.buildings = gpd.GeoDataFrame(buildings)
        self.sindex = self.buildings.sindex

    def _check_collision(self, x, y, z, route_width):
        if self.buildings is None: return False
        point = Point(x, y)
        query_area = point.buffer((route_width/2.0) + self.safety_margin)
        possible_idx = list(self.sindex.intersection(query_area.bounds))
        possible_matches = self.buildings.iloc[possible_idx]
        for _, row in possible_matches.iterrows():
            if row.geometry.intersects(query_area):
                if row['height_val'] + self.safety_margin > z: return True
        return False

    def generate_graph(self):
        print("构建三维航路网...")
        min_x, min_y, max_x, max_y = self.bounds
        x_range = np.arange(min_x, max_x, self.step)
        y_range = np.arange(min_y, max_y, self.step)
        node_id = 0
        
        # 1. 水平
        for layer_name, config in self.layers.items():
            z = config['height']
            width = config['width']
            for x in x_range:
                for y in y_range:
                    if not self._check_collision(x, y, z, width):
                        self.G.add_node(node_id, pos=(x, y, z), layer=layer_name, grid_idx=(x, y), type='horizontal')
                        node_id += 1
                        
        nodes = list(self.G.nodes(data=True))
        pos_map = {(d['grid_idx'][0], d['grid_idx'][1], d['layer']): n for n, d in nodes if d.get('type')=='horizontal'}
        
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

        # 2. 垂直
        layer_levels = [0] + [self.layers[k]['height'] for k in ['Low', 'Mid', 'High']]
        layer_names_ordered = ['Ground'] + ['Low', 'Mid', 'High']
        
        for x in x_range:
            for y in y_range:
                if (abs(x - min_x) % self.vert_spacing < 1.0) and (abs(y - min_y) % self.vert_spacing < 1.0):
                    if self._check_collision(x, y, 320, self.vert_width): continue
                    offset = 25 
                    for direction, offset_val in [('Up', offset), ('Down', -offset)]:
                        vx, vy = x + offset_val, y
                        prev_node = None
                        for i, h in enumerate(layer_levels):
                            current_layer = layer_names_ordered[i]
                            v_node_id = node_id
                            node_id += 1
                            self.G.add_node(v_node_id, pos=(vx, vy, h), layer=current_layer, type='vertical_node', direction=direction)
                            if prev_node is not None:
                                self.G.add_edge(prev_node, v_node_id, weight=h-layer_levels[i-1], type='vertical_shaft', direction=direction)
                            prev_node = v_node_id
                            if current_layer in ['Low', 'Mid', 'High'] and (x, y, current_layer) in pos_map:
                                h_node = pos_map[(x, y, current_layer)]
                                dist = np.sqrt((vx-x)**2 + (vy-y)**2)
                                self.G.add_edge(v_node_id, h_node, weight=dist, type='access_link', direction=direction)
        print(f"路网构建完成: {self.G.number_of_nodes()} 节点")

    def save_network(self, filename="route_network.pkl"):
        filepath = MODEL_DIR / filename
        with open(filepath, 'wb') as f:
            pickle.dump({'graph': self.G, 'buildings': self.buildings, 'layers': self.layers, 'bounds': self.bounds}, f)
        print(f"数据已保存: {filepath}")

    def _create_building_mesh(self, geom, height, center_offset_x=0, center_offset_y=0):
        """生成网格，支持坐标平移归零"""
        if geom.geom_type != 'Polygon': return None, None, None, None, None, None
        x, y = geom.exterior.xy
        x = [val - center_offset_x for val in list(x)[:-1]] # 平移 X
        y = [val - center_offset_y for val in list(y)[:-1]] # 平移 Y
        n = len(x)
        avg_x, avg_y = sum(x)/n, sum(y)/n
        vx = x + x + [avg_x]
        vy = y + y + [avg_y]
        vz = [0]*n + [height]*n + [height]
        i, j, k = [], [], []
        for idx in range(n):
            next_idx = (idx + 1) % n
            i.extend([idx, next_idx]); j.extend([idx+n, idx+n]); k.extend([next_idx, next_idx+n])
            i.append(idx+n); j.append(next_idx+n); k.append(2*n)
        return vx, vy, vz, i, j, k

    def visualize(self, filename="Hangzhou_Route_Relative.html"):
        """
        [可视化优化版] 
        将坐标轴转换为相对中心点的米数 (例如 0, 0 为中心)，方便直观查看。
        """
        print("正在渲染可视化 (相对坐标模式)...")
        fig = go.Figure()

        # 1. 计算中心点，用于坐标归零
        min_x, min_y, max_x, max_y = self.bounds
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        
        # 计算中心点的真实经纬度，用于标题显示
        title_text = "杭州城市低空三维立体航路网"
        if self.crs:
            try:
                trans = Transformer.from_crs(self.crs, "EPSG:4326", always_xy=True)
                lon, lat = trans.transform(center_x, center_y)
                title_text += f"<br><sup>中心点: ({lon:.4f}E, {lat:.4f}N) | 坐标单位: 米 (相对中心)</sup>"
            except: pass

        # === 1. 建筑物合并渲染 (带平移) ===
        if self.buildings is not None:
            all_x, all_y, all_z = [], [], []
            all_i, all_j, all_k = [], [], []
            v_offset = 0
            
            for _, row in self.buildings.iterrows():
                # 传入偏移量
                vx, vy, vz, i, j, k = self._create_building_mesh(
                    row.geometry, row['height_val'], center_x, center_y
                )
                if vx:
                    all_x.extend(vx); all_y.extend(vy); all_z.extend(vz)
                    all_i.extend([idx + v_offset for idx in i])
                    all_j.extend([idx + v_offset for idx in j])
                    all_k.extend([idx + v_offset for idx in k])
                    v_offset += len(vx)

            fig.add_trace(go.Mesh3d(
                x=all_x, y=all_y, z=all_z,
                i=all_i, j=all_j, k=all_k,
                color='lightgray', opacity=1.0, flatshading=True,
                name='建筑物', showlegend=True
            ))

        # === 2. 航路网渲染 (带平移) ===
        node_pos = nx.get_node_attributes(self.G, 'pos')
        
        # 水平航路
        for layer_name, config in self.layers.items():
            edge_x, edge_y, edge_z = [], [], []
            for u, v, d in self.G.edges(data=True):
                if d.get('type') == 'horizontal' and d.get('layer') == layer_name:
                    x0, y0, z0 = node_pos[u]
                    x1, y1, z1 = node_pos[v]
                    # 减去中心点坐标
                    edge_x.extend([x0 - center_x, x1 - center_x, None])
                    edge_y.extend([y0 - center_y, y1 - center_y, None])
                    edge_z.extend([z0, z1, None])
            
            width_px = (config['width'] - 60) / 15 + 2
            fig.add_trace(go.Scatter3d(
                x=edge_x, y=edge_y, z=edge_z, mode='lines',
                line=dict(color=config['color'], width=width_px),
                opacity=0.6, name=config['name'], hoverinfo='skip'
            ))

        # 垂直通道
        dirs = {'Up': {'color': 'cyan', 'name': '上行通道'}, 'Down': {'color': 'magenta', 'name': '下行通道'}}
        for direction, style in dirs.items():
            vx, vy, vz = [], [], []
            for u, v, d in self.G.edges(data=True):
                if d.get('type') in ['vertical_shaft', 'access_link'] and d.get('direction') == direction:
                    x0, y0, z0 = node_pos[u]
                    x1, y1, z1 = node_pos[v]
                    vx.extend([x0 - center_x, x1 - center_x, None])
                    vy.extend([y0 - center_y, y1 - center_y, None])
                    vz.extend([z0, z1, None])
            fig.add_trace(go.Scatter3d(
                x=vx, y=vy, z=vz, mode='lines',
                line=dict(color=style['color'], width=4), name=style['name']
            ))

        # 地面基站
        port_x, port_y = [], []
        for n, d in self.G.nodes(data=True):
            if d.get('type') == 'vertical_node' and d['pos'][2] == 0:
                port_x.append(d['pos'][0] - center_x)
                port_y.append(d['pos'][1] - center_y)
        
        fig.add_trace(go.Scatter3d(
            x=port_x, y=port_y, z=[0]*len(port_x), mode='markers',
            marker=dict(size=4, color='black', symbol='circle-open'), name='地面起降点'
        ))

        # === 3. 布局优化 ===
        fig.update_layout(
            title=title_text,
            scene=dict(
                # 修改坐标轴标题
                xaxis_title='相对距离 X (m)', 
                yaxis_title='相对距离 Y (m)', 
                zaxis_title='高度 Z (m)',
                xaxis=dict(backgroundcolor="rgb(245, 245, 245)"),
                yaxis=dict(backgroundcolor="rgb(245, 245, 245)"),
                zaxis=dict(backgroundcolor="rgb(240, 240, 240)"),
                aspectmode='manual', aspectratio=dict(x=1, y=1, z=0.35),
                camera=dict(eye=dict(x=1.4, y=-1.4, z=0.6))
            ),
            margin=dict(t=60, b=0, l=0, r=0)
        )
        
        save_path = RESULT_MAP_DIR / filename
        fig.write_html(str(save_path))
        print(f"可视化文件已生成: {save_path}")

# === 主程序入口 ===
if __name__ == "__main__":
    net = LowAltitudeRouteNetwork(bounds=None, grid_step=200)
    shp_path = DATA_DIR / "Hangzhou_Buildings_DWG-Polygon.shp"
    
    try:
        if shp_path.exists():
            net.load_buildings_from_file(shp_path, use_small_area=True, area_size_m=5000)
        else:
            print(f"未找到 {shp_path}，使用虚拟数据。")
            net.bounds = [0, 0, 5000, 5000]
            net.create_mock_buildings(count=70)
    except Exception as e:
        import traceback
        traceback.print_exc()
        net.bounds = [0, 0, 5000, 5000]
        net.create_mock_buildings(count=70)
    
    net.generate_graph()
    net.save_network("hangzhou_route_graph.pkl")
    net.visualize("Route_Network.html")