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

# === 1. 配置路径管理 ===
BASE_DIR = Path(".")
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

    def select_area(self, gdf, area_size_m=5000, center_point=None):
        """
        智能选择区域：优先寻找建筑物密度最高的区域 (CBD)
        """
        print(f"\n=== 智能选取核心区域 ({area_size_m/1000:.1f}km x {area_size_m/1000:.1f}km) ===")
        
        # 1. 统一转为投影坐标系 (UTM) 以便计算米
        if gdf.crs and gdf.crs.is_geographic:
            try:
                # 尝试自动计算 UTM 带，或者直接用杭州所在的 51N (EPSG:32651)
                gdf_proj = gdf.to_crs(epsg=32651) 
                print("已转换为 UTM Zone 51N 投影坐标系")
            except:
                gdf_proj = gdf.to_crs(epsg=3857) # Web Mercator
        else:
            gdf_proj = gdf.copy()
        
        # 2. 确定中心点
        if center_point is None:
            print("正在计算建筑物密度热力图，寻找最密集区域(CBD)...")
            try:
                from sklearn.neighbors import KernelDensity
                
                # 提取所有建筑中心点
                gdf_proj['centroid'] = gdf_proj.geometry.centroid
                points = np.array([[p.x, p.y] for p in gdf_proj['centroid']])
                
                # 使用核密度估计 (KDE) 寻找密度峰值
                # bandwidth=500m 意味着搜索半径
                kde = KernelDensity(bandwidth=500, metric='euclidean', kernel='gaussian', algorithm='ball_tree')
                kde.fit(points)
                
                # 创建搜索网格 (降低分辨率以加快速度)
                xmin, ymin, xmax, ymax = gdf_proj.total_bounds
                xx, yy = np.meshgrid(
                    np.linspace(xmin, xmax, 50),
                    np.linspace(ymin, ymax, 50)
                )
                grid_points = np.vstack([xx.ravel(), yy.ravel()]).T
                
                # 获取密度评分
                log_density = kde.score_samples(grid_points)
                max_idx = np.argmax(log_density)
                center_x, center_y = grid_points[max_idx]
                
                print(f"已自动锁定最密集区域中心: ({center_x:.1f}, {center_y:.1f})")
                
            except ImportError:
                print("⚠️ sklearn 未安装，回退到几何中心选择。")
                minx, miny, maxx, maxy = gdf_proj.total_bounds
                center_x, center_y = (minx+maxx)/2, (miny+maxy)/2
        else:
            # 如果用户指定了经纬度
            p = gpd.GeoSeries([Point(center_point)], crs="EPSG:4326").to_crs(gdf_proj.crs)
            center_x, center_y = p[0].x, p[0].y
            print(f"使用指定中心点: {center_point}")

        # 3. 创建裁剪框并筛选
        half_size = area_size_m / 2
        bbox_geom = Point(center_x, center_y).buffer(half_size, cap_style=3) # cap_style=3 create square
        bbox = gpd.GeoDataFrame(geometry=[bbox_geom], crs=gdf_proj.crs)
        
        selected_buildings = gpd.sjoin(gdf_proj, bbox, how='inner', predicate='intersects')
        
        print(f"筛选结果: 从 {len(gdf)} 个建筑物中选出核心区 {len(selected_buildings)} 个")
        
        # 4. 更新类的边界属性 (用于后续网格生成)
        self.bounds = list(selected_buildings.total_bounds)
        
        return selected_buildings

    def load_buildings_from_file(self, shp_path, use_small_area=True, area_size_m=5000, center_point=None):
        file_path = Path(shp_path)
        if not file_path.exists():
            raise FileNotFoundError(f"找不到文件: {file_path}")
            
        print(f"正在读取 GIS 数据: {file_path} ...")
        gdf = gpd.read_file(file_path)
        
        # 1. 区域选择 (包含坐标系转换)
        if use_small_area:
            gdf = self.select_area(gdf, area_size_m, center_point)
        else:
            # 如果不裁剪，也要确保坐标系正确
            if gdf.crs and gdf.crs.is_geographic:
                gdf = gdf.to_crs(epsg=32651)
            self.bounds = list(gdf.total_bounds)
            
        # 2. 高度处理
        target_col = None
        is_floor = False
        possible_cols = ['height', 'Height', 'HEIGHT', 'Elevation', 'elevation', 'floor', 'Floor', '层数']
        
        for col in gdf.columns:
            if col in possible_cols:
                target_col = col
                if 'floor' in col.lower() or '层' in col: is_floor = True
                break
        
        if target_col:
            print(f"使用字段 '{target_col}' 提取高度...")
            gdf['height_val'] = pd.to_numeric(gdf[target_col], errors='coerce').fillna(15)
            if is_floor: gdf['height_val'] *= 3.0
        else:
            print("未找到高度字段，生成随机城市天际线...")
            # 使用正态分布生成高度，模拟真实城市：大部分低，少量高
            gdf['height_val'] = np.abs(np.random.normal(40, 20, size=len(gdf))) + 15

        self.buildings = gdf
        self.sindex = self.buildings.sindex
        print(f"数据准备就绪。范围: {self.bounds}")

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
            height = random.uniform(30, 200)
            
            poly = Polygon([
                (cx-w/2, cy-h_geo/2), (cx+w/2, cy-h_geo/2), 
                (cx+w/2, cy+h_geo/2), (cx-w/2, cy+h_geo/2)
            ])
            buildings.append({'geometry': poly, 'height_val': height})
        
        self.buildings = gpd.GeoDataFrame(buildings)
        self.sindex = self.buildings.sindex

    def _check_collision(self, x, y, z, route_width):
        """避障检测"""
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
        
        # 1. 水平航路
        for layer_name, config in self.layers.items():
            z = config['height']
            width = config['width']
            for x in x_range:
                for y in y_range:
                    if not self._check_collision(x, y, z, width):
                        self.G.add_node(node_id, pos=(x, y, z), layer=layer_name, 
                                      grid_idx=(x, y), type='horizontal')
                        node_id += 1
                        
        # 水平连接
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

        # 2. 垂直通道
        layer_levels = [0] + [self.layers[k]['height'] for k in ['Low', 'Mid', 'High']]
        layer_names_ordered = ['Ground'] + ['Low', 'Mid', 'High']
        
        for x in x_range:
            for y in y_range:
                # 相对坐标取模，生成垂直通道
                if (abs(x - min_x) % self.vert_spacing < 1.0) and (abs(y - min_y) % self.vert_spacing < 1.0):
                    if self._check_collision(x, y, 320, self.vert_width): 
                        continue

                    offset = 25 
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
        """生成单个建筑的网格数据 (顶点+面)"""
        if geom.geom_type != 'Polygon': return None, None, None, None, None, None
        x, y = geom.exterior.xy
        x, y = list(x)[:-1], list(y)[:-1]
        n = len(x)
        avg_x = sum(x) / n
        avg_y = sum(y) / n
        
        # 顶点：底面环 + 顶面环 + 顶面中心
        vx = x + x + [avg_x]
        vy = y + y + [avg_y]
        vz = [0]*n + [height]*n + [height]
        
        i, j, k = [], [], []
        for idx in range(n):
            next_idx = (idx + 1) % n
            # 侧面 1
            i.append(idx); j.append(idx+n); k.append(next_idx)
            # 侧面 2
            i.append(next_idx); j.append(idx+n); k.append(next_idx+n)
            # 顶面
            i.append(idx+n); j.append(next_idx+n); k.append(2*n)
        return vx, vy, vz, i, j, k

    def visualize(self, filename="Hangzhou_Route_Final_Optimized.html"):
        """
        [高性能渲染版]
        使用 Mesh Merging 技术合并所有建筑为一个对象。
        即使在 5km x 5km 密集区域，文件依然小巧流畅。
        """
        print("正在进行高性能可视化渲染...")
        fig = go.Figure()

        # === 1. 建筑物超级网格 (Mesh Merging) ===
        if self.buildings is not None:
            print(f"  - 正在合并 {len(self.buildings)} 个建筑物网格 (这可能需要几秒钟)...")
            
            all_x, all_y, all_z = [], [], []
            all_i, all_j, all_k = [], [], []
            v_offset = 0
            
            # 遍历所有建筑物，将顶点和面数据放入一个大数组
            for _, row in self.buildings.iterrows():
                vx, vy, vz, i, j, k = self._create_building_mesh(row.geometry, row['height_val'])
                
                if vx:
                    all_x.extend(vx)
                    all_y.extend(vy)
                    all_z.extend(vz)
                    
                    # 关键：面的索引必须加上当前的偏移量
                    all_i.extend([idx + v_offset for idx in i])
                    all_j.extend([idx + v_offset for idx in j])
                    all_k.extend([idx + v_offset for idx in k])
                    
                    v_offset += len(vx)

            # 只绘制这一个 Trace，大大减轻浏览器负担
            fig.add_trace(go.Mesh3d(
                x=all_x, y=all_y, z=all_z,
                i=all_i, j=all_j, k=all_k,
                color='lightgray',
                opacity=1.0,
                flatshading=True,
                name='City Buildings',
                showlegend=True
            ))

        # === 2. 航路网渲染 (Scatter3d 性能较好，保留) ===
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
                    vx.extend([x0, x1, None])
                    vy.extend([y0, y1, None])
                    vz.extend([z0, z1, None])
            fig.add_trace(go.Scatter3d(
                x=vx, y=vy, z=vz, mode='lines',
                line=dict(color=style['color'], width=4), name=style['name']
            ))

        # 地面基站
        port_x, port_y = [], []
        for n, d in self.G.nodes(data=True):
            if d.get('type') == 'vertical_node' and d['pos'][2] == 0:
                port_x.append(d['pos'][0])
                port_y.append(d['pos'][1])
        
        fig.add_trace(go.Scatter3d(
            x=port_x, y=port_y, z=[0]*len(port_x), mode='markers',
            marker=dict(size=4, color='black', symbol='circle-open'), name='地面起降点'
        ))

        # === 3. 布局 ===
        fig.update_layout(
            title="杭州城市低空三维立体航路网设计 (Optimized)",
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
        print("提示: 文件已通过网格合并优化，请直接在浏览器打开。")

# === 主程序入口 ===
if __name__ == "__main__":
    net = LowAltitudeRouteNetwork(bounds=None, grid_step=200)
    
    # 真实 GIS 数据路径
    shp_path = DATA_DIR / "Hangzhou_Buildings_DWG-Polygon.shp"
    
    try:
        if shp_path.exists():
            # 开启 use_small_area=True 和 center_point=None
            # 这会触发 select_area 里的密度扫描逻辑，自动找到最密集区域
            net.load_buildings_from_file(
                shp_path, 
                use_small_area=True, 
                area_size_m=5000, 
                center_point=None 
            )
        else:
            print(f"未找到 {shp_path}，使用虚拟数据。")
            net.bounds = [0, 0, 5000, 5000]
            net.create_mock_buildings(count=70)
    except Exception as e:
        print(f"处理出错: {e}")
        import traceback
        traceback.print_exc()
        net.bounds = [0, 0, 5000, 5000]
        net.create_mock_buildings(count=70)
    
    net.generate_graph()
    net.save_network("hangzhou_route_graph.pkl")
    net.visualize("Route_Network_Optimized.html")