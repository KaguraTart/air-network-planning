import numpy as np
import geopandas as gpd
import networkx as nx
from shapely.geometry import Point
import pickle
from pathlib import Path
import plotly.graph_objects as go # 新增：用于画路网
import random

# === 配置 ===
BASE_DIR = Path(".")
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
RESULT_MAP_DIR = BASE_DIR / "results/maps" # 新增：保存可视化结果
for p in [DATA_DIR, MODEL_DIR, RESULT_MAP_DIR]: p.mkdir(parents=True, exist_ok=True)

class AirHighwayBuilder:
    def __init__(self):
        self.G = nx.Graph()
        self.buildings = None
        self.bounds = None
        self.map_center = None
        
        # 航路分层参数 (单位: 米)
        self.layers = {
            'Low':  {'z': 120, 'step': 200, 'color': 'blue'}, 
            'Mid':  {'z': 200, 'step': 400, 'color': 'orange'},
            'High': {'z': 280, 'step': 800, 'color': 'red'}
        }

    def find_densest_area(self, gdf):
        print("   -> 正在扫描全城寻找 CBD 核心区...")
        try:
            from sklearn.neighbors import KernelDensity
            points = np.array([[p.x, p.y] for p in gdf.geometry.centroid])
            sample = points[::10] if len(points) > 5000 else points
            kde = KernelDensity(bandwidth=500, metric='euclidean', kernel='gaussian')
            kde.fit(sample)
            
            minx, miny, maxx, maxy = gdf.total_bounds
            xx, yy = np.meshgrid(np.linspace(minx, maxx, 50), np.linspace(miny, maxy, 50))
            grid_pts = np.vstack([xx.ravel(), yy.ravel()]).T
            
            scores = kde.score_samples(grid_pts)
            best_idx = np.argmax(scores)
            cx, cy = grid_pts[best_idx]
            print(f"   -> 锁定 CBD 中心坐标: ({cx:.1f}, {cy:.1f})")
            return cx, cy
        except ImportError:
            print("   ⚠️ 警告: 未安装 scikit-learn，使用几何中心。")
            return (gdf.total_bounds[0]+gdf.total_bounds[2])/2, (gdf.total_bounds[1]+gdf.total_bounds[3])/2

    def load_city_data(self, shp_path):
        print("1. 读取城市 GIS 数据...")
        gdf = gpd.read_file(shp_path)
        
        if gdf.crs is None: gdf.set_crs(epsg=4326, inplace=True)
        if gdf.crs.to_epsg() != 32651:
            print("   🔄 转换为米制坐标系 (EPSG:32651)...")
            gdf = gdf.to_crs(epsg=32651)
            
        cx, cy = self.find_densest_area(gdf)
        
        # 裁剪
        area_size = 5000
        bbox = gpd.GeoDataFrame(geometry=[Point(cx, cy).buffer(area_size/2, cap_style=3)], crs=gdf.crs)
        self.buildings = gpd.sjoin(gdf, bbox, how='inner', predicate='intersects')
        
        # === [核心] 坐标归一化 (0,0居中) ===
        print(f"   🔄 执行坐标归一化 (原中心: {cx:.1f}, {cy:.1f})...")
        self.buildings['geometry'] = self.buildings['geometry'].translate(xoff=-cx, yoff=-cy)
        self.map_center = (0, 0)
        self.bounds = [-2500, -2500, 2500, 2500]
        
        if 'height' not in self.buildings.columns:
            self.buildings['height'] = np.abs(np.random.normal(60, 30, len(self.buildings))) + 15
            
        print(f"   -> 核心区建筑物数量: {len(self.buildings)}")

    def build_topology(self):
        print("2. 构建三维分层路网...")
        min_x, min_y, max_x, max_y = self.bounds
        node_registry = {} 
        node_id_counter = 0
        
        for layer_name, config in self.layers.items():
            z = config['z']; step = config['step']
            
            # 从 0,0 向四周扩展对齐
            start_x = - (int(2500/step) * step)
            start_y = - (int(2500/step) * step)
            
            xs = np.arange(start_x, 2500, step)
            ys = np.arange(start_y, 2500, step)
            
            for x in xs:
                for y in ys:
                    # 简单碰撞检测：如果点在建筑物内，则不生成
                    # 这里为了速度暂略，或者你可以加回来
                    self.G.add_node(node_id_counter, pos=(x, y, z), layer=layer_name)
                    node_registry[(int(x), int(y), layer_name)] = node_id_counter
                    node_id_counter += 1
            
            # 水平连接
            current_nodes = [n for n, d in self.G.nodes(data=True) if d['layer'] == layer_name]
            for u in current_nodes:
                ux, uy, uz = self.G.nodes[u]['pos']
                neighbors = [(int(ux + step), int(uy), layer_name), (int(ux), int(uy + step), layer_name)]
                for k in neighbors:
                    if k in node_registry:
                        v = node_registry[k]
                        dist = ((ux-k[0])**2 + (uy-k[1])**2)**0.5
                        self.G.add_edge(u, v, weight=dist, type='horizontal', layer=layer_name)

        print("3. 构建垂直通道...")
        high_nodes = [n for n, d in self.G.nodes(data=True) if d['layer'] == 'High']
        for h_node in high_nodes:
            hx, hy, hz = self.G.nodes[h_node]['pos']
            prev_node = h_node
            for lower_layer in ['Mid', 'Low']:
                key = (int(hx), int(hy), lower_layer)
                if key in node_registry:
                    curr_node = node_registry[key]
                    dist = abs(self.G.nodes[curr_node]['pos'][2] - self.G.nodes[prev_node]['pos'][2])
                    self.G.add_edge(curr_node, prev_node, weight=dist, type='vertical')
                    prev_node = curr_node
            
            # 地面
            ground_id = node_id_counter
            node_id_counter += 1
            self.G.add_node(ground_id, pos=(hx, hy, 0), layer='Ground')
            dist = self.G.nodes[prev_node]['pos'][2] - 0
            self.G.add_edge(prev_node, ground_id, weight=dist, type='vertical')

    def visualize_network(self):
        """
        [新增] 生成包含建筑物和路网的 3D 渲染图
        """
        print("4. 生成路网与建筑渲染图...")
        fig = go.Figure()

        # 1. 绘制建筑物
        if self.buildings is not None:
            # 抽样显示以防太卡
            b_show = self.buildings.sample(min(len(self.buildings), 5000))
            all_x, all_y, all_z, all_i, all_j, all_k = [], [], [], [], [], []
            v_offset = 0
            
            for _, r in b_show.iterrows():
                if r.geometry.geom_type != 'Polygon': continue
                x, y = r.geometry.exterior.xy
                x = list(x[:-1]); y = list(y[:-1])
                h = r.get('height', 30)
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
            
            fig.add_trace(go.Mesh3d(x=all_x, y=all_y, z=all_z, i=all_i, j=all_j, k=all_k, 
                                  color='lightgray', opacity=0.5, name='Buildings'))

        # 2. 绘制路网 (水平层)
        for layer, cfg in self.layers.items():
            # 提取该层所有边
            edge_x, edge_y, edge_z = [], [], []
            layer_nodes = [n for n, d in self.G.nodes(data=True) if d.get('layer') == layer]
            
            # 为了绘图效率，只画水平边
            subgraph = self.G.subgraph(layer_nodes)
            for u, v in subgraph.edges():
                p1 = self.G.nodes[u]['pos']
                p2 = self.G.nodes[v]['pos']
                edge_x.extend([p1[0], p2[0], None])
                edge_y.extend([p1[1], p2[1], None])
                edge_z.extend([p1[2], p2[2], None])
                
            fig.add_trace(go.Scatter3d(
                x=edge_x, y=edge_y, z=edge_z,
                mode='lines',
                line=dict(color=cfg['color'], width=2),
                name=f'{layer} Layer ({cfg["z"]}m)'
            ))

        # 3. 绘制垂直通道 (本次修改：全部绘制，增强可视化)
        vert_x, vert_y, vert_z = [], [], []
        vert_edges = [(u, v) for u, v, d in self.G.edges(data=True) if d.get('type') == 'vertical']
        
        # [修改点] 不再抽样，画出全部垂直通道，以便清晰看到上升/下降路径
        for u, v in vert_edges:
            p1 = self.G.nodes[u]['pos']; p2 = self.G.nodes[v]['pos']
            vert_x.extend([p1[0], p2[0], None])
            vert_y.extend([p1[1], p2[1], None])
            vert_z.extend([p1[2], p2[2], None])
            
        fig.add_trace(go.Scatter3d(
            x=vert_x, y=vert_y, z=vert_z,
            mode='lines',
            line=dict(color='magenta', width=3), # 使用显眼的洋红色，加粗线条
            name='上升/下降通道'
        ))

        fig.update_layout(
            title="杭州核心区三维分层航路网 (Relative Coords)",
            scene=dict(
                xaxis=dict(title='横坐标 (m)', backgroundcolor="white"),
                yaxis=dict(title='纵坐标 (m)', backgroundcolor="white"),
                zaxis=dict(title='高度 (m)', backgroundcolor="#f0f0f0"),
                aspectmode='manual', aspectratio=dict(x=1, y=1, z=0.3)
            ),
            margin=dict(t=40, b=0, l=0, r=0)
        )
        
        save_path = RESULT_MAP_DIR / "Network_Structure_Viz.html"
        fig.write_html(str(save_path))
        print(f"   📊 可视化图已保存: {save_path}")

    def save(self):
        print("5. 保存模型数据...")
        path = MODEL_DIR / "hangzhou_network.pkl"
        with open(path, 'wb') as f:
            pickle.dump({
                'graph': self.G,
                'buildings': self.buildings,
                'bounds': self.bounds,
                'center': self.map_center
            }, f)
        print(f"   ✅ 模型已保存: {path}")

if __name__ == "__main__":
    builder = AirHighwayBuilder()
    shp = DATA_DIR / "Hangzhou_Buildings_DWG-Polygon.shp"
    if shp.exists():
        builder.load_city_data(shp)
        builder.build_topology()
        builder.visualize_network() # 生成您要的图
        builder.save()
    else:
        print(f"❌ 错误: 找不到文件 {shp}")