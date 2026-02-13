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
from pyproj import Transformer

# === 1. 配置 ===
BASE_DIR = Path(".")
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
for p in [DATA_DIR, MODEL_DIR]: p.mkdir(parents=True, exist_ok=True)

try:
    from sklearn.neighbors import KernelDensity
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

class LowAltitudeRouteNetwork:
    def __init__(self):
        self.bounds = [0, 0, 5000, 5000]
        self.G = nx.Graph()
        self.buildings = None
        self.crs = None
        
        # 稀疏分层参数
        self.layers = {
            'Low':  {'height': 120, 'step': 200, 'width': 80,  'color': '#1f77b4'},
            'Mid':  {'height': 200, 'step': 500, 'width': 100, 'color': '#2ca02c'},
            'High': {'height': 280, 'step': 1000,'width': 120, 'color': '#d62728'}
        }
        self.vert_spacing = 1000
        self.vert_width = 20
        self.safety_margin = 10

    def load_buildings(self, shp_path, size=5000):
        print(f"读取数据: {shp_path}")
        gdf = gpd.read_file(shp_path)
        
        if gdf.crs and gdf.crs.is_geographic:
            try: gdf = gdf.to_crs(epsg=32651)
            except: gdf = gdf.to_crs(epsg=3857)
        self.crs = gdf.crs

        # 智能找 CBD
        if HAS_SKLEARN:
            print("正在扫描 CBD 核心区...")
            pts = np.array([[p.x, p.y] for p in gdf.geometry.centroid])
            sample = pts[::10] if len(pts)>5000 else pts
            kde = KernelDensity(bandwidth=500).fit(sample)
            mx, my, xx, xy = gdf.total_bounds
            x_grid, y_grid = np.meshgrid(np.linspace(mx, xx, 30), np.linspace(my, xy, 30))
            grid_pts = np.vstack([x_grid.ravel(), y_grid.ravel()]).T
            center = grid_pts[np.argmax(kde.score_samples(grid_pts))]
        else:
            center = [(gdf.total_bounds[0]+gdf.total_bounds[2])/2, (gdf.total_bounds[1]+gdf.total_bounds[3])/2]
            
        print(f"中心点: {center}")
        bbox = gpd.GeoDataFrame(geometry=[Point(center).buffer(size/2, cap_style=3)], crs=gdf.crs)
        self.buildings = gpd.sjoin(gdf, bbox, how='inner', predicate='intersects')
        
        # 高度处理
        col = next((c for c in self.buildings.columns if c in ['height', 'Height', 'Floor', 'floor']), None)
        if col:
            self.buildings['h'] = pd.to_numeric(self.buildings[col], errors='coerce').fillna(15)
            if 'floor' in col.lower(): self.buildings['h'] *= 3.0
        else:
            self.buildings['h'] = np.random.normal(40, 20, len(self.buildings)) + 10
            
        self.sindex = self.buildings.sindex
        self.bounds = list(self.buildings.total_bounds)

    def _check(self, x, y, z, w):
        q = Point(x, y).buffer(w/2 + self.safety_margin)
        hits = self.buildings.iloc[list(self.sindex.intersection(q.bounds))]
        for _, r in hits.iterrows():
            if r.geometry.intersects(q) and r['h'] + self.safety_margin > z: return True
        return False

    def generate(self):
        print("构建 8-邻域 对角线航路网...")
        nid = 0
        pos_map = {} # (x,y,layer) -> id
        minx, miny, maxx, maxy = self.bounds

        # 1. 水平层 (支持对角线)
        for lname, cfg in self.layers.items():
            print(f"  - {lname} 层 ({cfg['step']}m)")
            z, step, w = cfg['height'], cfg['step'], cfg['width']
            xs = np.arange(minx, maxx, step)
            ys = np.arange(miny, maxy, step)
            
            nodes = []
            for x in xs:
                for y in ys:
                    if not self._check(x, y, z, w):
                        self.G.add_node(nid, pos=(x, y, z), layer=lname, type='hor')
                        pos_map[(round(x), round(y), lname)] = nid
                        nodes.append((round(x), round(y), nid))
                        nid += 1
            
            # 连接逻辑：不仅连上下左右，还连对角线
            diag_step = step * 1.414 # 根号2
            
            for x, y, u in nodes:
                # 8个方向的邻居偏移量
                # (dx, dy, distance)
                neighbors_def = [
                    (step, 0, step), (0, step, step), # 正向
                    (step, step, diag_step), (step, -step, diag_step) # 对角向 (新加的!)
                ]
                
                for dx, dy, dist in neighbors_def:
                    key = (x + dx, y + dy, lname)
                    if key in pos_map:
                        v = pos_map[key]
                        # 碰撞检测 (检测中点)
                        mx, my = x + dx/2, y + dy/2
                        if not self._check(mx, my, z, w):
                            self.G.add_edge(u, v, weight=dist, layer=lname, type='hor')

        # 2. 垂直通道
        print("  - 垂直通道...")
        v_xs = np.arange(minx, maxx, self.vert_spacing)
        v_ys = np.arange(miny, maxy, self.vert_spacing)
        levels = [0, 120, 200, 280]
        lnames = ['G', 'Low', 'Mid', 'High']
        
        for vx in v_xs:
            for vy in v_ys:
                if self._check(vx, vy, 300, 20): continue
                
                # 上下两根柱子
                for direct, off in [('Up', 30), ('Down', -30)]:
                    rx, ry = vx + off, vy
                    pid = None
                    for i, h in enumerate(levels):
                        cid = nid
                        nid += 1
                        l = lnames[i]
                        self.G.add_node(cid, pos=(rx, ry, h), layer=l, type='vert', dir=direct)
                        
                        if pid is not None:
                            self.G.add_edge(pid, cid, weight=h-levels[i-1], type='shaft', dir=direct)
                        pid = cid
                        
                        # 吸附水平网格
                        if l in self.layers:
                            step = self.layers[l]['step']
                            # 找最近点
                            best, min_d = None, 9999
                            # 局部搜索优化
                            search_x = round(rx / step) * step
                            search_y = round(ry / step) * step
                            
                            # 检查周围9宫格
                            for dx in [-step, 0, step]:
                                for dy in [-step, 0, step]:
                                    k = (search_x+dx, search_y+dy, l)
                                    if k in pos_map:
                                        target = pos_map[k]
                                        p = self.G.nodes[target]['pos']
                                        d = ((p[0]-rx)**2 + (p[1]-ry)**2)**0.5
                                        if d < min_d:
                                            min_d = d
                                            best = target
                            
                            if best:
                                self.G.add_edge(cid, best, weight=min_d, type='link', dir=direct)

        with open(MODEL_DIR / "hangzhou_route_graph.pkl", 'wb') as f:
            pickle.dump({'graph': self.G, 'buildings': self.buildings, 'bounds': self.bounds, 'crs': self.crs}, f)
        print(f"完成。节点: {self.G.number_of_nodes()}, 边: {self.G.number_of_edges()}")

if __name__ == "__main__":
    net = LowAltitudeRouteNetwork()
    shp = DATA_DIR / "Hangzhou_Buildings_DWG-Polygon.shp"
    if shp.exists(): net.load_buildings(shp)
    else: 
        # Mock data logic if needed
        pass
    net.generate()