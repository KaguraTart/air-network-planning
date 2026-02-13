import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
from shapely.geometry import Point, Polygon
import pickle
import random
import os
from pathlib import Path
from pyproj import Transformer

# === 配置 ===
BASE_DIR = Path(".")
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
for p in [DATA_DIR, MODEL_DIR]: p.mkdir(parents=True, exist_ok=True)

class CBDRouteNetwork:
    def __init__(self):
        self.G = nx.Graph()
        self.buildings = None
        self.bounds = None
        self.crs = None
        self.map_center = None # (x, y) 绝对坐标
        
        # 稀疏路网配置
        self.layers = {
            'Low':  {'height': 120, 'step': 200, 'w': 80},
            'Mid':  {'height': 200, 'step': 500, 'w': 100},
            'High': {'height': 280, 'step': 1000,'w': 120}
        }
        self.vert_dist = 1000

    def find_cbd_center(self, gdf):
        """核心修复：强制使用核密度估计寻找最密集点"""
        print("正在进行全域核密度扫描 (KDE) 以锁定 CBD...")
        
        try:
            from sklearn.neighbors import KernelDensity
            # 提取所有建筑中心
            points = np.array([[p.x, p.y] for p in gdf.geometry.centroid])
            
            # 降采样防卡死
            sample = points[::10] if len(points) > 10000 else points
            
            # 带宽设为 500m，寻找大范围的密集区
            kde = KernelDensity(bandwidth=500, metric='euclidean', kernel='gaussian')
            kde.fit(sample)
            
            # 创建扫描网格
            minx, miny, maxx, maxy = gdf.total_bounds
            xx, yy = np.meshgrid(np.linspace(minx, maxx, 50), np.linspace(miny, maxy, 50))
            grid_pts = np.vstack([xx.ravel(), yy.ravel()]).T
            
            # 评分
            z = kde.score_samples(grid_pts)
            best_idx = np.argmax(z)
            cx, cy = grid_pts[best_idx]
            
            print(f"✅ 成功锁定 CBD 中心坐标 (UTM): ({cx:.1f}, {cy:.1f})")
            return cx, cy
            
        except ImportError:
            print("❌ 严重警告：未安装 scikit-learn，无法自动寻找 CBD！将使用几何中心代替。")
            print("请运行: pip install scikit-learn")
            minx, miny, maxx, maxy = gdf.total_bounds
            return (minx+maxx)/2, (miny+maxy)/2

    def build(self, shp_path):
        print(f"读取数据: {shp_path}")
        gdf = gpd.read_file(shp_path)
        
        # 1. 坐标系标准化
        if gdf.crs.is_geographic:
            gdf = gdf.to_crs(epsg=32651)
        self.crs = gdf.crs
        
        # 2. 锁定 CBD 并裁剪 5km
        cx, cy = self.find_cbd_center(gdf)
        self.map_center = (cx, cy)
        
        box = gpd.GeoDataFrame(geometry=[Point(cx, cy).buffer(2500, cap_style=3)], crs=gdf.crs)
        self.buildings = gpd.sjoin(gdf, box, how='inner', predicate='intersects')
        self.bounds = list(self.buildings.total_bounds)
        
        # 3. 补全高度
        if 'height_val' not in self.buildings.columns:
            # 尝试从常见字段恢复
            col = next((c for c in self.buildings.columns if c.lower() in ['height', 'floor']), None)
            if col:
                self.buildings['height_val'] = pd.to_numeric(self.buildings[col], errors='coerce').fillna(20)
                if 'floor' in col.lower(): self.buildings['height_val'] *= 3
            else:
                self.buildings['height_val'] = np.abs(np.random.normal(60, 30, len(self.buildings))) + 20

        self.sindex = self.buildings.sindex
        self._generate_topology()

    def _check(self, x, y, z, w):
        # 碰撞检测
        q = Point(x, y).buffer(w/2 + 10)
        hits = list(self.sindex.intersection(q.bounds))
        sub = self.buildings.iloc[hits]
        for _, r in sub.iterrows():
            if r.geometry.intersects(q) and r['height_val'] + 10 > z: return True
        return False

    def _generate_topology(self):
        print("构建稀疏路网图...")
        nid = 0
        pos_map = {} # (x_int, y_int, layer) -> node_id
        minx, miny, maxx, maxy = self.bounds
        
        # 1. 水平层 (支持8邻域连接点，但边在后面连)
        nodes_by_layer = {l: [] for l in self.layers}
        
        for lname, cfg in self.layers.items():
            z, step, w = cfg['height'], cfg['step'], cfg['w']
            xs = np.arange(minx, maxx, step)
            ys = np.arange(miny, maxy, step)
            
            for x in xs:
                for y in ys:
                    if not self._check(x, y, z, w):
                        self.G.add_node(nid, pos=(x, y, z), layer=lname)
                        pos_map[(int(x), int(y), lname)] = nid
                        nodes_by_layer[lname].append((int(x), int(y), nid))
                        nid += 1
            
            # 连接水平边 (8邻域：直线+对角线)
            diag_step = step * 1.414
            for x, y, u in nodes_by_layer[lname]:
                # 8个方向: (dx, dy, cost)
                dirs = [
                    (step, 0, step), (0, step, step), # 直线
                    (step, step, diag_step), (step, -step, diag_step) # 对角
                ]
                for dx, dy, cost in dirs:
                    k = (x+dx, y+dy, lname)
                    if k in pos_map:
                        v = pos_map[k]
                        # 检查连线中点是否碰撞
                        mx, my = x + dx/2, y + dy/2
                        if not self._check(mx, my, z, w):
                            self.G.add_edge(u, v, weight=cost, type='hor', layer=lname)

        # 2. 垂直通道
        vx_range = np.arange(minx, maxx, self.vert_dist)
        vy_range = np.arange(miny, maxy, self.vert_dist)
        levels = [0, 120, 200, 280]
        lnames = ['G', 'Low', 'Mid', 'High']
        
        for vx in vx_range:
            for vy in vy_range:
                if self._check(vx, vy, 300, 20): continue
                
                # 双向通道
                for d_tag, offset in [('Up', 30), ('Down', -30)]:
                    rx, ry = vx+offset, vy
                    prev = None
                    for i, h in enumerate(levels):
                        cid = nid
                        nid += 1
                        l = lnames[i]
                        self.G.add_node(cid, pos=(rx, ry, h), layer=l, type='vert')
                        
                        if prev is not None:
                            self.G.add_edge(prev, cid, weight=h-levels[i-1], type='shaft')
                        prev = cid
                        
                        # 吸附
                        if l in self.layers:
                            step = self.layers[l]['step']
                            # 找最近
                            best, min_d = None, 9999
                            # 局部搜
                            sx = round(rx/step)*step
                            sy = round(ry/step)*step
                            for dx in [-step, 0, step]:
                                for dy in [-step, 0, step]:
                                    k = (sx+dx, sy+dy, l)
                                    if k in pos_map:
                                        target = pos_map[k]
                                        tp = self.G.nodes[target]['pos']
                                        d = ((tp[0]-rx)**2 + (tp[1]-ry)**2)**0.5
                                        if d < min_d:
                                            min_d = d; best = target
                            if best:
                                self.G.add_edge(cid, best, weight=min_d, type='link')

        print(f"路网生成完毕: {self.G.number_of_nodes()} 节点")
        with open(MODEL_DIR / "hangzhou_network.pkl", 'wb') as f:
            pickle.dump({
                'G': self.G, 
                'buildings': self.buildings, 
                'center': self.map_center,
                'bounds': self.bounds,
                'crs': self.crs
            }, f)

if __name__ == "__main__":
    net = CBDRouteNetwork()
    shp = DATA_DIR / "Hangzhou_Buildings_DWG-Polygon.shp"
    if shp.exists():
        net.build(shp)
    else:
        print("未找到 Shapefile")