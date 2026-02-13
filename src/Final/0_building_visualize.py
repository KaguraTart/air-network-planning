import numpy as np
import geopandas as gpd
import plotly.graph_objects as go
import pickle
from pathlib import Path
from shapely.geometry import Polygon
import random

# === 路径配置 ===
BASE_DIR = Path(".")
MODEL_DIR = BASE_DIR / "models"
RESULT_MAP_DIR = BASE_DIR / "results" / "maps"
RESULT_MAP_DIR.mkdir(parents=True, exist_ok=True)

class CityEnvironmentVisualizer:
    def __init__(self, bounds=[-2500, -2500, 2500, 2500]):
        self.bounds = bounds
        self.buildings = None

    def load_environment(self, pickle_path="hangzhou_network.pkl"): # 注意这里文件名统一
        filepath = MODEL_DIR / pickle_path
        if filepath.exists():
            print(f"正在加载现有环境数据: {filepath} ...")
            try:
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                    self.buildings = data.get('buildings')
                    if 'bounds' in data: self.bounds = data['bounds']
                print(f"成功加载 {len(self.buildings)} 个建筑物。")
                return True
            except Exception as e:
                print(f"加载失败: {e}。")
        return False

    def _create_building_mesh(self, geom, height):
        if geom.geom_type != 'Polygon': return None, None, None, None, None, None
        x, y = geom.exterior.xy
        x = list(x)[:-1]; y = list(y)[:-1]
        n = len(x)
        avg_x = sum(x)/n; avg_y = sum(y)/n
        vx = x*2 + [avg_x]*2; vy = y*2 + [avg_y]*2; vz = [0]*n + [height]*n + [height, 0]
        i, j, k = [], [], []
        for idx in range(n):
            next_idx = (idx + 1) % n
            i.extend([idx, next_idx]); j.extend([idx+n, idx+n]); k.extend([next_idx, next_idx+n])
            i.append(idx+n); j.append(next_idx+n); k.append(2*n)
        return vx, vy, vz, i, j, k

    def visualize(self, filename="City_Environment_Only.html"):
        print("正在渲染环境图...")
        fig = go.Figure()

        if self.buildings is not None:
            # 抽样渲染
            b_show = self.buildings.sample(min(len(self.buildings), 5000))
            for _, row in b_show.iterrows():
                h = row.get('height', 30)
                vx, vy, vz, i, j, k = self._create_building_mesh(row.geometry, h)
                if vx:
                    fig.add_trace(go.Mesh3d(x=vx, y=vy, z=vz, i=i, j=j, k=k,
                        color='lightgray', opacity=1.0, flatshading=True, name='Obstacle', showlegend=False))

        # 地面
        min_x, min_y, max_x, max_y = self.bounds
        fig.add_trace(go.Mesh3d(
            x=[min_x, max_x, max_x, min_x],
            y=[min_y, min_y, max_y, max_y],
            z=[0, 0, 0, 0],
            color='rgb(240, 240, 240)', opacity=0.5, name='Ground'
        ))

        fig.update_layout(
            title="杭州城市低空环境 (无路网)",
            scene=dict(
                xaxis=dict(title='X (m)', backgroundcolor="white"),
                yaxis=dict(title='Y (m)', backgroundcolor="white"),
                zaxis=dict(title='Z (m)', backgroundcolor="white"),
                aspectmode='manual', aspectratio=dict(x=1, y=1, z=0.35)
            ),
            margin=dict(t=40, b=0, l=0, r=0)
        )
        
        save_path = RESULT_MAP_DIR / filename
        fig.write_html(str(save_path))
        print(f"可视化文件已生成: {save_path}")

if __name__ == "__main__":
    viz = CityEnvironmentVisualizer()
    viz.load_environment("hangzhou_network.pkl")
    viz.visualize("City_Environment_Only.html")