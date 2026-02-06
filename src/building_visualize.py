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

# 确保输出目录存在
RESULT_MAP_DIR.mkdir(parents=True, exist_ok=True)

class CityEnvironmentVisualizer:
    def __init__(self, bounds=[0, 0, 5000, 5000]):
        self.bounds = bounds
        self.buildings = None

    def load_environment(self, pickle_path="hangzhou_route_graph.pkl"):
        """尝试从保存的模型文件中加载建筑物数据，保证上下文一致"""
        filepath = MODEL_DIR / pickle_path
        if filepath.exists():
            print(f"正在加载现有环境数据: {filepath} ...")
            try:
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                    self.buildings = data.get('buildings')
                    if 'bounds' in data:
                        self.bounds = data['bounds']
                print(f"成功加载 {len(self.buildings)} 个建筑物。")
                return True
            except Exception as e:
                print(f"加载失败: {e}，将使用随机生成。")
        else:
            print(f"未找到 {filepath}，将生成新的随机环境。")
        return False

    def create_mock_buildings(self, count=80):
        """如果没有存档，生成随机建筑物"""
        print(f"生成 {count} 个虚拟建筑物...")
        buildings = []
        min_x, min_y, max_x, max_y = self.bounds
        for _ in range(count):
            cx = random.uniform(min_x + 200, max_x - 200)
            cy = random.uniform(min_y + 200, max_y - 200)
            w = random.uniform(80, 250)
            h_geo = random.uniform(80, 250)
            
            # 高度分布
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

    def _create_building_mesh(self, geom, height):
        """构建实心建筑物的 Mesh 数据"""
        if geom.geom_type != 'Polygon': return None, None, None, None, None, None
        
        x, y = geom.exterior.xy
        x = list(x)[:-1]
        y = list(y)[:-1]
        n = len(x)
        
        avg_x = sum(x) / n
        avg_y = sum(y) / n
        
        vx = x * 2 + [avg_x, avg_x]
        vy = y * 2 + [avg_y, avg_y]
        vz = [0]*n + [height]*n + [height, 0]
        
        i, j, k = [], [], []
        for idx in range(n):
            next_idx = (idx + 1) % n
            # Walls
            i.append(idx); j.append(idx+n); k.append(next_idx)
            i.append(next_idx); j.append(idx+n); k.append(next_idx+n)
            # Roof
            i.append(idx+n); j.append(next_idx+n); k.append(2*n)
            
        return vx, vy, vz, i, j, k

    def visualize(self, filename="City_Environment_Only.html"):
        """只渲染环境（静态障碍场）"""
        print("正在渲染城市环境图...")
        fig = go.Figure()

        # 1. 绘制建筑物
        if self.buildings is not None:
            print("  - 渲染建筑物实体...")
            for _, row in self.buildings.iterrows():
                vx, vy, vz, i, j, k = self._create_building_mesh(row.geometry, row['height_val'])
                if vx:
                    fig.add_trace(go.Mesh3d(
                        x=vx, y=vy, z=vz, i=i, j=j, k=k,
                        color='lightgray', 
                        opacity=1.0, 
                        flatshading=True, # 让棱角分明
                        lighting=dict(
                            ambient=0.5, 
                            diffuse=0.8, 
                            roughness=0.1,
                            specular=0.1
                        ),
                        name='Obstacle',
                        hoverinfo='z', # 只显示高度
                        showlegend=False
                    ))
            
            # 增加一个图例项
            fig.add_trace(go.Scatter3d(
                x=[None], y=[None], z=[None], 
                mode='markers',
                marker=dict(size=15, color='lightgray', symbol='square'),
                name='城市建筑物 (Static Obstacles)'
            ))

        # 2. 绘制地面 (Ground Plane) - 增加视觉参考
        min_x, min_y, max_x, max_y = self.bounds
        fig.add_trace(go.Mesh3d(
            x=[min_x, max_x, max_x, min_x],
            y=[min_y, min_y, max_y, max_y],
            z=[0, 0, 0, 0],
            color='rgb(240, 240, 240)', # 极浅的灰色地面
            opacity=0.5,
            name='Ground',
            showlegend=False
        ))

        # 3. 布局设置
        fig.update_layout(
            title="杭州城市低空环境可视化",
            scene=dict(
                xaxis_title='经度/X (m)', 
                yaxis_title='纬度/Y (m)', 
                zaxis_title='高度/Z (m)',
                # 背景更加干净，适合论文截图
                xaxis=dict(backgroundcolor="white", gridcolor="lightgrey", showbackground=True),
                yaxis=dict(backgroundcolor="white", gridcolor="lightgrey", showbackground=True),
                zaxis=dict(backgroundcolor="white", gridcolor="lightgrey", showbackground=True),
                # 视觉比例
                aspectmode='manual', 
                aspectratio=dict(x=1, y=1, z=0.35),
                camera=dict(
                    eye=dict(x=1.2, y=-1.2, z=0.5), # 稍微低一点的视角，突出建筑高度
                    up=dict(x=0, y=0, z=1)
                )
            ),
            margin=dict(t=50, b=0, l=0, r=0)
        )
        
        save_path = RESULT_MAP_DIR / filename
        fig.write_html(str(save_path))
        print(f"可视化文件已生成: {save_path}")

if __name__ == "__main__":
    # 初始化
    viz = CityEnvironmentVisualizer()
    
    # 优先加载之前生成的模型 (保证一致性)
    loaded = viz.load_environment("hangzhou_route_graph.pkl")
    
    # 如果没找到文件，才生成新的随机数据
    if not loaded:
        viz.create_mock_buildings(count=80)
    
    # 执行可视化
    viz.visualize("City_Environment_Visualization.html")