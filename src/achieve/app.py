import streamlit as st
import pydeck as pdk
import pandas as pd
import geopandas as gpd
import networkx as nx
import pickle
import numpy as np
from pyproj import Transformer

# === 1. 页面配置 ===
st.set_page_config(layout="wide", page_title="低空航路网驾驶舱")

# 注入 CSS 缩小顶部空白
st.markdown("""
    <style>
        .block-container { padding-top: 1rem; padding-bottom: 0rem; }
        h3 { margin-top: 0; padding-top: 0; }
    </style>
""", unsafe_allow_html=True)

# === 2. 数据加载与缓存 ===
@st.cache_data
def load_data():
    """加载 Pickle 模型文件"""
    pkl_path = "models/hangzhou_route_graph.pkl" 
    
    try:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        return data
    except FileNotFoundError:
        st.error(f"未找到数据文件: {pkl_path}。请先运行 generate_network.py 生成数据。")
        return None

data_pack = load_data()

if data_pack:
    buildings_gdf = data_pack['buildings']
    G = data_pack['graph']
    bounds = data_pack['bounds'] # [minx, miny, maxx, maxy]
    
    # 获取原始 CRS
    source_crs = buildings_gdf.crs
    
    # 创建坐标转换器: UTM -> WGS84
    transformer = Transformer.from_crs(source_crs, "EPSG:4326", always_xy=True)

    # === 3. 侧边栏：控制台 ===
    st.sidebar.markdown("### 🎮 驾驶舱控制台")
    
    st.sidebar.info("""
    **👆 如何 3D 旋转地图？**
    * **旋转/倾斜**：按住 **Ctrl + 左键** 拖动 (或鼠标右键)
    * **平移**：鼠标左键拖动
    * **缩放**：鼠标滚轮
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.caption("🗺️ 视域范围 (ROI)")
    
    # 强制 float 转换
    min_x, min_y, max_x, max_y = float(bounds[0]), float(bounds[1]), float(bounds[2]), float(bounds[3])
    pad_x = (max_x - min_x) * 0.05
    pad_y = (max_y - min_y) * 0.05
    
    view_x = st.sidebar.slider("经度范围 (X)", min_x - pad_x, max_x + pad_x, (min_x, max_x))
    view_y = st.sidebar.slider("纬度范围 (Y)", min_y - pad_y, max_y + pad_y, (min_y, max_y))

    st.sidebar.markdown("---")
    st.sidebar.caption("👁️ 图层开关")
    show_buildings = st.sidebar.checkbox("🏙️ 城市建筑", value=True)
    show_routes = st.sidebar.checkbox("🛣️ 空中航路", value=True)
    
    # === 4. 数据处理：建筑物 ===
    layers = []
    
    # 初始化中心点
    center_lon = 120.15
    center_lat = 30.28
    has_valid_center = False

    if show_buildings:
        filtered_buildings = buildings_gdf.cx[view_x[0]:view_x[1], view_y[0]:view_y[1]].copy()
        
        st.sidebar.text(f"渲染建筑数: {len(filtered_buildings)}")
        if len(filtered_buildings) > 3000:
            st.sidebar.warning("⚠️ 建筑密集，建议缩小范围")

        if not filtered_buildings.empty:
            # 计算几何中心 (米)
            centroid = filtered_buildings.geometry.centroid
            avg_x = centroid.x.mean()
            avg_y = centroid.y.mean()
            clon, clat = transformer.transform(avg_x, avg_y)
            center_lon = float(clon)
            center_lat = float(clat)
            has_valid_center = True

        # 坐标转换
        if filtered_buildings.crs and not filtered_buildings.crs.is_geographic:
            filtered_buildings = filtered_buildings.to_crs(epsg=4326)

        # 提取坐标
        def get_poly_coords(geom):
            if geom.geom_type == 'Polygon':
                return [list(p) for p in geom.exterior.coords]
            return []
        
        # 清洗数据
        clean_building_data = []
        for _, row in filtered_buildings.iterrows():
            coords = get_poly_coords(row.geometry)
            if coords:
                clean_building_data.append({
                    "coordinates": coords,
                    "height_val": float(row['height_val'])
                })
        
        layer_buildings = pdk.Layer(
            "PolygonLayer",
            clean_building_data,
            get_polygon="coordinates",
            get_fill_color=[50, 60, 70, 200], # 深灰
            get_line_color=[100, 255, 218],   # 青色描边
            get_line_width=1,
            get_elevation="height_val",
            extruded=True,
            wireframe=True,
            pickable=True,
            auto_highlight=True,
            opacity=0.8
        )
        layers.append(layer_buildings)

    # === 5. 数据处理：航路网 ===
    @st.cache_data
    def process_graph_nodes(_graph):
        """缓存节点坐标转换"""
        node_positions = {}
        nodes_raw = nx.get_node_attributes(_graph, 'pos')
        for node_id, (x, y, z) in nodes_raw.items():
            lon, lat = transformer.transform(x, y)
            node_positions[node_id] = [float(lon), float(lat), float(z)]
        return node_positions

    if show_routes:
        node_pos_wgs84 = process_graph_nodes(G)
        route_data = []
        
        for u, v, d in G.edges(data=True):
            p1 = node_pos_wgs84[u]
            p2 = node_pos_wgs84[v]
            
            # 默认样式
            color = [255, 255, 255]
            width = 2
            
            edge_type = d.get('type', 'unknown')
            layer = d.get('layer', 'unknown')
            direction = d.get('direction', 'unknown')
            
            # --- 样式逻辑 ---
            if edge_type == 'horizontal':
                if layer == 'Low':
                    color = [0, 243, 255] # Cyan
                    width = 3
                elif layer == 'Mid':
                    color = [0, 255, 157] # Green
                    width = 5
                elif layer == 'High':
                    color = [255, 0, 85]  # Red
                    width = 7
            
            # 垂直通道高亮显示逻辑
            elif edge_type in ['vertical_shaft', 'access_link']:
                width = 1 # 特别加粗
                if direction == 'Up':
                    color = [255, 215, 0]   # Gold (上行)
                else:
                    color = [180, 0, 255]   # Vivid Purple (下行)
            
            route_data.append({
                "path": [p1, p2],
                "color": color,
                "width": int(width), 
                "type": str(edge_type)
            })
            
        layer_routes = pdk.Layer(
            "PathLayer",
            route_data,
            get_path="path",
            get_color="color",
            get_width="width",
            width_scale=1,
            width_min_pixels=2, # 保证最小可见度
            pickable=True,
            auto_highlight=True,
            billboard=True # 线条始终面向相机，看起来更立体
        )
        layers.append(layer_routes)
        
        # 地面起降点
        vertiports = []
        for n, d in G.nodes(data=True):
            if d.get('type') == 'vertical_node' and d['pos'][2] == 0:
                vertiports.append({
                    "position": node_pos_wgs84[n],
                    "type": "Vertiport"
                })
        
        if vertiports:
            layer_ports = pdk.Layer(
                "ScatterplotLayer",
                vertiports,
                get_position="position",
                get_fill_color=[255, 255, 255],
                get_radius=20,
                pickable=True
            )
            layers.append(layer_ports)

    # === 6. 渲染地图 ===
    st.markdown("### 🏙️ 杭州低空航路网", unsafe_allow_html=True)
    
    if not has_valid_center:
        mid_x = (view_x[0] + view_x[1]) / 2
        mid_y = (view_y[0] + view_y[1]) / 2
        clon, clat = transformer.transform(mid_x, mid_y)
        center_lon = float(clon)
        center_lat = float(clat)

    # [关键修复]：controller=True 移入 ViewState
    view_state = pdk.ViewState(
        latitude=center_lat,
        longitude=center_lon,
        zoom=14.5,
        pitch=60, # 默认倾斜，展示3D效果
        bearing=15,
        controller=True # <--- 移动到这里
    )

    r = pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        map_style="mapbox://styles/mapbox/dark-v10",
        # controller=True,  <--- 从这里删除了
        tooltip={
            "html": "<b>类型:</b> {type}<br><b>高度:</b> {height_val}m",
            "style": {"color": "white"}
        }
    )

    st.pydeck_chart(r)

else:
    st.warning("请先运行 generate_network.py 生成模型数据。")