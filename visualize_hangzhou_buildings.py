import geopandas as gpd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import platform
import pandas as pd
from sklearn.neighbors import KernelDensity
# 根据操作系统设置字体
system = platform.system()
if system == 'Darwin':  # macOS
    # macOS系统使用系统自带的中文字体
    plt.rcParams['font.family'] = ['Arial Unicode MS', 'Heiti TC', 'STHeiti']
elif system == 'Windows':
    # Windows系统使用微软雅黑或黑体
    plt.rcParams['font.family'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
else:
    # Linux或其他系统
    plt.rcParams['font.family'] = ['DejaVu Sans', 'WenQuanYi Micro Hei', 'SimHei']

print(f"当前操作系统: {system}")
print(f"使用的字体族: {plt.rcParams['font.family']}")

def check_and_transform_coordinates(gdf):
    """检查并转换坐标系统，确保使用地理坐标系"""
    print("\n=== 坐标系统检查 ===")
    print(f"原始坐标系统: {gdf.crs}")
    
    # 获取坐标范围
    bounds = gdf.total_bounds
    print(f"原始坐标范围:")
    print(f"  最小X: {bounds[0]:.6f}")
    print(f"  最小Y: {bounds[1]:.6f}")
    print(f"  最大X: {bounds[2]:.6f}")
    print(f"  最大Y: {bounds[3]:.6f}")
    
    # 判断是否为投影坐标（值很大）
    if abs(bounds[0]) > 180 or abs(bounds[2]) > 180 or abs(bounds[1]) > 90 or abs(bounds[3]) > 90:
        print("检测到投影坐标系统，正在转换为地理坐标系...")
        if gdf.crs and gdf.crs.is_projected:
            gdf_geo = gdf.to_crs('EPSG:4326')  # WGS84地理坐标系
            bounds_geo = gdf_geo.total_bounds
            print(f"转换后的地理坐标范围:")
            print(f"  最小经度: {bounds_geo[0]:.6f}°")
            print(f"  最小纬度: {bounds_geo[1]:.6f}°")
            print(f"  最大经度: {bounds_geo[2]:.6f}°")
            print(f"  最大纬度: {bounds_geo[3]:.6f}°")
            return gdf_geo
        else:
            print("无法确定坐标系统，尝试使用默认转换")
            gdf_geo = gdf.set_crs('EPSG:3857')  # 假设是Web墨卡托投影
            gdf_geo = gdf_geo.to_crs('EPSG:4326')
            return gdf_geo
    else:
        print("已使用地理坐标系")
        return gdf

def calculate_building_area(gdf):
    """计算建筑物实际面积（平方米）"""
    # 如果是地理坐标系，先转换为等面积投影再计算面积
    if gdf.crs and gdf.crs.is_geographic:
        # 使用适合中国区域的等面积投影
        # 方法1: 使用Albers等面积投影，适合中国区域
        try:
            # 创建适合杭州区域的Albers等面积投影
            albers_crs = 'PROJCS["Asia_North_Albers_Equal_Area_Conic",GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,298.257223563]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],PROJECTION["Albers"],PARAMETER["False_Easting",0.0],PARAMETER["False_Northing",0.0],PARAMETER["Central_Meridian",120.0],PARAMETER["Standard_Parallel_1",25.0],PARAMETER["Standard_Parallel_2",47.0],PARAMETER["Latitude_Of_Origin",0.0],UNIT["Meter",1.0]]'
            gdf_area = gdf.to_crs(albers_crs)
            gdf['area_m2'] = gdf_area.geometry.area
            print("使用Albers等面积投影计算面积")
        except:
            # 方法2: 使用UTM投影
            try:
                # 根据经度范围确定UTM分区
                bounds = gdf.total_bounds
                center_lon = (bounds[0] + bounds[2]) / 2
                utm_zone = int((center_lon + 180) // 6) + 1
                utm_crs = f'EPSG:326{utm_zone}'  # 北半球的UTM分区
                gdf_area = gdf.to_crs(utm_crs)
                gdf['area_m2'] = gdf_area.geometry.area
                print(f"使用UTM分区 {utm_zone} 投影计算面积")
            except:
                # 方法3: 使用Web墨卡托投影
                gdf_area = gdf.to_crs('EPSG:3857')
                gdf['area_m2'] = gdf_area.geometry.area
                print("使用Web墨卡托投影计算面积")
    else:
        # 如果已经是投影坐标系，直接计算面积
        gdf['area_m2'] = gdf.geometry.area
        print("使用当前投影坐标系计算面积")
    
    # 检查面积计算结果是否合理
    min_area = gdf['area_m2'].min()
    max_area = gdf['area_m2'].max()
    avg_area = gdf['area_m2'].mean()
    
    # 如果面积值看起来不合理（过小或过大），可能需要调整投影
    if avg_area < 1:  # 平均面积小于1平方米，可能单位错误
        print("警告：计算出的面积值过小，可能单位不是平方米")
        gdf['area_m2'] = gdf['area_m2'] * 1000000  # 尝试转换为平方米
        print("已将面积值乘以1,000,000以转换为平方米")
    
    print(f"\n=== 建筑物面积统计 ===")
    print(f"  最小面积: {gdf['area_m2'].min():.2f} m²")
    print(f"  最大面积: {gdf['area_m2'].max():.2f} m²")
    print(f"  平均面积: {gdf['area_m2'].mean():.2f} m²")
    print(f"  中位数面积: {gdf['area_m2'].median():.2f} m²")
    
    # 显示面积分布
    print(f"\n=== 面积分布 ===")
    print(f"  小于50m²: {len(gdf[gdf['area_m2'] < 50])} 个建筑物")
    print(f"  50-200m²: {len(gdf[(gdf['area_m2'] >= 50) & (gdf['area_m2'] < 200)])} 个建筑物")
    print(f"  200-1000m²: {len(gdf[(gdf['area_m2'] >= 200) & (gdf['area_m2'] < 1000)])} 个建筑物")
    print(f"  大于1000m²: {len(gdf[gdf['area_m2'] >= 1000])} 个建筑物")
    
    return gdf

def select_small_area(gdf, area_size_m=500, center_point=None):
    """选择指定大小的区域进行可视化
    
    参数:
    - gdf: 建筑物GeoDataFrame
    - area_size_m: 区域大小（米），默认500m x 500m
    - center_point: 中心点坐标 [经度, 纬度]，如果为None则选择建筑物最密集的区域
    
    返回:
    - 选取的子区域GeoDataFrame
    """
    print(f"\n=== 选取 {area_size_m}m x {area_size_m}m 的区域 ===")
    
    # 确保使用投影坐标系以便计算距离
    if gdf.crs and gdf.crs.is_geographic:
        # 转换为适合杭州区域的UTM投影
        try:
            bounds = gdf.total_bounds
            center_lon = (bounds[0] + bounds[2]) / 2
            utm_zone = int((center_lon + 180) // 6) + 1
            utm_crs = f'EPSG:326{utm_zone}'
            gdf_proj = gdf.to_crs(utm_crs)
            print(f"使用UTM分区 {utm_zone} 投影进行区域选择")
        except:
            # 备选方案：使用Web墨卡托投影
            gdf_proj = gdf.to_crs('EPSG:3857')
            print("使用Web墨卡托投影进行区域选择")
    else:
        gdf_proj = gdf.copy()
        print("使用当前投影坐标系进行区域选择")
    
    # 计算每个建筑物的中心点
    gdf_proj['centroid'] = gdf_proj.geometry.centroid
    
    # 如果没有指定中心点，选择建筑物最密集的区域
    if center_point is None:
        # 使用核密度估计找到建筑物最密集的点
        from sklearn.neighbors import KernelDensity
        
        # 获取所有建筑物的中心点坐标
        points = np.array([[p.x, p.y] for p in gdf_proj['centroid']])
        
        # 计算核密度
        kde = KernelDensity(bandwidth=100).fit(points)  # 带宽设为100米
        
        # 在网格上评估密度
        xmin, ymin, xmax, ymax = gdf_proj.total_bounds
        xx, yy = np.meshgrid(
            np.linspace(xmin, xmax, 100),
            np.linspace(ymin, ymax, 100)
        )
        grid_points = np.vstack([xx.ravel(), yy.ravel()]).T
        log_density = kde.score_samples(grid_points)
        
        # 找到密度最高的点
        max_idx = np.argmax(log_density)
        center_x, center_y = grid_points[max_idx]
        
        print(f"自动选择建筑物最密集的区域作为中心点")
    else:
        # 使用指定的中心点
        if gdf.crs and gdf.crs.is_geographic:
            # 将地理坐标转换为投影坐标
            center_point_transformed = gpd.GeoSeries(
                [gpd.points_from_xy([center_point[0]], [center_point[1]])[0]], 
                crs='EPSG:4326'
            ).to_crs(gdf_proj.crs)[0]
            center_x, center_y = center_point_transformed.x, center_point_transformed.y
        else:
            center_x, center_y = center_point
        
        print(f"使用指定的中心点: ({center_point[0]:.6f}°, {center_point[1]:.6f}°)")
    
    # 定义选择区域（正方形）
    half_size = area_size_m / 2
    bbox = gpd.GeoDataFrame(
        geometry=[gpd.points_from_xy([center_x], [center_y])[0].buffer(half_size, cap_style=3)],
        crs=gdf_proj.crs
    )
    
    # 选择区域内的建筑物
    selected_buildings = gpd.sjoin(gdf_proj, bbox, how='inner', predicate='within')
    
    # 转换回原始坐标系
    if gdf.crs and gdf.crs.is_geographic:
        selected_buildings = selected_buildings.to_crs(gdf.crs)
    
    print(f"在 {area_size_m}m x {area_size_m} 的区域内找到 {len(selected_buildings)} 个建筑物")
    
    # 显示区域的地理范围
    if gdf.crs and gdf.crs.is_geographic:
        bounds = selected_buildings.total_bounds
        print(f"区域地理范围:")
        print(f"  经度: {bounds[0]:.6f}° - {bounds[2]:.6f}°")
        print(f"  纬度: {bounds[1]:.6f}° - {bounds[3]:.6f}°")
    
    return selected_buildings

def visualize_buildings_3d(shapefile_path, save_path='hangzhou_3d_vis.png', max_buildings=20, use_small_area=True, area_size_m=500, center_point=None):
    # 1. 读取数据
    print(f"正在读取数据: {shapefile_path} ...")
    try:
        gdf = gpd.read_file(shapefile_path)
        print(f"成功读取 {len(gdf)} 个建筑物")
        print(f"数据字段: {list(gdf.columns)}")
    except Exception as e:
        print(f"读取失败，请检查文件是否完整 (.shp, .shx, .dbf): {e}")
        return

    # 2. 检查并转换坐标系统
    gdf = check_and_transform_coordinates(gdf)
    
    # 3. 计算建筑物面积
    gdf = calculate_building_area(gdf)

    # 4. 自动识别高度字段
    height_col = None
    possible_cols = ['Elevation', 'ELEVATION', 'Height', 'HEIGHT', 'Floor', 'FLOOR', '层数', '楼层']
    for col in possible_cols:
        if col in gdf.columns:
            height_col = col
            break
    
    if height_col:
        print(f"\n使用高度字段: {height_col}")
        # 确保高度是数值类型，并处理缺失值
        gdf[height_col] = pd.to_numeric(gdf[height_col], errors='coerce').fillna(0)
        
        # 显示高度统计
        heights = gdf[height_col]
        print(f"  最小高度: {heights.min():.2f} m")
        print(f"  最大高度: {heights.max():.2f} m")
        print(f"  平均高度: {heights.mean():.2f} m")
    else:
        print("\n未找到高度字段，将使用随机高度进行演示")
        gdf['mock_height'] = np.random.randint(20, 100, size=len(gdf))
        height_col = 'mock_height'

    # 5. 选择区域
    if use_small_area:
        # 选择指定大小的区域
        subset = select_small_area(gdf, area_size_m, center_point)
        
        # 如果区域内建筑物仍然太多，进一步限制数量
        if len(subset) > max_buildings:
            print(f"区域内建筑物数量({len(subset)})超过最大限制({max_buildings})，随机选择")
            subset = subset.sample(n=max_buildings, random_state=42)
    else:
        # 不使用小区域，随机选择建筑物
        if len(gdf) > max_buildings:
            print(f"\n数据包含 {len(gdf)} 个建筑物，随机选择 {max_buildings} 个进行可视化")
            subset = gdf.sample(n=max_buildings, random_state=42)
        else:
            subset = gdf
    
    # 6. 设置 3D 绘图
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # 设置颜色映射 (越高越红)
    max_h = subset[height_col].max()
    min_h = subset[height_col].min()
    cmap = plt.cm.Spectral_r
    
    print("\n开始绘制 3D 模型...")
    for idx, row in subset.iterrows():
        geom = row.geometry
        h = row[height_col]
        area = row['area_m2']
        
        # 跳过空几何体
        if not geom: continue
            
        # 根据高度计算颜色
        if max_h > min_h:
            norm_h = (h - min_h) / (max_h - min_h)
        else:
            norm_h = 0.5
        color = cmap(norm_h)
        
        # 处理 Polygon 类型
        if geom.geom_type == 'Polygon':
            polys = [geom]
        elif geom.geom_type == 'MultiPolygon':
            polys = geom.geoms
        else:
            continue
            
        for poly in polys:
            # 获取多边形坐标
            x, y = poly.exterior.coords.xy
            x = list(x)
            y = list(y)
            
            # 绘制底面
            base_verts = [list(zip(x, y, [0]*len(x)))]
            ax.add_collection3d(Poly3DCollection(base_verts, facecolors='gray', alpha=0.5, edgecolors='k', linewidths=0.2))
            
            # 绘制顶面 (Roof)
            roof_verts = [list(zip(x, y, [h]*len(x)))]
            ax.add_collection3d(Poly3DCollection(roof_verts, facecolors=color, alpha=0.9, edgecolors='k', linewidths=0.5))
            
            # 绘制侧面 (Walls)
            for i in range(len(x)-1):
                wall_verts = [[
                    (x[i], y[i], 0),
                    (x[i+1], y[i+1], 0),
                    (x[i+1], y[i+1], h),
                    (x[i], y[i], h)
                ]]
                ax.add_collection3d(Poly3DCollection(wall_verts, facecolors=color, alpha=0.7, edgecolors='k', linewidths=0.1))

    # 7. 设置坐标轴和视角
    bounds = subset.total_bounds
    ax.set_xlim(bounds[0], bounds[2])
    ax.set_ylim(bounds[1], bounds[3])
    ax.set_zlim(0, max_h * 1.2)
    
    # 添加单位
    ax.set_xlabel('经度 (°)', fontsize=12)
    ax.set_ylabel('纬度 (°)', fontsize=12)
    ax.set_zlabel('高度 (m)', fontsize=12)
    ax.set_title(f'杭州建筑物 3D 可视化\n(n={len(subset)}, 高度范围: {min_h:.1f}-{max_h:.1f}m)', fontsize=14)
    
    # 调整视角
    ax.view_init(elev=30, azim=-45)
    
    # 添加颜色条
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min_h, vmax=max_h))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.1, shrink=0.8)
    cbar.set_label('建筑物高度 (m)', fontsize=12)
    
    # 保存结果
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n图片已保存至: {save_path}")
    # plt.show()

# 运行函数 (请修改路径为您本地的实际路径)
if __name__ == "__main__":
    visualize_buildings_3d(
    "杭州/Hangzhou_Buildings_DWG-Polygon.shp",
    use_small_area=True,
    area_size_m=500,
    max_buildings=200
    )