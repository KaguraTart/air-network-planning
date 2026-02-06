import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

# 假设 gdf 是您加载的 GeoDataFrame
# 选取局部区域 (例如前 50 个建筑物，或者根据坐标筛选)
# gdf_subset = gdf.cx[120.1:120.2, 30.2:30.3]  # 示例：根据经纬度切片
# 或者简单选取前 N 个：
# gdf_subset = gdf.head(50) 

def plot_3d_subset(gdf_subset, height_col='Height', save_path='partial_area_3d.png'):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 获取高度范围用于颜色映射
    heights = gdf_subset[height_col].fillna(10) # 填充缺失值为默认10米
    max_h = heights.max()
    cmap = plt.cm.get_cmap('Spectral_r')
    
    for idx, row in gdf_subset.iterrows():
        geom = row.geometry
        h = row[height_col]
        if np.isnan(h): h = 10
        
        # 颜色计算
        color = cmap(h / max_h)
        
        if geom.geom_type == 'Polygon':
            x, y = geom.exterior.coords.xy
            x = list(x)
            y = list(y)
            
            # 1. 绘制顶面 (Roof)
            roof_verts = [list(zip(x, y, [h]*len(x)))]
            ax.add_collection3d(Poly3DCollection(roof_verts, facecolors=color, alpha=0.9, edgecolors='k', linewidths=0.5))
            
            # 2. 绘制侧面 (Walls)
            for i in range(len(x)-1):
                wall_verts = [[
                    (x[i], y[i], 0),
                    (x[i+1], y[i+1], 0),
                    (x[i+1], y[i+1], h),
                    (x[i], y[i], h)
                ]]
                ax.add_collection3d(Poly3DCollection(wall_verts, facecolors=color, alpha=0.7, edgecolors='k', linewidths=0.1))

    # 自动调整坐标轴范围
    bounds = gdf_subset.total_bounds
    ax.set_xlim(bounds[0], bounds[2])
    ax.set_ylim(bounds[1], bounds[3])
    ax.set_zlim(0, max_h * 1.2)
    
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_zlabel('Height (m)')
    ax.set_title('局部区域建筑物 3D 可视化')
    
    plt.savefig(save_path, dpi=300)
    plt.show()

# 使用说明：
# 在您的脚本中加载数据后，调用此函数即可：
plot_3d_subset(gdf.iloc[:50])