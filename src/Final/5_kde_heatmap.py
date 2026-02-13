import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import matplotlib.patches as patches
from pathlib import Path
from shapely.geometry import Point, box
from pyproj import Transformer, Geod

# === 配置中文显示 ===
# 尝试使用常见的中文字体，防止乱码
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

def main():
    # 1. 文件路径配置
    shp_path = Path("data/Hangzhou_Buildings_DWG-Polygon.shp")
    
    if not shp_path.exists():
        print(f"错误：找不到文件 {shp_path}，请确认文件路径是否正确。")
        return

    print(f"正在读取 GIS 数据: {shp_path} ...")
    gdf = gpd.read_file(shp_path)
    
    # 2. 坐标系处理 (确保是米制单位)
    if gdf.crs is None:
        print("警告：数据缺失坐标系，默认假定为 EPSG:4326 (WGS84)...")
        gdf.set_crs(epsg=4326, inplace=True)
    
    # 转换为适合杭州的投影坐标系 (UTM zone 51N)，单位为米
    target_crs = "EPSG:32651"
    if gdf.crs.to_string() != target_crs:
        print(f"正在转换坐标系至 {target_crs} ...")
        gdf = gdf.to_crs(target_crs)
        
    print(f"成功加载 {len(gdf)} 个建筑物要素。")

    # 3. 提取中心点用于 KDE 计算
    print("正在提取建筑物中心点...")
    # 使用质心代表建筑物位置
    centroids = gdf.geometry.centroid
    x = centroids.x.values
    y = centroids.y.values
    
    # 4. 核密度估计 (KDE)
    print("正在计算核密度 (这可能需要几秒钟)...")
    
    # 为了加快计算，如果数据量太大，可以进行降采样
    if len(x) > 10000:
        indices = np.random.choice(len(x), 10000, replace=False)
        x_sample = x[indices]
        y_sample = y[indices]
    else:
        x_sample = x
        y_sample = y
        
    # 定义网格范围 (在数据边界基础上外扩一些)
    pad = 1000 # 外扩 1km
    x_min, x_max = x.min() - pad, x.max() + pad
    y_min, y_max = y.min() - pad, y.max() + pad
    
    # 生成 100x100 的网格用于绘图
    xi, yi = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
    positions = np.vstack([xi.ravel(), yi.ravel()])
    values = np.vstack([x_sample, y_sample])
    
    kernel = gaussian_kde(values)
    # 计算网格上的密度值
    zi = kernel(positions).reshape(xi.shape)
    
    # 5. 寻找峰值 (CBD 中心)
    max_idx = np.unravel_index(np.argmax(zi), zi.shape)
    center_x = xi[max_idx]
    center_y = yi[max_idx]
    print(f"计算得出的 CBD 中心坐标: ({center_x:.1f}, {center_y:.1f})")

    # 将投影坐标转换为经纬度 (WGS84) 便于定位
    center_point = gpd.GeoSeries([Point(center_x, center_y)], crs=target_crs)
    center_lonlat = center_point.to_crs(epsg=4326).iloc[0]
    print(
        "CBD 中心经纬度 (WGS84): "
        f"({center_lonlat.y:.6f}, {center_lonlat.x:.6f})"
    )

    # 6. 绘图可视化 (转换为经纬度显示)
    print("正在生成可视化图表...")
    fig, ax = plt.subplots(figsize=(12, 10))

    # 转换网格到经纬度，用于等值线绘制
    transformer = Transformer.from_crs(target_crs, "EPSG:4326", always_xy=True)
    lon_grid, lat_grid = transformer.transform(xi, yi)

    # 底图转换为经纬度显示
    gdf_lonlat = gdf.to_crs(epsg=4326)
    
    # (A) 绘制建筑物底图 (浅灰色，体现城市肌理)
    gdf_lonlat.plot(ax=ax, color='#DDDDDD', edgecolor='none', alpha=0.8, label='建筑物 (Buildings)')
    
    # (B) 绘制 KDE 热力图 (半透明叠加)
    # levels=20 让渐变更平滑
    cf = ax.contourf(lon_grid, lat_grid, zi, levels=20, cmap='Reds', alpha=0.6)
    
    # (C) 绘制 5km x 5km 核心仿真区
    rect_w = 5000
    rect_h = 5000
    rect_x = center_x - rect_w / 2
    rect_y = center_y - rect_h / 2
    
    rect_geom = box(rect_x, rect_y, rect_x + rect_w, rect_y + rect_h)
    rect_lonlat = gpd.GeoSeries([rect_geom], crs=target_crs).to_crs(epsg=4326)
    rect_lonlat.boundary.plot(
        ax=ax,
        linewidth=3,
        edgecolor='blue',
        linestyle='--',
        label='核心仿真区域 (5km×5km)'
    )
    
    # (D) 标记 CBD 中心
    ax.plot(
        center_lonlat.x,
        center_lonlat.y,
        'b+',
        markersize=15,
        markeredgewidth=3,
        label='密度极值中心 (CBD)'
    )
    
    # (E) 添加文字标注
    label_pt = gpd.GeoSeries(
        [Point(rect_x + rect_w / 2, rect_y + rect_h + 200)],
        crs=target_crs
    ).to_crs(epsg=4326).iloc[0]
    ax.text(
        label_pt.x,
        label_pt.y,
        "核心仿真区域",
        color='blue',
        fontsize=16,
        ha='center',
        fontweight='bold'
    )

    # 设置图形属性
    ax.set_title("杭州市核心区建筑物核密度估计与仿真范围选取", fontsize=16, pad=20)
    ax.set_xlabel("经度 (°)", fontsize=16)
    ax.set_ylabel("纬度 (°)", fontsize=16)
    ax.legend(loc='upper right', fontsize=16, frameon=True, framealpha=0.9)
    # 设置坐标轴刻度字体大小
    ax.tick_params(axis='both', which='major', labelsize=14)
    
    
    # 添加颜色条
    cbar = plt.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('建筑密度 (Kernel Density)', fontsize=16)
    
    # 设置颜色条刻度字体大小
    cbar.ax.tick_params(labelsize=16)
    
    # 限制显示范围，聚焦在数据区域
    ax.set_xlim(lon_grid.min(), lon_grid.max())
    ax.set_ylim(lat_grid.min(), lat_grid.max())
    
    # 添加比例尺 (简单的文本示意)
    scale_len = 2000 # 2km
    scale_x = x_min + 1000
    scale_y = y_min + 1000
    scale_start = gpd.GeoSeries([Point(scale_x, scale_y)], crs=target_crs).to_crs(epsg=4326).iloc[0]
    geod = Geod(ellps="WGS84")
    scale_end_lon, scale_end_lat, _ = geod.fwd(scale_start.x, scale_start.y, 90, scale_len)
    ax.plot([scale_start.x, scale_end_lon], [scale_start.y, scale_end_lat], 'k-', linewidth=2)
    

    # 保存图片
    output_file = "results/img/5_kde_heatmap_real.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"可视化图表已保存至: {output_file}")
    # plt.show()

if __name__ == "__main__":
    main()