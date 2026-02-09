20260207今日更新:
1.  generate_network.py文件更新
更新了相对位置坐标米(m),重新运行generate_network.py, 生成result/maps/Route_Network.html文件和models/hangzhou_route_graph.pkl文件
可以查看result/maps/Route_Network.html文件, 查看航路网.

![Route_Network](results/img/111.jpg)

2. path_planning.py文件更新
更新了A*路径规划算法,可以根据航路网,规划出一个无人机的最优路径.
输出:
results/maps/Trajectory_Plan.html文件

3. animate_trajectory_single.py文件更新
更新了动画可视化,可以根据规划出的路径,可视化展示无人机的飞行轨迹.
输出:   
results/maps/Trajectory_Animation.html文件


一定要安装
pip install scikit-learn

1. 优先运行generate_network.py, 生成result/maps/Route_Network.html文件和models/hangzhou_route_graph.pkl文件

2. 之后运行building_visualize.py, 生成results/maps/City_Environment_Visualization.html文件


可以用google浏览器打开,加载了杭州的地图.

要运行另外一个可视化需要如下:
aap.py 在命令行使用：
    streamlit run app.py

    前提：
20260207需要安装的安装依赖:
pip install streamlit pydeck pandas geopandas networkx pyproj

在命令行里面输入streamlit run app.py, 会弹出来网页,这个比较好看,但是实际航路距离是暂时无法按照实际设计,只是用粗细表示了.





可以不适用conda了,直接使用python3.11

(不用了)
conda create -n air python=3.11

先运行，有报错之后ModuleNotFoundError: No module named 'networkx'
使用
pip install networkx这个东西的方式安装。
numpy
plotly
networkx

这个库有问题。
pip install -r requirements.txt
