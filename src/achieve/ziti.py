import platform
import matplotlib.pyplot as plt
# 根据操作系统设置字体
system = platform.system()
if system == 'Darwin':  # macOS
    # macOS系统使用系统自带的中文字体
    plt.rcParams['font.family'] = ['Arial Unicode MS', 'Heiti TC', 'STHeiti', 'SimHei']
elif system == 'Windows':
    # Windows系统使用微软雅黑或黑体
    plt.rcParams['font.family'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
else:
    # Linux或其他系统
    plt.rcParams['font.family'] = ['DejaVu Sans', 'WenQuanYi Micro Hei', 'SimHei']

print(f"当前操作系统: {system}")
print(f"使用的字体族: {plt.rcParams['font.family']}")