import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 读取Excel文件
file_path = 'data_after.xlsx'

sheet_names = ['CVAF', 'CVAR', 'HFF', 'HDF']

# 创建一个空的DataFrame来存储所有工作表的数据
all_data = pd.DataFrame()

# 读取每个工作表的数据并合并到all_data中
for sheet_name in sheet_names:
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    df['Sheet'] = sheet_name  # 添加一个新列来表示工作表名
    all_data = all_data._append(df, ignore_index=True)
print(all_data)

# 创建一个新的图形窗口
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# 定义从蓝色到红色的颜色范围
colorscale = [
    [0, "rgb(0,0,255)"],      # 蓝色
    [0.25, "rgb(0,255,255)"], # 青色
    [0.5, "rgb(0,255,0)"],    # 绿色
    [0.75, "rgb(255,255,0)"], # 黄色
    [1, "rgb(255,0,0)"]       # 红色
]
# 为每个类型和工作表的组合绘制散点图
colors = ['r', 'g', 'b', 'y']
markers = ['o', '^', 's', 'D']
for i, (type_name, type_group) in enumerate(all_data.groupby('type')):
    for j, (sheet_name, sheet_group) in enumerate(type_group.groupby('Sheet')):
        ax.scatter(sheet_group['Qv'], sheet_group['DP'], sheet_group['RPM'], c=colors[j], marker=markers[j], label=f'{type_name} - {sheet_name}')

# 设置图形属性
ax.set_xlabel('Input1')
ax.set_ylabel('Input2')
ax.set_zlabel('Input3')
ax.legend()

# 显示图形
plt.show()