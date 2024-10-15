import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import webbrowser
import os

# 设置文件路径
file_path = 'data_after.xlsx'

# 定义sheet名称和数据列名称
sheet_names = ['CVAF', 'CVAR', 'HFF', 'HDF']
data_names = ['Qv', 'DP', 'RPM', 'M1']

# 读取数据
mode_data = {}
for sheet_name in sheet_names:
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    mode_data[sheet_name] = {
        data_name: df[data_name].tolist() for data_name in data_names
    }
print(mode_data)
# 创建子图
fig = make_subplots(
    rows=2, cols=2,
    specs=[[{'type': 'scene'}, {'type': 'scene'}],
           [{'type': 'scene'}, {'type': 'scene'}]],
    subplot_titles=sheet_names,  # 使用sheet名称作为子图标题
    horizontal_spacing=0.05,  # 减小水平间距
    vertical_spacing=0.05     # 减小垂直间距
)

# 定义从蓝色到红色的颜色范围
colorscale = [
    [0, "rgb(0,0,255)"],      # 蓝色
    [0.25, "rgb(0,255,255)"], # 青色
    [0.5, "rgb(0,255,0)"],    # 绿色
    [0.75, "rgb(255,255,0)"], # 黄色
    [1, "rgb(255,0,0)"]       # 红色
]

# 为每个模式创建3D散点图
for i, sheet_name in enumerate(sheet_names):
    row = i // 2 + 1
    col = i % 2 + 1
    
    data = mode_data[sheet_name]
    
    scatter = go.Scatter3d(
        x=data['Qv'],
        y=data['DP'],
        z=data['RPM'],
        mode='markers',
        marker=dict(
            size=5,
            color=data['M1'],
            colorscale=colorscale,
            colorbar=dict(title='M1', len=0.6),  # 调整颜色条长度
            showscale=True
        ),
        text=[f'M1: {m1}' for m1 in data['M1']],
        hoverinfo='text',
        name=sheet_name  # 设置trace的名称为sheet名
    )
    
    fig.add_trace(scatter, row=row, col=col)
    
    fig.update_scenes(
        xaxis_title='Qv',
        yaxis_title='DP',
        zaxis_title='RPM',
        aspectmode='cube',  # 保持坐标轴比例一致
        row=row, col=col
    )

# 更新布局
fig.update_layout(
    title='4D Visualization of Qv, DP, RPM, and M1',
    height=1200,  # 增加高度
    width=1600,   # 增加宽度
    margin=dict(l=0, r=0, t=50, b=0)  # 减小边距
)

# 调整子图标题的位置和样式
for i in fig['layout']['annotations']:
    i['font'] = dict(size=16, color='black', family="Arial, sans-serif")
    i['y'] = i['y'] - 0.05  # 稍微降低标题位置

# 保存为HTML文件
output_path = 'data preprocessing.html'
fig.write_html(output_path)

# 自动打开生成的HTML文件
webbrowser.open('file://' + os.path.realpath(output_path))