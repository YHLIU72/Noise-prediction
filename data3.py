import pandas as pd  
import plotly.express as px  
  
# 读取Excel文件  
excel_file = 'data_after.xlsx'  # 替换为你的Excel文件名  
sheets = ['CVAF', 'CVAR', 'HFF', 'HDF']  # 替换为你的工作表名  
  
# 初始化一个空的DataFrame来存储所有数据  
all_data = pd.DataFrame()  
  
# 遍历每个工作表并读取数据  
for sheet in sheets:  
    df = pd.read_excel(excel_file, sheet_name=sheet)  
    df['method'] = sheet  # 添加一个列来表示工作表名  
    all_data = pd.concat([all_data, df], ignore_index=True)  
print(all_data)
# 准备数据用于plotly  
fig = px.scatter_3d(all_data,  
                    x='Qv',   
                    y='DP',   
                    z='RPM',   
                    color='type', 
                    size='M1',  # size参数控制点的大小  
                    symbol='method',  # 使用工作表名作为符号（点形状）的分类  
                    hover_name='type',  # 鼠标悬停时显示的名称  
                    category_orders={'method': sheets},  # 设置工作表的显示顺序  
                    labels={  
                        'Qv': 'Qv',  # 标签  
                        'DP': 'DP',  # 标签  
                        'RPM': 'RPM',  # 标签  
                        'M1': 'M1',  # 标签  
                        'type': 'type',  
                    },  
                    title='3D Scatter Plot by Type and Sheet'  
                   )  
  
# 显示图表  
fig.show()