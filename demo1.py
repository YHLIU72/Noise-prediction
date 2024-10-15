# import pandas as pd
 
# # 读取Excel文件
# xls = pd.ExcelFile('data_after.xlsx',)
 
# # 存储工作表数据的字典
# sheet_data = {}
 
# # 遍历所有工作表
# for sheet_name in xls.sheet_names:
#     # 读取工作表数据
#     df = xls.parse(sheet_name)
    
#     # 对数据进行整理，例如，选择特定列，过滤数据等
#     # 假设我们只需要整理某一列数据
#     processed_column = df['ColumnName'].unique()  # 获取唯一值列表
    
#     # 将整理后的数据存储在字典中
#     sheet_data[sheet_name] = processed_column
 
# # 关闭Excel文件
# xls.close()
 
# # 现在sheet_data字典包含了所有工作表的整理后数据
# print(sheet_data)

# import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
# # 假设输入数据如下
# input1 = [1, 2, 3, 4, 5]
# input2 = [2, 3, 4, 5, 6]
# input3 = [3, 4, 5, 6, 7]
# output = ['A', 'B', 'C', 'D', 'E']
 
# # 创建DataFrame
# data = {
#     'Input1': input1,
#     'Input2': input2,
#     'Input3': input3,
#     'Output': output
# }
# df = pd.DataFrame(data)
 
# # 可视化
# sns.scatterplot(x='Input1', y='Input2', hue='Output', data=df)
# plt.show()
import pandas as pd
 
# 从Excel文件创建DataFrame
df = pd.read_excel('data_after.xlsx', sheet_name='CVAF')  # 修改为你的文件路径和工作表名称
#print(df) 
# 将DataFrame中的特定列转换为列表
list_of_data1 = df['Qv'].tolist()  # 修改为你想要转换的列名
list_of_data2 = df['DP'].tolist()  # 修改为你想要转换的列名
list_of_data3 = df['RPM'].tolist()
list_of_data4 = df['M1'].tolist()
data = {
    'Input1': list_of_data1,
    'Input2': list_of_data2,
    'Input3': list_of_data3,
    'Output': list_of_data4
}
df1 = pd.DataFrame(data)

# 创建3D散点图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df1['Input1'], df1['Input2'], df1['Input3'], c=df1['Output'])
 
# # 设置颜色条
# sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=10, vmax=80))
# sm.set_array([])
# fig.colorbar(sm)
 
plt.show()
 





# sns.scatterplot(x='Input1', y='Input2', hue='Output', data=df1)
# plt.show()
# 打印列表
# print(list_of_data1)
# print(list_of_data2)