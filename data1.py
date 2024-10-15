import pandas as pd


# 设置文件路径
file_path = 'data_after.xlsx'

# 定义sheet名称和数据列名称
sheet_names = ['CVAF', 'CVAR', 'HFF', 'HDF']
data_names = ['type','Qv', 'DP', 'RPM', 'M1']

# 读取Excel文件里所有工作表
mode_data = {}
for sheet_name in sheet_names:
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    mode_data[sheet_name] = {
        data_name: df[data_name].tolist() for data_name in data_names
    }

# 打印第一个工作表的type
# print(mode_data['CVAF']['type'])

# 获得相同元素的索引值
def get_all_indices(lst, value):
    return [i for i, x in enumerate(lst) if x == value]


def get_type_index():
    all_type_index={}
    for sheet_name in sheet_names:
        type_lst = list(set(mode_data[sheet_name]['type']))
        print(type_lst)
        for type in type_lst:
            all_type_index[sheet_name] = {
            type:[get_all_indices(mode_data[sheet_name]['type'],type)]
            }
    return all_type_index
mysole=get_type_index()
print(mysole)







# for sheet_name in sheet_names:
#     type_lst = list(set(mode_data[sheet_name]['type'])) 
#     print(type_lst)
# for type in type_lst:
#     print(type)
# 示例使用
# my_list = [1, 2, 3, 2, 4, 2, 5]
# element_to_find = 2
# indices = get_all_indices(my_list, element_to_find)
# print(indices)  # 输出元素2的所有索引，例如: [1, 3, 5]