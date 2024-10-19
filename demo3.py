import pandas as pd  
import plotly.express as px  
import plotly.graph_objects as go
import webbrowser
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

def create_3d_plot(sheet_data, sheet_name):
    # 定义从蓝色到红色的颜色范围
    colorscale = [
        [0, "rgb(0,0,255)"],
        [0.25, "rgb(0,255,255)"],
        [0.5, "rgb(0,255,0)"],
        [0.75, "rgb(255,255,0)"],
        [1, "rgb(255,0,0)"]
    ]

    fig = go.Figure()#提供一个画布

    for t in sheet_data['types']:
        data = sheet_data['data'][t]
        
        scatter = go.Scatter3d(
            x=data['Qv'],
            y=data['DP'],
            z=data['RPM'],
            mode='markers',
            marker=dict(
                size=5,#点的大小
                color=data['M1'],#根据M1的数据值确定颜色
                colorscale=colorscale,#设定颜色条
                colorbar=dict(title='M1'),#颜色条标题M1
                showscale=True#默认颜色条显示
            ),
            text=[f'Type: {t}, M1: {m1}' for m1 in data['M1']],
            hoverinfo='text',
            name=t,
            visible=True  # 默认所有type都可见
        )
        
        fig.add_trace(scatter)

    # 更新布局
    fig.update_layout(
        scene=dict(
            xaxis_title='Qv',
            yaxis_title='DP',
            zaxis_title='RPM',
            aspectmode='cube'
        ),
        title=f'3D Visualization of {sheet_name}: Qv, DP, RPM, and M1',
        height=900,  # 增加高度
        width=1600,  # 增加宽度
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        margin=dict(l=0, r=0, t=40, b=0)  # 减小边距
    )
    return fig
def load_data():
    # 读取Excel文件  
    excel_file = 'data_after.xlsx'  # 替换为你的Excel文件名  
    sheet_names = ['CVAF', 'CVAR', 'HFF', 'HDF']  # 替换为你的工作表名  
    
    # 初始化一个空的字典来存储所有数据  
    all_data =dict() 
    # 初始化一个空的DataFrame来存储所有数据  
    all_data1 = pd.DataFrame()
    for sheet_name in sheet_names:
        # 读取数据
        df = pd.read_excel(excel_file, sheet_name=sheet_name)
        df1=df
        df1['method'] = sheet_name  # 添加一个列来表示工作表名  
        all_data1 = pd.concat([all_data1, df1], ignore_index=True) 
        types = df['type'].unique().tolist()
        all_data[sheet_name] = {'types': types,'data': {t: df[df['type'] == t] for t in types}}
    return all_data1,all_data
        # # 创建3D图
        # fig = create_3d_plot(all_data[sheet_name], sheet_name)

        # # 保存为HTML文件，使用相对路径
        # output_path = f'3D_plot_{sheet_name}.html'
        # fig.write_html(output_path, full_html=False, include_plotlyjs='cdn')

        # # 自动打开生成的HTML文件
        # webbrowser.open('file://' + os.path.abspath(output_path)) 
        # return all_data1,all_data

#数据处理得到训练集和测试集，拟用各类型数据作为测试集，总数据作为训练集



def dataPreprocess_train(alldata_lst):
    QV_data = alldata_lst['Qv']
    DP_data = alldata_lst['DP']
    RPM_data = alldata_lst['RPM']
    M1_data = alldata_lst['M1']

    xdata = []
    for i in range(len(QV_data)):
        xdata.append([QV_data[i],DP_data[i],RPM_data[i]])

    X_train = np.array(xdata)
    y_train = np.array(M1_data)

    #X_train, X_test, y_train, y_test = train_test_split(X, M1_data, test_size=0.3)
    return X_train, y_train
def data_test(data, X_train, y_train,key):
    
    
    QV_data = data['Qv'].tolist()
    DP_data = data['DP'].tolist()
    RPM_data = data['RPM'].tolist()
    M1_data = data['M1'].tolist()

    xdata = []
    for i in range(len(QV_data)):
        xdata.append([QV_data[i],DP_data[i],RPM_data[i]])

    diabetes_X_test = np.array(xdata)
    diabetes_y_test = np.array(M1_data)
    result=Linear_Regression(X_train,y_train,diabetes_X_test,diabetes_y_test)
    print(key)
    print(result)
    


#最小二乘法回归
def Linear_Regression(diabetes_X_train,diabetes_y_train,diabetes_X_test,diabetes_y_test):
    # Code source: Jaques Grobler
# License: BSD 3 clause

# # Load the diabetes dataset
# diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

# # Use only one feature
# diabetes_X = diabetes_X[:, np.newaxis, 2]

# # Split the data into training/testing sets
# diabetes_X_train = diabetes_X[:-20]
# diabetes_X_test = diabetes_X[-20:]

# # Split the targets into training/testing sets
# diabetes_y_train = diabetes_y[:-20]
# diabetes_y_test = diabetes_y[-20:]

# Create linear regression object
    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(diabetes_X_train, diabetes_y_train)

    # Make predictions using the testing set
    diabetes_y_pred = regr.predict(diabetes_X_test)

    # The coefficients
    # print("Coefficients: \n", regr.coef_)
    # The mean squared error
    # print("Mean squared error: %.2f" % mean_squared_error(diabetes_y_test, diabetes_y_pred))
    # The coefficient of determination: 1 is perfect prediction
    # print("Coefficient of determination: %.2f" % r2_score(diabetes_y_test, diabetes_y_pred))
    dic={'Coefficients':regr.coef_,
         'Mean squared error':mean_squared_error(diabetes_y_test, diabetes_y_pred),
         'Coefficient of determination': r2_score(diabetes_y_test, diabetes_y_pred)
         }
    return dic

    # # Plot outputs
    # plt.scatter(diabetes_X_test, diabetes_y_test, color="black")
    # plt.plot(diabetes_X_test, diabetes_y_pred, color="blue", linewidth=3)

    # plt.xticks(())
    # plt.yticks(())

    # plt.show()
def Ridge(diabetes_X_train,diabetes_y_train):
    # \alpha \geq大于等于 0 是控制系数收缩量的复杂性参数： \alpha 的值越大，收缩量越大，模型对共线性的鲁棒性也更强。
    reg = linear_model.Ridge (alpha = .5)
    reg.fit (diabetes_X_train,diabetes_y_train)
    Ridge(alpha=0.5, copy_X=True, fit_intercept=True, max_iter=None,
 normalize=False, random_state=None, solver='auto', tol=0.001)
   

if __name__ == "__main__":
    all_data1,all_data=load_data()
    X_train, y_train=dataPreprocess_train(alldata_lst=all_data1)
    sheet_names = ['CVAF', 'CVAR', 'HFF', 'HDF']
    types = all_data1['type'].unique().tolist()
    all_result={}
    result={}
    excel_file = 'data_after.xlsx'  # 替换为你的Excel文件名  
    sheet_names = ['CVAF', 'CVAR', 'HFF', 'HDF']  # 替换为你的工作表名  
    
    # 初始化一个空的字典来存储所有数据  
    all_data =dict()  
    # 初始化一个空的DataFrame来存储所有数据  
    all_data1 = pd.DataFrame()
    for sheet_name in sheet_names:
        # 读取数据
        df = pd.read_excel(excel_file, sheet_name=sheet_name)
        types = df['type'].unique().tolist()
        sheet_data = {'types': types,'data': {t: df[df['type'] == t] for t in types}}
        dic=dict()
        dic=sheet_data['data']
        for key, value in dic.items():
            data_test(value, X_train, y_train,key)    

        # print(f"Processing DataFrame for key: {key}")
        # print(df)

    # for sheet_name in sheet_names:
    #     for type in types:
    #         t=all_data[sheet_name]['data'][type]
    #         diabetes_X_test,diabetes_y_test=data_test(t)

   
        


   
    
# def traverse_dict(nested_dict, level=0):
#     """递归遍历嵌套字典"""
#     for key, value in nested_dict.items():
#         print(' ' * level * 4 + f"{key}: {value}")  # 打印当前键值对，使用缩进表示层级
#         if isinstance(value, dict):  # 如果值是字典，递归调用
#             traverse_dict(value, level + 1)

# # 遍历嵌套字典
# traverse_dict(all_data)

