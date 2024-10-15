import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

from pyecharts.charts import Line # type: ignore
from pyecharts import options as opts # type: ignore

# 数据结构按照
# { 
#     "method"    :   ["CVAF","CVAF",...],
#     "type"  :   ["T1X","T1X",...],
#     "Qv"    :   [375,375,...],
#     "DP"    :   [-138,-67,...],
#     "RPM"   :   [2797,2553,...],
#     "M1"    :   [57.03,55.88,...]
# }
# 排布，一共1889条数据
def load_data(dir = "."):
    all_sheets = pd.read_excel(os.path.join(dir, "data_after.xlsx"), sheet_name=None)
    dataFrame = dict()
    for sheet_name, df in all_sheets.items():
        print(f"Sheet name: {sheet_name}")
        for key in df.keys():
            for i in df[key]:
                dataFrame.setdefault(key,[]).append(i)
        for i in ([sheet_name] * len(df[key])):
            dataFrame.setdefault("method",[]).append(i) 
    # print(dataFrame.keys())
    # print(len(dataFrame["method"]))
    # print(len(dataFrame["type"]))
    # print(len(dataFrame["Qv"]))
    # print(len(dataFrame["DP"]))
    # print(len(dataFrame["RPM"]))
    # print(len(dataFrame["M1"]))
    return dataFrame
 
def plot_demo(): 
    # 数据
    data = [10, 20, 30, 40, 50, 60]
    
    # 创建折线图对象
    line = Line()
    
    # 添加数据
    line.add_xaxis(["A", "B", "C", "D", "E", "F"])
    line.add_yaxis("系列1", data)
    
    # 设置全局选项
    line.set_global_opts(title_opts=opts.TitleOpts(title="折线图示例"))
    
    # 渲染图表到文件
    line.render("line_chart.html")

def plot_debug(dataFrame, method='CVAF'):
    method_list = np.asarray(dataFrame["method"])
    type_list = np.asarray(dataFrame["type"])
    method_ind = np.where(method_list==method)[0].tolist()


    # 创建折线图对象
    line = Line()
    
    for type in set(dataFrame["type"]):
        type_ind = np.where(type_list==type)[0].tolist()
        inter_ind = [x for x in method_ind if x in type_ind]

        x_data = [dataFrame["Qv"][i] for i in inter_ind]
        y_data = [dataFrame["M1"][i] for i in inter_ind]

        # 添加数据
        line.add_xaxis(xaxis_data = x_data)
        line.add_yaxis(series_name = type, y_axis = y_data)

    # 设置全局选项
    line.set_global_opts(
        # title_opts=opts.TitleOpts(title="折线图堆叠"),
        tooltip_opts=opts.TooltipOpts(trigger="axis"),
        yaxis_opts=opts.AxisOpts(
            type_="value",
            axistick_opts=opts.AxisTickOpts(is_show=True),
            splitline_opts=opts.SplitLineOpts(is_show=True),
        ),
        xaxis_opts=opts.AxisOpts(type_="value", boundary_gap=False))

        # 渲染图表到文件
    line.render(f"{method}_Qv_M1_data.html")

if __name__ == '__main__':
    df = load_data()
    # plot_debug(df)
    plot_debug(df,'CVAR')
    plot_debug(df,'HFF')
    plot_debug(df,'HDF')
    # plot_demo()