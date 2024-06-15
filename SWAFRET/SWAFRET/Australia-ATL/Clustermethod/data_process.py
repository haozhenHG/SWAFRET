import datetime
import pandas as pd
import numpy as np
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

def data_timeAload():
    data = pd.read_csv('..\\CSVData\\datas.csv', header=0, index_col=0, parse_dates=['date'])

    data_index = data.index.unique()

    group_data = data.groupby(by='date')

    def timegroup(x_data):
        loaddata = []

        loaddata.append(x_data['load'].values.tolist()[:])

        return loaddata
    loaddata = group_data.apply(timegroup)
    def timegroup2(x_data):
        tempdata = []
        tempdata.append(x_data['temp'].values.tolist()[:])
        return tempdata
    tempdata = group_data.apply(timegroup2)
    load=[]
    temp=[]
    for i in range(len(loaddata)):
        onedayload =np.array(loaddata.values.flatten()[i])
        onedayload = onedayload.reshape(48,1)
        load.append(onedayload)
    for i in range(len(tempdata)):
        onedaytemp =np.array(tempdata.values.flatten()[i])
        onedaytemp = onedaytemp.reshape(48,1)
        temp.append(onedaytemp)

    stack_data = TimeSeriesScalerMeanVariance(mu=0.0, std=1.0).fit_transform(load)
    stack_data = np.column_stack((stack_data, TimeSeriesScalerMeanVariance(mu=0.0, std=1.0).fit_transform(temp)))
    return stack_data



def data_load():
    data = pd.read_csv('E:\\pythonfile\\Copy_LIu\\CSVData\\Summer.csv', header=0, index_col=0, parse_dates=['date'])

    data_index = data.index.unique()

    group_data = data.groupby(by='date')

    def timegroup(x_data):
        loaddata = []

        loaddata.append(x_data['load'].values.tolist()[:])

        return loaddata

    loaddata = group_data.apply(timegroup)
    load = []
    for i in range(len(loaddata)):
        onedayload =np.array(loaddata.values.flatten()[i])
        onedayload = onedayload.reshape(48,1)
        load.append(onedayload)

    return load


# 设置累积效应标签   ：连续几天累积
# 部分数据的
def Tem_accumult():
    # start = ''
    zero_day_list = []  # 没有累积的那一天的集合

    # 分析 数据  存在两天连续温度降低情况 则第二天设为 0
    Continuous_cooling = [] # 存放整个的累积效应标签

    Avg_temp_data = pd.read_csv('..\\CSVData\\average_temp.csv', header=0, index_col=0, parse_dates=['date'])
    print(Avg_temp_data.head())

    Avg_temp_data = Avg_temp_data.loc[:]





    index = Avg_temp_data.index

    dict = {}

    for i,date in enumerate(index):
        # 第一天没有体现 累积   这天温度加入 0 list中
        # 只有第一天进入该条件
        if i == 0:
            zero_day_list.append(Avg_temp_data.loc[date,'AvgTemp'])

            # accumulated_label.append(i)

            dict[date] = 0

            continue
        # 如果 明天 数据 高于今天  进行累计

        if Avg_temp_data.loc[date,'AvgTemp'] > Avg_temp_data.loc[date+datetime.timedelta(days=-1),'AvgTemp']:
            dict[date] = dict[date+datetime.timedelta(days=-1)] + 1  # 及今天比昨天气温高
        elif Avg_temp_data.loc[date, 'AvgTemp'] > zero_day_list[-1] and abs(Avg_temp_data.loc[date,'AvgTemp']-Avg_temp_data.loc[date+datetime.timedelta(days=-1),'AvgTemp']) <= 5:
            # 虽然今天没有昨天高 但是比最近 0 标签对应的那天高 也是温度累积
            dict[date] = dict[date+datetime.timedelta(days=-1)] + 1
            # 将本次降温记录
            Continuous_cooling.append(date)
        else:
            dict[date] = 0
            zero_day_list.append(Avg_temp_data.loc[date,'AvgTemp'])

    accumulated_label = list(dict.values())

    # 累积标签作为新的数据列写入文件
    Avg_temp_data['AcTemplabel'] = accumulated_label
    # 保存文件
    df = pd.DataFrame(Avg_temp_data, index=index,columns=['AvgTemp','AcTemplabel'])
    # df.to_csv('..\\CSVData\\average_temp_August_label.csv')
    # df.to_csv('..\\CSVData\\average_temp_May_label.csv')
    # df.to_csv('..\\CSVData\\average_temp_February_label.csv')
    # df.to_csv('..\\CSVData\\average_temp_November_label.csv')

    df.to_csv('..\\CSVData\\average_temp_label.csv')

if  __name__ == '__main__':
    Tem_accumult()