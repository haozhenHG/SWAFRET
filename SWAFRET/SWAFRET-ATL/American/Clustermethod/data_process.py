
import pandas as pd
import numpy as np
from tslearn.preprocessing import TimeSeriesScalerMeanVariance #时间序列分析 tslearn


# -----------------本文件旨在 对数据文件进行 数据提取、标准化处理 方便Kshapeclusterget.py文件进行聚类------------------------

def data_timeAload():
    data = pd.read_csv('..\\CSVData\\datas.csv', header=0, index_col=0, parse_dates=['date'])
    # header=0使用数据的默认表头
    # index_col=0   index_col : int, sequence or bool 官方文档  =0，直接将第一列作为索引
    # parse_dates ：指定某些列为时间类型

    # data_index = data.index.unique()

    group_data = data.groupby(by='date')  #按照列data分组 返回DataFramegroupby对象


    # x_data实际就是group_data 尤其注意x_data['load']的值
    # 通过对 temp1  temp2   temp3 研究 找到温度对load的关系
    def timegroup(x_data):
        loaddata = []
        loaddata.append(x_data['load'].values.tolist()[:]) #对 列 load 操作
        return loaddata

    # 对列temp1 方法同上
    def timegroup2(x_data):
        tempdata = []
        tempdata.append(x_data['temp1'].values.tolist()[:])
        return tempdata

    def timegroup3(x_data):
        tempdata = []
        tempdata.append(x_data['temp2'].values.tolist()[:])
        return tempdata

    def timegroup4(x_data):
        tempdata = []
        tempdata.append(x_data['temp3'].values.tolist()[:])
        return tempdata

    # apply参数传入的是  函数方法名
    '''
    loaddata = group_data.apply(timegroup) 结果如下
    
    2015/7/2    [[77437, 72863, 69862, 67951]]
    2015/7/3    [[67153, 68405, 71204]]
    2015/7/4    [[75033]]
    2015/7/5    [[78999]]
    dtype: object
    '''
    # apply方法  按照参数进行分组  timegroup传入的参数就是group_data
    loaddata = group_data.apply(timegroup)      # x_data['load']
    tempdata1 = group_data.apply(timegroup2)    # x_data['temp1']
    tempdata2 = group_data.apply(timegroup3)    # x_data['temp2']
    tempdata3 = group_data.apply(timegroup4)    # x_data['temp3']
    print('loaddata:',loaddata)
    load=[]
    temp1=[]
    temp2=[]
    temp3=[]

    # len()获得长度  每个元素的长度又不相同  如[[1,1,1,],[2,2]]
    # for  函数的作用是维度转换  将datas.csv中的(1,24)数据 转换为 (24,1)
    for i in range(len(loaddata)):
        onedayload =np.array(loaddata.values.flatten()[i]) # 转换为numpy类型
        # print('before reshape =', onedayload.shape)  # (1, 24)  每天的载荷数据  有24组
        # print(onedayload.shape, i)
        onedayload = onedayload.reshape(24, 1)
        # print('after reshape =', onedayload.shape)
        load.append(onedayload)
    # print('load content:',load)  单独的那一列  [array([]),array([]),...,array([])] 共有2447组数据
    # print('load:',len(load))  # 共有2447组数据

    for i in range(len(tempdata1)):
        onedaytemp =np.array(tempdata1.values.flatten()[i])
        onedaytemp = onedaytemp.reshape(24,1) #维度转化
        temp1.append(onedaytemp)
    # print('temp1:', len(temp1)) # [[1 temp1(24组)],...,[n temp1(24组)],[2447 temp1(24组)]]

    for i in range(len(tempdata2)):
        onedaytemp =np.array(tempdata1.values.flatten()[i])
        onedaytemp = onedaytemp.reshape(24,1)
        temp2.append(onedaytemp)
    # print('temp2:', len(temp2))

    for i in range(len(tempdata3)):
        onedaytemp =np.array(tempdata1.values.flatten()[i])
        onedaytemp = onedaytemp.reshape(24,1)
        temp3.append(onedaytemp)
    # print('temp3:', len(temp3))
    # 时间序列平均方差
    '''
    TimeSeriesScalerMeanVariance方法：
    mu：输出时间序列的平均值。
    std：输出时间序列的标准偏差 
    
    此方法需要大小相等的时间序列数据集。
    在计算mu和std时，会忽略时间序列中的NaN。
    '''
    # TimeSeriesScalerMeanVariance(mu=0.0, std=1.0).fit_transform 搭配使用
    '''
    fit_transform()先拟合数据，再转化它将其转化为标准形式  通过 (a - mean) / var 拟合
    不仅计算训练数据的均值和方差，还会基于计算出来的均值和方差来转换训练数据，从而把数据转换成标准的正态分布。
    X ~ N(0,1) 时, 称X服从标准正态分布  
    此处是方便 K-Shape 使用    （通常采用 z-normalization）
    '''
    stack_load = TimeSeriesScalerMeanVariance(mu=0.0, std=1.0).fit_transform(load)  # 返回numpy.ndarray类型
    # 查看元素信息 shape:(2447, 24, 1) size:58728 ndim:3:   size 是 shape各数据相乘  2447*24*1
    # 2447行  24列  每列1个元素     每行代表一个date
    print('first shape:%s size:%s ndim:%s:'% (stack_load.shape,stack_load.size,stack_load.ndim))
    # print('choice tow row:',stack_load[0:2,:,:])
    # print('fit_transform(load) :', stack_data)
    '''
    shape:(2447, 24, 1)
    1       24列数据   每列 1 个信息
    2       24列数据   每列 1 个信息
    3       24列数据   每列 1 个信息
    ..
    2447    24列数据
    
    (3,5,2)表示，第三维的2表示每组有姓名和国籍两个属性；第二维的5表示每组有五个参赛者；最后的第一维3就代表者有3个小组。
    '''
    stack_temp1 = TimeSeriesScalerMeanVariance(mu=0.0, std=1.0).fit_transform(temp1)
    stack_temp2 = TimeSeriesScalerMeanVariance(mu=0.0, std=1.0).fit_transform(temp2)
    stack_temp3 = TimeSeriesScalerMeanVariance(mu=0.0, std=1.0).fit_transform(temp3)


    # ? 不理解 为什么 np.column_stack 合并 第三位是1  观察jupyter notebook 179-之后代码

    # column_stack 将2个矩阵按列合并  维度不变
    # column_stack 参数只能传入参数1个   所以np.column_stack(())   用元组表示
    # 4个 shape:(2447, 24, 1) 拼接  shape:(2447, 96, 1)
    stack_data = np.column_stack((np.column_stack((np.column_stack((stack_load, stack_temp1)),stack_temp2)),stack_temp3))
    # column_stack 本人推测  ：拼接后应该是 shape:(2447, 24, 4)
    #                 实际  ：shape:(2447, 96, 1) size:234912 ndim:3
    #
    # print('shape:%s size:%s ndim:%s:' % (stack_data.shape, stack_data.size, stack_data.ndim))
    # print('choice one row:', stack_data[0:1, :, :].shape)  # 取第一个数据
    # print('choice one row:', stack_data[:1, :, :])

    return stack_data
