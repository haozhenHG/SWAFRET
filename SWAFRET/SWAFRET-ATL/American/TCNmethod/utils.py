import torch
import numpy as np
from torch.autograd import Variable
import pandas as pd
import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn import model_selection


def reshape_data(datapsf, data, psf_index, pre_index):
    '''

    Args:
        datapsf:'..\\CSVData\\psfdk=6w=6.csv'   PSF序列文件  且 data作为列索引
        data:
                上述两个data.drop的结果是
                      load	temp1	temp2	temp3
                date
        psf_index:  '2018-01-07'  '2018-12-31'
        pre_index:  2018-01-06     2018-12-30

    Returns:

    '''
    sc_load = MinMaxScaler()
    data['load'] = sc_load.fit_transform(data['load'].values.reshape(-1, 1))
    # data['load'].values : array([77437, 72863, 69862, ..., 75081, 72716, 72716], dtype=int64)  reshape : n*1

    sc_temp = MinMaxScaler()

    # fit_transform和transform方法可以结合使用
    data['temp1'] = sc_temp.fit_transform(data['temp1'].values.reshape(-1, 1))
    data['temp2'] = sc_temp.transform(data['temp2'].values.reshape(-1, 1))
    data['temp3'] = sc_temp.transform(data['temp3'].values.reshape(-1, 1))

    datapsf['PSF'] = sc_load.transform(datapsf['PSF'].values.reshape(-1, 1))

    # datapsf['PSF'] = datapsf['PSF'].values.reshape(-1, 1)

    data_real = data.copy() #  load	temp1	temp2	temp3

    # axis=1 或 axis='columns'：沿着列方向进行操作，即删除列
    data = data.drop(data.columns[0], axis=1)  # 删除负荷，只剩下时间   指定axis 否则会报错
    # 	temp1	temp2	temp3
    # date

    length = psf_index.shape[0]  # shape(359,)   359
    # --------------------------------------------------划分数据集--------------------------------------------------
    # pre_index:2018-01-06  2018-12-30
    # psf_index:'2018-01-07'  '2018-12-31'
    x = np.column_stack((
        np.column_stack((
            data_real.loc[pre_index].iloc[:length * 24], data.loc[psf_index].iloc[:length * 24]
        )),
            datapsf.loc[psf_index, 'PSF'].iloc[:length * 24]
    )).reshape(-1, 48, 4) # length * 24 =  8616
    #x.shape (359, 48, 4)


    # 有没有.iloc[:length * 24]一样
    # data_real.loc[psf_index, 'load'] = data_real.loc[psf_index, 'load'].iloc[:length * 24]
    y = data_real.loc[psf_index, 'load'].iloc[:length * 24].values.reshape(-1, 24)# 一天的24个特征放在一起


    # shuffle: 是否在分割前对数据进行随机打乱。默认为True
    # test_size: 测试集所占的比例，可以是浮点数（表示比例，如0.2表示20%）或整数
    X_train, X_test, y_train, y_test = model_selection.train_test_split(x, y, shuffle=True, test_size=0.33)

    print('X_train.shape=', X_train.shape)
    return [X_train, X_test, y_train, y_test, sc_temp, sc_load]

def data_generator(psftrainfilename,dataname):
    """
    Args:
        psf_name: psftrainfilename = '..\\CSVData\\psfdk=6w=6.csv'
        dataname: dataname = '..\\CSVData\\datas.csv'
    """
    '''
    psf_name : '..\\CSVData\\psfdk=6w=6.csv'   PSF序列文件
    dataname : '..\\CSVData\\datas.csv'        源数据文件
    '''
    datapsf = pd.read_csv(psftrainfilename, header=0, index_col=1, parse_dates=['date'])# psftrainfilename 并将date列作为行索引
    data = pd.read_csv(dataname, header=0, index_col=0, parse_dates=['date'])  # 将dataname读入 并将date列作为行索引

    # 对时间索引去重   一个时间有24条数据
    psf_index = datapsf.index.unique()
    # # psf_index  '2018-01-07'  '2018-12-31'

    data = data.drop(data.columns[4], axis=1)  # 删除  weekday 一列
    data = data.drop(data.columns[4], axis=1)  # 删除  season 一列
    '''
    上述两个data.drop的结果是	
          load	temp1	temp2	temp3
    date			
    '''

    pre_index = []
    # psf_index  '2018-01-07'  '2018-12-31'
    for i in psf_index:
        pre_index.append(i + datetime.timedelta(days=-1))
    # for循环的结果是 pre_index ：2018-01-06  2018-12-30


    # datapsf    '..\\CSVData\\psfdk=6w=6.csv'   PSF序列文件
    # data    :         load	 temp1	temp2	temp3
    #           date
    # psf_index  '2018-01-07'  '2018-12-31'
    # pre_index   2018-01-06    2018-12-30
    a = reshape_data(datapsf, data, psf_index, pre_index)
    #  a =  [X_train, X_test, y_train, y_test, sc_temp, sc_load]

    return torch.tensor(a[0]), torch.tensor(a[2]), torch.tensor(a[1]), torch.tensor(a[3]), a[-1], a[4]

