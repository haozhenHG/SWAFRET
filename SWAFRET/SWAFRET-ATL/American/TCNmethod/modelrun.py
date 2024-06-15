import torch

import sys
sys.path.append("../../")
# sys.path.append('报错模块的上一级的路径')
sys.path.append(r"..//")#定位到上一目录

import American.PSFmethod.PSF_Sequence_generation as PSFSg
from American.TCNmethod.utils import data_generator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# 标红无所谓  不影响
import American.PSFmethod.PSF_Inputdate as PSF_Inputdate
import PSFmethod.PSF as PSF

model = torch.load('3.28_2.tar')
# model = torch.load('best_model.tar')
psftrainfilename = '..\\CSVData\\psfdk=6w=6.csv'   # PSF序列文件
# psffilename = '..\\CSVData\\PSF_sequence_2Y.csv'   # PSF序列文件

psffilename = '..\\CSVData\\cluster6LT.csv'        # 聚类标签文件
dataname = '..\\CSVData\\datas.csv'                # 源数据文件
savefigname = 'kshapetcnwin.jpg'
savecsvname = 'wintermape.csv'
wsname = '..\\CSVData\\weekday_season.csv'
label = '..\\CSVData\\label.csv'

# sc_load :sc_load = MinMaxScaler()
# sc_temp ：sc_temp = MinMaxScaler()
X_train, Y_train, X_test, Y_test, sc_load, sc_temp = data_generator(psftrainfilename,dataname)
print('X_train.shape:%s Y_train.shape:%s X_test.shape:%s Y_test.shape:%s'%(X_train.shape,Y_train.shape,X_test.shape,Y_test.shape))
# return [X_train, X_test, y_train, y_test, sc_temp, sc_load]

# dataname = '..\\CSVData\\datas.csv'
# data_index：['2021-04-30':'2021-05-30']
season,targ_list, aim_list = PSF_Inputdate.novtarg_aim_listw7(dataname)  # novtarg_aim_listw6
# targ_list = ['2021-04-30':'2021-05-30']
# aim_list = ['2021-05-01':'2021-05-31']  目标   预测电力数据

# dataname = '..\\CSVData\\datas.csv'
data_real = pd.read_csv(dataname, header=0, index_col=0, parse_dates=['date'])  # 这样date作为索引  不作为列
wsdata = pd.read_csv(wsname, header=0, index_col=0, parse_dates=['date'])
label = pd.read_csv(label, header=0, index_col=0, parse_dates=['date'])




data_real = data_real.drop(data_real.columns[4], axis=1)  # 删除weekday
data_real = data_real.drop(data_real.columns[4], axis=1)  # 删除season
# drop操作后 data_real 只存在  load temp1  temp2  temp3  四列数据  date作为索引 不算列

# 复制数据
data_real1 = data_real.copy()
data = data_real.copy()

# 删除  列 load   保留温度数据列
data_real1 = data_real1.drop(data_real1.columns[0], axis=1)
# data_real1 只存在   temp1  temp2  temp3  三列数据   date作为索引 不算列

# data_real  存在  load temp1  temp2  temp3  四列数据
# data_real1 存在   temp1  temp2  temp3  三列数据

# 使用了fit_transform后可以直接使用transform [特征缩放] 因为已经进行fit了
#对数据集中的特征进行“调整 ”，使得所有特征的值都落在一个指定的范围内
# 将调整后的特征替换掉原始数据
data_real['load'] = sc_load.transform(data_real['load'].values.reshape(-1, 1))

data_real['temp1'] = sc_temp.transform(data_real['temp1'].values.reshape(-1, 1))
data_real['temp2'] = sc_temp.transform(data_real['temp2'].values.reshape(-1, 1))
data_real['temp3'] = sc_temp.transform(data_real['temp3'].values.reshape(-1, 1))
# data_real  只存在   load temp1  temp2  temp3  列数据
# load temp1  temp2  temp3  替换成对应的   特征


data_real1['temp1'] = sc_temp.transform(data_real1['temp1'].values.reshape(-1, 1))
data_real1['temp2'] = sc_temp.transform(data_real1['temp2'].values.reshape(-1, 1))
data_real1['temp3'] = sc_temp.transform(data_real1['temp3'].values.reshape(-1, 1))
# data_real1 只存在   temp1  temp2  temp3  三列数据
#  temp1  temp2  temp3  替换成对应的   特征


# 选择两年的数据集['2019-01-01':'2020-12-31']
# data_real  只存在 data  load temp1  temp2  temp3  五列数据
data_index = data_real.loc['2019-01-01':'2020-12-31'].index.unique()

pred = np.array([])
true = np.array([])

mape_values = []
PSF1 = []
ok_list=[]
nopsf=0

# targ_list = ['2021-04-30':'2021-05-30']
# aim_list = ['2021-05-01':'2021-05-31']
# data_index：['2019-01-01':'2020-12-31']
for i, j in zip(targ_list, aim_list):
    try:
        # psffilename = '..\\CSVData\\cluster6LT.csv'
        # ps_index = PSF.PSF_dayouttryw6T(psffilename, i, data_index)
        # psfpre = PSF.PSF_result(data_real, dataname,j,ps_index)
        psfpre = PSFSg.PSF_Sequence(wsdata,data_real,label,3,j)
        # print('psfpre:',psfpre)
        ok_list.append(j)
        # 将psfpre 缩放特征值
        psfpre = sc_load.transform(psfpre.reshape(-1,1))

        # data_real  只存在   load temp1  temp2  temp3  四列数据
        # data_real1 只存在   temp1  temp2  temp3  三列数据
        # 确保所有输入数组的长度相同是使用 column_stack 的前提条件
        for_pred = np.column_stack((np.column_stack((data_real.loc[i],data_real1.loc[j])),psfpre)).reshape(-1,48,4)      #把前一天数据变成二维形式跑模型得到预测值preddata

        # print(for_pred)
        pred_data = model(torch.tensor(for_pred).float().cuda())

        pred_data = np.array((list(pred_data.cpu().flatten().detach())))

        pred_data_true = sc_load.inverse_transform(pred_data.reshape(-1, 1)).flatten()   #逆归一化 得到风速数据 逆归一化只支持2维

        true_value = data.loc[j,'load'].values.flatten()
        print(i, 'to predict', j)

        # plt.plot(psfpre,'g')
        plt.plot(pred_data_true, 'b')
        plt.plot(true_value,'r')
        # plt.show()
        mapes = np.mean(np.abs(true_value - pred_data_true) / true_value) * 100
        print('mape:', mapes)
        pred = np.append(pred, pred_data_true)
        true = np.append(true, true_value)
        mape_values.append(mapes)
    except ZeroDivisionError:
        nopsf = nopsf + 1


current_time = datetime.now().strftime('%Y-%m-%d~%H-%M-%S')
dict = {'spring':'Nov','summer':'Feb','autumn':'May','winter':'Aug'}

gggg = 'pre_'+season+ dict[season]+':'
print(gggg,np.mean(mape_values))
# plt.figure(figsize=(12,5))
# plt.plot(pred,'b')
# plt.plot(true,'r')
# plt.show()
# plt.savefig('..\\Figure\\kshapetcn8.jpg')
print(mape_values)
pd.DataFrame(mape_values, index=ok_list).to_csv('..\\Result\\mape'+season+ dict[season]+str(current_time)+'.csv')
pd.DataFrame(pred.reshape(-1,24), columns=[str(x) + 'h' for x in range(1,25)], index=aim_list).to_csv('..\\Result\\pre_'+season+ dict[season]+str(current_time)+str(np.mean(mape_values))+'.csv')