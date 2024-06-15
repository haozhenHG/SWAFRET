import torch
import argparse
import torch.optim as optim
import torch.nn.functional as F
import sys
sys.path.append("../../")
from model import TCN
from GEFCom2012.TCNmethod.utils import data_generator,data_traindeltatemp,data_generatordeltaT,data_deltatemp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import GEFCom2012.PSFmethod.PSF_Inputdate as PSF_Inputdate
import GEFCom2012.PSFmethod.PSF_Sequence_generation as PSFSg



model = torch.load('Paper Model.tar')
# model = torch.load('best_model.tar')
psftrainfilename = '..\\CSVData\\psfdk=8w=6.csv'
psffilename = '..\\CSVData\\cluster8LTwhole.csv'
dataname = '..\\CSVData\\data_load.csv'
tempname = '..\\CSVData\\data_temp.csv'
savefigname = 'kshapetcnwin.jpg'
savecsvname = 'wintermape.csv'
savepredname = 'win_prevalue.csv'
wsname = '..\\CSVData\\weekday_season.csv'
label = '..\\CSVData\\label.csv'

datan = '..\\CSVData\\datas.csv'

data = pd.read_csv(dataname, header=0, index_col=0, parse_dates=['date'])
tempdata = pd.read_csv(tempname, header=0, index_col=0, parse_dates=['date'])
data = data.loc['2004-01-01':'2008-06-28'].dropna()
tempdata = tempdata.loc['2004-01-01':'2008-06-28'].dropna()


X_train, Y_train, X_test, Y_test, sc_load, sc_temp = data_generator(psftrainfilename,data, tempdata)

# targ_list, aim_list = PSF_Inputdate.AUGtarg_aim_list0711(dataname)
# targ_list, aim_list = PSF_Inputdate.FEBtarg_aim_list0802(dataname)
# targ_list, aim_list = PSF_Inputdate.MAYtarg_aim_list0805(dataname)
targ_list, aim_list = PSF_Inputdate.AUGtarg_aim_list0711(dataname)
# targ_list = ['2010-04-21']
# aim_list = ['2010-04-22']
# targ_list = ['2010-09-08','2010-09-23','2010-10-24','2010-11-01','2010-11-06']
# aim_list = ['2010-09-09','2010-09-24','2010-10-25','2010-11-02','2010-11-07']
data_real = pd.read_csv(dataname, header=0, index_col=0, parse_dates=['date'])
tempdata = pd.read_csv(tempname, header=0, index_col=['date'], parse_dates=['date'])
wsdata = pd.read_csv(wsname, header=0, index_col=0, parse_dates=['date'])
label = pd.read_csv(label, header=0, index_col=0, parse_dates=['date'])
data_lload = pd.read_csv(datan, header=0, index_col=0, parse_dates=['date'])

load_train_pre = pd.DataFrame(sc_load.transform(data_real.loc[targ_list].values.reshape(-1, 1)).reshape(-1, 24))
load_test = pd.DataFrame(data_real.loc[aim_list].values.reshape(-1, 1).reshape(-1, 24))
temp_train_pre = pd.DataFrame(sc_temp.transform(tempdata.loc[targ_list].values.reshape(-1, 1)).reshape(-1, 24))
temp_train = pd.DataFrame(sc_temp.transform(tempdata.loc[aim_list].values.reshape(-1, 1)).reshape(-1, 24))
load_train_pre.index, load_train_pre.columns = targ_list, data_real.columns
temp_train_pre.index, temp_train_pre.columns = targ_list, data_real.columns
temp_train.index,load_test.index = aim_list,aim_list
data_real = data_real.loc['2006-11-1':'2007-10-30'].dropna()
data_index = data_real.index.unique()
pred = np.array([])
true = np.array([])
psf = np.array([])
mape_values = []
psfmape =[]
for i, j, k in zip(targ_list, aim_list, range(0,len(aim_list))):

    # ps_index = PSF.PSF_dayouttryw6T(psffilename, i, data_index)
    # psfpre = np.array(PSF.PSF_result(data_real,dataname, ps_index, j))
    psfpre = PSFSg.PSF_Sequence(wsdata, data_real,data_lload, label, 9, j)

    true_value = load_test.iloc[k].values.flatten()
    # mapes1 = np.mean(np.abs(true_value - psfpremape) / true_value) * 100
    psfpre = sc_load.transform(psfpre.reshape(-1, 1))
    for_pred = np.column_stack(
        (load_train_pre.iloc[k].values.reshape(-1, 1), temp_train_pre.iloc[k].values.reshape(-1, 1)))
    for_pred = np.column_stack((for_pred, psfpre))
    for_pred = np.column_stack((for_pred, temp_train.iloc[k].values.reshape(-1, 1))).reshape(-1, 48, 2)
    # for_pred = psfpre.reshape(-1,48)      #把前一天数据变成二维形式跑模型得到预测值preddata
    # print(for_pred)
    pred_data = model(torch.tensor(for_pred).float().cuda())
    pred_data = np.array((list(pred_data.cpu().flatten().detach())))
    pred_data_true = sc_load.inverse_transform(pred_data.reshape(-1, 1)).flatten()   #逆归一化 得到风速数据 逆归一化只支持2维


    print(i, 'to predict', j)
    # plt.plot(psfpre,'g')
    # plt.plot(pred_data_true, 'b')
    # plt.plot(true_value,'r')
    # plt.show()
    mapes = np.mean(np.abs(true_value - pred_data_true) / true_value) * 100
    print('mape:', mapes)
    # print('mape1:', mapes1)
    pred = np.append(pred, pred_data_true)
    true = np.append(true, true_value)
    mape_values.append(mapes)
    # psfmape.append(mapes1)
print(np.mean(mape_values))
plt.figure(figsize=(15,5))
plt.plot(pred,'b')
plt.plot(true,'r')
# plt.show()
# plt.savefig(r'E:\pythonfile\kshape3tcn.jpg')
print(mape_values)

# pd.DataFrame(mape_values, index=aim_list).to_csv('mape_AUG.csv')
# pd.DataFrame(pred.reshape(-1,24), columns=[str(x) + 'h' for x in range(1,25)], index=aim_list).to_csv('pred_AUTUMN_AUG.csv')