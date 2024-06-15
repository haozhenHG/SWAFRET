import torch
import argparse
import torch.optim as optim
import torch.nn.functional as F
import sys
sys.path.append("../../")
sys.path.append("..\\")
from model import TCN
from TCNmethod.utils import data_generator,seq_jiwen,data_traindeltatemp,data_generatordeltaT,data_deltatemp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from sklearn.preprocessing import MinMaxScaler
import PSFmethod.PSF_Inputdate as PSF_Inputdate
import PSFmethod.PSF as PSF
import PSFmethod.PSF_Sequence_generation as PSFSg
from datetime import datetime as ddtime

parser = argparse.ArgumentParser(description='Sequence Modeling - The Adding Problem')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='batch size (default: 32)')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA (default: True)')
parser.add_argument('--dropout', type=float, default=0.0,
                    help='dropout applied to layers (default: 0.0)')
parser.add_argument('--clip', type=float, default=-1,
                    help='gradient clip, -1 means no clip (default: -1)')
parser.add_argument('--epochs', type=int, default=200,
                    help='upper epoch limit (default: 100)')
parser.add_argument('--ksize', type=int, default=2,     #4
                    help='kernel size (default: 4)')
parser.add_argument('--levels', type=int, default=15,
                    help='# of levels (default: 8)')
parser.add_argument('--seq_len', type=int, default=48,
                    help='sequence length (default: 48)')
parser.add_argument('--log-interval', type=int, default=2, metavar='N',    #多少个batch打印一次训练状态
                    help='report interval (default: 100')
parser.add_argument('--lr', type=float, default=4e-3,
                    help='initial learning rate (default: 4e-3)')
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use (default: Adam)')
parser.add_argument('--nhid', type=int, default=32,    #40
                    help='number of hidden units per layer (default: 32)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed (default: 1111)')
args = parser.parse_args()
psftrainfilename = '..\\CSVData\\psfdk=8w=6.csv'
psffilename = '..\\CSVData\\cluster8LTwhole.csv'
dataname = '..\\CSVData\\data_load.csv'
tempname = '..\\CSVData\\data_temp.csv'
wsname = '..\\CSVData\\weekday_season.csv'
label = '..\\CSVData\\label.csv'
datan = '..\\CSVData\\datas.csv'
ATLabelname = '..\\CSVData\\average_temp_label.csv'
data = pd.read_csv(dataname, header=0, index_col=0, parse_dates=['date'])
tempdata = pd.read_csv(tempname, header=0, index_col=0, parse_dates=['date'])
dataATL = pd.read_csv(ATLabelname, header=0, index_col=0, parse_dates=['date'])
data = data.loc['2004-01-01':'2008-06-28'].dropna()
tempdata = tempdata.loc['2004-01-01':'2008-06-28'].dropna()
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

input_channels = 48
n_classes = 24                  #分类类别
batch_size = args.batch_size
seq_length = args.seq_len
epochs = args.epochs

print(args)
print("Producing data...")
# temp1A,temp2A,temp3P,temp4P = data_traindeltatemp(psftrainfilename, dataname)
jiwenlist_pre,jiwenlist_psf = seq_jiwen(dataname,psftrainfilename,ATLabelname)
X_train, Y_train, X_test, Y_test, sc_load, sc_temp,sc_jiwen = data_generator(psftrainfilename, data, tempdata,jiwenlist_pre,jiwenlist_psf )

# Note: We use a very simple setting here (assuming all levels have the same # of channels.
channel_sizes = [args.nhid]*args.levels    #256
kernel_size = args.ksize
dropout = args.dropout
model = TCN(input_channels, n_classes, channel_sizes, kernel_size=kernel_size, dropout=dropout)

if args.cuda:
    model.cuda()
    X_train = X_train.float().cuda()
    Y_train = Y_train.float().cuda()
    X_test = X_test.float().cuda()
    Y_test = Y_test.float().cuda()

lr = args.lr
optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr)     #getattr 从目标对象取东西


model = torch.load('Paper Model.tar')

# ['2008-05-31':'2008-06-28']
season,targ_list, aim_list = PSF_Inputdate.AUGtarg_aim_list0711(dataname)

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
ok_list = []
# ['2008-05-31': '2008-06-28']
for i, j, k in zip(targ_list, aim_list, range(0,len(aim_list))):
    ok_list.append(j)

    # ps_index = PSF.PSF_dayouttryw6T(psffilename, i, data_index)
    # psfpre = np.array(PSF.PSF_result(data_real,dataname, ps_index, j))
    psfpre = PSFSg.PSF_Sequence(wsdata, data_real,data_lload, label, 3, j)

    jiwen = dataATL.loc[i + datetime.timedelta(days=-1), 'AcTemplabel']
    ilist1 = [jiwen] * 24

    jiwen = dataATL.loc[i, 'AcTemplabel']
    ilist2 = [jiwen] * 24

    ilist1 = sc_jiwen.fit_transform(np.array(ilist1).reshape(-1, 1))
    ilist2 = sc_jiwen.fit_transform(np.array(ilist2).reshape(-1, 1))

    true_value = load_test.iloc[k].values.flatten()
    # mapes1 = np.mean(np.abs(true_value - psfpremape) / true_value) * 100
    psfpre = sc_load.transform(psfpre.reshape(-1, 1))
    for_pred = np.column_stack(
        (load_train_pre.iloc[k].values.reshape(-1, 1), temp_train_pre.iloc[k].values.reshape(-1, 1)))
    for_pred = np.column_stack((for_pred, ilist1))
    for_pred = np.column_stack((for_pred, psfpre))
    for_pred = np.column_stack((for_pred, temp_train.iloc[k].values.reshape(-1, 1)))
    for_pred = np.column_stack((for_pred, ilist2)).reshape(-1, 48, 3)

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

current_time = ddtime.now().strftime('%Y-%m-%d~%H-%M-%S')
dict = {'spring':'May','summer':'June','autumn':'Nov','winter':'Feb'}

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