import torch
import argparse
import torch.optim as optim
import torch.nn.functional as F
import sys
sys.path.append("../../")
from model import TCN
from TCNmethod.utils import data_generator,seq_jiwen
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from datetime import datetime as ddtime

from sklearn.preprocessing import MinMaxScaler
import PSFmethod.PSF_Inputdate as PSF_Inputdate
import PSFmethod.PSF as PSF
psftrainfilename = '..\\CSVData\\psfdk=6w=6.csv'
psffilename = '..\\CSVData\\cluster6LT.csv'
dataname = '..\\CSVData\\datas.csv'
ATLabelname = '..\\CSVData\\average_temp_label.csv'
jiwenlist_pre,jiwenlist_psf = seq_jiwen(dataname,psftrainfilename,ATLabelname)
X_train, Y_train, X_test, Y_test, sc_load, sc_temp,sc_jiwen = data_generator(psftrainfilename,dataname,jiwenlist_pre,jiwenlist_psf)

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
parser.add_argument('--log-interval', type=int, default=3, metavar='N',    #多少个batch打印一次训练状态
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

# psftrainfilename = 'E:\\pythonfile\\auspring\\CSVData\\psfdk=6w=6.csv'
# psffilename = 'E:\\pythonfile\\auspring\\CSVData\\cluster6LT.csv'
# dataname = 'E:\\pythonfile\\auspring\\CSVData\\spring.csv'

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

# X_train, Y_train, X_test, Y_test, sc_load, sc_hour, sc_temp = data_generatordeltaT(psftrainfilename, temp1A, temp2A, temp3P, temp4P)

###(((seq_len - kernel_size)+2padding)/strides or dilation)+1 = out(向下取整)
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


def train(epoch):
    global lr
    model.train()
    batch_idx = 1
    total_loss = 0
    for i in range(0, X_train.size(0), batch_size):       #size0 第0个维度
        if i + batch_size > X_train.size(0):
            x, y = X_train[i:], Y_train[i:]
        else:
            x, y = X_train[i:(i+batch_size)], Y_train[i:(i+batch_size)]
        optimizer.zero_grad()               #清空过往梯度
        output = model(x)
        loss = F.mse_loss(output, y)
        loss.backward()                 #反向传播
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()            #根据梯度更新网络参数
        batch_idx += 1          #完成一次batch
        total_loss += loss.item()

        if batch_idx % args.log_interval == 0:
            cur_loss = total_loss / args.log_interval
            processed = min(i+batch_size, X_train.size(0))
            print('Train Epoch: {:2d} [{:6d}/{:6d} ({:.0f}%)]\tLearning rate: {:.4f}\tLoss: {:.6f}'.format(
                epoch, processed, X_train.size(0), 100.*processed/X_train.size(0), lr, cur_loss))
            total_loss = 0
    return model


def evaluate():
    model.eval()    #评估模式 与下句组合使用
    with torch.no_grad():
        output = model(X_test)
        test_loss = F.mse_loss(output, Y_test)
        print('\nTest set: Average loss: {:.6f}\n'.format(test_loss.item()))
        return test_loss.item()

# for ep in range(1, epochs+1):
#     train(ep)
#     tloss = evaluate()
tlossy=100
#
flag=0

model = torch.load('Paper Model.tar')
season,targ_list, aim_list = PSF_Inputdate.novtarg_aim_listw9(dataname)

data_real = pd.read_csv(dataname, header=0, index_col=0, parse_dates=['date'])
dataATL = pd.read_csv(ATLabelname, header=0, index_col=0, parse_dates=['date'])
# data_real = data_real.drop(data_real.columns[3], axis=1)
data_real = data_real.drop(data_real.columns[4], axis=1)
data_real = data_real.drop(data_real.columns[4], axis=1)
data_real1 = data_real.copy()
data = data_real.copy()
data_real1 = data_real1.drop(data_real1.columns[0], axis=1)
# data_real['hour'] = sc_hour.fit_transform(data_real['hour'].values.reshape(-1, 1))
data_real['load'] = sc_load.transform(data_real['load'].values.reshape(-1, 1))
data_real['temp1'] = sc_temp.transform(data_real['temp1'].values.reshape(-1, 1))
data_real['temp2'] = sc_temp.transform(data_real['temp2'].values.reshape(-1, 1))
data_real['temp3'] = sc_temp.transform(data_real['temp3'].values.reshape(-1, 1))
data_real1['temp1'] = sc_temp.transform(data_real1['temp1'].values.reshape(-1, 1))
data_real1['temp2'] = sc_temp.transform(data_real1['temp2'].values.reshape(-1, 1))
data_real1['temp3'] = sc_temp.transform(data_real1['temp3'].values.reshape(-1, 1))
# data_real1['hour'] = sc_hour.fit_transform(data_real1['hour'].values.reshape(-1, 1))
# data_real = data_real.drop(data_real.columns[1], axis=1)
# data = data.drop(data.columns[1], axis=1)
data_index = data_real.loc['2019-01-01':'2020-12-31'].index.unique()
pred = np.array([])
true = np.array([])
mape_values = []
PSF1 = []
ok_list=[]
nopsf=0
for i, j in zip(targ_list, aim_list):
    # i = datetime.datetime.strptime(i, "%Y-%m-%d")
    try:
        ps_index = PSF.PSF_dayouttryw6T(psffilename, i, data_index)
        psfpre = PSF.PSF_result(data_real, dataname,j,ps_index)
        ok_list.append(j)
        jiwen = dataATL.loc[i + datetime.timedelta(days=-1), 'AcTemplabel']
        ilist1 = [jiwen] * 24
        jiwen = dataATL.loc[i, 'AcTemplabel']
        ilist2 = [jiwen] * 24
        ilist1 = sc_jiwen.fit_transform(np.array(ilist1).reshape(-1, 1))
        ilist2 = sc_jiwen.fit_transform(np.array(ilist2).reshape(-1, 1))
    # print('没有psf的天数有', nopsf, '天')
        psfpre = sc_load.transform(psfpre.reshape(-1,1))
    # predtemp1,predtemp2,predtemp3,predtemp4 = data_deltatemp(data_real, j)


        for_pred = np.column_stack((np.column_stack((np.column_stack((np.column_stack((data_real.loc[i],ilist1)),data_real1.loc[j])),psfpre)),ilist2)).reshape(-1,48,5)      #把前一天数据变成二维形式跑模型得到预测值preddata
        # for_pred = psfpre.reshape(-1,48)      #把前一天数据变成二维形式跑模型得到预测值preddata
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
current_time = ddtime.now().strftime('%Y-%m-%d~%H-%M-%S')
dict = {'spring':'May','summer':'Aug','autumn':'Nov','winter':'Feb'}

gggg = 'pre_'+season+ dict[season]+':'
print(gggg,np.mean(mape_values))

print(mape_values)
pd.DataFrame(mape_values, index=ok_list).to_csv('..\\Result\\mape'+season+ dict[season]+str(current_time)+'.csv')
pd.DataFrame(pred.reshape(-1,24), columns=[str(x) + 'h' for x in range(1,25)], index=aim_list).to_csv('..\\Result\\pre_'+season+ dict[season]+str(current_time)+'-'+str(np.mean(mape_values))+'.csv')