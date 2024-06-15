#通过设置滑动窗口进行序列匹配寻找合适的标签值
import pandas as pd
import datetime
import numpy as np
import os

from matplotlib import pyplot as plt

# from American.Readcsv import read_labelcsv, read_27col_featurecsv, read_weekday_seasoncsv


def Average_Relative_Error(true_values, predicted_values):
    """
    计算真实值和预测值之间的平均相对误  。

    参数:
    true_values -- 真实值列表
    predicted_values -- 预测值列表

    返回:
    average_relative_error -- 平均相对误差
    """
    # 检查列表长度是否相同
    if len(true_values) != len(predicted_values):
        raise ValueError("真实值和预测值的数量必须相同")

    # 计算每个数据点的相对误差
    relative_errors = []
    for X, Y in zip(true_values, predicted_values):
        if X == 0:
            # 如果真实值为0，相对误差未定义，可以选择将其视为0或者跳过
            relativeerror = 0
        else:
            relativeerror = np.abs(Y - X) / X * 100
        relative_errors.append(relativeerror)

    # 计算平均相对误差
    average_relative_error = sum(relative_errors) / len(relative_errors)

    return average_relative_error

# 设置软窗口
def SetsoftWindow(SW_te):
    Soft_W = [ [] for _ in range(len(SW_te)) ]
    for i, sublist in enumerate(Soft_W):
        sublist.extend([ SW_te[i] + x for x in range(-1,2) ])
    return Soft_W

# 匹配  判断是否在软窗口内
def SelectEs_d(SW_tr,Soft_W):
    Is_true=[]
    for i, sublist in enumerate(Soft_W):
        if SW_tr[i] in sublist:
            Is_true.append(True)
        else:
            Is_true.append(False)
    return Is_true


def PSF(Ydat, traindata, label, w, testset):
    '''
    :param Ydat:        只有weekday season 两列的数据集
    :param traindata:   要训练的数据集
    :param label:       标签所有数据集
    :param w:           滑动窗口
    :param testset:     测试集
    :return:
    '''

    ES_d = []
    MER = []
    # First_label  # 通过w窗口第一匹配到的数据记录下来  作为预测数据的标签
    for testday in testset.index:  # 测试集里看数据
        print('now testday is :', testday)
        print('now window is :', w)

        # 记录两个序列  一个用于测试集  一个用于训练集
        SW_te = []
        SW_tr = []

        # 窗口w已知  则通过要预测的数据标签集  去训练集   查找类似标签
        # 要生成w个标签组成的集合
        for i in range(0, w):
            # print(testday + datetime.timedelta(days=-(i + 1)))
            # print(label.loc[testday + datetime.timedelta(days=-(i + 1)), 'label'])
            la = label.loc[testday + datetime.timedelta(days=-(i + 1)), 'label']
            # print('la:', la)
            SW_te.append(la)
        SW_te.reverse()  # 逆序输出  ----> 标签：[d-w  ... d-1 ]  d   =====  w个标签
        print('SW_te:', SW_te)

        # 生成软窗口序列    为了更灵活的匹配数据
        Soft_W = SetsoftWindow(SW_te)

        for trainday in traindata.index:  # 训练集里遍历查找标签相同的
            if trainday + datetime.timedelta(days=w) in traindata.index:  # 最大窗口范围日期必须在源数据集里
                for i in range(0, w):
                    SW_tr = [label.loc[trainday + datetime.timedelta(days=i), 'label'] for i in range(0, w)]  # w个标签

                # 判断是否是软窗口内的数值
                Is_true = SelectEs_d(SW_tr, Soft_W)

                # 每个位置必须符合
                if all(Is_true):  # 则增加 weekday = [] 和 season 特征特征 进行权重赋值
                    ES_d.append(trainday + datetime.timedelta(days=w))  # 记录相似序列后的标签 即真正意义上的标签

        weekday = []
        season = []
        yuan = []
        # 对 ES_d记录的数据 进一步进行相关性判断  记录的数据都在traindata里找
        if len(ES_d) > 1:
            for n in ES_d:
                # 下面两个if语句 将更符合条件的数据筛选出来  精简数据
                if Ydat.loc[n]['weekday'] == Ydat.loc[testday]['weekday'] and Ydat.loc[n]['season'] == \
                        Ydat.loc[testday]['season']:
                    yuan.append(n)  # 强相关  寻找的数据 与 预测的数据  weekday  season 一样
                elif Ydat.loc[n]['weekday'] == Ydat.loc[testday]['weekday'] and Ydat.loc[n]['season'] != \
                        Ydat.loc[testday]['season']:
                    weekday.append(n)  # 星期相关
                else:
                    season.append(n)  # 季节相关
        else:
            ES_d = ES_d

        print('ES_d length:%s  yuan  length:%s weekday length:%s season length:%s' % (
        len(ES_d), len(yuan), len(weekday), len(season)))

        sum_yuan = traindata.loc[yuan, :].sum()
        sum_weekday = traindata.loc[weekday, :].sum()
        sum_season = traindata.loc[season, :].sum()

        y_true = testset.loc[testday]['load1':'load24'].values  # 真实值
        # 存在相关日期的集合是进行一切操作的前提
        if len(ES_d):
            if len(yuan):  # 强相关数据存在
                y_pred = sum_yuan / len(yuan)
                # print('y_pred:\n',y_pred['load1':'load24'])
            else:  # 不存在强相关数据
                if len(weekday) != 0 and len(season) != 0:
                    y_pred = (sum_weekday / len(weekday)) * 0.6 + (sum_season / len(season)) * 0.4
                elif len(weekday) != 0:
                    y_pred = sum_weekday / len(weekday)
                else:
                    y_pred = sum_season / len(season)
            mse = Average_Relative_Error(y_true, y_pred['load1':'load24'].values)
        else:
            mse = -1

        MER.append(mse)
        # big_MER.append(MER)

    return MER

# if __name__ == "__main__":
#     # 打印当前工作目录
#     # print("Current working directory:", os.getcwd())
#     dat = read_27col_featurecsv()
#     label = read_labelcsv()
#     wsdata = read_weekday_seasoncsv()
#     # 分割数据
#     traindata = dat[:'2021-04-30'] # (2130, 27)    [2015-07-02  2021-04-30]
#     trainlabel = label[:'2021-04-30'] # (2130, 1)
#     # - 测试数据
#     testdata= dat['2021-05-01':'2021-05-31'] # (31, 27)
#     testlabel = label['2021-05-01':'2021-05-31']  # (31, 1)
#     # print(traindata.shape,trainlabel.shape,testdata.shape,testlabel.shape)
#
#     # 选择最优的w   根据结果最后选择6
#     errors = []
#     for i in range(2, 9):
#         MER = PSF(Ydat=wsdata, traindata=traindata, label=label, w=i, testset=testdata)
#         errors.append(MER)
#
#     # 创建一个新的图表
#     plt.figure(figsize=(15, 6))
#
#     # 使用for循环遍历数据并绘制每条线
#     for i, sublist in enumerate(errors):
#         plt.plot(testdata.index, sublist, label='w is ' + str(i + 2))
#
#     plt.xlabel('Window')
#     plt.xticks(testdata.index, rotation=90)  # 旋转 度
#     plt.ylabel('error')
#     # 添加图例
#     plt.legend()
#     plt.savefig('..\\CSVData\\MER_2_9.jpg')
#     # 显示图表
#     plt.show()