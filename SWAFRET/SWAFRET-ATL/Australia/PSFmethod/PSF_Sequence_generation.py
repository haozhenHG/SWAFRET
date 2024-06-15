import datetime
import numpy as np
import pandas as pd
def read_48_columns_featurecsv():
    wsname = '..\\CSVData\\48_columns_feature.csv'
    feadata = pd.read_csv(wsname, header=0, index_col=0, parse_dates=['date'])

    return  feadata
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
    true_values = np.array(true_values)
    predicted_values = np.array(predicted_values)
    # 计算每个数据点的相对误差
    mapes = np.mean(np.abs(true_values - predicted_values) / true_values) * 100

    return mapes



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


def Data_filtering(ES_d_COPY,data, true_values, num=2):
    ES_d = []
    if num == 2:
        for date in ES_d_COPY:
            predicted_values = data.loc[date]['load1':'load48'].values
            mape = Average_Relative_Error(true_values, predicted_values)
            if mape < 4.5:
                ES_d.append(date)
    if num == 3:
        for date in ES_d_COPY:
            predicted_values = data.loc[date]['load1':'load48'].values
            mape = Average_Relative_Error(true_values, predicted_values)
            if mape < 4:
                ES_d.append(date)
    if num == 4:
        for date in ES_d_COPY:
            predicted_values = data.loc[date]['load1':'load48'].values
            mape = Average_Relative_Error(true_values, predicted_values)
            if mape <3.5:
                ES_d.append(date)
    if num == 5:
        for date in ES_d_COPY:
            predicted_values = data.loc[date]['load1':'load48'].values
            mape = Average_Relative_Error(true_values, predicted_values)
            if mape <3.8:
                ES_d.append(date)
    if num == 6:
        for date in ES_d_COPY:
            predicted_values = data.loc[date]['load1':'load48'].values
            mape = Average_Relative_Error(true_values, predicted_values)
            if mape < 3.5:
                ES_d.append(date)
    if num == 7:
        for date in ES_d_COPY:
            predicted_values = data.loc[date]['load1':'load48'].values
            mape = Average_Relative_Error(true_values, predicted_values)
            if mape < 2.5:
                ES_d.append(date)
    if num == 8:
        for date in ES_d_COPY:
            predicted_values = data.loc[date]['load1':'load48'].values
            mape = Average_Relative_Error(true_values, predicted_values)
            if mape < 2:
                ES_d.append(date)
    if num == 9:
        for date in ES_d_COPY:
            ES_d.append(date)
    return ES_d,num

def PSF_Sequence(Ydat, traindata, label, w, testday):
    '''
    :param Ydat:        只有weekday season 两列的数据集
    :param traindata:   要训练的数据集
    :param label:       标签所有数据集
    :param w:           滑动窗口
    :param testset:     测试集
    :return:
    '''
    traindata = traindata.loc['2007-01-01':'2010-01-31']
    if w < 1:
        return 0

    ES_d = []
    num=2
    # print('now testday is :', testday)
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

    dataindex = traindata.index.unique()
    for trainday in dataindex:  # 训练集里遍历查找标签相同的
        if trainday + datetime.timedelta(days=w) in dataindex:  # 最大窗口范围日期必须在源数据集里
            for i in range(0, w):
                SW_tr = [label.loc[trainday + datetime.timedelta(days=i), 'label'] for i in range(0, w)]  # w个标签

            # 判断是否是软窗口内的数值
            Is_true = SelectEs_d(SW_tr, Soft_W)

            # 每个位置必须符合
            if all(Is_true):  # 则增加 weekday = [] 和 season 特征特征 进行权重赋值
                ES_d.append(trainday + datetime.timedelta(days=w))  # 记录相似序列后的标签 即真正意义上的标签

    print('current ES_d length:',len(ES_d))

    ES_d_COPY = ES_d
    data = read_48_columns_featurecsv()
    true_values = data.loc[testday]['load1':'load48'].values  # 真实值    # 根据一天的长度  要改 这是48

    if len(ES_d) >= 5:
        ES_d = []
        for date in ES_d_COPY:
            predicted_values = data.loc[date]['load1':'load48'].values  # 根据一天的长度  要改 这是48
            mape = Average_Relative_Error(true_values, predicted_values)
            print('compare - mape : ',mape)
            if mape < 3: # 6->8
                ES_d.append(date)
            elif w == 1 and mape < 3.3:
                ES_d.append(date)
            elif w == 1 and mape < 3.5:
                ES_d.append(date)
            # elif w == 1 and mape < 6:
            #     ES_d.append(date)

        print('now ES_d length:', len(ES_d))
    else:
        print('now ES_d length:', len(ES_d))

    # 初始 num=2
    while(len(ES_d)>=10):
        print('num:', num)
        if num >= 9:
            break
        # elif num >= 5 and len(ES_d)<=20:
        #     break
        else:

            ES_d,num = Data_filtering(ES_d,data,true_values, num)
            num = num +1

        print('while  ES_d length:', len(ES_d))

    weekday = []
    season = []
    yuan = []
    no_fea = []

    Strong_cor_value = []
    weekday_cor_value = []
    season_cor_value = []
    # 对 ES_d记录的数据 进一步进行相关性判断  记录的数据都在traindata里找
    if len(ES_d) > 0:
        for n in ES_d:
            # 下面两个if语句 将更符合条件的数据筛选出来  精简数据
            if Ydat.loc[n]['weekday'] == Ydat.loc[testday]['weekday'] and Ydat.loc[n]['season'] == \
                    Ydat.loc[testday]['season']:
                # onedayload = traindata.loc[n, 'load'].values.reshape(24, 1) # 24小时负载
                # Strong_cor_value.append(onedayload)
                # print('Strong_cor_value:',Strong_cor_value)
                # print(onedayload,onedayload.shape)  # (24, 1)
                yuan.append(n)  # 强相关  寻找的数据 与 预测的数据  weekday  season 一样
            elif Ydat.loc[n]['weekday'] == Ydat.loc[testday]['weekday'] and Ydat.loc[n]['season'] != \
                    Ydat.loc[testday]['season']:

                weekday.append(n)  # 星期相关
            elif Ydat.loc[n]['weekday'] != Ydat.loc[testday]['weekday'] and Ydat.loc[n]['season'] == \
                    Ydat.loc[testday]['season']:
                season.append(n)  # 季节相关
            else:
                no_fea.append(n)  # 都不相关

        print('yuanlength:%s weekdaylength:%s seasonlength:%s no_fealength:%s' % ( \
                len(yuan), len(weekday), len(season), len(no_fea)))

        sum_yuan = traindata.loc[yuan, 'load'].values.reshape(-1, 48)  # shape :(len(yuan),24)
        sum_weekday = traindata.loc[weekday, 'load'].values.reshape(-1, 48)
        sum_season = traindata.loc[season, 'load'].values.reshape(-1, 48)
        sum_no_fea = traindata.loc[no_fea, 'load'].values.reshape(-1, 48)

        # 存在相关日期的集合是进行一切操作的前提

        if len(yuan):  # 强相关数据存在
            y_pred = sum(sum_yuan) / len(yuan)
            # print('y_pred:\n',y_pred['load1':'load24'])
        else:  # 不存在强相关数据  根据随机森林了解到 season 特征比重更大
            if len(weekday) != 0 and len(season) != 0:
                y_pred = (sum(sum_weekday) / len(weekday)) * 0.5 + (sum(sum_season) / len(season)) * 0.5
            elif len(weekday) != 0 and len(season) == 0:
                y_pred = sum(sum_weekday) / len(weekday) * 0.9
            elif len(weekday) == 0 and len(season) != 0:
                y_pred = sum(sum_season) / len(season)   * 0.8
            elif len(sum_no_fea) !=0:
                y_pred = (sum(sum_no_fea) / len(no_fea)) * 0.85
            else:
                w = w - 1
                if w < 1:
                    raise ValueError("参数w必须大于或等于1")
                    return 1
                else:
                    y_pred = PSF_Sequence(Ydat, traindata, label, w, testday=testday)
    else:
        w = w - 1
        if w < 1:
            raise ValueError("参数w必须大于或等于1")
            return 1
        else:
            y_pred = PSF_Sequence(Ydat, traindata, label, w, testday=testday)

    y_pred = y_pred.reshape(-1,1)
    return y_pred