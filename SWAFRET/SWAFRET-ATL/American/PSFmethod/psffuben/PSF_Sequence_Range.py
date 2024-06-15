import pandas as pd
import datetime


def get_18_0101_1231(data):
    borrow_date_list = []
    psf_date_list = []

    # 要生成PSF序列的数据
    data = data.loc['2018-01-01':'2018-12-31']
    dataindex = data.index.unique()  # 取不重复的所有日期

    for i in dataindex:
        # x = datetime.datetime.strptime(i, "%Y-%m-%d")
        # timedelta类：主要用于做时间加减的
        borrow_date = i+datetime.timedelta(days=-1)
        borrow_date_list.append(borrow_date)
        psf_date_list.append(i)

    return borrow_date_list, psf_date_list


