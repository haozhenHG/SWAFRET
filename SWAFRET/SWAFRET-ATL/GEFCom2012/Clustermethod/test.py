import numpy as np
import pandas as pd

labelset = '..\\CSVData\\weekday_season.csv'  # 源数据集

datname = '..\\CSVData\\datas.csv'  # 源数据集

label = pd.read_csv(labelset, header=0, index_col=0, parse_dates=['date'])
data = pd.read_csv(datname, header=0, index_col=0, parse_dates=['date'])

labelnew = pd.DataFrame(np.repeat(label.values,24,axis=0)) # 5

data['weekday'] = labelnew[:][0].values
data['season'] = labelnew[:][1].values
data.to_csv('..\\CSVData\\datas.csv')