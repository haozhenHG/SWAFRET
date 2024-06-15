import numpy as np
import pandas as pd


data = pd.read_csv('..\\CSVData\\cluster6LT.csv', header=0, index_col=0)

print(data['cluster_label'].value_counts())
'''
cluster_label
1    1082
5     465
0     419
2     271
4     116
3      94
'''