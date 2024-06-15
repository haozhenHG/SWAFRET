import pandas as pd
import numpy as np
from tslearn.clustering import KShape
import matplotlib.pyplot as plt
import math
import data_process

# --------------------此文件目的是最终生成聚类标签文件  即每天对应一个标签值

outputclucsv = '..\\CSVData\\cluster6LT.csv'  # 标签文件

num_cluster = 6  # 此值应是通过Kshape_silhouette_score.py得出最佳聚类数值
seed = 0

stack_data = data_process.data_timeAload()  # 获取预处理的数据  ：shape:(2447, 96, 1) size:234912 ndim:3

'''
# 保存到stack.csv文件下进行查看
arr = stack_data.reshape(stack_data.shape[0],-1)   # 转化为 二维数据  (2447, 96)
df = pd.DataFrame(arr)
df.to_csv(r'..\\CSVData\\stack_data.csv')
'''

np.random.seed(seed)

data = pd.read_csv('..\\CSVData\\datas.csv', header=0, index_col=0, parse_dates=['date'])

data_index = data.index.unique()  # 返回维一值   在索引中找到所有唯一值

# n_init  如何理解？
# 算法随机运行 n_init 次，最好的一个聚类结果做为最终结果
# 希望结果可以重现，固定random_state是非常重要的 设置random_state每次构建的模型是相同的、生成的数据集是相同的、每次的拆分结果也是相同的
# random_state=？看自己  本例中=0   以后每次运行结果都会复现
# 我们可以使用参数n_init来选择，每个随机数种子random_state下运行的次数。
ks = KShape(n_clusters= num_cluster,n_init= 5 ,verbose= True ,random_state=seed)

y_pred = ks.fit_predict(stack_data)  # 2447 个标签
# print('len y_pred ',y_pred.shape)    len y_pred  (2447,)
'''
# 查看ks_cluster_centers数据
arr = ks.cluster_centers_.reshape(ks.cluster_centers_.shape[0],-1)   # 转化为 二维数据  (6, 96)
df = pd.DataFrame(arr)
df.to_csv(r'..\\CSVData\\ks_cluster_centers_.csv')
'''

fin_cluster = pd.DataFrame({"id": data_index, "cluster_label": y_pred})
# print('fin_cluster:',fin_cluster)

fin_cluster.to_csv(outputclucsv)    # 保存聚类后的标签数据

# 5张图像
for yi in range(num_cluster):

    plt.subplot(math.ceil(num_cluster / 2), 2, yi + 1) # 计算需要多少行来放置所有的聚类图 2表示每行放置两个子图，yi + 1是当前聚类的子图位置。

    for xx in stack_data[y_pred == yi]:   #  len y_pred  (2447,)  stack_data(2447,96,1)
        plt.plot(xx.ravel(), "k-", alpha=.3) # xx.ravel()将数据点展平为一维数组

    plt.plot(ks.cluster_centers_[yi].ravel(), "r-")  # 6个聚类中心 ks.cluster_centers_ (6,96,1) # 绘制当前聚类的中心
    # plt.text()函数用于设置文字说明  transform就是转换的意思，是不同坐标系的转换 左边距离横坐标轴长的0.55倍，下面距离纵坐标轴的0.85倍
    # transform=plt.gca().transAxes表示文本位置是相对于子图轴的，这样可以确保文本位置在子图之间保持一致。
    plt.text(0.55, 0.85, 'Cluster %d' % (yi + 1),
             transform=plt.gca().transAxes) # plt.gca( )挪动坐标轴 gca就是get current axes的意思

    if yi == 1:
        plt.title("SBD" + "  $k$-shape")


plt.tight_layout()
plt.show()