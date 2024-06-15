import math
import numpy as np
import matplotlib.pyplot as plt
from tslearn.clustering import KShape
from tslearn.clustering import silhouette_score  # 轮廓系数
import data_process
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

# ---此代码应该是判断cluster数值的  获得cluster最优解
# CSDN： https://blog.csdn.net/qq_37960007/article/details/107937212

# datas:'..\\CSVData\\datas.csv
stack_data = data_process.data_timeAload()
stack_data = TimeSeriesScalerMeanVariance(mu=0.0, std=1.0).fit_transform(stack_data)
seed = 0
np.random.seed(seed)


sil_score = []
plt.figure()
# 测试num_cluster  数值 判断效果
for i in range(2,21):  # 为什么21？
    num_cluster = i

    # 为什么 n_init= 5 ？ 依据？
    ks = KShape(n_clusters= num_cluster ,n_init= 5 ,verbose= True ,random_state=seed)   # --->

    y_pred = ks.fit_predict(stack_data)
    dists = ks._my_cross_dist(stack_data)
    # dists = ks._cross_dists(stack_data)

    print('dists:',dists)

    # #这步重要，由于计算出的距离矩阵只是负8次方，还不是真正的零，得替换，否则会报错
    np.fill_diagonal(dists,0)

    # 师兄源码源码 score = silhouette_score(dists,y_pred,metric='precomputed')
    # #计算轮廓系数
    score = silhouette_score(dists, y_pred)
    sil_score.append(score)

    print(stack_data.shape)
    print(y_pred.shape)

    print("silhouette_score: " + str(score))
    for yi in range(num_cluster):
        plt.subplot(math.ceil(num_cluster/2), 2, yi + 1)
        for xx in stack_data[y_pred == yi]:
            plt.plot(xx.ravel(), "k-", alpha=.3)
        plt.plot(ks.cluster_centers_[yi].ravel(), "r-")
        plt.text(0.55, 0.85,'Cluster %d' % (yi + 1),
                transform=plt.gca().transAxes)
        if yi == 1:
            plt.title("SBD" + "  $k$-shape")

    plt.tight_layout()
    plt.show()


plt.plot(range(2,21), sil_score, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('silhouette_score')
plt.savefig('.\\CSVData\\Kshape2-21-silscrealrealonlyload.jpg')
plt.show()