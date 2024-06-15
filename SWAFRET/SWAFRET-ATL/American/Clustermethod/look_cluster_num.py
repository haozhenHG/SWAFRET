import numpy as np
import glob
from tslearn.clustering import KShape
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
import matplotlib.pyplot as plt
from tslearn.generators import random_walks
import tslearn.metrics as metrics
# 自定义数据处理
import data_process

#  --------------------寻找最佳的cluster数值  此代码结果数据  以look_cluster_num.png的形式存放在CSVData文件夹下
# 时间过长  不要运行


# 加载数据
stack_data = data_process.data_timeAload()
X = TimeSeriesScalerMeanVariance(mu=0.0, std=1.0).fit_transform(stack_data)

seed = 0

plt.figure()

# elbow = 4（elbow法则寻找最佳聚类个数）
def test_elbow():
    global X, seed
    distortions = []
    # X = TimeSeriesScalerMeanVariance(mu=0.0, std=1.0).fit_transform(X)
    # 1 to 10 clusters
    for i in range(2, 21):
        ks = KShape(n_clusters=i, n_init=5, verbose=True, random_state=seed)
        # Perform clustering calculation
        ks.fit(X)
        # ks.fit will give you ks.inertia_ You can
        # inertia_
        distortions.append(ks.inertia_)
    plt.plot(range(2, 21), distortions, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    plt.show()


if __name__ == '__main__':
    test_elbow()