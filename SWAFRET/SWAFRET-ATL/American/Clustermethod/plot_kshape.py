# -*- coding: utf-8 -*-
"""
KShape
======

This example uses the KShape clustering method [1] that is based on
cross-correlation to cluster time series.


[1] J. Paparrizos & L. Gravano. k-Shape: Efficient and Accurate Clustering \
of Time Series. SIGMOD 2015. pp. 1855-1870.
"""

# Author: Romain Tavenard
# License: BSD 3 clause

# -----------------------------官方示例代码--------------------------------------

import numpy
import matplotlib.pyplot as plt

from tslearn.clustering import KShape
from tslearn.datasets import CachedDatasets
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

seed = 0
numpy.random.seed(seed)
X_train, y_train, X_test, y_test = CachedDatasets().load_dataset("Trace")
print('X_train shape:%s y_train shape:%s  X_test shape:%s y_test shape:%s '%(X_train.shape ,y_train.shape, X_test.shape ,y_test.shape))
# X_train shape:(100, 275, 1) y_train shape:(100,)  X_test shape:(100, 275, 1) y_test shape:(100,)


# Keep first 3 classes and 50 first time series
X_train = X_train[y_train < 4]

print('X_train[y_train < 4]:',X_train.shape)  # (69, 275, 1)  有69条符合条件数据
'''
y_train < 4  返回一个列表 (100,)   [True/False  True/False  ....   True/False  ...True/False]

X_train[True/False  True/False  ....   True/False  ...True/False]  获得对应的数据
'''

X_train = X_train[:50]   # (69, 275, 1)  有69条符合条件数据  选择50条数据

numpy.random.shuffle(X_train)  # 打乱顺序 X_train   多维矩阵中，只对第一维（行）做打乱顺序操作
# print(X_train[:1,:20,:])
# For this method to operate properly, prior scaling is required
X_train = TimeSeriesScalerMeanVariance().fit_transform(X_train)
print('X_train fit_transform shape: ',X_train.shape)

sz = X_train.shape[1] # 取第二维  275

# kShape clustering    KShape类的实例化  使用它需要实例化
ks = KShape(n_clusters=3, verbose=True, random_state=seed)   # n_init 默认是 10
# print('type(ks):',type(ks))
# print(ks)  # 返回的是Kshape  类型
# print('shape:%s size:%s ndim:%s:'% (ks.shape,ks.size,ks.ndim))   这句报错  因为kashpe类型

# fit_predict() 返回每个数据对应的标签，并将标签值对应到相应的簇
# 计算簇的中心并且预测每个样本对应的簇类别，相当于先调用fit(X)再调用predict(X)，提倡这种方法，返回labels标签（0,1,2……）
y_pred = ks.fit_predict(X_train)
# X_train = X_train[:50]  所以 y_pred 50个标签  (50,1)
print('y_pred:%s  len(y_pred):%s y_pred shape:%s '%(y_pred,len(y_pred),type(y_pred)))
'''
y_pred:n_clusters=3
[2 0 1 1 1 1 1 0 2 2 0 0 1 0 2 2 0 0 2 0 2 1 2 2 0 0 0 1 0 1 2 0 1 1 0 2 2 1 1 1 0 2 2 0 0 1 2 2 0 2]  标签
y_pred:n_clusters=4
[2 1 3 3 3 3 3 0 1 1 0 0 3 1 2 2 0 0 1 1 2 3 1 1 0 0 0 3 0 3 2 0 3 3 0 2 23 3 3 0 1 1 0 0 3 1 1 0 2]  标签

'''

#  len(y_pred):50 y_pred shape:<class 'numpy.ndarray'>
plt.figure()

for yi in range(3):
    # 表示将整个图像窗口分为3行1列   当前位置为1 + yi.
    plt.subplot(3, 1, 1 + yi)  #定图像位置 直接指定划分方式和位置进行绘图

    for xx in X_train[y_pred == yi]:
        plt.plot(xx.ravel(), "k-", alpha=.2)  #  #用ravel()方法将数组a拉成一维数组  ravel函数的作用是让多维数组变成一维数组 alpha透明度

    # print('ks.cluster_centers_[yi]:',ks.cluster_centers_[yi])  ks.cluster_centers_ (3,275,1)

    # 源码：self.cluster_centers_ = TimeSeriesScalerMeanVariance(mu=0., std=1.).fit_transform(self.cluster_centers_)

    plt.plot(ks.cluster_centers_[yi].ravel(), "r-")  #   输出  cluster_centers_  个类的聚类中心

    '''
    ks.cluster_centers_   为什么是 (3,275,1) 三维数组
    print('ks.cluster_centers_[yi]:',ks.cluster_centers_[yi].shape)
    ks.cluster_centers_[yi]: (275, 1)
    ks.cluster_centers_[yi]: (275, 1)
    ks.cluster_centers_[yi]: (275, 1)

    '''
    # 定 x y 轴
    plt.xlim(0, sz)   # 0-275
    plt.ylim(-4, 4)
    # 定标题
    plt.title("Cluster %d" % (yi + 1))

plt.tight_layout()  # 简易的调整多图、单图标签  避免标题覆盖等问题
plt.show()
