import numpy as np
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.clustering import KShape


# -------------学习理解  Kshape方法  ------------
# 示例数据
data = np.array([
    [1, 2, 3, 1, 2, 3],
    [2, 3, 4, 2, 3, 4],
    [1, 1, 1, 2, 2, 2],
    [3, 3, 3, 4, 4, 4]
])
print('data shape:%s size:%s ndim:%s:'% (data.shape,data.size,data.ndim))  # 二维数组

data = data.reshape(-1,6,1) # 将数据重塑为 (n_samples, n_timesteps, n_features) 格式   三维数组

# print(data) 查看 reshape 的数据
print('reshape data shape:%s size:%s ndim:%s:'% (data.shape,data.size,data.ndim))

data_normal = (data - np.mean(data)) / np.std(data)
# print('标准化后: \n', data_normal)

# 数据预处理：进行 z-normalization，使得每个时间序列具有相同的均值和方差
data_scaled = TimeSeriesScalerMeanVariance(mu=0.0, std=1.0).fit_transform(data)
# data_scaled1 = TimeSerie
# sScalerMeanVariance().fit_transform(data)  本例中mu=0.0, std=1.0 有无结果一样
print("使用 fit_transform 方法标准化的结果：\n", data_scaled)


# 使用 K-Shape 聚类
'''
n_clusters : Number of clusters to form
verbose : 是否打印有关惯性的信息
n_init :使用不同质心种子运行k-Shape算法的时间。最终结果将是n_init连续运行在惯性方面的最佳输出。
'''
# n_clusters = 2
'''
n_clusters: 即我们的k值，一般需要多试一些值以获得较好的聚类效果
n_init : 默认是10，一般不需要改。
'''
kshape_model = KShape(n_clusters=2, n_init=10, verbose=True, random_state=42)

labels = kshape_model.fit_predict(data_scaled)
print('cluster: ',kshape_model.cluster_centers_.shape)  # n_clusters=3和cluster_centers_ 相关的 cluster:  (3, 6, 1)
'''
n_clusters=1 
labels 聚类标签： [0 0 0 0]
n_clusters=2 
聚类标签： [0 0 1 1]
n_clusters=3
聚类标签： [0 0 1 1]
n_clusters 设置大了  但是标签不变？
'''
# 输出聚类结果
print("聚类标签：", labels)