# 本代码 用来进行特征重要性分析

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import data_process
import seaborn as sns

dataset = '..\\CSVData\\datas.csv'  # 源数据集
data = pd.read_csv(dataset, header=0, index_col=0, parse_dates=['date'])
# print(data.values.shape)


data  = data.iloc[:,1:] #列上删除
print(data)


labelset = '..\\CSVData\\label.csv'  # 源数据集
label = pd.read_csv(labelset, header=0, index_col=0, parse_dates=['date'])

labelnew = pd.DataFrame(np.repeat(label.values,24,axis=0))
labelnew.columns = label.columns
print(labelnew.size)
feat_labels = data.columns[:]
print(feat_labels) # Index(['temp1', 'temp2', 'temp3', 'weekday', 'season'], dtype='object')
#
#
# 初始化随机森林模型
# 注意：虽然我们用RandomForestClassifier，但实际上我们并不需要它进行分类
# 我们只是用它来构建随机森林，并获取特征重要性
rf = RandomForestClassifier(n_estimators=50, random_state=42)

# 训练模型（即使没有标签，我们仍然可以“训练”模型）
# 在训练过程中，模型会评估特征的重要性
rf.fit(data.values,labelnew.values.ravel())
# 获取特征重要性
feature_importances = rf.feature_importances_
# 将特征重要性排序，并获取排序后的索引
indices = np.argsort(feature_importances)[::-1]

for f in range(data.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], feature_importances[indices[f]]))

# 创建特征重要性的dataframe
importance_df = pd.DataFrame({'Feature': feat_labels, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.savefig('..\\CSVData\\Feature_Importance.jpg')
plt.show()
