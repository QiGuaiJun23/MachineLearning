import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
def kmeans():
	"""
	Kmeams聚类分析啤酒
	:return:
	"""
	#读取数据
	data = pd.read_table('../../../数据集/机器学习/聚类算法/啤酒成分和价格/beer.txt',sep=' ',error_bad_lines=False)
	# print(data)

	#提取特征值
	X = data.loc[:,['calories','sodium','alcohol','cost']]
	# print(x)

	#使用Kmeans聚类
	kmean = KMeans(n_clusters=3)
	km = kmean.fit(X)
	data['cluster'] = km.labels_

	# cluster_centers = km.cluster_centers_
	# print(data.groupby("cluster").mean())#计算每个分类的均值
	# print(cluster_centers)
	# print(km.labels_)
	print(data)

	# centers = data.groupby("cluster").mean().reset_index()
	# # print(centers)
	# # 画图，四个特征量量比较
	# plt.rcParams['font.size'] = 14
	# colors = np.array(['red','green','blue','yellow'])
	# plt.scatter(data['calories'],data['alcohol'],c=colors[data["cluster"]])
	# plt.scatter(centers.calories,centers.alcohol,linewidths=3,marker='+',s=300,c='black')#中心点
	# plt.xlabel("Calories")
	# plt.ylabel("Alcohol")
	# #
	# scatter_matrix(data[['calories','sodium','alcohol','cost']],s=100,alpha=1,c = colors[data["cluster"]],figsize=(15,15))
	# plt.suptitle("初始化成3类")
	# plt.show()
	#
	#我们进行标准化看看效果
	# std = StandardScaler()
	# x = std.fit_transform(X)
	# # print(x)
	# kms = kmean.fit(x)
	# data["cluster_std"] = kms.labels_
	# print(data)
	#使用轮廓系数进行评估
	# sht = silhouette_score(X,data.cluster)  #没有进行标准化的评估值
	# sht2 = silhouette_score(x,data.cluster_std)
	# print(sht)
	# print(sht2)
	# scores = []
	# for i in range(2,19):
	# 	labels = KMeans(n_clusters=i).fit(X).labels_
	# 	score = silhouette_score(X,labels)
	# 	scores.append(score)
	# print(scores)
	#
	# plt.plot(list(range(2,19)),scores)
	# plt.xlabel("Number of Clusters Initialized")
	# plt.ylabel("Sihouette Score")
	# plt.show()
	return None

if __name__ == "__main__":
	kmeans()