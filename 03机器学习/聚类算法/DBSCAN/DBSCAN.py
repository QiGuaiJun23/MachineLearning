from pandas.plotting import scatter_matrix
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score
def dbscan():
	"""
	使用DBSCAN算法对啤酒进行聚类
	:return:
	"""
	#读取数据集
	data = pd.read_table('../../../数据集/机器学习/聚类算法/啤酒成分和价格/beer.txt', sep=' ', error_bad_lines=False)
	# print(data)

	#提取特征值
	X = data.loc[:,['calories','sodium','alcohol','cost']]


	db = DBSCAN(eps=20,min_samples=2).fit(X)
	labels =db.labels_
	data['cluster_db'] = labels
	colors = np.array(['red', 'green', 'blue', 'yellow'])
	print(data.sort_values('cluster_db'))

	print(data.groupby('cluster_db').mean())

	scatter_matrix(data[['calories', 'sodium', 'alcohol', 'cost']], s=100, alpha=1, c=colors[data["cluster_db"]],
	               figsize=(15, 15))
	# plt.show()

	sht = silhouette_score(X, data.cluster_db)
	print(sht)
	return None

if __name__ == "__main__":
	dbscan()