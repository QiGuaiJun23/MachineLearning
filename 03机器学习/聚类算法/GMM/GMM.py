import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
def dataset():
	"""
	读取数据并处理
	:return:
	"""
	data = pd.read_csv('../../../数据集/机器学习/EM算法/FremontHourly.csv',index_col='Date',parse_dates=True)
	# print(data.head(30))
	# data.plot()
	# plt.show()
	# #在时间上进行重采样
	# data.resample('w').sum().plot()
	# data.resample('D').sum().rolling(365).sum().plot()
	# plt.show()
	# data.groupby(data.index.time).mean().plot()
	# plt.xticks(rotation=45)
	# plt.show()
	data.columns=['East','West']
	data['Total']=data['East']+data['West']
	pivoted = data.pivot_table('Total',index=data.index.time,columns=data.index.date)
	print(pivoted.iloc[:5,:5])
	pivoted.plot(legend=False,alpha=0.01)
	plt.xticks(rotation=45)
	plt.show()
	# print(pivoted.shape)
	X = pivoted.fillna(0).T.values
	print(X.shape)
	#PCA降维成2维
	pca = PCA(n_components=2)
	Y = pca.fit_transform(X)
	# print(Y.shape)
	plt.scatter(Y[:,0],Y[:,1])
	plt.show()

	return Y,pivoted



def gmm(data,pivoted):
	"""
	利用GMM聚类算法进行聚类
	:return:
	"""
	gmm = GaussianMixture(n_components=2)#两种高斯分布
	gmm.fit(data)
	labels = gmm.predict_proba(data)
	x = gmm.predict(data)
	print(labels)
	plt.scatter(data[:,0],data[:,1],c=x,cmap='rainbow')
	plt.show()
	fig, ax = plt.subplots(1, 2, figsize=(14, 6))
	pivoted.T[labels == 0].T.plot(legend = False,alpha=0.1,ax=ax[0])
	pivoted.T[labels == 1].T.plot(legend=False, alpha=0.1, ax=ax[1])
	ax[0].set_title('Purple Cluster')
	ax[1].set_title('Red Cluster')
	plt.show()
	return None


if __name__=="__main__":
	data,pivoted = dataset()
	gmm(data,pivoted)