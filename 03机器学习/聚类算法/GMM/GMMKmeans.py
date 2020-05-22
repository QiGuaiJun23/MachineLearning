from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import numpy as np
def datasets():
	"""
	使用datasets包产生一些数据
	:return:
	"""
	plt.rcParams['axes.unicode_minus'] = False#解决不显示负数问题
	X,y_true = make_blobs(n_samples=800,centers=4,random_state=11)
	plt.scatter(X[:,0],X[:,1])
	plt.show()
	rng = np.random.RandomState(13)
	Y = np.dot(X,rng.randn(2,2))
	plt.scatter(Y[:,0],Y[:,1])
	plt.show()
	return X,Y

def GmmKmean(data,dataY):
	"""
	GMM算法与Kmeans算法对比
	:return:
	"""
	kmeans = KMeans(n_clusters=4)
	kmeans.fit(data)
	y_kmeans = kmeans.predict(data)

	plt.scatter(data[:,0],data[:,1],c=y_kmeans,s=50,cmap='viridis')
	plt.show()
	centers = kmeans.cluster_centers_
	print(centers)

	gmm = GaussianMixture(n_components=4,random_state=1)
	gmm.fit(data)
	labels = gmm.predict(data)
	plt.scatter(data[:,0],data[:,1],c=labels,s=40,cmap='viridis')
	plt.show()

	kmeansy = KMeans(n_clusters=4,random_state=1)
	kmeansy.fit(dataY)
	datay_kmeans = kmeansy.predict(dataY)
	plt.scatter(dataY[:,0],dataY[:,1],c=datay_kmeans,s=40,cmap='viridis')
	plt.show()

	# gmmY = GaussianMixture(n_components=4)
	gmm.fit(dataY)
	labelsY = gmm.predict(dataY)
	plt.scatter(dataY[:,0],dataY[:,1],c=labelsY,s=40,cmap='viridis')
	plt.show()

	return None



if __name__ == "__main__":
	data,dataY = datasets()

	GmmKmean(data,dataY)