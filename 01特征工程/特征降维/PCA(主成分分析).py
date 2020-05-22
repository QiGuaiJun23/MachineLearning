from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def pca():
	"""
	主成分分析进行特征降维
	:return:
	"""

	#读取数据集
	data = pd.read_csv('../../数据集/机器学习/分类算法/鸢尾花数据集/iris.csv')
	#划分特征值与目标值
	x = data.iloc[:,0:4].values
	y = data.iloc[:,4].values
	# print(x,y)
	#数据标准化
	std = StandardScaler()
	x_std = std.fit_transform(x)
	# print(x_std)
	#主成分分析
	#求每列的平均值
	mean_vec = np.mean(x_std,axis=0)
	# 求协方差矩阵（直接求）
	# cov_mat = (x_std-mean_vec).T.dot((x_std-mean_vec)) / (x_std.shape[0] -1)
	#使用numpy中自带的公式求协方差矩阵np.cov(x_std.T)
	cov_mat = np.cov(x_std.T)
	# print('Covariance matrix \n%s' %cov_mat)
	#计算协方差的特征值和特征向量
	eig_vals,eig_vecs = np.linalg.eig(cov_mat)
	# print('Eigenvectors \n%s' %eig_vecs)
	# print('\nEigenvalues \n%s' %eig_vals)
	#将特征值与特征向量对应起来
	eig_pairs = [(np.abs(eig_vals[i]),eig_vecs[:,i]) for i in range(len(eig_vals))]
	# print(eig_pairs)
	#从高到底按特征值对eig_pairs排序
	eig_pairs.sort(key=lambda x:x[0],reverse=True)
	# print('Eigvalues in descending order:')
	# for i in eig_pairs:
	# 	print(i[0])
	#通过累加，确定将特征值降到几维
	tot = sum(eig_vals)
	var_exp = [(i/tot)*100 for i in sorted(eig_vals,reverse=True)]
	cum_var_exp = np.cumsum(var_exp)
	# print(cum_var_exp)
	#因为前两维的数据变化比较大，所以我们决定降到2维数据，150x4->150x2  需要一个4x2的矩阵，前两维的特征向量是我们需要的2维数据
	matrix_w = np.hstack((eig_pairs[0][1].reshape(4,1),eig_pairs[1][1].reshape(4,1)))#水平方向平铺
	# print('Matrix W:\n',matrix_w)
	Y = x_std.dot(matrix_w)
	# print(Y)

	#画图比较进行PCA之前和之后的变化
	#之前
	# plt.figure(figsize=(6,6))
	# for lab,col in zip(('setosa','versicolor','virginica'),('blue','red','green')):
	# 	plt.scatter(x[y == lab,0],
	# 	            x[y == lab,1],
	# 	            label=lab,
	# 	            c = col)
	# plt.xlabel('Sepal.Length')
	# plt.ylabel('Sepal.Width')
	# plt.legend(loc='best')
	# plt.tight_layout()
	# plt.show()
	#
	# #之后
	# plt.figure(figsize=(6,6))
	# for lab,col in zip(('setosa','versicolor','virginica'),('blue','red','green')):
	# 	plt.scatter(Y[y == lab,0],
	# 	            Y[y == lab,1],
	# 	            label=lab,
	# 	            c = col)
	# plt.xlabel('Sepal.Length')
	# plt.ylabel('Sepal.Width')
	# plt.legend(loc='best')
	# plt.tight_layout()
	# plt.show()

	# 第二种方法：使用sklearn包
	pca=PCA(n_components=2)

	Y=pca.fit_transform(x_std)
	print(Y)
	return None

if __name__=="__main__":
	pca()