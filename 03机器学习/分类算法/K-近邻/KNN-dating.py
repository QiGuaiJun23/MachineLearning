# -*- coding: UTF-8 -*-
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import operator
def dataset():
	"""
	打开并解析文件，对数据进行分类:
	1代表不喜欢
	2代表魅力一般
	3代表极具魅力
	:return:
	"""
	# data = pd.read_table('../../数据集/机器学习/分类算法/海伦约会/datingTestSet.txt',sep='\s+',header=None)
	# data[4]=1
	# for i in range(0,1000):
	# 	if(data[3][i] == 'smallDoses'):
	# 		data[4][i] = 2
	# 	elif(data[3][i] == 'didntLike'):
	# 		data[4][i] = 3
	# # print(data)
	# #把特征值与目标值分开
	# y = data[4]
	# x = data.drop([3],axis=1).drop([4],axis=1)
	# # print(y)

	#numpy矩阵的形式

	#打开文件
	fr = open('../../数据集/机器学习/分类算法/海伦约会/datingTestSet.txt')
	#读取文件的所有内容
	arrayOne = fr.readlines()
	#得到文件行数
	numoflines = len(arrayOne)
	#返回numpy矩阵，解析完成的数据：numoflines
	returnMat = np.zeros((numoflines,3))
	#返回分类标签向量
	classLabelVector = []
	#行的索引
	index = 0
	for line in arrayOne:
		# s.strip(rm)，当rm空时,默认删除空白符(包括'\n','\r','\t',' ')
		line = line.strip()
		# 使用s.split(str="",num=string,cout(str))将字符串根据'\t'分隔符进行切片。
		listFromLine = line.split('\t')
		# 将数据前三列提取出来,存放到returnMat的NumPy矩阵中,也就是特征矩阵
		returnMat[index,:] = listFromLine[0:3]#每三个赋值给矩阵的每一行
		if listFromLine[-1] == 'didntLike':
			classLabelVector.append(1)
		elif listFromLine[-1] == 'smallDoses':
			classLabelVector.append(2)
		elif listFromLine[-1] == 'largeDoses':
			classLabelVector.append(3)
		index += 1

	return returnMat,classLabelVector

def showdatas(x,y):
	"""
	数据可视化
	:return:
	"""

	fig = plt.figure(figsize=(15,15))
	#第一张散点图显示视频游戏与飞机里程数占比关系
	ax1 = fig.add_subplot(2,2,1)
	colors = []
	#设置图例
	didntLike = mlines.Line2D([], [], color='black', marker='.',markersize=6, label='didntLike')
	smallDoses = mlines.Line2D([], [], color='orange', marker='.',markersize=6, label='smallDoses')
	largeDoses = mlines.Line2D([], [], color='red', marker='.',markersize=6, label='largeDoses')
	for i in y:
		if i == 1:
			colors.append('black')
		if i == 2:
			colors.append('orange')
		if i == 3:
			colors.append('red')
	ax1.scatter(x=x[:,0],y=x[:,1],color=colors,s=15)
	ax1.set_title('每年获得的飞行常客里程数与玩视频游戏所消耗时间占比')
	ax1.set_xlabel('每年获得的飞行常客里程数')
	ax1.set_ylabel('玩视频游戏所消耗时间占比')
	# 添加图例
	ax1.legend(handles=[didntLike,smallDoses,largeDoses])

	#第二张散点图显示视频游戏与冰激凌之间的关系

	ax2 = fig.add_subplot(2,2,2)
	ax2.scatter(x=x[:,1],y = x[:,2],color=colors,s=15)
	ax2.set_title('视频游戏消耗时间与每周消费的冰激凌公升数')
	ax2.set_xlabel('玩视频游戏消耗时间')
	ax2.set_ylabel('每周消费的冰激凌公升数')
	# 添加图例
	ax2.legend(handles=[didntLike,smallDoses,largeDoses])
	# plt.show()
	# print(colors)

	#第三张散点图显示飞机里程数与冰激凌公升数的关系
	ax3 = fig.add_subplot(2,2,3)
	ax3.scatter(x = x[:,0],y = x[:,2],color=colors,s = 15)
	ax3.set_title('每年飞机飞行里程数与每周消费的冰激凌公升数')
	ax3.set_xlabel('每年获得的飞行常客里程数')
	ax3.set_ylabel('每周消费的冰激凌公升数')
	# 添加图例
	ax3.legend(handles=[didntLike, smallDoses, largeDoses])
	plt.show()
	return None

def autoNorm(x):
	"""
	数据归一化
	newValue = (oldValue - min) / (max - min)
	:param x:
	:return:
	"""
	# #这边还可以做一些改进，筛掉一些数据
# 	# for i in range(0,3):
# 	# 	x[:,i] = (x[:,i]-x[:,i].min())/(x[:,i].max()-x[:,i].min())
# 	# return x
	#获得数据的最小值
	minvals = x.min(0)
	maxvals = x.max(0)
	#最大值和最小值的范围
	ranges = maxvals - minvals
	#shape（x）返回x的矩阵行列数
	normx = np.zeros(np.shape(x))
	#返回x的行数
	m = x.shape[0]
	#原始值减去最小值
	normx = x - np.tile(minvals,(m,1))
	#除以最大和最小值的差，得到归一化的数据
	normx = normx / np.tile(ranges,(m,1))
	#返回归一化数据结果，数据范围，最小值
	return normx,ranges,minvals

def classify(x_data,y_data,labels,k):
	"""
	Knn算法，分类器
	x_data：训练集
	y_data:测试集
	labels:分类标签
	k：选取的分类区域
	:return:
	"""
	#返回训练集的行数
	xdatasize = y_data.shape[0]
	#将测试集在行列上进行复制，并减去训练集
	diffMat = np.tile(x_data,(xdatasize,1)) - y_data
	#求特征矩阵差的平方
	sqdiffMat = diffMat**2
	#平方求和
	sqDistance = sqdiffMat.sum(axis=1)
	#开方计算距离
	distance = sqDistance ** 0.5
	#返回距离从小到大排序后的索引值
	sortedDistance = distance.argsort()
	#定一个记录类别次数的字典
	classified = {}
	#将排序后的类别记录
	for i in range(k):
		#选出前k个元素的类别
		votedlabels = labels[sortedDistance[i]]
		# dict.get(key,default=None),字典的get()方法,返回指定键的值,如果值不在字典中返回默认值。
		# 计算类别次数
		classified[votedlabels] = classified.get(votedlabels,0) + 1
	# python3中用items()替换python2中的iteritems()
	# key=operator.itemgetter(1)根据字典的值进行排序
	# key=operator.itemgetter(0)根据字典的键进行排序
	# reverse降序排序字典
	sortedCounts = sorted(classified.items(), key=operator.itemgetter(1), reverse=True)
	#第一个存放的就是出现次数最多的类别
	return sortedCounts[0][0]


def classifyDataset(normx,labels):
	"""
	划分测试集与训练集
	将训练集的10%划分为测试集
	:return:
	"""
	alpha = 0.1
	#获得归一化后数据集的行数
	m = normx.shape[0]
	#10%的数据为测试集
	numtest = int (m * alpha)
	#分类错误计数
	errorcount = 0.0

	#调用算法 进行分类
	for i in range(numtest):
		# 前numtest为测试集，后m-numtest为训练
		classfyresult = classify(normx[i,:],normx[numtest:m,:],labels[numtest:m],4)
		print("分类结果:%d\t真实类别:%d" % (classfyresult,labels[i]))
		if (classfyresult != labels[i]):
			errorcount += 1
	print("错误率为：%f%%" % (errorcount / float(numtest)* 100))

	return None


if __name__ == "__main__":
	returnMat,classLabelVector = dataset()
	normx, ranges, minvals = autoNorm(returnMat)
	classifyDataset(normx,classLabelVector)
	# showdatas(returnMat,classLabelVector)
	# print(normx)
