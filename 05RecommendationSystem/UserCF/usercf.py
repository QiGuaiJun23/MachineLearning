import pandas as pd

def dataset():
	"""
	数据集读取、预处理
	:return:
	"""
	names = ['userId','itemId','rating','time']
	data = pd.read_csv('../../DataSets/RecommendationSystem/ml-100k/u.data',sep='\t',names=names)
	#删掉时间
	data = data.drop('time',axis=1)
	#利用pandas的透视函数pivot()转换成用户评分矩阵
	user_item=data.pivot(index='userId',columns='itemId',values='rating')
	print(user_item)
	return user_item

def build_xy(user_id1,user_id2):
	"""
	构建共同的评分向量
	:param data:
	:param user_id1:
	:param user_id2:
	:return:
	"""
	bool_array = data.loc[user_id1].notnull() & data.loc[user_id2].notnull()
	return data.loc[user_id1,bool_array],data.loc[user_id2,bool_array]

def pearson(user_id1,user_id2):
	"""
	计算皮尔逊相关系数
	:return:
	"""
	x,y = build_xy(user_id1,user_id2)
	mean1,mean2 = x.mean(),y.mean()
	#分母
	denominator = (sum((x-mean1)**2)*sum((y-mean2)**2))**0.5
	#计算
	try:
		per = (sum(x-mean1)*sum(y-mean2)) / denominator
	except ZeroDivisionError:
		per = 0

	return per

def computeNearestNeighbor(user_id,k=3):
	"""
	计算最相似的邻居，取前三个用户
	:param data:
	:param user_id:
	:param k:
	:return:
	"""
	return data.drop(user_id).index.to_series().apply(pearson,args=(user_id,)).nlargest(k)

def recommand(user_id):
	"""
	找到最相似的用户di
	:return:
	"""
	nearest_user_id = computeNearestNeighbor(user_id).index[0]
	print('最相似用户ID：')
	print(nearest_user_id)

	#找出邻居评价过，但是自己未曾评价的项目
	#结果：index是项目名称，values是评价
	print(data.loc[nearest_user_id,data.loc[user_id].isnull() & data.loc[nearest_user_id].notnull()])

if __name__ == '__main__':
    data = dataset()
    recommand(2)
    # print(computeNearestNeighbor(196, k=3))
    # computeNearestNeighbor(data, '149', k=3)