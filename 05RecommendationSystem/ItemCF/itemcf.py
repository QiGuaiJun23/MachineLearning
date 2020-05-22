import pandas as pd

def dataset():
	"""
	读取数据集，并进行预处理
	:return:
	"""
	names = ['userId','itemId','ratings','timestamp']
	data = pd.read_csv('../../DataSets/RecommendationSystem/ml-100k/u.data',sep='\t',names=names)
	data = data.drop('timestamp',axis=1)
	# print(data.head(10))
	trainset = data.drop('ratings',axis=1)
	# print(trainset.head(10))
	return data,trainset

def create_item_list_by_user(trainset):
	"""
	生成物品同现矩阵
	:return:
	"""
	movies_popular = {}
	for user,movies in trainset.items():
		for movie in movies:
			if movie not in movies_popular:
				movies_popular[movie] = 0
			movies_popular[movie] += 1
	print(movies_popular)



if __name__ == '__main__':
    data,trainset = dataset()
    create_item_list_by_user(trainset)