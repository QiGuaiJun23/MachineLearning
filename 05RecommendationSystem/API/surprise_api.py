from surprise import KNNBasic,SVD,accuracy,NormalPredictor
from surprise import Dataset
from surprise.model_selection import cross_validate,train_test_split,GridSearchCV
from surprise import Reader
import os
import pandas as pd
from collections import defaultdict
def dataset():
	"""
	读取数据集
	:return:
	"""
	#1、直接下载，数据存储在c/user/.surprise_data中
	# data = Dataset.load_builtin('ml-100k')
	#2、使用自己下载的数据集
	#(文本文件)
	#指定数据路径
	file_path = os.path.expanduser('../../DataSets/RecommendationSystem/ml-100k/u.data')
	#指定分隔符
	reader = Reader(line_format='user item rating timestamp',sep='\t')
	#加载数据集
	data = Dataset.load_from_file(file_path=file_path,reader=reader)

	#(CSV文件)
	rating_dict = {'itemID':[1,1,1,2,2],
	               'userID':[9,32,2,45,'user_foo'],
	               'rating':[3,2,4,3,1]}
	df = pd.DataFrame(rating_dict)
	reader = Reader(rating_scale=(1,5)) #rating_scale必须指定参数
	datacsv = Dataset.load_from_df(df[['userID','itemID','rating']],reader)


	return data,datacsv

def model(data,datacsv):
	"""
	建立模型
	:return:
	"""
	#使用SVD模型
	algo = SVD()
	#进行5折交叉验证
	# print(cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True))
	print(cross_validate(NormalPredictor(),datacsv,cv=2))
def train_test(data):
	"""
	分割训练集和测试集

	如果不想运行完整的交叉验证过程，则可以使用 train_test_split()
	来给定大小的训练集和测试集采样.
	您将需要使用将在训练集上训练算法的方法，
	以及将返回从测试集得出的预测的方法：accuracy metric fit()test()
	:return:
	"""
	trainset,testset = train_test_split(data,test_size=0.25)
	algo = SVD()
	algo.fit(trainset)
	#预测
	predictions = algo.test(testset)
	#准确率
	print(accuracy.rmse(predictions))
	return None

def predict_ratings(data):
	"""
	可以简单地将算法适合整个数据集，
	而不是运行交叉验证。
	这可以通过使用build_full_trainset()将创建trainset对象的方法来完成
	可以通过直接调用该predict()方法来预测收视率
	:return:
	"""
	trainset = data.build_full_trainset()

	svg = SVD()
	svg.fit(trainset)
	testset = trainset.build_anti_testset()
	predictions = svg.test(testset)

	algo = KNNBasic()
	algo.fit(trainset)

	#收视率预测：假设对用户196和项目302感兴趣（确保它们在trainset中！），并且知道真实的评分rui=4
	uid = str(196)
	iid = str(302)

	# algo.predict(uid,iid,r_ui=4,verbose=True)

	return predictions

def grid_search_usage(data):
	"""
	使用GridSearchCV调整算法参数
	给定一个dict参数，该类穷举尝试所有参数组合，并报告任何精度度量（在不同分割中取平均值）的最佳参数
	:return:
	"""
	param_grid = {'n_epochs':[5,10],'lr_all':[0.002,0.005],
	              'reg_all':[0.4,0.6]}

	gs = GridSearchCV(SVD,param_grid=param_grid,measures=['rmse','mae'],cv=3)
	gs.fit(data)
	print(gs.best_score['rmse'])
	print(gs.best_params['rmse'])
	algo = gs.best_estimator['rmse']
	algo.fit(data.build_full_trainset())

def top_n_recommendations(predictions,n=10):
	"""
	获取每个用户的前N条建议

	在整个数据集上训练SVD算法，
	然后预测不在训练集中的对（用户，项目）的所有评分。
	然后，我们为每个用户检索前10位的预测。

	:return:
	"""
	top_n = defaultdict(list)
	for uid,iid,true_r,est,_ in predictions:
		top_n[uid].append((iid,est))
	for uid,user_ratings in top_n.items():
		user_ratings.sort(key = lambda x:x[1],reverse=True)
		top_n[uid] = user_ratings[:n]

	return top_n



if __name__ == '__main__':
     data,datacsv = dataset()
     # model(data,datacsv)
     # train_test(data)
     predictions = predict_ratings(data)
     # grid_search_usage(data)
     top_n = top_n_recommendations(predictions,n=10)
     for uid,user_ratings in top_n.items():
	     print(uid,[iid for (iid,_) in user_ratings])
     # print(top_n)