import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler

#这项比赛的目的是预测一个人想签到哪个地方。
# 为了比赛的目的，Facebook创建了一个人工世界，
# 由10万乘10公里的正方形中的100,000多个地方组成。
# 对于给定的一组坐标，您的任务是返回最可能的位置的排名列表。
# 数据被制作成类似于来自移动设备的位置信号，
# 从而使您具有处理真实数据（复杂的值和不准确的噪声）所需的风味。
# 不一致和错误的位置数据可能会破坏Facebook Check In等服务的体验。

# 实例分析：
#
# 特征值：x,y坐标，定位准确性，时间    目标值：入住位置的id
#
# 处理：  0<x<10    0<y<10
#
# 1、由于数据量大，节省时间x,y缩小
#
# 2、时间戳进行（年、月、日、周、时分秒），当做新的特征
#
# 3、入住类别（几千~几万），少于指定签到人数的位置删除
def knn():
	"""
	knn预测分类
	:return:
	"""
	# 读取数据
	data = pd.read_csv('../../数据集/机器学习/分类算法/facebook-v-predicting-check-ins/train.csv')
	# print(data.count(axis=0))

	# 数据预处理
	#1、缩小数据范围
	data = data.query(" x > 0.25 & x < 1.25 & y > 2.5 &y < 2.75")
	# print(data.count(axis=0))
	#2、处理时间数据
	data['time'] = pd.to_datetime(data['time'], unit='s')
	#把日期格式转换成  字典格式
	time_value = pd.DatetimeIndex(data['time'])
	# print(time_value)
	#3、增加分割的日期数据
	data['day'] = time_value.day
	data['hour']= time_value.hour
	data['weekday'] = time_value.weekday
	# print(data.head())

	#4、删除没用的数据
	data = data.drop(['time'],axis=1)
	# print(data.head())
	#5、将签到位置少于n个用户的删除
	place_count = data.groupby('place_id').count()
	# print(place_count)
	tf = place_count[place_count.row_id > 3].reset_index()
	data = data[data['place_id'].isin(tf.place_id)]
	# print(data)
	#
	#取出数据当中的特征值和目标值
	y=data['place_id']
	x=data.drop(['place_id'],axis=1)
	x=data.drop(['row_id'],axis=1)
	#
	#进行数据集的分割
	x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25)

	# 特征工程(标准化)
	std=StandardScaler()

	# 对测试集和训练集的特征值进行标准化
	x_train = std.fit_transform(x_train)
	x_test = std.fit_transform(x_test)


	# 进行算法流程
	knn = KNeighborsClassifier(n_neighbors=5)#n_neighbors=5

	#fit predict  score
	knn.fit(x_train,y_train)

	#得出预测结果
	y_predict = knn.predict(x_test)
	print("预测的目标签到位置为：",y_predict)

	#得出准确率
	print("预测的准确率：",knn.score(x_test,y_test))

	#接下来我们进行网格搜索（超参数调优）
	# neighbors = {"n_neighbors": [3,5,10]}
	#
	# gc = GridSearchCV(knn,param_grid=neighbors,cv = 5)
	#
	# gc.fit(x_train,y_train)
	# #预测准确率
	# print("在测试集上的准确率：",gc.score(x_test,y_test))
	#
	# print("在交叉验证中验证的最好结果：",gc.best_score_)
	#
	# print("选择最好的模型是：",gc.best_estimator_)
	#
	# print("每次交叉验证的结果",gc.cv_results_)


	return None

if __name__ == "__main__":
	knn()