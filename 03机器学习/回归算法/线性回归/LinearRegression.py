from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression,SGDRegressor,Ridge
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib
def linear():
	"""
	线性回归预测房价
	"""
	#获取数据集
	lb = load_boston()
	# print(lb.target)
	#分割数据集
	x_train,x_test,y_train,y_test = train_test_split(lb.data,lb.target,test_size=0.25)

	#特征标准化(特征值与目标值都需要标准化,实例化两个标准化API)
	std_x = StandardScaler()
	x_train = std_x.fit_transform(x_train)
	x_test = std_x.transform(x_test)

	std_y = StandardScaler()
	y_train = std_y.fit_transform(y_train.reshape(-1,1))#0.19版本之后要求数据必须是二维的
	y_test = std_y.transform(y_test.reshape(-1,1))

	#estimator预测
	#正规方程求解方式预测结果
	lr = LinearRegression()
	lr.fit(x_train,y_train)
	# print(lr.coef_)
	#模型的保存和加载
	joblib.dump(lr,'test.pkl')
	#加载
	estimator = joblib.load('test.pkl')
	y_predict = std_y.inverse_transform(estimator.predict(x_test))
	print(y_predict)
	# #预测房子价格
	# y_predict = std_y.inverse_transform(lr.predict(x_test))
	# # print("房价预测为：",y_predict)
	# # print(std_y.inverse_transform(y_test))
	#
	# #使用梯度下降算法进行预测
	# sgd = SGDRegressor()
	# sgd.fit(x_train,y_train)
	# #预测房子价格
	# sgd_y_predict = std_y.inverse_transform(sgd.predict(x_test))
	# print("房价预测为：",sgd_y_predict)
	# # print(std_y.inverse_transform(y_test))
	#
	# #使用岭回归进行预测分析
	# rd = Ridge(alpha=1.0)
	# rd.fit(x_train,y_train)
	# rd_y_predict=std_y.inverse_transform(rd.predict(x_test))
	#
	#
	# #使用均方误差计算两种预测方法的好坏
	# print("使用正规方程计算的误差：",mean_squared_error(std_y.inverse_transform(y_test),y_predict))
	# print("使用梯度下降算法计算的误差：",mean_squared_error(std_y.inverse_transform(y_test),sgd_y_predict))
	# print("使用岭回归计算的误差：", mean_squared_error(std_y.inverse_transform(y_test), rd_y_predict))
	#

	return None

if __name__ == "__main__":
	linear()