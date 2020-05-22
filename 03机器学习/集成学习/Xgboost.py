from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import plot_importance
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV,StratifiedKFold
def dataset():
	"""
	获取数据集并进行处理
	:return:
	"""
	# 获取数据集
	data = pd.read_csv('../../数据集/机器学习/集成学习/糖尿病患者/PimaIndiansdiabetes.csv')
	# print(data)
	# 将特征值与目标值分开
	y = data.iloc[:, 8]
	x = data.drop(columns='Outcome', axis=1)
	# 分割成训练集与测试集
	seed = 7
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=seed)
	return x,y,x_train, x_test, y_train, y_test


def xgboost(x_train, x_test, y_train, y_test):
	"""
	利用xgboost对癌症数据进行分类
	:return:
	"""
	#xgboost进行训练
	xgb = XGBClassifier()
	xgb.fit(x_train,y_train)
	y_predict = xgb.predict(x_test)
	accuracy = accuracy_score(y_test, y_predict)
	print("准确率：%.2f%%" % (accuracy*100))
	return None

def xgboot_steps(x_train, x_test, y_train, y_test):
	"""
	将xgboost选择树的过程分步展示
	:return:
	"""
	xgb = XGBClassifier()
	eval_set = [(x_test,y_test)]
	xgb.fit(x_train, y_train,early_stopping_rounds=10,eval_metric="logloss",eval_set=eval_set,verbose=True)
	y_predict = xgb.predict(x_test)
	accuracy = accuracy_score(y_test, y_predict)
	print("准确率：%.2f%%" % (accuracy*100))

def show_importance(x,y):
	"""
	画图展示出特征的重要程度
	:return:
	"""
	xgb = XGBClassifier()
	xgb.fit(x,y)
	plot_importance(xgb)
	plt.show()
	return None

def adaboost(x_train, x_test, y_train, y_test):
	"""
	使用adaboost算法进行分类预测
	:return:
	"""
	ada = AdaBoostClassifier(learning_rate=0.01)
	ada.fit(x_train,y_train)
	y_predict = ada.predict(x_test)
	accuracy = accuracy_score(y_test, y_predict)
	print("准确率：%.2f%%" % (accuracy * 100))
	return None




def gridsearchCV(x_train, y_train):
	"""
	使用网格搜索进行超参数调优
	:return:
	"""
	#1、learning rate 学习率
	#2、tree(max_depth、min_child_weight、subsample、colsample_bytree、gamma
	#3、gamma
	#4、正则化参数(lambda、alpha)
	learning_rate = [0.0001,0.001,0.01,0.1,0.2,0.3]
	param_rate = dict(learning_rate=learning_rate)  #必须是字典格式

	#StratifiedKFold是一种将数据集中每一类样本数据按均等方式拆分的方法。
	kfold = StratifiedKFold(n_splits=10,shuffle=True,random_state=7)

	xgb = XGBClassifier()
	grid_search = GridSearchCV(xgb,param_grid=param_rate,scoring="neg_log_loss",n_jobs=-1,cv=kfold)
	grid_result = grid_search.fit(x_train,y_train)
	print("Best:%f using %s" % (grid_result.best_score_, grid_result.best_params_))

	ada = AdaBoostClassifier()
	grid_search_ada = GridSearchCV(ada, param_grid=param_rate, scoring="neg_log_loss", n_jobs=-1, cv=kfold)
	grid_result_ada = grid_search_ada.fit(x_train, y_train)
	print("Best:%f using %s" % (grid_result_ada.best_score_, grid_result_ada.best_params_))

	#
	# means =grid_result.cv_results_['mean_test_score']
	# params = grid_result.cv_results_['params']
	# for mean,param in zip(means,params):
	# 	print("%f with: %r" % (mean,param))




if __name__ == "__main__":
	x,y,x_train, x_test, y_train, y_test = dataset()
	xgboost(x_train, x_test, y_train, y_test)
	# xgboot_steps(x_train, x_test, y_train, y_test)
	# show_importance(x,y)
	# gridsearchCV(x_train, y_train)
	adaboost(x_train, x_test, y_train, y_test)