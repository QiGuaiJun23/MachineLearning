import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn.ensemble import RandomForestClassifier
def Decision():
	"""
	决策树预测泰坦尼克生死
	:return:
	"""
	#读取数据
	data = pd.read_csv('../../数据集/机器学习/分类算法/Titanic/titanic.csv')
	# print(data.columns.values)
	#处理数据，找出特征值和目标值
	x = data.loc[:,['sex','age','pclass']]
	y = data.loc[:,['survived']]

	#有缺失值，处理缺失值，平均值填充
	x['age'].fillna(x['age'].mean(),inplace = True)

	#划分训练集和测试集
	x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25)

	#进行字典特征的抽取,特征->类别->one_hot编码（特征里面是类别，值的类型都不同，要进行one_hot编码）
	dict = DictVectorizer(sparse=False)

	x_train = dict.fit_transform(x_train.to_dict(orient="records"))

	x_test = dict.transform(x_test.to_dict(orient="records"))

	# 利用决策树进行分类
	# dec = DecisionTreeClassifier(max_depth=5)
	#
	# dec.fit(x_train,y_train)
	#
	# #预测准确率
	# print("准确率为：",dec.score(x_test,y_test))
	# print(dict.get_feature_names())
	#
	# #导出树的结构
	# export_graphviz(dec,out_file='../../数据集/机器学习/分类算法/Titanic/tree.dot',feature_names=['年龄','pclass','女性','男性'])

	#随机森林预测分类（超参数调优）
	rf = RandomForestClassifier()

	param = {"n_estimators":[120,200,300,500,800],"max_depth":[5,8,15,25,30]}

	#网格搜索与交叉验证
	gc = GridSearchCV(rf,param_grid=param,cv=5)

	gc.fit(x_train,y_train)

	#输出准确率
	print("准确率为：",gc.score(x_test,y_test))

	print("选择最好的模型是：",gc.best_estimator_)

	print("每次交叉验证的结果",gc.cv_results_)

	return None


if __name__ == "__main__":
	Decision()