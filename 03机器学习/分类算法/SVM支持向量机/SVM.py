from sklearn.datasets import fetch_lfw_people
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
def svm():
	"""
	SVM进行简单人脸分类
	:return:
	"""
	#获取数据集
	faces =fetch_lfw_people(min_faces_per_person=60)
	# print(faces.target_names)
	# print(faces.images.shape)
	#每个图的大小是[62x47]
	#将数据集划分为测试集与训练集
	x_train,y_train,x_test,y_test = train_test_split(faces.data,faces.target,random_state=40)
	#
	#使用PCA降维
	#我们降维成150维
	# 　whiten： 白化。所谓白化，就是对降维后的数据的每个特征进行标准化，让方差都为1。
	# random_state:伪随机数发生器的种子,在混洗数据时用于概率估计
	pca = PCA(n_components=150,whiten=True,random_state=42)
	#实例化SVM
	svc = SVC(kernel='rbf',class_weight='balanced')
	model = make_pipeline(pca,svc)

	#交叉验证确定系数
	param = {'svc__C':[1,5,10],
	         'svc__gamma':[0.0001,0.0005,0.001]}
	grid = GridSearchCV(model,param_grid =param)
	grid.fit(x_train,x_test)
	# print(grid.best_params_)

	model=grid.best_estimator_
	yfit = model.predict(y_train)
	# print(yfit.shape)

	# 图形
	# fig,ax=plt.subplots(3,5)
	# for i,axi in enumerate(ax.flat):
	# 	axi.imshow(faces.images[i],cmap='bone')
	# 	axi.set(xticks=[],yticks=[],xlabel=faces.target_names[faces.target[i]])
	# plt.show()
	#
	#
	#算法分类之后的图形
	# fig,ax=plt.subplots(4,6)
	# for i,axi in enumerate(ax.flat):
	# 	axi.imshow(y_train[i].reshape(62,47),cmap='bone')
	# 	axi.set(xticks=[],yticks=[])
	# 	axi.set_ylabel(faces.target_names[yfit[i]].split()[-1],
	# 	               color='black' if yfit[i] == y_test[i] else 'red')
	#
	# fig.suptitle('Predicted Names:Incorrect Labels in Red',size=14)
	# plt.show()
	print(classification_report(y_test,yfit,target_names=faces.target_names))

	#混淆矩阵
	mat = confusion_matrix(y_test,yfit)
	sns.heatmap(mat.T,square=True,annot=True,fmt='d',
	            xticklabels=faces.target_names,
	            yticklabels=faces.target_names)
	plt.xlabel('true label')
	plt.ylabel('predicted label')
	plt.show()


	return None

if __name__ == "__main__":
	svm()