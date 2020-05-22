from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

iris = load_iris()

#注意返回值，训练集train   x_train,y_train    测试集 test   x_test  y_test
x_train,x_test,y_train,y_test=train_test_split(iris.data,iris.target,test_size=0.25)

print("训练集特征值和目标值：",x_train,y_train)

print("测试集特征值和目标值：",x_test,y_test)
