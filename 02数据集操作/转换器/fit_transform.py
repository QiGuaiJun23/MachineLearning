#fit_trandform()    最常用
#fit():输入数据,但不做事情    计算平均值和标准差
#transform()：进行数据的转换
from sklearn.preprocessing import StandardScaler

s=StandardScaler()

print(s.fit_transform([[1,2,3],[4,5,6]]))

ss=StandardScaler()

print(ss.fit([[1,2,3],[4,5,6]]))

print(ss.transform([[1,2,3],[4,5,6]]))

print(ss.fit([[1,2,3],[4,5,6]]))

print(ss.transform([[2,3,4],[5,6,7]]))

