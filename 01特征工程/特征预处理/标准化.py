from sklearn.preprocessing import StandardScaler,Imputer
import numpy as np
def stand():
	"""
	标准化
	:return:
	"""
	std=StandardScaler()

	data=std.fit_transform([[1,-1,3],[2,4,2],[4,6,-1]])

	print(data)

	return None

#缺失值处理
def im():
	"""
	缺失值处理
	:return:
	"""
	#naN,nan
	im=Imputer(missing_values='NaN',strategy='mean',axis=0)#0是列，1是行

	data=im.fit_transform([[1,2],[np.nan,3],[7,6]])

	print(data)

	return None

if __name__=="__main__":
	# stand()
	im()