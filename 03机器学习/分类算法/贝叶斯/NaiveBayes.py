from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics.classification import classification_report
def naivebayes():
	"""
	朴素贝叶斯进行分本分类
	:return:
	"""
	#获取数据集
	news = fetch_20newsgroups(subset='all')
	print(news)
	#对数据集进行分割
	x_train,x_test,y_train,y_test = train_test_split(news.data,news.target,test_size=0.25)
	#进行特征抽取
	tf = TfidfVectorizer()
	x_train = tf.fit_transform(x_train)
	print(tf.get_feature_names())#输出特征名
	x_test = tf.transform(x_test)
	#进行贝叶斯算法预测
	mlt = MultinomialNB(alpha=1.0)
	print(x_train.toarray())

	mlt.fit(x_train,y_train)
	y_predict = mlt.predict(x_test)
	print("预测的文章类型为：",y_predict)

	#得出准确率
	print("准确率为：",mlt.score(x_test,y_test))

	#得出精确率召回率
	print("每个类别的精确率和召回率：",classification_report(y_test,y_predict,target_names=news.target_names))#target_names表示每个类别的名称（比如文章分科技、娱乐等）

	return None


if __name__ == "__main__":
	naivebayes()