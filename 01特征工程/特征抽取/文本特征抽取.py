from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import jieba
def countvec():
	"""
	对文本进行特征值化
	:return:
	"""
	cv=CountVectorizer()

	data=cv.fit_transform(["life is is short , i like python","life is too long, i dislike python"])
	dta=cv.fit_transform(["人生苦短，我用python","人生漫长，不用python"])
	print(cv.get_feature_names())

	print(dta.toarray())
	#统计所有文章当中所有的词，重复的只看一次
	#对每篇文章，在词的列表里面进行统计每个词出现的次数
	#对单个英文字母不统计：没有分类依据
	return None

def cutword():
	#分词
	con1 = jieba.cut("今天很残酷，明天更残酷，后天很美好，但绝大部分是死在明天晚上，看不到后天的太阳。")
	con2 = jieba.cut("我们看到的从很远星系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。")
	con3 = jieba.cut("如果只有一种方式了解某样事物，你就不会真正了解它，了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系")

	#转换成列表
	content1=list(con1)
	content2=list(con2)
	content3=list(con3)

	#把列表转换成字符串
	c1= ' '.join(content1)
	c2= ' '.join(content2)
	c3= ' '.join(content3)
	return c1,c2,c3

def chinesevec():
	"""
	中文特征化
	:return:
	"""
	c1,c2,c3=cutword()

	print(c1,c2,c3)

	cv = CountVectorizer()

	data = cv.fit_transform([c1, c2,c3])

	print(cv.get_feature_names())#单个汉字不统计

	print(data.toarray())

	return None

def tfidfvec():
	"""
	中文特征化
	:return:
	"""
	c1,c2,c3=cutword()

	print(c1,c2,c3)

	tf = TfidfVectorizer()

	data = tf.fit_transform([c1, c2,c3])

	print(tf.get_feature_names())#单个汉字不统计

	print(data.toarray())

	return None

if __name__=="__main__":
	tfidfvec()