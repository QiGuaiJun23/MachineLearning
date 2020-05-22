from sklearn.datasets import fetch_20newsgroups
news=fetch_20newsgroups(subset='all')
print(news.data)
print(news.target)