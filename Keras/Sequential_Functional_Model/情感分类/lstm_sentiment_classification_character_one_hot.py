#encoding:utf-8

'''
https://kexue.fm/archives/3863
character的one hot测试
Dropout不能用太多，否则信息损失太严重。
'''

import numpy as np
import pandas as pd

pos = pd.read_excel("./sentences/pos.xls", header=None)
pos['label'] = 1
neg = pd.read_excel("./sentences/neg.xls", header=None)
neg['label'] = 0
all_ = pos.append(neg, ignore_index=True)

# print all_
print "all_ shape: ", all_.shape
print "all_[0] shape: ", all_[0].shape
# print type(all_[0][2])

# 截断字数
maxlen = 200
# 出现次数少于该值的扔掉，最简单的降维方法
min_count = 20

content = ''.join(all_[0])
# 生成Series，index为单词，值为出现次数
abc = pd.Series(list(content)).value_counts()
print abc.index
print "abc shape", abc.shape
abc = abc[abc >= min_count]
print "after dimension reduction: ", abc.shape
# 将Series中的出现次数变为one_hot编码
abc[:] = list(range(len(abc)))
# word_set里面包含着所有的单词
word_set = set(abc.index)

# 将文本中的字转化成one_hot串
def doc2num(s, maxlen):
	s = [i for i in s if i in word_set]
	s = s[:maxlen]
	return list(abc[s])

all_['doc2num'] = all_[0].apply(lambda s: doc2num(s, maxlen))

# 打乱数据
idx = list(range(len(all_)))
np.random.shuffle(idx)
all_ = all_.loc[idx]

# 生成数据
x = np.array(list(all_['doc2num']))
y = np.array(list(all_['label']))
y = y.reshape((-1,1)) #调整标签形状，原先是list形式，一维，变成二维

print "x,y shape: "
print x.shape, y.shape
print "x[0]'s type: "
print type(x[0])

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM

# 建立模型
model = Sequential()
model.add(LSTM(128, input_shape=(maxlen,len(abc)))) # 128个 200*2417
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

#单个one hot矩阵的大小是maxlen*len(abc)的，非常消耗内存
#为了方便低内存的PC进行测试，这里使用了生成器的方式来生成one hot矩阵
#仅在调用时才生成one hot矩阵
#可以通过减少batch_size来降低内存使用，但会相应地增加一定的训练时间
batch_size = 128
train_num = 15000

# 不足则补全0行
# 传入转化为01串的句子，例如句子长度为157，传入长度为157的list；通过to_categorical转化为157*2417的矩阵，然后通过vstack转化为200*2417的矩阵。
gen_matrix = lambda z: np.vstack((np_utils.to_categorical(z, len(abc)), np.zeros((maxlen-len(z), len(abc)))))

def data_generator(data, labels, batch_size): 
	batches = [list(range(batch_size*i, min(len(data), batch_size*(i+1)))) for i in range(len(data)/batch_size+1)]
	while True:
		for i in batches:
			xx = np.zeros((maxlen, len(abc)))
			# data[i]的shape为128，因为二维是句子包含的词的个数，每个句子都不一样
			# map中gen_matrix中传入的是句子，返回的是200*2417的张量。
			# 最后xx的shape为128*200*2417
			xx, yy = np.array(map(gen_matrix, data[i])), labels[i]
			# print "xx yy shape: "
			# print data[i].shape, xx.shape, yy.shape
			yield (xx, yy)

#a = data_generator(x[:train_num], y[:train_num], batch_size)
#a.next()
model.fit_generator(data_generator(x[:train_num], y[:train_num], batch_size), samples_per_epoch=train_num, nb_epoch=30)
model.evaluate_generator(data_generator(x[train_num:], y[train_num:], batch_size), val_samples=len(x[train_num:]))

def predict_one(s): #单个句子的预测函数
	s = gen_matrix(doc2num(s, maxlen))
	s = s.reshape((1, s.shape[0], s.shape[1]))
	return model.predict_classes(s, verbose=0)[0][0]
