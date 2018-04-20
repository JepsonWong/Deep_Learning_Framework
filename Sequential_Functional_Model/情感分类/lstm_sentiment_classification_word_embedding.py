#encoding:utf-8

'''
word 的 word2vec测试
Dropout不能用太多，否则信息损失太严重
'''

'''
batch_size = 128
train_num = 15000
30 epochs
['loss', 'acc']
[0.41771692529894305, 0.9066339067120126]
'''
import numpy as np
import pandas as pd
import jieba

pos = pd.read_excel('./sentences/pos.xls', header=None)
pos['label'] = 1
neg = pd.read_excel('./sentences/neg.xls', header=None)
neg['label'] = 0
all_ = pos.append(neg, ignore_index=True)
all_['words'] = all_[0].apply(lambda x: list(jieba.cut(x)))

# 句子为100个词，超过100个词截断，少于100个词补上
maxlen = 100
# 出现次数少于该值的词扔掉，最简单的降维方法
min_count = 5

content = []
for i in all_['words']:
	content.extend(i)

# 生成单词表的长度
abc = pd.Series(content).value_counts()
abc = abc[abc >= min_count]
abc[:] = list(range(1, len(abc) + 1))
abc[''] = 0
word_set = set(abc.index)

# 根据单词表将句子转化为one_hot串
def doc2num(s, maxlen):
	s = [i for i in s if i in word_set]
	s = s[:maxlen] + [''] * max(0, maxlen-len(s))
	return list(abc[s])

all_['doc2num'] = all_['words'].apply(lambda x: doc2num(x, maxlen))

# 手动打乱数据
idx = list(range(len(all_)))
np.random.shuffle(idx)
# 打乱数据
all_ = all_.loc[idx]

# 按照keras的要求生成输入数据
x = np.array(list(all_['doc2num']))
y = np.array(list(all_['label']))
y = y.reshape(-1, 1)
print "x y shape: "
print x.shape
print y.shape

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Embedding
from keras.layers import LSTM

# 建立模型
model = Sequential()
model.add(Embedding(len(abc), 256, input_length=maxlen))
model.add(LSTM(128))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

batch_size = 128
train_num = 15000

model.fit(x[:train_num], y[:train_num], batch_size=batch_size, epochs=30)
out = model.evaluate(x[train_num:], y[train_num:], batch_size = batch_size)
print(model.metrics_names)
print out

def predict_one(s): #单个句子的预测函数
    s = np.array(doc2num(list(jieba.cut(s)), maxlen))
    s = s.reshape((1, s.shape[0]))
    return model.predict_classes(s, verbose=0)[0][0]
