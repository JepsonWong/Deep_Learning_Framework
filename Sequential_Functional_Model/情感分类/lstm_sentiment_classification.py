#encoding:utf-8

'''
https://spaces.ac.cn/archives/3414
'''

import pandas as pd
import numpy as np
import jieba
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
import tensorflow as tf

# 读取训练语料
neg = pd.read_excel("./sentences/neg.xls", header=None, index=None)
pos = pd.read_excel("./sentences/pos.xls", header=None, index=None)

# 训练语料加标签
pos['mark'] = 1
neg['mark'] = 0

# 计算语料数目
neglen=len(neg)
poslen=len(pos) #计算语料数目

# 合并语料
pn = pd.concat([pos, neg], ignore_index=True)

# 定义分词函数
cw = lambda x: list(jieba.cut(x))
pn['words'] = pn[0].apply(cw)

# 读入评论内容
comment = pd.read_excel("sum.xls");
# 仅读取非空评论
comment = comment[comment['rateContent'].notnull()]
# 评论分词
comment['words'] = comment['rateContent'].apply(cw)

d2v_train = pd.concat([pn['words'], comment['words']], ignore_index=True)

w = []
for i in d2v_train:
	w.extend(i)

dict = pd.DataFrame(pd.Series(w).value_counts())
del w, d2v_train
dict['id'] = list(range(1,len(dict)+1))

# 将语料转化为数字串 不如一句5个词的句子转化为ont-hot的编码[1, 200, 355, 21, 211]
get_sent = lambda x: list(dict['id'][x])
pn['sent'] = pn['words'].apply(get_sent)

# 将语料转化成的数字串填充成相同长度的数字串
maxlen = 50
print("Pad sequences (samples x time)")
pn['sent'] = list(sequence.pad_sequences(pn['sent'], maxlen=maxlen))

'''
batch_size=128, epochs=2
0.8495
'''
x = np.array(list(pn['sent']))[::2] # 训练集
y = np.array(list(pn['mark']))[::2]
xt = np.array(list(pn['sent']))[1::2] #测试集
yt = np.array(list(pn['mark']))[1::2]

'''
batch_size=128, epochs=2
0.8618
'''
'''
idx = list(range(len(pn)))
np.random.shuffle(idx)
pn = pn.loc[idx]

x = np.array(list(pn['sent']))[::2]
y = np.array(list(pn['mark']))[::2]
xt = np.array(list(pn['sent']))[1::2]
yt = np.array(list(pn['mark']))[1::2]
'''
'''
batch_size=128, epochs=2
0.8858
'''
'''
x = np.array(list(pn['sent']))[:15000]
y = np.array(list(pn['mark']))[:15000]
xt = np.array(list(pn['sent']))[15000:]
yt = np.array(list(pn['mark']))[15000:]
'''

yt = yt.reshape(-1, 1)

print x.shape, y.shape
print xt.shape, yt.shape

print('Build model...')
model = Sequential()
model.add(Embedding(len(dict) + 1, 256))
# 也可以使用GRU
model.add(LSTM(128))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])

model.fit(x, y, batch_size=16, epochs=2, validation_split = 0.1) #训练时间为若干个小时 # nb_epoch=1

classes = model.predict_classes(xt)
print "classes: "
print type(classes)
print classes.shape
# print classes[0:10]
print "yt: "
print type(yt)
print yt.shape
# print yt[0:10]

#acc = np_utils.accuracy(classes, yt)
acc_bool = np.equal(classes, yt)
acc = np.mean(acc_bool)
print('Test accuracy:', acc)
