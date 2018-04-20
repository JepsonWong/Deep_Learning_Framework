#encoding:utf-8

'''
语料库：https://s3.amazonaws.com/text-datasets/babi_tasks_1-20_v1-2.tar.gz
'''

import os
import tarfile
import re
import numpy as np

import keras
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers import Input, Activation, Dense, Merge, Permute, Dropout
from keras.layers import LSTM
from keras.utils.data_utils import get_file
from keras.preprocessing.sequence import pad_sequences
from functools import reduce
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from keras.models import Model

def tokenize(sent):
	'''Return the tokens of a sentence including punctuation.
	>>> tokenize('Bob dropped the apple. Where is the apple?')
    	['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    	'''
	return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]

def parse_stories(lines, only_supporting=False):
	'''Parse stories provided in the bAbi tasks format

    	If only_supporting is true, only the sentences that support the answer are kept.
    	'''
	# data里面的元素是一个tuple
	# eg. [(substory1, q1, a1), (substory2, q2, a2), ……]
    	data = []
    	story = []
    	for line in lines:
        	line = line.decode('utf-8').strip()
        	nid, line = line.split(' ', 1)
        	nid = int(nid)
        	if nid == 1:
            		story = []
        	if '\t' in line:
            		q, a, supporting = line.split('\t')
            		q = tokenize(q)
           	 	substory = None
			if only_supporting:
                		# Only select the related substory
                		supporting = map(int, supporting.split())
                		substory = [story[i - 1] for i in supporting]
            		else:
                		# Provide all the substories
                		substory = [x for x in story if x]
            		data.append((substory, q, a))
            		# 如果是only_supporting模式，那么需要找到依赖的句子。但是问句所在的句子没有添加到story里面，所以要加入''来假装是问句所在的句子，要不然上方[story[i - 1]就会出错。
			story.append('')
        	else:
            		sent = tokenize(line)
            		story.append(sent)
    	return data

# 将story、query和answer转换成ont hot向量
def vectorize_stories(data, word_idx, story_maxlen, query_maxlen):
	X = []
    	Xq = []
    	Y = []
	for story, query, answer in data:
		x = [word_idx[w] for w in story]
		xq = [word_idx[w] for w in query]
		y = np.zeros(len(word_idx) + 1)  # let's not forget that index 0 is reserved
		y[word_idx[answer]] = 1
		X.append(x)
		Xq.append(xq)
		Y.append(y)
	return (pad_sequences(X, maxlen=story_maxlen), pad_sequences(Xq, maxlen=query_maxlen), np.array(Y))

def get_stories(f, only_supporting=False, max_length=None):
	'''Given a file name, read the file, retrieve the stories, and then convert the sentences into a single story.
	If max_length is supplied, any stories longer than max_length tokens will be discarded.
    	'''
	data = parse_stories(f.readlines(), only_supporting=only_supporting)
    	# 将data中的list据连成一个整体的list
	flatten = lambda data: reduce(lambda x, y: x + y, data)
    	data = [(flatten(story), q, answer) for story, q, answer in data if not max_length or len(flatten(story)) < max_length]
    	return data

path = ''
try:
	if not os.path.isfile("babi_tasks_1-20_v1-2.tar.gz"):
		path = get_file('babi_tasks_1-20_v1-2.tar.gz', origin='https://s3.amazonaws.com/text-datasets/babi_tasks_1-20_v1-2.tar.gz')
	else:
		path = 'babi_tasks_1-20_v1-2.tar.gz'
except:
	print('Error downloading dataset, please download it manually')
	raise
if not os.path.isfile(path):
	print("babi_tasks_1-20_v1-2.tar.gz downlaod faild")
	exit()

tar = tarfile.open(path)

challenges = {
    # QA1 with 10,000 samples
    'single_supporting_fact_10k': 'tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_{}.txt',
    # QA2 with 10,000 samples
    'two_supporting_facts_10k': 'tasks_1-20_v1-2/en-10k/qa2_two-supporting-facts_{}.txt',
}
challenge_type = 'single_supporting_fact_10k'
challenge = challenges[challenge_type]

print('Extracting stories for the challenge:', challenge_type)

train_stories = get_stories(tar.extractfile(challenge.format('train')))
print len(train_stories)
test_stories = get_stories(tar.extractfile(challenge.format('test')))
print len(test_stories)

vocab = sorted(reduce(lambda x, y: x | y, (set(story + q + [answer]) for story, q, answer in train_stories + test_stories)))
print('vocab')
print(vocab)

# Reserve 0 for masking via pad_sequences
vocab_size = len(vocab) + 1
story_maxlen = max(map(len, (x for x, _, _ in train_stories + test_stories)))
query_maxlen = max(map(len, (x for _, x, _ in train_stories + test_stories)))
print "(vocab_size: {})".format(vocab_size)
print "(story_maxlen: {})".format(story_maxlen)
print "(query_maxlen: {})".format(query_maxlen)
print('Number of training stories:', len(train_stories))
print('Number of test stories:', len(test_stories))

print('Here\'s what a "story" tuple looks like (input, query, answer):')
print(train_stories[0])

# 向量化 
word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
inputs_train, queries_train, answers_train = vectorize_stories(train_stories, word_idx, story_maxlen, query_maxlen)
inputs_test, queries_test, answers_test = vectorize_stories(test_stories, word_idx, story_maxlen, query_maxlen)

print('inputs_train[0]')
print(inputs_train[0])
print('queries_train[0]')
print(queries_train[0])
print('answers_train[0]')
print(answers_train[0])
print ('inputs_test[0]')
print(inputs_test[0])
print('queries_test[0]')
print(queries_test[0])
print('answers_test[0]')
print(answers_test[0])

print('-')
print('inputs: integer tensor of shape (samples, max_length)')
print('inputs_train shape:', inputs_train.shape)
print('inputs_test shape:', inputs_test.shape)
print('-')
print('queries: integer tensor of shape (samples, max_length)')
print('queries_train shape:', queries_train.shape)
print('queries_test shape:', queries_test.shape)
print('-')
print('answers: binary (1 or 0) tensor of shape (samples, vocab_size)')
print('answers_train shape:', answers_train.shape)
print('answers_test shape:', answers_test.shape)
print('-')

# 构建模型
# embed the input sequence into a sequence of vectors
inputs = Input(shape=(68,))
input_encoder_m = Embedding(input_dim=vocab_size, output_dim=64, input_length=story_maxlen)(inputs)
input_encoder_m = Dropout(0.3)(input_encoder_m) # output: (samples, story_maxlen, embedding_dim)

# embed the question into a sequence of vectors
queries = Input(shape=(4,))
question_encoder = Embedding(input_dim=vocab_size, output_dim=64, input_length=query_maxlen)(queries)
question_encoder = Dropout(0.3)(question_encoder) # output: (samples, query_maxlen, embedding_dim)

# compute a 'match' between input sequence elements (which are vectors)
# and the question vector sequence
match = keras.layers.merge([input_encoder_m, question_encoder], mode='dot', dot_axes=[2, 2])
match = Activation('softmax')(match) # output: (samples, story_maxlen, query_maxlen)

# embed the input into a single vector with size = story_maxlen:
input_encoder_c = Embedding(input_dim=vocab_size, output_dim=query_maxlen, input_length=story_maxlen)(inputs)
input_encoder_c = Dropout(0.3)(input_encoder_c) # output: (samples, story_maxlen, query_maxlen)

# sum the match vector with the input_encoder_c
response = keras.layers.merge([match, input_encoder_c], mode='sum') # output: (samples, story_maxlen, query_maxlen)
response = Permute((2, 1))(response)  # output: (samples, query_maxlen, story_maxlen)

# concatenate the match vector with the question vector,
# and do logistic regression on top
# (none, 4, 68) + (none, 4, 64) = (none, 4, 132)
answer = keras.layers.merge([response, question_encoder], mode='concat', concat_axis=-1)

answer = LSTM(32)(answer)
answer = Dropout(0.3)(answer)
answer = Dense(vocab_size)(answer)
answer = Activation('softmax')(answer)

answer_model = Model(inputs=[inputs, queries], outputs=[answer])

checkpointer = ModelCheckpoint(filepath="./checkpoint.hdf5", verbose=1)
lrate = ReduceLROnPlateau(min_lr=0.00001)

answer_model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
               metrics=['accuracy'])

answer_model.fit(
	[inputs_train, queries_train], answers_train,
	batch_size=32,
	epochs=5,
	validation_data=([inputs_test, queries_test], answers_test),
	callbacks=[checkpointer, lrate]
) # epochs=5000

from keras.utils import plot_model
plot_model(answer_model, to_file='model1.png')
