#encoding:utf-8

'''
语料库：https://s3.amazonaws.com/text-datasets/babi_tasks_1-20_v1-2.tar.gz
'''

import os
import tarfile
import re
import numpy as np

from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers import Activation, Dense, Merge, Permute, Dropout
from keras.layers import LSTM
from keras.utils.data_utils import get_file
from keras.preprocessing.sequence import pad_sequences
from functools import reduce
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau

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
print('inputs_test[0]')
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
input_encoder_m = Sequential()
input_encoder_m.add(Embedding(input_dim=vocab_size, output_dim=64, input_length=story_maxlen))
input_encoder_m.add(Dropout(0.3)) # output: (samples, story_maxlen, embedding_dim)

# embed the question into a sequence of vectors
question_encoder = Sequential()
question_encoder.add(Embedding(input_dim=vocab_size, output_dim=64, input_length=query_maxlen))
question_encoder.add(Dropout(0.3)) # output: (samples, query_maxlen, embedding_dim)

# compute a 'match' between input sequence elements (which are vectors)
# and the question vector sequence
match = Sequential()
match.add(Merge([input_encoder_m, question_encoder], mode='dot', dot_axes=[2, 2]))
match.add(Activation('softmax')) # output: (samples, story_maxlen, query_maxlen)

# embed the input into a single vector with size = story_maxlen:
input_encoder_c = Sequential()
input_encoder_c.add(Embedding(input_dim=vocab_size,
                              output_dim=query_maxlen,
                              input_length=story_maxlen))
input_encoder_c.add(Dropout(0.3)) # output: (samples, story_maxlen, query_maxlen)

# sum the match vector with the input_encoder_c
response = Sequential()
response.add(Merge([match, input_encoder_c], mode='sum')) # output: (samples, story_maxlen, query_maxlen)
response.add(Permute((2, 1)))  # output: (samples, query_maxlen, story_maxlen)

# concatenate the match vector with the question vector,
# and do logistic regression on top
answer = Sequential()
# (none, 4, 68) + (none, 4, 64) = (none, 4, 132)
answer.add(Merge([response, question_encoder], mode='concat', concat_axis=-1))

answer.add(LSTM(32))
answer.add(Dropout(0.3))
answer.add(Dense(vocab_size))
answer.add(Activation('softmax'))

checkpointer = ModelCheckpoint(filepath="./checkpoint.hdf5", verbose=1)
lrate = ReduceLROnPlateau(min_lr=0.00001)

answer.compile(optimizer='rmsprop', loss='categorical_crossentropy',
               metrics=['accuracy'])

config = answer.get_config()
print config

answer.fit(
	[inputs_train, queries_train, inputs_train], answers_train,
	batch_size=32,
	epochs=1,
	validation_data=([inputs_test, queries_test, inputs_test], answers_test),
	callbacks=[checkpointer, lrate]
) # epochs=5000

print "after training: "

config = answer.get_config()
print config

'''
from keras.utils import plot_model
plot_model(answer, to_file='model.png')
'''
