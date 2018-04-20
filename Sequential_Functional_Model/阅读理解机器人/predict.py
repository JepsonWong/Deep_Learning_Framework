#encoding:utf-8

from __future__ import print_function
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers import Activation, Dense, Merge, Permute, Dropout
from keras.layers import LSTM
from keras.utils.data_utils import get_file
from keras.preprocessing.sequence import pad_sequences
from functools import reduce
import tarfile
import numpy as np
import re
import os
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from keras.models import load_model

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
            story.append('')
        else:
            sent = tokenize(line)
            story.append(sent)
    return data

def parse_stories_target(lines, only_supporting=False):
    '''Parse stories provided in the bAbi tasks format

    If only_supporting is true, only the sentences that support the answer are kept.
    '''
    data = []
    story = []
    for line in lines:
        line = line.decode('utf-8').strip()
        nid, line = line.split(' ', 1)
        nid = int(nid)
        if nid == 1:
            story = []
        print(line)
        sent = tokenize(line)
        story.append(sent)
    substory = story[:len(story) - 1]
    query = story[-1]
    data.append((substory, query))
    return data

def get_stories(f, only_supporting=False, max_length=None):
    '''Given a file name, read the file, retrieve the stories, and then convert the sentences into a single story.

    If max_length is supplied, any stories longer than max_length tokens will be discarded.
    '''
    data = parse_stories(f.readlines(), only_supporting=only_supporting)
    flatten = lambda data: reduce(lambda x, y: x + y, data)
    data = [(flatten(story), q, answer) for story, q, answer in data if not max_length or len(flatten(story)) < max_length]
    return data

def get_stories_target(f, only_supporting=False, max_length=None):
    '''Given a file name, read the file, retrieve the stories, and then convert the sentences into a single story.

    If max_length is supplied, any stories longer than max_length tokens will be discarded.
    '''
    data = parse_stories_target(f.readlines(), only_supporting=only_supporting)
    print('parse_stories_target')
    print(data)
    flatten = lambda data: reduce(lambda x, y: x + y, data)
    data = [(flatten(story), q) for story, q in data if not max_length or len(flatten(story)) < max_length]
    return data

def vectorize_stories(data, word_idx, story_maxlen, query_maxlen):
    X = []
    Xq = []
    for story, query in data:
        x = [word_idx[w] for w in story]
        xq = [word_idx[w] for w in query]
        X.append(x)
        Xq.append(xq)
    return (
            pad_sequences(X, maxlen=story_maxlen),
            pad_sequences(Xq, maxlen=query_maxlen)
        )

path = 'babi_tasks_1-20_v1-2.tar.gz'
if not os.path.isfile(path):
	print("babi_tasks_1-20_v1-2.tar.gz not exist")
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

train_stories = get_stories(tar.extractfile(challenge.format('train')))
test_stories = get_stories(tar.extractfile(challenge.format('test')))

vocab = sorted(reduce(lambda x, y: x | y, (set(story + q + [answer]) for story, q, answer in train_stories + test_stories)))

vocab_size = len(vocab) + 1
story_maxlen = max(map(len, (x for x, _, _ in train_stories + test_stories)))
query_maxlen = max(map(len, (x for _, x, _ in train_stories + test_stories)))

word_idx = dict((c, i + 1) for i, c in enumerate(vocab))

# get the vec of the target story we input
with open("./target.story") as fs:
	target_input = get_stories_target(fs)
	target_storys, target_queries = vectorize_stories(target_input, word_idx, story_maxlen, query_maxlen)

# laod the model
answer_model = load_model('./checkpoint.hdf5')
answer_output = answer_model.predict([target_storys, target_queries], batch_size=1, verbose=1)
print(answer_output)
print(type(answer_output))
print(answer_output.shape)
