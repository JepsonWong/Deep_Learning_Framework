#coding:utf-8

import numpy as np
import math
import tensorflow as tf
import collections
import random
import zipfile
import os
import urllib

# Download the data.
url = 'http://mattmahoney.net/dc/'

def maybe_download(filename, expected_bytes):
    """Download a file if not present, and make sure it's the right size."""
    if not os.path.exists(filename):
        filename, _ = urllib.urlretrieve(url + filename, filename)

    # 查看下载的文件的信息
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified', filename)
    else:
        print(statinfo.st_size)
        raise Exception(
            'Failed to verify ' + filename + '. Can you get to it with a browser?')
    return filename

filename = maybe_download('text8.zip', 31344016)

# Read the data into a list of strings.
def read_data(filename):
    """Extract the first file enclosed in a zip file as a list of words"""
    with zipfile.ZipFile(filename) as f:
        print f.namelist()
        print f.namelist()[0]
        # 读取压缩包中的第一个文件f.namelist()[0]
        first = f.read(f.namelist()[0])
        print type(first)
        data = tf.compat.as_str(first).split()
    return data

words = read_data(filename)
print('Data size', len(words))

# 单词表大小
vocabulary_size = 50000

# Read the data into a list of strings.
def build_dataset(words):
    count = [['UNK', -1]]
    # collections的Counter类，求用来跟踪值出现了多少次。
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    # 单词字典，word和单词的编号
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    # dictionary是key为单词，value为数字的dict
    # reverse_dictionary是key为数字，value为单词的dict
    # data是我们得到的单词集合
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary

data, count, dictionary, reverse_dictionary = build_dataset(words)
print "size of count、dictionary、reverse_dictionary:"
print len(count)
print len(dictionary)
print len(reverse_dictionary)
del words  # Hint to reduce memory.
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

data_index = 0

# Function to generate a training batch for the skip-gram model.
def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    # batch_size是num_skips的整数倍，保证每个batch包含了一个词汇对应的所有样本
    assert batch_size % num_skips == 0
    # 一个词汇对应的样本数量不能大雨skip_window的两倍
    assert num_skips <= 2 * skip_window
    # 初始化batch和labels
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    # 初始化deque，大小为span
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    for i in range(batch_size // num_skips):
        target = skip_window  # target label at the center of the buffer
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            # 随机选取一个目标单词，但要求不能是wkip_window上的单词或者已经选取过的单词
            while target in targets_to_avoid:
            # 随机初始化数字，得到目标单词的索引
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            # 生成batch和labels单个样本
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, labels

batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
for i in range(8):
  print(batch[i], reverse_dictionary[batch[i]],
      '->', labels[i, 0], reverse_dictionary[labels[i, 0]])


batch_size = 128
# 单词向量的维度
embedding_size = 128
skip_windows = 1
num_skips = 2

# 验证时候需要参数
valid_size = 16
valid_window = 100
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
# 计算nce_loss需要参数
num_sampled = 64

graph = tf.Graph()
with graph.as_default():

    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    # 限定计算在CPU上执行
    with tf.device('/cpu:0'):
        # 随机生成所有单词的词向量 50000*128
        embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, -1.0))
        # 生成batch_size*128的batch样本
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)

        print embeddings.shape
        print embed.shape

        # 50000*128的nce_loss参数
        nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size],
                                                      stddev=1.0 / math.sqrt(embedding_size)))
        # 50000维度的nce_bias
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

    # tf.nn.nce_loss()计算学习出的词向量embeddind在训练数据上的loss
    loss = tf.reduce_mean(
        tf.nn.nce_loss(weights=nce_weights,
                       biases=nce_biases,
                       labels=train_labels,
                       inputs=embed,
                       num_sampled=num_sampled,
                       num_classes=vocabulary_size))

    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
    similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

    init = tf.global_variables_initializer()

num_steps = 100001
with tf.Session(graph=graph) as session:
    init.run()
    print("Initialized")

    average_loss = 0
    for step in range(num_steps):
        batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_windows)
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

        _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += loss_val

        if step % 2000 == 0:
            if step > 0:
                average_loss /= 2000
            print("Average loss at step", step, ": ", average_loss)
            average_loss = 0

        if step % 10000 == 0:
            sim = similarity.eval()
            for i in range(valid_size):
                valid_word = reverse_dictionary[valid_examples[i]]
                top_k = 8
                nearst = (-sim[i, :]).argsort()[1:top_k+1]
                log_str = "Nearst to %s" % valid_word
                for k in range(top_k):
                    closed_word = reverse_dictionary[nearst[k]]
                    log_str = "%s, %s" % (log_str, closed_word)
                print(log_str)
    final_embeddings = normalized_embeddings.eval()

# Visualize the embeddings.
def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    plt.figure(figsize=(18, 18))  #in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i,:]
        plt.scatter(x, y)
        plt.annotate(label,
                    xy=(x, y),
                    xytext=(5, 2),
                    textcoords='offset points',
                    ha='right',
                    va='bottom')
    plt.savefig(filename)

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
plot_only = 100
low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only,:])
labels = [reverse_dictionary[i] for i in range(plot_only)]
plot_with_labels(low_dim_embs, labels)
