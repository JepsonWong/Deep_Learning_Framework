#coding:utf-8
import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("/Users/wangzhongpu/Library/Mobile Documents/com~apple~CloudDocs/Documents/GitHub/Tensor-Flow-Examples/Auto_Encoder/MNIST_data", one_hot=True)

class Multi_Layer_Percetron(object):
	def __init__(self, in_units, h1_units, out_units):
		self.in_units = in_units
		self.h1_units = h1_units
		self.out_units = out_units
		all_weights = self._initialize_weights()
		self.w1 = all_weights['w1']
		self.b1 = all_weights['b1']
		self.w2 = all_weights['w2']
		self.b2 = all_weights['b2']
		self.x = tf.placeholder(tf.float32, [None, in_units])
		self.keep_prob = tf.placeholder(tf.float32)
		
		# 定义神经网络forward
		self.hidden1 = tf.nn.relu(tf.matmul(self.x, self.w1) + self.b1)
		self.hidden1_drop = tf.nn.dropout(self.hidden1, self.keep_prob)
		self.y = tf.nn.softmax(tf.matmul(self.hidden1_drop, self.w2) + self.b2)
		
		# 定义损失函数和选择优化器来优化loss
		self.y_ = tf.placeholder(tf.float32, [None, self.out_units])
		self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.log(self.y), reduction_indices=[1]))
		self.train_step = tf.train.AdagradOptimizer(0.3).minimize(self.cross_entropy)
		
		# 训练步骤1
		self.sess = tf.InteractiveSession()
		tf.global_variables_initializer().run()
		
		# 评价模型
		self.correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
		self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

	# 参数初始化
	def _initialize_weights(self):
		all_weights = dict()
		all_weights['w1'] = tf.Variable(tf.truncated_normal([self.in_units, self.h1_units], stddev=0.1))
		all_weights['b1'] = tf.Variable(tf.zeros([self.h1_units]))
		all_weights['w2'] = tf.Variable(tf.zeros([self.h1_units, self.out_units]))
		all_weights['b2'] = tf.Variable(tf.zeros([self.out_units]))
		return all_weights
	
	# 训练步骤2
	def partial_fit(self, X, y, keep_prob):
		self.train_step.run({self.x: X, self.y_: y, self.keep_prob: keep_prob})

	# 评价模型
	def accuracy_all(self, X, y, keep_prob):
		return self.accuracy.eval({self.x: X, self.y_: y, self.keep_prob: keep_prob})

# 参数初始化
in_units = 784
h1_units = 300
out_units = 10
keep_prob = 0.75

multi_layer_perception = Multi_Layer_Percetron(in_units, h1_units, out_units)

for i in range(30):
	print i
	batch_xs, batch_ys = mnist.train.next_batch(100)
	multi_layer_perception.partial_fit(batch_xs, batch_ys, keep_prob)

Test1 = mnist.test.images
Test2 = mnist.test.labels
print multi_layer_perception.accuracy_all(Test1, Test2, 1.0)

# print accuracy
