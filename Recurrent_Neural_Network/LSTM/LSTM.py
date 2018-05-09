# encoding:utf-8
'''
https://blog.csdn.net/a343902152/article/details/54429096
https://blog.csdn.net/u014595019/article/details/52759104#reply 代码分析
'''
import reader
import tensorflow as tf
import time
import numpy as np


class PTBInput(object):
    def __init__(self, config, data, name=None):
        self.batch_size = batch_size = config.batch_size;
        self.num_steps = num_steps = config.num_steps;
        self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
        self.input_data, self.targets = reader.ptb_producer(data, batch_size, num_steps, name=name)


class PTBModel(object):
    def __init__(self, is_training, config, input_):
        self._input = input_
        batch_size = input_.batch_size
        num_steps = input_.num_steps
        size = config.hidden_size
        vocab_size = config.vocab_size
        '''
		使用BasicLSTMCell构建一个基础LSTM单元，然后根据keep_prob来为cell配置dropout。最后通过MultiRNNCell将num_layers个lstm_cell连接起来。
		在LSTM单元中，有2个状态值，分别是c和h。
		'''

        def lstm_cell():
            return tf.contrib.rnn.BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=True)

        attn_cell = lstm_cell
        if is_training and config.keep_prob < 1:
            def attn_cell():
                return tf.contrib.rnn.dropoutWrapper(lstm_cell(), output_keep_prob=config.keep_prob)

        cell = tf.contrib.rnn.MultiRNNCell([attn_cell() for _ in range(config.num_layers)], state_is_tuple=True)
        self._initial_state = cell.zero_state(batch_size, tf.float32)

        # embedding部分 如果为训练状态，后面加一层Dropout
        with tf.device("/cpu:0"):
            embedding = tf.get_variable("embedding", [vocab_size, size + 50], dtype=tf.float32)
            inputs = tf.nn.embedding_lookup(embedding, input_.input_data)
        if is_training and config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, config.keep_prob)
        # 定义输出output，lstm的中间变量复用
        # LSTM循环
        outputs = []
        state = self._initial_state
        with tf.variable_scope("RNN"):
            for time_step in range(num_steps):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(inputs[:, time_step, :], state)
                outputs.append(cell_output)
        print "cell_output shape: ", cell_output.shape
        print "state shape: ", len(state)
        print "state[0] type: ", type(state[0])
        print "state[1] type: ", type(state[1])
        print "state[0]: ", (state[0])
        print "state[1]: ", (state[1])

        # 得到网络最后的输出
        # # 把之前的list展开，成[batch, hidden_size*num_steps]，然后 reshape，成[batch*numsteps, hidden_size]。
        output = tf.reshape(tf.concat(outputs, 1), [-1, size])
        softmax_w = tf.get_variable("softmax_w", [size, vocab_size], dtype=tf.float32)
        softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=tf.float32)
        # softmax_w，shape=[hidden_size, vocab_size]，用于将distributed表示的单词转化为one-hot表示。
        logits = tf.matmul(output, softmax_w) + softmax_b
        # 带权重的交叉熵计算
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits], [tf.reshape(input_.targets, [-1])],
                                                                  [tf.ones([batch_size * num_steps], dtype=tf.float32)])
        self._cost = cost = tf.reduce_sum(loss) / batch_size
        self._final_state = state
        if not is_training:
            return

        # 定义学习率，学习率用tf.Variable定义，模型之间并没有复用
        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        # 梯度修剪
        # 修正梯度值，用于控制梯度爆炸的问题。梯度爆炸和梯度弥散的原因一样，都是因为链式法则求导的关系，导致梯度的指数级衰减。为了避免梯度爆炸，需要对梯度进行修剪。
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), config.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self._lr)
        self._train_op = optimizer.apply_gradients(zip(grads, tvars),
                                                   global_step=tf.contrib.framework.get_or_create_global_step())

        # 更改学习率
        self._new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    @property
    def input(self):
        return self._input

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def cost(self):
        return self._cost

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op


class SmallConfig(object):
    """
	Small config.
	num_steps为LSTM展开的步数。
	batch_size为一个batch的数据个数。
	num_layers为LSTM的层数。
	hidden_size为隐层单元数目，每个词会表示成[hidden_size]大小的向量，当然也不一定单词成都必须为[hidden_size]。
	max_epoch：当epoch < max_epoch时，lr_decay值=1；epoch > max_epoch时，lr_decay逐渐减小。
	max_max_epoch为epoch次数。
	"""
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 20
    hidden_size = 200
    max_epoch = 4
    max_max_epoch = 13
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    vocab_size = 10000


def run_epoch(session, model, eval_op=None, verbose=False):
    start_time = time.time()
    costs = 0.0
    iters = 0
    state = session.run(model.initial_state)

    fetches = {"cost": model.cost, "final_state": model.final_state, }
    if eval_op is not None:
        fetches["eval_op"] = eval_op

    for step in range(model.input.epoch_size):
        feed_dict = {}
        for i, (c, h) in enumerate(model.initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h
        vals = session.run(fetches, feed_dict)
        cost = vals["cost"]
        state = vals["final_state"]

        costs += cost
        iters += model.input.num_steps

        if verbose and step % (model.input.epoch_size // 10) == 10:
            print("%.3f perplexity: %.3f speed: %.0f wps" % (step * 1.0 / model.input.epoch_size, np.exp(costs / iters),
                                                             iters * model.input.batch_size / (
                                                             time.time() - start_time)))
    return np.exp(costs / iters)


# 读取reader读取数据内容，将单词转为唯一的数字编码，以便神经网络进行处理。返回的raw_data是包含4个元素的tuple，前三个元素均为list，分别是训练、验证、测试文本的转化为数字编码的单词串。最后一个元素为数字，表示文本库中的单词数量。

raw_data = reader.ptb_raw_data('simple-examples/data/')

print type(raw_data)
print len(raw_data)
print type(raw_data[0])
print len(raw_data[0])
print type(raw_data[1])
print len(raw_data[1])
print type(raw_data[2])
print len(raw_data[2])
print type(raw_data[3])
print raw_data[3]

train_data, valid_data, test_data, _ = raw_data

config = SmallConfig()
eval_config = SmallConfig()
eval_config.batch_size = 1
eval_config.num_steps = 1

with tf.Graph().as_default():
    initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)
    with tf.name_scope("Train"):
        train_input = PTBInput(config=config, data=train_data, name="TrainInput")
        print type(train_input)
        print type(train_input.input_data)
        print train_input.input_data.shape
        print type(train_input.targets)
        print train_input.targets.shape
        with tf.variable_scope("Model", reuse=None, initializer=initializer):
            m = PTBModel(is_training=True, config=config, input_=train_input)
    with tf.name_scope("Valid"):
        valid_input = PTBInput(config=config, data=valid_data, name="ValidInput")
        with tf.variable_scope("Model", reuse=True, initializer=initializer):
            mvalid = PTBModel(is_training=False, config=config, input_=valid_input)
    with tf.name_scope("Test"):
        test_input = PTBInput(config=eval_config, data=test_data, name="TestInput")
        with tf.variable_scope("Model", reuse=True, initializer=initializer):
            mtest = PTBModel(is_training=False, config=eval_config, input_=test_input)

    sv = tf.train.Supervisor()
    with sv.managed_session() as session:
        for i in range(config.max_max_epoch):
            lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
            m.assign_lr(session, config.learning_rate * lr_decay)
            print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
            train_perplexity = run_epoch(session, m, eval_op=m.train_op, verbose=True)
            print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
            valid_perplexity = run_epoch(session, mvalid)
            print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

        test_perplexity = run_epoch(session, mtest)
        print("Test Perplexity: %.3f" % test_perplexity)
