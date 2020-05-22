import tensorflow as tf

v1 = tf.Variable(tf.constant(3.0, shape = [1]), name = "v1")
v2 = tf.Variable(tf.constant(4.0, shape = [1]), name = "v2")

result = v1 + v2
init_op = tf.global_variables_initializer()

saver = tf.train.Saver()

with tf.Session() as sess:
	sess.run(init_op)
	saver.save(sess, "/Users/wangzhongpu/Library/Mobile Documents/com~apple~CloudDocs/Documents/GitHub/Tensor-Flow-Examples/Tensorflow_persistence/model/model.ckpt", global_step = 2)
	saver.save(sess, "/Users/wangzhongpu/Library/Mobile Documents/com~apple~CloudDocs/Documents/GitHub/Tensor-Flow-Examples/Tensorflow_persistence/model/model.ckpt", global_step = 3)
	print sess.run(result)
