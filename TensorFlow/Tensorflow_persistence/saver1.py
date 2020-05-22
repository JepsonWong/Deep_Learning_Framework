import tensorflow as tf

v1 = tf.constant(3.0, shape = [1])
v2 = tf.constant(4.0, shape = [1])

result = v1 + v2
init_op = tf.global_variables_initializer()

# saver = tf.train.Saver()

with tf.Session() as sess:
	sess.run(init_op, feed_dict = {v1:[6]})
	#saver.save(sess, "/Users/wangzhongpu/Library/Mobile Documents/com~apple~CloudDocs/Documents/GitHub/Tensor-Flow-Examples/Tensorflow_persistence/model/model.ckpt", global_step = 2)
	#saver.save(sess, "/Users/wangzhongpu/Library/Mobile Documents/com~apple~CloudDocs/Documents/GitHub/Tensor-Flow-Examples/Tensorflow_persistence/model/model.ckpt", global_step = 3)
	print sess.run(result, feed_dict = {result:[20]})
