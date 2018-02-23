import tensorflow as tf

v1 = tf.Variable(tf.constant(1.0, shape = [1]), name = "v1")
v2 = tf.Variable(tf.constant(2.0, shape = [1]), name = "v2")

result_w = v1 * v2
# init_op = tf.global_variables_initializer()

saver = tf.train.Saver()

with tf.Session() as sess:
	# sess.run(init_op)
	
	print "hello"	
	ckpt = tf.train.get_checkpoint_state("./model")
	print ckpt
	print ckpt.model_checkpoint_path
	
	saver.restore(sess, "/Users/wangzhongpu/Library/Mobile Documents/com~apple~CloudDocs/Documents/GitHub/Tensor-Flow-Examples/Tensorflow_persistence/model/model.ckpt-2")
	print sess.run(result_w)
