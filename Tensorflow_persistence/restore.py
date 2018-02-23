import tensorflow as tf

saver = tf.train.import_meta_graph("/Users/wangzhongpu/Library/Mobile Documents/com~apple~CloudDocs/Documents/GitHub/Tensor-Flow-Examples/Tensorflow_persistence/model/model.ckpt.meta")

with tf.Session() as sess:
	saver.restore(sess, "/Users/wangzhongpu/Library/Mobile Documents/com~apple~CloudDocs/Documents/GitHub/Tensor-Flow-Examples/Tensorflow_persistence/model/model.ckpt")
	print sess.run(tf.get_default_graph().get_tensor_by_name("add:0"))
