#encoding:utf-8
import tensorflow as tf

x = tf.Variable(0.0)
x_plus_1 = tf.assign_add(x, 1)#对x进行加1操作
#tf.control_dependencies的作用是:在执行y=x前，先执行x_plus_1
with tf.control_dependencies([x_plus_1]):
	y = tf.identity(x) #y = x #y = x_plus_1
init = tf.initialize_all_variables()

with tf.Session() as session:
    init.run()
    for i in xrange(5):
        print(y.eval())
