import tensorflow as tf

a = tf.gfile.FastGFile("./110472418_87b6a3aa98_m.jpg","rb").read()
print len(a)
# print a.shape

b = open("./110472418_87b6a3aa98_m.jpg","rb")
b_all = b.read()
print len(b_all)
# print b_all
b.close()

c = open("./110472418_87b6a3aa98_m.jpg","r")
c_all = c.read()
print len(c_all)
c.close()

with tf.Session() as sess:
	img_data = tf.image.decode_jpeg(a)
	a1 = sess.run(img_data)
	print a1.shape
	print a1.ndim
