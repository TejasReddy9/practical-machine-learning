import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

x1 = tf.constant(5)
x2 = tf.constant(6)
# result = x1 * x2
result = tf.multiply(x1,x2) 
print(result)

# sess = tf.Session()
# print(sess.run(result))

with tf.Session() as sess:
	output = sess.run(result)   # python variable output
	print(output)

print(output)
# print(sess.run(result))  # outside session.. error