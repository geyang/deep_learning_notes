import tensorflow as tf
import numpy as np

# Some tensor we want to print the value of
x = tf.placeholder(tf.float32, shape=[2, 2, 2])
a = np.array([[[1., 1.], [1., 1.]], [[2., 2.], [2., 2.]]])

m = tf.Print(x, [x])

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    m_eval = m.eval(session=sess, feed_dict={x: a})
