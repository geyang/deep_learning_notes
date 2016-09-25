import tensorflow as tf

xys = tf.constant([
    [0, 0],
    [0, 1],
    [0, 2]
])

r = tf.reshape(tf.reduce_sum(xys * xys, 1), [-1, 1])

D = r - 2 * tf.matmul(xys, tf.transpose(xys)) + tf.transpose(r)

with tf.Session() as sess:
    result = sess.run(D)

print(result)
