import tensorflow as tf

with tf.Graph().as_default() as g:
    a = tf.constant([[1]])
    b = tf.constant([[2]])

    conv = tf.matmul(a, b)

    with tf.Session() as sess:
        result = sess.run(conv)
        print(result)
