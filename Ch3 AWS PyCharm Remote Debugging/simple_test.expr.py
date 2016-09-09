import tensorflow as tf

with tf.Graph().as_default() as g:
    a = tf.constant([[1], [2]])
    b = tf.constant([[3, 5]])

    conv = tf.matmul(a, b)

    with tf.Session() as sess:
        result = sess.run(conv)
        print(result)
        # result => [[3 5]
        #            [6 10]]
