import tensorflow as tf

all_zero = [0, 0, 0, 0]
with tf.Graph().as_default(), tf.Session() as sess:
    argmax = tf.argmax(
        tf.constant(all_zero, dtype=tf.float64),
        dimension=0
    )
    result = sess.run(argmax)
    assert result != 0, 'argmax returning the first index is a wrong behavior.'
    print(result)
