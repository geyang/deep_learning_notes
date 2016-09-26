import tensorflow as tf

r_ij = tf.constant([
    [0, 0.1],
    [0.01, -0.1],
    [-1, -1],

])

mask = tf.cast(
    r_ij,
    dtype=tf.bool
)

subject = \
    tf.reduce_sum(
        tf.select(
            mask,
            1 / r_ij,
            tf.zeros(shape=r_ij.get_shape())
        ),
        reduction_indices=[0, 1]
    )

with tf.Session() as sess:
    result = sess.run(subject)

print(result)
