import tensorflow as tf


def r2_ij(xys):
    """
    calculate the distance^2 matrix between each pair of the xy locations.

    :param xys:
    :return:
    """
    with tf.name_scope('r2_ij'):
        r = tf.reshape(
            tf.reduce_sum(
                tf.mul(xys, xys),
                1
            ),
            [-1, 1]
        )

        return r + tf.transpose(r) - 2.0 * tf.matmul(
                xys,
                tf.transpose(xys)
            )
