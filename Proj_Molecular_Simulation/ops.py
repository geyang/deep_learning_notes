import tensorflow as tf


def r_ij(xys):
    """
    calculate the distance matrix between each pair of the xy locations.

    :param xys:
    :return:
    """
    with tf.name_scope('r_ij'):
        r = tf.reshape(tf.reduce_sum(xys * xys, 1), [-1, 1])
        D = tf.sqrt(r - 2 * tf.matmul(xys, tf.transpose(xys)) + tf.transpose(r))
        return D
