import tensorflow as tf


def weight_variable(shape, name="W"):
    # initial = tf.truncated_normal(shape, stddev=0.1, mean=1 / sqrt(fan_out))
    # return tf.Variable(initial)
    xavier = tf.contrib.layers.xavier_initializer()
    return tf.get_variable(name, shape=shape, initializer=xavier)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def conv2d(x, W, name="conv2d"):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME', name=name)


def conv2d_layer(image, L, channels, n_filters, layer_name):
    with tf.name_scope(layer_name):
        weight = tf.Variable(tf.truncated_normal([L, L, channels, n_filters], 0.1, 0.1))
        bias = tf.Variable(tf.truncated_normal([n_filters], 0.1, 0.1))
        # output = tf.nn.relu(
        #      conv2d(image, weight) + bias
        # )
        output = conv2d(image, weight) + bias
        return weight, bias, output
