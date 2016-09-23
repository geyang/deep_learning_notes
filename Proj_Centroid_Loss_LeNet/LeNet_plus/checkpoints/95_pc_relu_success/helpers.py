import tensorflow as tf
from math import sqrt


def weight_variable(shape, name="W"):
    # initial = tf.truncated_normal(shape, stddev=1e-2, mean=1e-2)
    # return tf.Variable(initial, name)
    xavier = tf.contrib.layers.xavier_initializer()
    return tf.get_variable(name, shape=shape, initializer=xavier)


def bias_variable(shape, name):
    initial = tf.truncated_normal(shape=shape, stddev=0.1, mean=1 / sqrt(shape[0]))
    return tf.Variable(initial, name)


def prelu(z, alpha):
    return tf.sub(tf.nn.relu(z), tf.mul(alpha, tf.nn.relu(-z)))


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
