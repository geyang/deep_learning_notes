import math
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

from . import helpers

### helper functions
from functools import reduce


def fc_layer(x, weight_shape, bias_shape, layer_name):
    with tf.name_scope(layer_name):
        # initializing at 0 is no-good.
        norm = math.sqrt(float(
            reduce(lambda v, e: v * e, weight_shape)
        ))
        weight = tf.Variable(
            tf.truncated_normal(weight_shape,
                                mean=0.5,
                                stddev=1.0 / norm),
            name='weight')
        bias = tf.Variable(tf.zeros(bias_shape), name='bias')
        activation = tf.matmul(x, weight) + bias
    return weight, bias, activation


# main network build stages
def inference():
    x = tf.placeholder(tf.float32, shape=[None, 784], name='input')
    image = tf.reshape(x, [-1, 28, 28, 1])

    W_conv1 = helpers.weight_variable([5, 5, 1, 32])
    b_conv1 = helpers.bias_variable([32])
    layer_conv_1 = tf.nn.relu(helpers.conv2d(image, W_conv1) + b_conv1)
    stage_1_pool = helpers.max_pool_2x2(layer_conv_1)

    W_conv3 = helpers.weight_variable([5, 5, 32, 64])
    b_conv3 = helpers.bias_variable([64])
    layer_conv_3 = tf.nn.relu(helpers.conv2d(stage_1_pool, W_conv3) + b_conv3)
    stage_3_pool = helpers.max_pool_2x2(layer_conv_3)
    stage_3_pool_flat = tf.reshape(stage_3_pool, [-1, 7 * 7 * 64])

    W_fc1 = helpers.weight_variable([7 * 7 * 64, 200])
    b_fc1 = helpers.bias_variable([200])
    h_fc1 = tf.nn.relu(tf.matmul(stage_3_pool_flat, W_fc1) + b_fc1)

    W_fc2 = helpers.weight_variable([200, 2])
    b_fc2 = helpers.bias_variable([2])
    h_fc2 = tf.matmul(h_fc1, W_fc2) + b_fc2

    W_output = helpers.weight_variable([2, 10])
    b_output = helpers.bias_variable([10])
    output = tf.nn.relu(tf.matmul(h_fc2, W_output) + b_output)

    return x, output


def loss(logits):
    batch_labels = tf.placeholder(tf.float32, name='labels')
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits, tf.argmax(batch_labels, dimension=1), name='xentropy')
    return batch_labels, tf.reduce_mean(cross_entropy, name='xentropy_mean')


def training(loss, learning_rate):
    # tf.scalar_summary(loss.op.name, loss)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(loss)
    return train_op


def evaluation(logits, labels):
    correct = tf.nn.in_top_k(logits, tf.cast(tf.argmax(labels, dimension=1), dtype=tf.int32), 1)
    # Return the number of true entries.
    return tf.reduce_sum(tf.cast(correct, tf.int32))
