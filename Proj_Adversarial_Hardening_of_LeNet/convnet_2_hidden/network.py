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

    with tf.name_scope('conv_layer_1'):
        W_conv1 = helpers.weight_variable([5, 5, 1, 32], 'W_conv1')
        b_conv1 = helpers.bias_variable([32], 'bias_conv1')
        # alphas_conv1 = helpers.bias_variable([32], 'alpha_conv1')
        layer_conv_1 = tf.nn.softplus(helpers.conv2d(image, W_conv1) + b_conv1)
        stage_1_pool = helpers.max_pool_2x2(layer_conv_1)

    with tf.name_scope('conv_layer_2'):
        W_conv3 = helpers.weight_variable([5, 5, 32, 64], "W_conv3")
        b_conv3 = helpers.bias_variable([64], 'bias_conv3')
        # alphas_conv3 = helpers.bias_variable([64], 'alpha_conv3')
        layer_conv_3 = tf.nn.softplus(helpers.conv2d(stage_1_pool, W_conv3) + b_conv3)
        stage_3_pool = helpers.max_pool_2x2(layer_conv_3)
        stage_3_pool_flat = tf.reshape(stage_3_pool, [-1, 7 * 7 * 64])

    with tf.name_scope('fc_layer_1'):
        W_fc1 = helpers.weight_variable([7 * 7 * 64, 500], "W_fc1")
        b_fc1 = helpers.bias_variable([500], 'bias_fc1')
        h_fc1 = tf.nn.softplus(tf.matmul(stage_3_pool_flat, W_fc1) + b_fc1)

    with tf.name_scope('fc_output'):
        W_output = helpers.weight_variable([500, 2], "W_putput")
        b_output = helpers.bias_variable([2], 'bias_output')
        output = tf.nn.softplus(tf.matmul(h_fc1, W_output) + b_output)

    # with tf.name_scope('output'):
    #     W_output = helpers.weight_variable([2, 10], "W_output")
    #     b_output = helpers.bias_variable([10])
    #     output = tf.nn.relu(tf.matmul(h_fc2, W_output) + b_output)

    return x, output


def loss(deep_features):
    with tf.name_scope('softmax_loss'):
        batch_labels = tf.placeholder(tf.float32, name='labels')
        W_loss = helpers.weight_variable([2, 10], "W_loss")
        # Note: we don't use the bias here because it does not affect things. removing the
        #       bias also makes the analysis simpler.
        # tf.nn.
        logits = tf.nn.relu(tf.matmul(deep_features, W_loss))
        cross_entropy = - tf.reduce_mean(
                tf.mul(batch_labels, tf.nn.log_softmax(logits)),
                reduction_indices=[1]
        )

        return batch_labels, logits, tf.reduce_mean(cross_entropy)


def training(loss, learning_rate):
    # tf.scalar_summary(loss.op.name, loss)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(
        tf.clip_by_value(
            loss,
            1e-10,
            1e10,
        )
    )
    return train_op


def evaluation(logits, labels):
    correct = tf.nn.in_top_k(logits, tf.cast(tf.argmax(labels, dimension=1), dtype=tf.int32), 1)
    # Return the number of true entries.
    return tf.reduce_sum(tf.cast(correct, tf.int32))
