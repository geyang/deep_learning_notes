import math
import tensorflow as tf
from termcolor import colored as c, cprint
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import helpers

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
        alphas_conv1 = helpers.bias_variable([32], 'alpha_conv1')
        layer_conv_1 = helpers.prelu(helpers.conv2d(image, W_conv1) + b_conv1, alphas_conv1)

        W_conv1_b = helpers.weight_variable([5, 5, 32, 32], 'W_conv1_b')
        b_conv1_b = helpers.bias_variable([32], 'bias_conv1_b')
        alphas_conv1_b = helpers.bias_variable([32], 'alpha_conv1_b')

        layer_conv_1_b = helpers.prelu(helpers.conv2d(layer_conv_1, W_conv1_b) + b_conv1_b, alphas_conv1_b)

        stage_1_pool = helpers.max_pool_2x2(layer_conv_1_b)

    with tf.name_scope('conv_layer_2'):
        W_conv2 = helpers.weight_variable([5, 5, 32, 64], "W_conv2")
        b_conv2 = helpers.bias_variable([64], 'bias_conv2')
        alphas_conv2 = helpers.bias_variable([64], 'alpha_conv2')
        layer_conv_2 = helpers.prelu(helpers.conv2d(stage_1_pool, W_conv2) + b_conv2, alphas_conv2)

        W_conv2_b = helpers.weight_variable([5, 5, 64, 64], "W_conv2_b")
        b_conv2_b = helpers.bias_variable([64], 'bias_conv2_b')
        alphas_conv2_b = helpers.bias_variable([64], 'alpha_conv2_b')
        layer_conv_2_b = helpers.prelu(helpers.conv2d(layer_conv_2, W_conv2_b) + b_conv2_b, alphas_conv2_b)

        stage_2_pool = helpers.max_pool_2x2(layer_conv_2_b)
        # stage_2_pool_flat = tf.reshape(stage_2_pool, [-1, 7 * 7 * 64])

    with tf.name_scope('conv_layer_3'):
        W_conv3 = helpers.weight_variable([5, 5, 64, 128], "W_conv3")
        b_conv3 = helpers.bias_variable([128], 'bias_conv3')
        alphas_conv3 = helpers.bias_variable([128], 'alpha_conv3')
        layer_conv_3 = helpers.prelu(helpers.conv2d(stage_2_pool, W_conv3) + b_conv3, alphas_conv3)

        # stage_3_pool = helpers.max_pool_2x2(layer_conv_3)
        # stage_3_pool_flat = tf.reshape(stage_3_pool, [-1, 4 * 4 * 256])

        W_conv3_b = helpers.weight_variable([5, 5, 128, 128], "W_conv3_b")
        b_conv3_b = helpers.bias_variable([128], 'bias_conv3_b')
        alphas_conv3_b = helpers.bias_variable([128], 'alpha_conv3_b')
        layer_conv_3_b = helpers.prelu(helpers.conv2d(layer_conv_3, W_conv3_b) + b_conv3_b, alphas_conv3_b)

        stage_3_pool = helpers.max_pool_2x2(layer_conv_3_b)
        stage_3_pool_flat = tf.reshape(stage_3_pool, [-1, 4 * 4 * 128])

    with tf.name_scope('fc_layer_1'):
        W_fc1 = helpers.weight_variable([4 * 4 * 128, 2], "W_fc1")
        # W_fc1 = helpers.weight_variable([7 * 7 * 64, 2], "W_fc1")
        b_fc1 = helpers.bias_variable([2], 'bias_fc1')
        alphas_fc1 = helpers.bias_variable([2], 'alpha_conv3')
        output = helpers.prelu(tf.matmul(stage_3_pool_flat, W_fc1) + b_fc1, alphas_fc1)

    # with tf.name_scope('fc_output'):
    #     W_output = helpers.weight_variable([500, 10], "W_putput")
    #     b_output = helpers.bias_variable([10], 'bias_output')
    #     output = tf.nn.relu(tf.matmul(h_fc1, W_output) + b_output)

    # with tf.name_scope('output'):
    #     W_output = helpers.weight_variable([2, 10], "W_output")
    #     b_output = helpers.bias_variable([10])
    #     output = tf.nn.relu(tf.matmul(h_fc2, W_output) + b_output)

    return x, output


def loss(deep_features):
    with tf.name_scope('softmax_loss'):
        batch_labels = tf.placeholder(tf.float32, name='labels')
        W_loss = helpers.weight_variable([2, 10], "W_loss")
        bias_loss = tf.Variable(
            tf.truncated_normal(shape=[10], stddev=1e-4, mean=1e-1), 'bias_loss')
        # Note: we don't use the bias here because it does not affect things. removing the
        #       bias also makes the analysis simpler.
        logits = tf.matmul(deep_features, W_loss) + bias_loss
        cross_entropy = - tf.reduce_mean(
            tf.mul(batch_labels, tf.nn.log_softmax(logits)),
            reduction_indices=[1]
        )
        xentropy_mean = tf.reduce_mean(cross_entropy, name="xentropy_mean")
        tf.scalar_summary(xentropy_mean.op.name, xentropy_mean)

    return batch_labels, logits, xentropy_mean


def center_loss(deep_features, labels):
    with tf.name_scope('center_loss'):
        features_expanded = tf.reshape(deep_features, shape=[-1, 2, 1])
        labels_expanded = tf.reshape(labels, shape=[-1, 1, 10])

        samples_per_label = tf.reduce_sum(
            labels_expanded,
            reduction_indices=[0]
        )
        centroids = \
            tf.reduce_sum(
                tf.reshape(deep_features, shape=[-1, 2, 1]) * \
                labels_expanded,
                reduction_indices=[0]
            ) / samples_per_label

        centroids_expanded = tf.reshape(centroids, shape=[1, 2, 10]) * labels_expanded

        spread = \
            tf.reduce_mean(
                tf.reduce_sum(
                    tf.square(
                        features_expanded * labels_expanded - centroids_expanded
                    ),
                    reduction_indices=[1, 2]
                )
            ) / 2.0

        tf.scalar_summary(spread.op.name, spread)
    return spread, centroids, spread


def training(loss):
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')
    with tf.name_scope('training'):
        global_step = tf.Variable(0, name='global_step', trainable=False)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        # optimizer = tf.train.AdamOptimizer(learning_rate)
        train_op = optimizer.minimize(loss, global_step=global_step)
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        # grads_and_vars = optimizer.compute_gradients(loss, tf.trainable_variables())
        # capped_grads_and_vars = [(tf.clip_by_value(grads, 1e-10, 1e10), vars) for grads, vars in grads_and_vars]
        # train_op = optimizer.apply_gradients(capped_grads_and_vars)
    return learning_rate, train_op, global_step


def evaluation(logits, labels):
    correct = tf.nn.in_top_k(logits, tf.cast(tf.argmax(labels, dimension=1), dtype=tf.int32), 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float64), name='accuracy')
    tf.scalar_summary(accuracy.op.name, accuracy)
    # Return the number of true entries.
    return accuracy
