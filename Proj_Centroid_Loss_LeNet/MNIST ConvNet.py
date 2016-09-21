import math
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


### helper functions
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def conv2d(x, W, name=None):
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

    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    layer_conv_1 = tf.nn.relu(conv2d(image, W_conv1) + b_conv1)
    stage_1_pool = max_pool_2x2(layer_conv_1)

    W_conv3 = weight_variable([5, 5, 32, 64])
    b_conv3 = bias_variable([64])
    layer_conv_3 = tf.nn.relu(conv2d(stage_1_pool, W_conv3) + b_conv3)
    stage_3_pool = max_pool_2x2(layer_conv_3)
    stage_3_pool_flat = tf.reshape(stage_3_pool, [-1, 7 * 7 * 64])

    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_fc1 = tf.nn.relu(tf.matmul(stage_3_pool_flat, W_fc1) + b_fc1)

    W_output = weight_variable([1024, 10])
    b_output = bias_variable([10])
    output = tf.nn.relu(tf.matmul(h_fc1, W_output) + b_output)

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


import os

BATCH_SIZE = 250
SUMMARIES_DIR = os.path.dirname(os.path.abspath(__file__))

### configure devices for this eval script.
USE_DEVICE = '/gpu:3'
session_config = tf.ConfigProto(log_device_placement=True)
session_config.gpu_options.allow_growth = True
# this is required if want to use GPU as device.
# see: https://github.com/tensorflow/tensorflow/issues/2292
session_config.allow_soft_placement = True

if __name__ == "__main__":
    with tf.Graph().as_default() as g:
        # inference()
        input, logits = inference()
        labels, loss_op = loss(logits)
        train = training(loss_op, 1e-1)
        eval = evaluation(logits, labels)

        init = tf.initialize_all_variables()

        with tf.Session(config=session_config) as sess:
            # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
            # to see the tensor graph, fire up the tensorboard with --logdir="./train"
            merged = tf.merge_all_summaries()
            train_writer = tf.train.SummaryWriter(SUMMARIES_DIR + '/summaries/train', sess.graph)
            test_writer = tf.train.SummaryWriter(SUMMARIES_DIR + '/summaries/test')

            sess.run(init)
            for i in range(300):
                batch_xs, batch_labels = mnist.train.next_batch(BATCH_SIZE)
                sess.run(train, feed_dict={
                    input: batch_xs,
                    labels: batch_labels
                })
                if i % 10 == 0:
                    output, loss_value, accuracy = sess.run([logits, loss_op, eval], feed_dict={
                        input: batch_xs,
                        labels: batch_labels
                    })
                    print("training accuracy is ", accuracy / BATCH_SIZE)

            # now let's test!
            TEST_BATCH_SIZE = np.shape(mnist.test.labels)[0]
            output, loss_value, accuracy = sess.run([logits, loss_op, eval], feed_dict={
                input: mnist.test.images,
                labels: mnist.test.labels
            })
            print("MNIST Test accuracy is ", accuracy / TEST_BATCH_SIZE)
