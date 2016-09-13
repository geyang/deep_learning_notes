import sys, os

sys.path.append(os.getcwd())

import math
import tensorflow as tf
import tf_helpers as helpers
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


### helper functions
def layer(layer_name, x, dimension):
    # image_dimension = [784]
    image_dimension = helpers.get_shape_array(x)
    assert isinstance(image_dimension, list), 'dimension has to be list.'
    with tf.name_scope(layer_name):
        # initializing at 0 is no-good.
        weight = tf.Variable(
            tf.truncated_normal(image_dimension[1:] + dimension,
                                stddev=1.0 / math.sqrt(float(image_dimension[1]))),
            name='weight')
        bias = tf.Variable(tf.zeros(dimension), name='bias')
        activation = tf.matmul(x, weight) + bias
    return weight, bias, activation


def neural_net_inference():
    image = tf.placeholder(tf.float32, shape=[None, 784], name='input')
    w_0, b_0, hidden_layer = layer('hidden_layer', image, dimension=[30])
    w_out, b_out, output = layer('output_layer', hidden_layer, dimension=[10])
    return image, output


def loss(logits):
    batch_labels = tf.placeholder(tf.float32, name='labels')
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits, tf.argmax(batch_labels, dimension=1), name='xentropy')
    # cross_entropy = - tf.reduce_sum(
    #     batch_labels * tf.log(logits),
    #     reduction_indices=[1]
    # )
    loss_op = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    tf.scalar_summary(loss_op.op.name, loss_op)
    return batch_labels, loss_op


def training(loss, learning_rate):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(loss)
    return train_op


def evaluation(logits, labels):
    correct = tf.nn.in_top_k(logits, tf.cast(tf.argmax(labels, dimension=1), dtype=tf.int32), 1)
    # Return the number of true entries.
    return tf.reduce_sum(tf.cast(correct, tf.int32))


BATCH_SIZE = 250
SUMMARIES_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    with tf.Graph().as_default() as g:
        input, logits = neural_net_inference()
        labels, loss_op = loss(logits)
        train = training(loss_op, 1e-1)
        eval = evaluation(logits, labels)

        init = tf.initialize_all_variables()

        with tf.Session() as sess:
            # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
            # to see the tensor graph, fire up the tensorboard with --logdir="./train"
            merged = tf.merge_all_summaries()
            train_writer = tf.train.SummaryWriter(SUMMARIES_DIR + '/train', sess.graph)
            test_writer = tf.train.SummaryWriter(SUMMARIES_DIR + '/test')

            sess.run(init)
            for i in range(300):
                batch_xs, batch_labels = mnist.train.next_batch(BATCH_SIZE)
                sess.run(train, feed_dict={
                    input: batch_xs,
                    labels: batch_labels
                })
                if i % 10 == 0:
                    print('---------------------------------')
                    output, loss_value, accuracy = sess.run([logits, loss_op, eval], feed_dict={
                        input: batch_xs,
                        labels: batch_labels
                    })
                    print("accuracy is ", accuracy / BATCH_SIZE)
                    # print("loss is ", loss_value)
                    # print("output is ", output[0], batch_labels[0])
