import sys, os

sys.path.append(os.getcwd())

import math
import tensorflow as tf
import tf_helpers as helpers
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


### helper functions
def layer(layer_name, x):
    # image_dimension = [784]
    image_dimension = helpers.get_shape_array(x)
    assert isinstance(image_dimension, list), 'dimension has to be list.'
    with tf.name_scope(layer_name):
        # initializing at 0 is no-good.
        weight = tf.Variable(
            tf.truncated_normal(image_dimension[1:] + [10],
                                stddev=1.0 / math.sqrt(float(image_dimension[1]))),
            name='weight')
        bias = tf.Variable(tf.zeros([10]), name='bias')
        activation = tf.matmul(x, weight) + bias
    return weight, bias, activation


def inference():
    image = tf.placeholder(tf.float32, shape=[None, 784], name='input')
    w, b, output = layer('output_layer', image)
    return image, output


def loss(logits):
    batch_labels = tf.placeholder(tf.float32, name='labels')
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits, tf.argmax(batch_labels, dimension=1), name='xentropy')
    # cross_entropy = - tf.reduce_sum(
    #     batch_labels * tf.log(logits),
    #     reduction_indices=[1]
    # )
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


BATCH_SIZE = 250
if __name__ == "__main__":
    with tf.Graph().as_default() as g:
        input, logits = inference()
        labels, loss_op = loss(logits)
        train = training(loss_op, 1e-3)
        eval = evaluation(logits, labels)

        init = tf.initialize_all_variables()

        with tf.Session() as sess:
            sess.run(init)
            for i in range(30000):
                batch_xs, batch_labels = mnist.train.next_batch(BATCH_SIZE)
                sess.run(train, feed_dict={
                    input: batch_xs,
                    labels: batch_labels
                })
                if i % 1000 == 0:
                    print('---------------------------------')
                    output, loss_value, accuracy = sess.run([logits, loss_op, eval], feed_dict={
                        input: batch_xs,
                        labels: batch_labels
                    })
                    print("accuracy is ", accuracy / BATCH_SIZE)
                    # print("loss is ", loss_value)
                    # print("output is ", output[0], batch_labels[0])
