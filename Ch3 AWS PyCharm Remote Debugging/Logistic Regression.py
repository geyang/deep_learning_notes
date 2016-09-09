import sys, os

sys.path.append(os.getcwd())

import tensorflow as tf
import tf_helpers as helpers
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


### helper functions
def layer(layer_name, x):
    # image_dimension = [784]
    image_dimension = helpers.get_shape_array(x)
    assert isinstance(image_dimension, list), 'dimension has to be list.'
    # with tf.name_scope(layer_name):
    weight = tf.Variable(tf.zeros(image_dimension[1:] + [10]), name='weight')
    bias = tf.Variable(tf.zeros([10]), name='bias')
    activation = tf.nn.relu(tf.matmul(x, weight) + bias)
    return weight, bias, activation


def one_hot_encoding_batch(batch, batch_size, num_labels):
    sparse_labels = tf.reshape(batch, [batch_size, 1])
    indices = tf.reshape(tf.range(0, batch_size, 1), [batch_size, 1])
    concatenated = tf.concat(1, [indices, sparse_labels])
    concat = tf.concat(0, [[batch_size], [num_labels]])
    output_shape = tf.reshape(concat, [2])
    sparse_to_dense = tf.sparse_to_dense(concatenated, output_shape, 1.0, 0.0)
    return tf.reshape(sparse_to_dense, [batch_size, num_labels])


def inference():
    image = tf.placeholder(tf.float32, shape=[None, 784], name='input')
    print(image.get_shape())
    w, b, output = layer('output_layer', image)
    return image, output


def loss(logits):
    labels = tf.placeholder(tf.float32, shape=[None, 10], name='labels')
    cross_entropy = - tf.reduce_sum(
        logits * tf.log(labels),
        reduction_indices=[1]
    )
    return labels, tf.reduce_mean(cross_entropy, name='xentropy_mean')


def training(loss, learning_rate):
    # tf.scalar_summary(loss.op.name, loss)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(loss)
    return train_op


def evaluation(logits, labels):
    correct = tf.nn.in_top_k(logits, labels, 1)
    # Return the number of true entries.
    return tf.reduce_sum(tf.cast(correct, tf.int32))


if __name__ == "__main__":
    with tf.Graph().as_default() as g:
        input, logits = inference()
        labels, loss_op = loss(logits)
        train = training(loss_op, 1e-4)

        init = tf.initialize_all_variables()

        with tf.Session() as sess:
            sess.run(init)
            # for i in range(1000):
            #     batch_xs, batch_ys = mnist.train.next_batch(100)
            #     sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
            for i in range(100):
                batch_xs, batch_labels = mnist.train.next_batch(250)
                sess.run(train, feed_dict={
                    input: batch_xs,
                    labels: batch_labels
                })
                if i % 100 == 0:
                    eval = evaluation(logits, tf.int32(batch_labels))
                    accuracy = sess.run(eval)
                    print("accuracy is " + accuracy)
