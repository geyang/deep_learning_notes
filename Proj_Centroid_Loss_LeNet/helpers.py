import tensorflow as tf


def relu_layer(layer_name=None, images):
    with tf.name_scope(layer_name):
        weights = tf.Variable(
            tf.truncated_normal([IMAGE_PIXELS, hidden1_units],
                                stddev=1.0 / math.sqrt(float(IMAGE_PIXELS))),
            name='weights')
        biases = tf.Variable(tf.zeros([hidden1_units]),
                             name='biases')
        hidden_layer = tf.nn.relu(tf.matmul(images, weights) + biases)

        return weights, biases, hidden_layer
