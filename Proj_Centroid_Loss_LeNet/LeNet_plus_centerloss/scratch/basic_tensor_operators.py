import tensorflow as tf
import numpy as np
from termcolor import cprint, colored as c

deep_features = tf.constant([
    [0, 1],
    [1, 1.5],
    [1.5, 2],
    [-1, 0],
    [-1, 1.5],
    [-1.5, 2],
    [-1, 0],
    [-1, 1.5],
    [-1.5, 2]
])

labels = tf.constant([
    [1., 0, 0],
    [1., 0, 0],
    [1., 0, 0],
    [0, 1., 0],
    [0, 1., 0],
    [0, 1., 0],
    [0, 0, 1.],
    [0, 0, 1.],
    [0, 0, 1.]
])

features_expanded = tf.reshape(deep_features, shape=[-1, 2, 1])
labels_expanded = tf.reshape(labels, shape=[-1, 1, 3])

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

centroids_expanded = tf.reshape(centroids, shape=[1, 2, 3]) * labels_expanded

spread = \
    tf.reduce_mean(
        tf.reduce_sum(
            tf.square(
                features_expanded * labels_expanded - centroids_expanded
            ),
            reduction_indices=[1, 2]
        )
    )

with tf.Session() as sess:
    result, = sess.run([spread])
    cprint(c(result, 'red'))
