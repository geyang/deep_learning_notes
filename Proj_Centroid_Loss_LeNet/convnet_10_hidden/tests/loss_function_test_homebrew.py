import numpy as np, tensorflow as tf
from pprint import pprint
from termcolor import colored as c, cprint

outputs = [
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # correct
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # wrong
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # null
    [-268252.5625, 48779.19921875, 80110.6796875, 354422.34375,
     158246.78125, 192678.75, 251321.09375, 353138.5, 362559.59375,
     - 80943.828125]
]

labels = [
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
]

"""
per the discussion here:
http://stackoverflow.com/questions/33712178/tensorflow-nan-bug
"""
with tf.Graph().as_default(), tf.device('/cpu:0'):
    logits = tf.constant(outputs, dtype=tf.float64)
    batch_labels = tf.constant(labels, dtype=tf.float64)
    exp_logits = tf.clip_by_value(
        tf.exp(logits),
        0,
        1e10
    )
    numerator = tf.reduce_sum(
        tf.mul(batch_labels, exp_logits),
        reduction_indices=[1]
    )
    cross_entropy = - tf.log(
        tf.mul(
            tf.div(
                numerator,
                tf.reduce_sum(exp_logits, reduction_indices=[1])
            ),
            tf.reduce_sum(tf.square(logits), reduction_indices=[1])
        )
    )

    with tf.Session() as sess:
        print("here is the calculated loss before being summed up.")
        results = sess.run([logits, exp_logits, numerator, cross_entropy])
        print("======")
        print('logits\n', results[0])
        print("------")
        print('exp_logits\n', results[1])
        print("------")
        print('numerator\n', results[2])
        print("------")
        print('cross_entropy\n', results[3])
        print("======")
