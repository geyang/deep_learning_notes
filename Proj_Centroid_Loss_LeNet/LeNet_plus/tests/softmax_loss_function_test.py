import numpy as np, tensorflow as tf
from termcolor import colored as c, cprint

outputs = [
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # correct
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # wrong
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # null
    [-268252.5625, 48779.19921875, 80110.6796875, 354422.34375,
     158246.78125, 192678.75, 251321.09375, 353138.5, 362559.59375,
     - 80943.828125]  # from experiment
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
    cross_entropy = - tf.div(
        tf.reduce_mean(
            tf.mul(batch_labels, tf.nn.log_softmax(logits)),
            reduction_indices=[1]
        ),
        tf.reduce_mean(
            logits,
            reduction_indices=[1]
        )
    )

    with tf.Session() as sess:
        print("here is the calculated loss before being summed up.")
        results = sess.run([logits, cross_entropy])
        print("======")
        cprint(c('logits', 'green') + '\n' + str(results[0]))
        print("------")
        cprint(c('cross_entropy', 'green') + '\n' + str(results[1]))
        print("======")
