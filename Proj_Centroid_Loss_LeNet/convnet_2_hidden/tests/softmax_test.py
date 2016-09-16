import numpy as np, tensorflow as tf

# outputs = [
#     [0.01, 1, 0.01, 0.01, 0.01],  # correct
#     [0.01, 0.01, 0.01, 0.01, 1],  # wrong
#     [0.01, 0.01, 0.01, 0.01, 0.01]  # null
# ]
outputs = [
    [0, 1, 0, 0, 0],  # correct
    [0, 0, 0, 0, 1],  # wrong
    [0, 0, 0, 0, 0]  # null
]

batch_labels = [
    [0, 1, 0, 0, 0],
    [0, 1, 0, 0, 0],
    [0, 1, 0, 0, 0]
]

"""
per the discussion here:
http://stackoverflow.com/questions/33712178/tensorflow-nan-bug
"""
with tf.Graph().as_default(), tf.device('/cpu:0'):
    logits = tf.constant(outputs, dtype=tf.float32)

    cross_entropy_native = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits,
        tf.argmax(batch_labels, dimension=1),
        name='xentropy'
    )

    cross_entropy_bad = -tf.reduce_sum(
        batch_labels * tf.log(logits),
        reduction_indices=[1]
    )
    # problem with clipping is that it stops the gradient from propagating back.
    cross_entropy_with_clip = - tf.reduce_sum(
        batch_labels * tf.log(tf.clip_by_value(logits, 1e-4, 1.0)),
        reduction_indices=[1]
    )

    cross_entropy_with_padding = -tf.reduce_sum(
        batch_labels * tf.log(logits + 1e-10),
        reduction_indices=[1]
    )
    with tf.Session() as sess:
        print("here is the bad kind of way of calculating the cross entropy")
        result = sess.run(cross_entropy_bad)
        print(result)
        print("here is a BETTER way of calculating the cross entropy")
        result = sess.run(cross_entropy_with_clip)
        print(result)
        print("here is an EVEN BETTER way of calculating the cross entropy")
        result = sess.run(cross_entropy_with_padding)
        print(result)
