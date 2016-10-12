import tensorflow as tf
import numpy as np
import ops

electron_xys = tf.constant([
    [0, 0],
    [0, 1],
    [0, 2],
], dtype=tf.float32)

target_result = [
    [0., 1., 2.],
    [1., 0., 1.],
    [2., 1., 0.]
]

with tf.Session() as sess:
    r_ij = ops.r2_ij(electron_xys)
    result = sess.run(r_ij)

    np.testing.assert_array_almost_equal(
        result,
        target_result,
        3,
        'the resulting total should be {}, but got {} instead. '
            .format(target_result, result)
    )
