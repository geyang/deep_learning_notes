import tensorflow as tf
import numpy as np
from termcolor import colored as c, cprint
import energies

electron_xys = tf.constant([
    [0, 0],
    [-1, 0],
    [-2, 0]
], dtype=tf.float32)

target_result = 1.44e-9 * 2.5  # eV/m

with tf.Session() as sess:
    interaction_energy = energies.total(electron_xys, lambda xy: 0.0)
    result = sess.run(interaction_energy)

    cprint(c(result, 'red'))
    np.testing.assert_almost_equal(
        result,
        target_result,
        err_msg='the resulting total should be {}, but got {} instead. ' \
            .format(target_result, result),
        decimal=10
    )
