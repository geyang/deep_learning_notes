import tensorflow as tf
import numpy as np
import Proj_Molecular_Simulation.energies as energies

electron_xys = tf.constant([
    [0, 0],
    [0, 1]
], dtype=tf.float32)

target_result = 1.44e-9  # eV/m

with tf.Session() as sess:
    interaction_energy = energies.interaction(electron_xys)
    result = sess.run(interaction_energy)

    np.testing.assert_almost_equal(
        result,
        target_result,
        decimal=4,
        'the resulting energy should be {}, but got {} instead. '
            .format(target_result, result)
    )
