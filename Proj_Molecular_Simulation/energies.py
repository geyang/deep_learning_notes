import tensorflow as tf
import ops
import constants

INTERACTIVE_CONSTANT = 1.6e-19


def total(xys, static2):
    with tf.name_scope('Interaction_Energy'):
        r2_ij = ops.r2_ij(xys)
        v2_ii = 2.0 / tf.map_fn(static2, xys)
        k_ij_total = tf.add(r2_ij, tf.diag(v2_ii))

        interactive_energy = \
            tf.reduce_sum(
                0.5 * constants.k_qq / tf.sqrt(k_ij_total),
                reduction_indices=[0, 1]
            )
    return interactive_energy
