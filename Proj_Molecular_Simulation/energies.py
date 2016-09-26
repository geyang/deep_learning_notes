import tensorflow as tf
import Proj_Molecular_Simulation.ops as ops
import Proj_Molecular_Simulation.constants as constants

INTERACTIVE_CONSTANT = 1.6e-19


def energy(xys, static):
    with tf.name_scope('Interaction_Energy'):
        m = xys.get_shape()[0:1]
        r2_ij = ops.r2_ij(xys)

        v_ii = 2.0 / tf.map_fn(static, xys)
        k_ij_total = tf.add(r2_ij, tf.diag(v_ii))

        interactive_energy = \
            tf.reduce_sum(
                0.5 * constants.k_qq / tf.sqrt(k_ij_total),
                reduction_indices=[0, 1]
            )
    return interactive_energy
