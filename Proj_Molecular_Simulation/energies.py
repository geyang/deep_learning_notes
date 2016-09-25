import tensorflow as tf
import Proj_Molecular_Simulation.ops as ops
import Proj_Molecular_Simulation.constants as constants

INTERACTIVE_CONSTANT = 1.6e-19


def total(xys, static):
    with tf.name_scope('Energy'):
        static_energy = static(xys)
        interactive_energy = interaction(xys)
        total_energy = tf.add(static_energy, interactive_energy, name='total_energy')
    return total_energy, static_energy, interactive_energy


def interaction(xys):
    with tf.name_scope('Interaction_Energy'):
        r_ij = ops.r_ij(xys)

        shape = r_ij.get_shape()

        interactive_energy = \
            tf.reduce_sum(
                tf.constant(0.5 * constants.k_qq) * tf.select(
                    tf.cast(r_ij, dtype=tf.bool),
                    1 / r_ij,
                    tf.zeros(shape=shape)
                ),
                reduction_indices=[0, 1]
            )
        return interactive_energy
