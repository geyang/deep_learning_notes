import tensorflow as tf
from termcolor import colored as c, cprint
import pickle
import numpy as np
import Proj_Molecular_Simulation.energies as energies

import matplotlib

# Force matplotlib to not use any Xwindows backend.
# Has to be done *before* importing pyplot.
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def get_locations(n, name='xys', range=1):
    shape = [n, 2]
    xys = tf.Variable(
        tf.truncated_normal(shape,
                            mean=0.0,
                            stddev=10e-6 / range),
        name=name)
    return xys


def train(step, energy):
    # global_step can not be placed on GPU. with soft_placement will be placed on CPU.
    global_step = tf.Variable(0, name='global_step', dtype=tf.int32, trainable=False)

    optimizer = tf.train.AdamOptimizer(step, 0.9)
    # optimizer = tf.train.GradientDescentOptimizer(step)
    train_op = optimizer.minimize(energy, global_step=global_step)
    return train_op


def get_summary(energy):
    tf.scalar_summary(energy.op.name, energy)


def static(xy):
    return (4 * 0.0733e6 ** 2) * tf.reduce_sum(
        tf.square(xy),
        reduction_indices=[0]
    )


config = tf.ConfigProto(allow_soft_placement=True)
with tf.Session(config=config) as sess, tf.device('/gpu:0'):
    xys = get_locations(100)
    interactive_energy = energies.total(xys, static)

    step_size = tf.placeholder(dtype=tf.float32)
    train_op = train(step_size, interactive_energy)

    init = tf.initialize_all_variables()
    # all_summaries = tf.merge_all_summaries()

    sess.run(init)

    for i in range(100001):
        sess.run(train_op, feed_dict={step_size: 1e-1 * i / 1e5})

        if i % 1000 == 0:
            current_xys, interactive_energy_result = sess.run([xys, interactive_energy])
            cprint(c('interactive_energy_result ', 'grey') + c(interactive_energy_result, 'green') + ' eV')

            # xs = current_xys[:, 0]
            # ys = current_xys[:, 1]
            #
            # plt.figure(figsize=(6, 6))
            # plt.scatter(xs, ys)
            # # plt.xlim(-10e-5, 10e-5)
            # # plt.ylim(-10e-5, 10e-5)
            #
            # plt.xlim(np.min(xs), np.max(ys))
            # plt.ylim(np.min(ys), np.max(ys))
            # plt.savefig('dumps/temp_{}.png'.format(str(1000 + i)[-3:]))

            with open('dumps/xys_{}.dump.pkl'.format(str(1000000 + i)[-6:]), 'wb') as f:
                pickle.dump(current_xys, f)
