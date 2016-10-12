import tensorflow as tf
from termcolor import colored as c, cprint
import pickle
import numpy as np
import energies as energies

import matplotlib

import time

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
    return (4 * 0.0733e12) * tf.reduce_sum(
        tf.square(xy),
        reduction_indices=[0]
    )


config = tf.ConfigProto(allow_soft_placement=True)
with tf.Session(config=config) as sess, tf.device('/gpu:0'):
    xys = get_locations(21)
    interactive_energy = energies.total(xys, static)

    step_size = tf.placeholder(dtype=tf.float32)
    train_op = train(step_size, interactive_energy)

    init = tf.initialize_all_variables()
    # all_summaries = tf.merge_all_summaries()

    sess.run(init)

    total_steps = 70000
    for i in range(total_steps + 1):
        tick = time.clock()
        sess.run(train_op,
                 feed_dict={step_size: 1e-1 * np.min([12.5 * i, 700, 0.01 * (total_steps - i)]) / total_steps})

        if i % 100 == 0:
            lapsed = (time.clock() - tick) / 100.
            current_xys, interactive_energy_result = sess.run([xys, interactive_energy])
            cprint(c('{}sec '.format(str(lapsed)[:7]), 'yellow') + c('#{} '.format(i), 'red') + c('interactive_energy_result ', 'grey') + c(
                interactive_energy_result, 'green') + ' eV')

            # with open('dumps/xys_{}.dump.pkl'.format(str(1000000 + i)[-6:]), 'wb') as f:
            #     pickle.dump(current_xys, f)
