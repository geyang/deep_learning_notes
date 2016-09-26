import tensorflow as tf
from termcolor import colored as c, cprint
import pickle

import matplotlib

# Force matplotlib to not use any Xwindows backend.
# Has to be done *before* importing pyplot.
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def train(step, loss):
    global_step = tf.Variable(0, name='global_step', dtype=tf.int32, trainable=False)
    optimizer = tf.train.GradientDescentOptimizer(step)
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


with tf.Session() as sess:
    x = tf.Variable(10, name='x', dtype=tf.float32)
    energy = 0.5 * x ** 2
    train_op = train(0.01, energy)

    init = tf.initialize_all_variables()
    sess.run(init)

    results = sess.run([energy])
    cprint(c('initial total is ', 'grey') + c(results[0], 'green'))

    decay = []
    for i in range(500):
        results = sess.run([energy, train_op])
        # cprint(c('initial total is ', 'grey') + c(results[0], 'green'))

        decay.append(results[0])
    plt.plot(decay)
    plt.savefig('figures/1D_pendulum_energy_decay.png', dpi=300, bbox_inches='tight')
