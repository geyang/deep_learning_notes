import tensorflow as tf
import Proj_Molecular_Simulation.energies as energies


def get_locations(n, name='xys', range=1):
    shape = [2, n]
    weight = tf.Variable(
        tf.truncated_normal(shape,
                            mean=0.5,
                            stddev=1.0 / range),
        name=name)
    return weight


def static(xys):
    static_energy = tf.reduce_mean(
        tf.map_fn(
            lambda x2y2: tf.sqrt(
                tf.reduce_mean(x2y2, reduction_indices=[0])
            ),
            tf.square(xys)
        ),
        reduction_indices=[0]
    )
    return static_energy


def train(step, energy):
    # global_step can not be placed on GPU. with soft_placement will be placed on CPU.
    global_step = tf.Variable(0, name='global_step', dtype=tf.int32, trainable=False)

    optimizer = tf.train.GradientDescentOptimizer(step)
    train_op = optimizer.minimize(energy, global_step=global_step)
    return train_op


def get_summary(energy):
    tf.scalar_summary(energy.op.name, energy)


config = tf.ConfigProto(allow_soft_placement=True)
with tf.Session(config=config) as sess, tf.device('/gpu:0'):
    xys = get_locations(2000)
    total_energy, static_energy, interactive_energy = energies.total(xys, static)
    train_op = train(0.01, total_energy)

    # record the energy trajectory
    get_summary(total_energy)

    init = tf.initialize_all_variables()
    all_summaries = tf.merge_all_summaries()

    sess.run(init)
    energy_result = sess.run(total_energy)
    print(energy_result)

    for i in range(1000):
        sess.run([all_summaries, train_op])
        # writer.add_summary(summary_result, global_step=current_step)
