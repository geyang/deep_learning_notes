import os, sys, numpy as np, tensorflow as tf
from pathlib import Path
from termcolor import colored as c, cprint

sys.path.append(str(Path(__file__).resolve().parents[1]))
import LeNet_plus_centerloss

__package__ = 'LeNet_plus_centerloss'
from . import network

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

BATCH_SIZE = 200
FILENAME = os.path.basename(__file__)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SUMMARIES_DIR = SCRIPT_DIR
SAVE_PATH = SCRIPT_DIR + "/network.ckpt"

### configure devices for this eval script.
USE_DEVICE = '/gpu:0'
session_config = tf.ConfigProto(log_device_placement=True)
session_config.gpu_options.allow_growth = True
# this is required if want to use GPU as device.
# see: https://github.com/tensorflow/tensorflow/issues/2292
session_config.allow_soft_placement = True

if __name__ == "__main__":

    with tf.Graph().as_default() as g, tf.device(USE_DEVICE):
        # inference()
        input, deep_features = network.inference()
        labels, logits, loss_op = network.loss(deep_features)
        # train, global_step = network.training(loss_op, 0.1)
        # train, global_step = network.training(loss_op, 0.03)
        train, global_step = network.training(loss_op, 0.01)
        eval = network.evaluation(logits, labels)

        init = tf.initialize_all_variables()

        with tf.Session(config=session_config) as sess:
            # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
            # to see the tensor graph, fire up the tensorboard with --logdir="./train"
            all_summary = tf.merge_all_summaries()
            train_writer = tf.train.SummaryWriter(SUMMARIES_DIR + '/summaries/train', sess.graph)
            test_writer = tf.train.SummaryWriter(SUMMARIES_DIR + '/summaries/test')

            saver = tf.train.Saver()

            try:
                saver.restore(sess, SAVE_PATH)
                cprint(c('successfully loaded checkpoint file.', 'green'))
            except ValueError:
                cprint(c('checkpoint file not found. Moving on to initializing automatically.', 'red'))
                sess.run(init)
            # sess.run(init)

            for i in range(20000):
                batch_xs, batch_labels = mnist.train.next_batch(BATCH_SIZE)
                accuracy = 0
                if i % 100 == 0:
                    summaries, step, logits_output, loss_value, accuracy = \
                        sess.run(
                            [all_summary, global_step, logits, loss_op, eval],
                            feed_dict={
                                input: mnist.test.images[:5000],
                                labels: mnist.test.labels[:5000]
                            })
                    test_writer.add_summary(summaries, global_step=step)
                    cprint(
                        c("#" + str(i), 'grey') +
                        c(" training accuracy", 'green') + " is " +
                        c(accuracy, 'red') + ", " +
                        c("loss", 'green') + " is " +
                        c(loss_value, 'red')
                    )
                    print('logits => ', logits_output[0])

                if i % 500 == 0 and (accuracy > 0.6):
                    saver.save(sess, SAVE_PATH)
                    print('=> saved network in checkfile.')

                summaries, step, _ = sess.run([all_summary, global_step, train], feed_dict={
                    input: batch_xs,
                    labels: batch_labels
                })
                train_writer.add_summary(summaries, global_step=step)

            # now let's test!
            TEST_BATCH_SIZE = np.shape(mnist.test.labels)[0]
            summaries, step, logits_output, loss_value, accuracy = \
                sess.run(
                    [all_summary, global_step, logits, loss_op, eval], feed_dict={
                        input: mnist.test.images[:5000],
                        labels: mnist.test.labels[:5000]
                    })
            test_writer.add_summary(summaries, global_step=step)
            print("MNIST Test accuracy is ", accuracy)
