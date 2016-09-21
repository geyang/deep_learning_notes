import os, sys, numpy as np, tensorflow as tf
from pathlib import Path
from termcolor import colored as c, cprint

sys.path.append(str(Path(__file__).resolve().parents[1]))
import convnet_2_hidden

__package__ = 'convnet_2_hidden'
from . import network

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

BATCH_SIZE = 500
FILENAME = os.path.basename(__file__)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SUMMARIES_DIR = SCRIPT_DIR
SAVE_PATH = SCRIPT_DIR + "/network.ckpt"

### configure devices for this eval script.
USE_DEVICE = '/gpu:1'
session_config = tf.ConfigProto(log_device_placement=True)
session_config.gpu_options.allow_growth = True
# this is required if want to use GPU as device.
# see: https://github.com/tensorflow/tensorflow/issues/2292
session_config.allow_soft_placement = True

if __name__ == "__main__":

    with tf.Graph().as_default() as g, tf.device(USE_DEVICE):
        # inference()
        input, deep_feature = network.inference()
        labels, logits, loss_op = network.loss(deep_feature)
        train = network.training(loss_op, 1e-1)
        eval = network.evaluation(logits, labels)

        init = tf.initialize_all_variables()

        with tf.Session(config=session_config) as sess:
            # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
            # to see the tensor graph, fire up the tensorboard with --logdir="./train"
            all_summary = tf.merge_all_summaries()
            train_writer = tf.train.SummaryWriter(SUMMARIES_DIR + '/summaries/train', sess.graph)
            test_writer = tf.train.SummaryWriter(SUMMARIES_DIR + '/summaries/test')

            saver = tf.train.Saver()

            sess.run(init)
            try:
                saver.restore(sess, SAVE_PATH)
            except ValueError:
                print('checkpoint file not found. Moving on to training.')

            for i in range(500000):
                batch_xs, batch_labels = mnist.train.next_batch(BATCH_SIZE)
                if i % 100 == 0:
                    summaries, logits_output, loss_value, accuracy = sess.run([all_summary, logits, loss_op, eval], feed_dict={
                        input: mnist.test.images,
                        labels: mnist.test.labels
                    })
                    test_writer.add_summary(summaries, i)
                    cprint(
                        c("#" + str(i), 'grey') +
                        c(" training accuracy", 'green') + " is " +
                        c(accuracy, 'red') + ", " +
                        c("loss", 'green') + " is " +
                        c(loss_value, 'red')
                    )
                    print('logits => ', logits_output[0])

                if i % 500 == 0:
                    saver.save(sess, SAVE_PATH)
                    print('=> saved network in checkfile.')

                summaries, _ = sess.run([all_summary, train], feed_dict={
                    input: batch_xs,
                    labels: batch_labels
                })
                train_writer.add_summary(summaries, i)

            # now let's test!
            TEST_BATCH_SIZE = np.shape(mnist.test.labels)[0]
            logits_output, loss_value, accuracy = sess.run([logits, loss_op, eval], feed_dict={
                input: mnist.test.images,
                labels: mnist.test.labels
            })
            print("MNIST Test accuracy is ", accuracy)
