import os, sys, numpy as np, tensorflow as tf
from pathlib import Path

import time

sys.path.append(str(Path(__file__).resolve().parents[1]))
import LeNet_plus_centerloss

__package__ = 'LeNet_plus_centerloss'
from . import network

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

BATCH_SIZE = 250
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
    with tf.Graph().as_default() as g:
        # inference()
        input, logits = network.inference()
        labels, loss_op = network.loss(logits)
        train = network.training(loss_op, 1e-1)
        eval = network.evaluation(logits, labels)

        init = tf.initialize_all_variables()

        with tf.Session(config=session_config) as sess:
            # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
            # to see the tensor graph, fire up the tensorboard with --logdir="./train"
            merged = tf.merge_all_summaries()
            train_writer = tf.train.SummaryWriter(SUMMARIES_DIR + '/summaries/train', sess.graph)
            test_writer = tf.train.SummaryWriter(SUMMARIES_DIR + '/summaries/test')

            saver = tf.train.Saver()

            sess.run(init)
            saver.restore(sess, SAVE_PATH)

            # now let's test!
            TEST_BATCH_SIZE = np.shape(mnist.test.labels)[0]

            while True:
                output, loss_value, accuracy = sess.run([logits, loss_op, eval], feed_dict={
                    input: mnist.test.images,
                    labels: mnist.test.labels
                })
                print("- MNIST Test accuracy is ", accuracy / TEST_BATCH_SIZE)
                time.sleep(5.0)
