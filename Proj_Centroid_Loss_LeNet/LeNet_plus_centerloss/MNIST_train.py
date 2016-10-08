#!/usr/bin/env python3

import os, sys, numpy as np, tensorflow as tf
from pathlib import Path
from termcolor import colored as c, cprint
import h5py

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
LAMBDA = os.environ['LAMBDA']
DUMP_FILE = os.environ['DUMP_FILE'] # 'dumps/training_lambda_0.01.h5'
session_config = tf.ConfigProto(log_device_placement=True)
session_config.gpu_options.allow_growth = True
# this is required if want to use GPU as device.
# see: https://github.com/tensorflow/tensorflow/issues/2292
session_config.allow_soft_placement = True

if __name__ == "__main__":


    with tf.Graph().as_default() as g, tf.device(USE_DEVICE):
        # inference()
        input, deep_features = network.inference()
        labels, logits, cross_entropy = network.loss(deep_features)
        centroid_loss = network.center_loss(deep_features, labels)

        # combine the two losses
        _lambda = tf.placeholder(dtype=tf.float32)
        total_loss = cross_entropy + _lambda / 2. * centroid_loss

        learning_rate, train, global_step = network.training(total_loss)
        eval = network.evaluation(logits, labels)

        init = tf.initialize_all_variables()

        with tf.Session(config=session_config) as sess, h5py.File(DUMP_FILE, 'a', libver='latest', swmr=True) as h5_file:
            # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
            # to see the tensor graph, fire up the tensorboard with --logdir="./train"
            all_summary = tf.merge_all_summaries()
            train_writer = tf.train.SummaryWriter(SUMMARIES_DIR + '/summaries/train', sess.graph)
            test_writer = tf.train.SummaryWriter(SUMMARIES_DIR + '/summaries/test')

            saver = tf.train.Saver()

            # try:
            #     saver.restore(sess, SAVE_PATH)
            #     cprint(c('successfully loaded checkpoint file.', 'green'))
            # except ValueError:
            #     cprint(c('checkpoint file not found. Moving on to initializing automatically.', 'red'))
            #     sess.run(init)
            sess.run(init)

            step = global_step.eval()

            for i in range(20000):
                batch_xs, batch_labels = mnist.train.next_batch(BATCH_SIZE)
                if i % 50 == 0:
                    eval_labels = mnist.test.labels[:5000]
                    eval_images = mnist.test.images[:5000]
                    summaries, step, logits_outputs, deep_features_outputs, loss_value, accuracy = \
                        sess.run(
                            [all_summary, global_step, logits, deep_features, total_loss, eval], feed_dict={
                                _lambda: LAMBDA,
                                input: eval_images,
                                labels: eval_labels
                            })
                    test_writer.add_summary(summaries, global_step=step)

                    cprint(
                        c("#" + str(i), 'grey') +
                        c(" training accuracy", 'green') + " is " +
                        c(accuracy, 'red') + ", " +
                        c("loss", 'green') + " is " +
                        c(loss_value, 'red')
                    )

                    cprint(c('logits => ', 'yellow') + str(logits_outputs[0]))

                    group = h5_file.create_group('step_{}'.format(str(1000000 + step)[-6:]))
                    group.create_dataset('deep_features', data=deep_features_outputs)
                    group.create_dataset('logits', data=logits_outputs)
                    group.create_dataset('target_labels', data=eval_labels)

                if i % 500 == 0 and (accuracy > 0.6):
                    saver.save(sess, SAVE_PATH)
                    print('=> saved network in checkfile.')

                if step < 5000:
                    learning_rate_value = 0.1
                elif step < 10000:
                    learning_rate_value = 0.033
                elif step < 15000:
                    learning_rate_value = 0.01
                else:
                    learning_rate_value = 0.0033

                summaries, step, _ = sess.run(
                    [all_summary, global_step, train],
                    feed_dict={
                        _lambda: LAMBDA,
                        learning_rate: learning_rate_value,
                        input: batch_xs,
                        labels: batch_labels
                    })

                train_writer.add_summary(summaries, global_step=step)
