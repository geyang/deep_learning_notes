#!/usr/bin/env python3

import numpy as np, tensorflow as tf
from glob import glob
from analysis import plot_deep_features, plot, smooth_ranges_2d, text

import os
import h5py
from pprint import pprint
from tqdm import tqdm
from IPython import display
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def make_frames(filepath, frame_prefix, title):
    xranges = []
    yranges = []

    with h5py.File(filepath, 'r', libver='latest', swmr=True) as f:
        for i, step_key in enumerate(tqdm(sorted(f.keys())[::1])):

            step_entry = f[step_key]

            if len(step_entry.keys()) == 0:
                print('entry is empty')

            else:
                deep_features = step_entry['deep_features']
                xranges.append((min(deep_features[:, 0]), max(deep_features[:, 0])))
                yranges.append((min(deep_features[:, 1]), max(deep_features[:, 1])))

    xranges = np.array(xranges)
    yranges = np.array(yranges)

    floor = 5e-2
    xranges, yranges = smooth_ranges_2d(xranges, yranges, scale=1.25, floor=floor, window_length=51, polyorder=3)

    with h5py.File(filepath, 'r', libver='latest', swmr=True) as f:
        for i, (step_key, xrange, yrange) in enumerate(tqdm(
                list(zip(sorted(f.keys()),
                         xranges,
                         yranges))[::1])):

            step_entry = f[step_key]

            if len(step_entry.keys()) == 0:
                print('entry is empty')

            else:

                deep_features = step_entry['deep_features']
                logits = step_entry['logits']
                target_labels = step_entry['target_labels']

                target_labels_output = list(target_labels)

                # plt.cla()
                display.clear_output(wait=True)
                ax = plot_deep_features(
                    deep_features,
                    target_labels,
                    title=title,
                    xlim=xrange,
                    ylim=yrange
                )

                centroid = step_entry['centroid']
                plt.scatter(centroid[0], centroid[1], c='black')

                learning_rate = np.array(step_entry['learning_rate'])
                _lambda = np.array(step_entry['lambda'])
                # accuracy = np.array(step_entry['accuracy'])


                logits = np.array(step_entry['logits'])
                target_labels = np.array(step_entry['target_labels'])
                sample_number = np.shape(target_labels)[0]
                accuracy = np.sum(
                    np.equal(np.argmax(logits, axis=1), np.argmax(target_labels, axis=1)
                             )
                ) / sample_number

                text('lambda: {}\nlearning rate: {}\naccuracy: {}'.format(str(_lambda), str(learning_rate),
                                                                          str(accuracy)), (0.95, 0.05), xrange,
                     yrange)
                plot(save=True,
                     frame_prefix=frame_prefix,
                     frame_index=int(i))


if __name__ == "__main__":
    frame_prefix = os.environ['FRAME_PREFIX']
    filepath = os.environ['FILEPATH']
    title = os.environ['TITLE']

    make_frames(filepath, frame_prefix, title)
