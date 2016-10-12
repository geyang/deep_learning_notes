# import h5py
import numpy as np
# from pprint import pprint
# from termcolor import colored as c, cprint
# from IPython.display import FileLink
# from IPython import display
# from tqdm import tqdm

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

color_palette = ['#507ba6', '#f08e39', '#e0595c', '#79b8b3', '#5ca153',
                 '#edc854', '#af7ba2', '#fe9fa9', '#9c7561', '#bbb0ac']

from scipy.signal import savgol_filter


def smooth(samples, window_length=101, polyorder=3):
    return savgol_filter(samples, window_length=window_length, polyorder=polyorder)


def smooth_ranges(samples, scale, floor=1, **kwargs):
    centroids = np.mean(samples, axis=1)
    smoothed = np.transpose(np.vstack((
        np.minimum(smooth(samples[:, 0] - centroids, **kwargs), - floor),
        np.maximum(smooth(samples[:, 1] - centroids, **kwargs), floor)
    )))
    return smoothed * scale + np.expand_dims(centroids, 1)


def smooth_ranges_2d(xranges, yranges, aspect_ratio='equal', *args, **kwargs):
    smoothed_x = smooth_ranges(xranges, *args, **kwargs)
    smoothed_y = smooth_ranges(yranges, *args, **kwargs)

    if aspect_ratio == 'equal':
        x_centroids = np.mean(smoothed_x, axis=1)
        y_centroids = np.mean(smoothed_y, axis=1)
        half_spread = np.maximum(np.abs(smoothed_x[:, 0] - x_centroids),
                                 np.abs(smoothed_y[:, 0] - y_centroids))

        return np.transpose(np.vstack((
            x_centroids - half_spread, x_centroids + half_spread
        ))), np.transpose(np.vstack((
            y_centroids - half_spread, y_centroids + half_spread
        )))
    else:
        return smoothed_x, smoothed_y


def plot_deep_features(deep_features, labels, **kwargs):
    bins = {}
    for logit, deep_feature in zip(labels, deep_features):
        label = np.argmax(logit)

        # print(label)
        try:
            bins[str(label)].append(list(deep_feature))
        except KeyError:
            bins[str(label)] = [list(deep_feature)]

    fig = plt.figure(figsize=(5, 5))

    for numeral in map(str, range(10)):
        try:
            features = np.array(bins[numeral])
        except KeyError:
            print(numeral + " does not exist")
            features = []

        try:
            x, y = np.transpose(features)
        except ValueError:
            x = [];
            y = []
        plt.scatter(
            x,
            y,
            s=1,
            color=color_palette[int(numeral)],
            label=numeral
        )

    plt.legend(loc=(1.05, 0.1), frameon=False)

    if 'title' in kwargs:
        title = kwargs['title']
    else:
        title = 'MNIST LeNet++ with 2 Deep Features (PReLU)'

    plt.title(title)

    plt.xlabel('activation of hidden neuron 1')
    plt.ylabel('activation of hidden neuron 2')

    if 'xlim' in kwargs:
        plt.xlim(kwargs['xlim'][0], kwargs['xlim'][1])
    if 'ylim' in kwargs:
        plt.ylim(kwargs['ylim'][0], kwargs['ylim'][1])

    return fig


def text(message, loc, xrange, yrange,
         horizontalalignment='right',
         verticalalignment='bottom', **kwargs):
    x = xrange[0] * (1 - loc[0]) + xrange[1] * loc[0]
    y = yrange[0] * (1 - loc[1]) + yrange[1] * loc[1]
    plt.text(x, y, message, horizontalalignment=horizontalalignment,
             verticalalignment=verticalalignment, **kwargs)


def plot(save=False, **kwargs):
    if 'frame_prefix' in kwargs:
        frame_prefix = kwargs['frame_prefix']
    else:
        frame_prefix = 'frame'
    if save and 'frame_index' in kwargs:
        fname = './figures/animation/' + frame_prefix + "_" + str(1000 + kwargs['frame_index'])[-3:] + '.png'
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        plt.show()
        # plt.close(fig)
    elif save:
        fname = './figures/animation/' + title + '.png'
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        plt.show()
        # plt.close(fig)
    else:
        plt.show()
