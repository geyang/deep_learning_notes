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

class Simple:
    def __init__(self):
        print('simple class has been initiated!')

def plot_deep_features(deep_features, labels, save=False, **kwargs):
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