import os, sys, numpy as np, tensorflow as tf
from pathlib import Path

# this one treats the parent as the working directory.
# this hacky, yet simple and nice.
sys.path.append(str(Path(__file__).resolve().parents[2]))
import convnet_2_hidden

__package__ = 'convnet_2_hidden'
from . import network

with tf.Graph().as_default(), tf.device('/gpu:0'):

    network.loss(deep_features)
