# -*- coding: utf-8 -*-
import tensorflow as tf  # NOQA
from keras import backend as K  # NOQA

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

from wbia_pie import _plugin  # NOQA
from wbia_pie import __main__  # NOQA

try:
    from wbia_pie._version import __version__
except ImportError:
    __version__ = '0.0.0'
