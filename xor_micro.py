""" Sampling XOR for FLA.
Author: Anna Bosman
Project: https://github.com/arakitianskaia/fla-in-numpy
"""

from __future__ import print_function

import tensorflow as tf
import numpy as np
import nn_for_fla_tf as nns
from nn_for_fla_tf import FLANeuralNetwork
from fla_for_nn_tf import MetricGenerator

# Input/output data
X_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y_data = np.array([[0], [1], [1], [0]])


def get_data():
    return X_data, Y_data


# Network Parameters
n_hidden_1 = 2 # 1st layer number of neurons
num_input = 2 # two bits
num_classes = 1 # 1 bit
dimension = nns.get_dimensionality(num_input, [n_hidden_1], num_classes)

# Sampling parameters
num_steps = 1000   # Micro: 1000 step
macro = False      # micro
bounds = 10        # Variable: {1,10}

num_walks = dimension * 10   # make it equal to num weights (i.e. dimension)
num_sims = 1                 # Do 1 sim for now. See if we can get away with it. (Ask Prof!)

# Do the sampling!
nn_model = FLANeuralNetwork(num_input=num_input, num_classes=num_classes, num_hidden=[n_hidden_1],
                            act_fn=tf.nn.sigmoid, out_act_fn=tf.nn.sigmoid, error_function="ce", compute_eigens=True)

mgen = MetricGenerator(nn_model, get_data, "unbounded_gradient", num_steps, num_walks, num_sims, bounds,
                       macro=macro, print_to_screen=False)

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
init = tf.global_variables_initializer()

with tf.Session(config=config) as sess:
    tf.get_default_graph().finalize()
    sess.run(init)
    mgen.write_walks_to_file_sequentially("data/output/xor/hessian/xor_hessian",sess)
