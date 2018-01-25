""" Sampling MNIST for FLA: Ruggedness measures
Author: Anna Bosman
Project: https://github.com/arakitianskaia/fla-in-numpy
"""

from __future__ import print_function

import tensorflow as tf
import numpy as np
import fla_metrics as fla
import random_samplers_tf as rs
from nn_for_fla_tf import FLANeuralNetwork
from fla_for_nn_tf import MetricGenerator

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data

# Input/output data
mnist = input_data.read_data_sets("datasets", one_hot=True)


# Return (x, y) tuple: input and output data matrix
def get_data():
    x, y = mnist.train.next_batch(batch_size)
    return x, y

# Sampling parameters
num_steps = 200    # FEM, grad: 1000 steps; Neutrality: # steps proportionate to step size/search space
batch_size = 100

macro = True#MACRO_SH                                    # try micro and macro for all
bounds = 1#BOUNDS_SH                                  # 0.5, 1, 5

num_walks = 203530   # make it equal to num weights (i.e. dimension)
num_sims = 10        # 30 independent runs: for stats

# Network Parameters
n_hidden_1 = 256    # 1st layer number of neurons
num_input = 784     # MNIST data input (img shape: 28*28)
num_classes = 10    # MNIST total classes (0-9 digits)

# Do the sampling!
nn_model = FLANeuralNetwork(num_input=num_input, num_classes=num_classes, num_hidden=[n_hidden_1],
                            act_fn=tf.nn.sigmoid, out_act_fn=tf.nn.sigmoid)
#nn_model.build_random_walk_graph(walk_type="progressive", step_size=step_size, bounds=bounds)
mgen = MetricGenerator(nn_model, get_data, "progressive", num_steps, num_walks, num_sims, bounds,
                       macro=True, print_to_screen=True)

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
init = tf.global_variables_initializer()

with tf.Session(config=config) as sess:
    tf.get_default_graph().finalize()
    sess.run(init)
    mgen.calculate_ruggedness_metrics(sess=sess, filename_header="data/output/mnist/TEST_mnist")