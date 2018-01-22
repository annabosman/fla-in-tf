""" Sampling XOR for FLA.
Author: Anna Bosman
Project: https://github.com/arakitianskaia/fla-in-numpy
"""

from __future__ import print_function

import tensorflow as tf
import numpy as np
from nn_for_fla import FLANeuralNetwork
from fla_for_nn import MetricGenerator

# Input/output data
X_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y_data = np.array([[0], [1], [1], [0]])


def get_data():
    return X_data, Y_data

# Sampling parameters
num_steps = 100    # FEM: 1000 steps; Neutrality: # steps proportionate to step size/search space
macro = True                                       # try micro and macro for all
bounds = 1                                          # Also try: 0.5, 1

num_walks = 9   # make it equal to num weights (i.e. dimension)
num_sims = 2   # 30 independent runs: for stats

# Network Parameters
n_hidden_1 = 2 # 1st layer number of neurons
num_input = 2 # two bits
num_classes = 1 # 1 bit

# Define the initialisation op
nn_model = FLANeuralNetwork(n_hidden_1, num_input, num_classes, tf.nn.sigmoid, tf.nn.sigmoid)
mgen = MetricGenerator(get_data=get_data, num_steps=num_steps, bounds=bounds, macro=macro, num_walks=num_walks, num_sims=num_sims, nn_model=nn_model, print_to_screen=True)
mgen.calculate_neutrality_metrics("data/output/xor/xor")