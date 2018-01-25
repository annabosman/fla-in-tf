""" Sampling XOR for FLA.
Author: Anna Bosman
Project: https://github.com/arakitianskaia/fla-in-numpy
"""

from __future__ import print_function

import tensorflow as tf
import numpy as np
from nn_for_fla_tf import FLANeuralNetwork
from fla_for_nn_tf import MetricGenerator

# Input/output data
X_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y_data = np.array([[0], [1], [1], [0]])


def get_data():
    return X_data, Y_data

# Sampling parameters
num_steps = 200    # FEM: 1000 steps; Neutrality: # steps proportionate to step size/search space
macro = True                                       # try micro and macro for all
bounds = 1                                          # Also try: 0.5, 1                                      # try micro and macro for all
step_size = 0.3

num_walks = 9   # make it equal to num weights (i.e. dimension)
num_sims = 2   # 30 independent runs: for stats

# Network Parameters
n_hidden_1 = 2 # 1st layer number of neurons
num_input = 2 # two bits
num_classes = 1 # 1 bit

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
    mgen.calculate_ruggedness_metrics(sess=sess, filename_header="data/output/xor/TEST_xor")
    #all_w = nn_model.one_sim(sess, num_walks, num_steps, get_data, print_to_screen=True)
    #sess.run(init)

#mgen = MetricGenerator(get_data=get_data, num_steps=num_steps, bounds=bounds, macro=macro, num_walks=num_walks, num_sims=num_sims, nn_model=nn_model, print_to_screen=True)
#mgen.calculate_neutrality_metrics("data/output/xor/xor")