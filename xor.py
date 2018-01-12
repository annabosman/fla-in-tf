""" Sampling XOR for FLA.
Author: Anna Bosman
Project: https://github.com/arakitianskaia/fla-in-numpy
"""

from __future__ import print_function

import tensorflow as tf
import numpy as np
import fla_metrics as fla
import random_samplers as rs
from nn_for_fla import FLANeuralNetwork

# Input/output data
X_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y_data = np.array([[0], [1], [1], [0]])


def get_data():
    return X_data, Y_data

# NN Parameters
num_steps = 100    # FEM: 1000 steps; Neutrality: # steps proportionate to step size/search space
batch_size = X_data.shape[0]  # The whole data set; i.e. batch gradient descent.
display_step = 100

# Sampling parameters
macro = True                                       # try micro and macro for all
bounds = 1                                          # Also try: 0.5, 1
step_size = 0
if macro is True: step_size = (2 * bounds) * 0.1    # 10% of the search space
else: step_size = (2 * bounds) * 0.01               # 1% of the search space

num_walks = 9   # make it equal to num weights (i.e. dimension)
num_sims = 2   # 30 independent runs: for stats

# Network Parameters
n_hidden_1 = 2 # 1st layer number of neurons
num_input = 2 # two bits
num_classes = 1 # 1 bit

# Define the initialisation op
init = tf.global_variables_initializer()
nn_model = FLANeuralNetwork(n_hidden_1, num_input, num_classes, tf.nn.sigmoid, tf.nn.sigmoid)
X, Y = nn_model.X, nn_model.Y
# Do the sampling:
with tf.Session() as sess:
    # Run the initializer
    tf.get_default_graph().finalize()
    sess.run(init)

    grad_list = np.empty((num_sims, 2, 3))
    fem_list = np.empty((num_sims, 3))
    m_list = np.empty((num_sims, 2, 3))

    for i in range(0, num_sims):
        all_w, d = nn_model.one_sim(sess, num_walks, num_steps, bounds, step_size, "progressive", get_data)
        g, fem, m1, m2 = fla.calculate_metrics(all_w, d, step_size, bounds)
        #fem = calc_fem(all_w)
        # print("Avg Grad: ", g)
        #print("Avg FEM for walk ", i+1, ": ", fem)
        # print("Avg M1: ", m1)
        # print("Avg M2: ", m2)
        # grad_list[i] = g
        m_list[i][0] = m1
        m_list[i][1] = m2
        print("----------------------- Sim ",i+1," is done -------------------------------")
        
    #print("Gradients across sims: ", grad_list)
    #print("FEM across sims: ", fem_list)
    print("M1/M2 across sims: ", m_list)

    m1 = m_list[:,0,:]
    print("m1: ", m1)

    m2 = m_list[:,1,:]
    print("m2: ", m2)

    # g_avg = grad_list[:,0,:]
    # print("g_avg: ", g_avg)
    #
    # g_dev = grad_list[:,1,:]
    # print("g_dev: ", g_dev)

    #with open("data/xor_fem_micro_1.csv", "a") as f:
    #    np.savetxt(f, ["cross-entropy", "mse", "accuracy"], "%s")
    #    np.savetxt(f, fem_list, delimiter=",")
