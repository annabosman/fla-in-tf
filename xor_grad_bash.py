""" Sampling XOR for FLA: Neutral measures
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
num_steps = 1000    # FEM: 1000 steps; Neutrality: # steps proportionate to step size/search space
batch_size = X_data.shape[0]  # The whole data set; i.e. batch gradient descent.
display_step = 100

# Sampling parameters
macro = MACRO_SH                                    # try micro and macro for all
bounds = BOUNDS_SH                                  # 0.5, 1, 5
step_size = 0
if macro is True: step_size = (2 * bounds) * 0.1    # 10% of the search space
else: step_size = (2 * bounds) * 0.01               # 1% of the search space

num_walks = 9   # make it equal to num weights (i.e. dimension)
num_sims = 30   # 30 independent runs: for stats

# Network Parameters
n_hidden_1 = 2 # 1st layer number of neurons
num_input = 2 # two bits
num_classes = 1 # 1 bit

# Define the initialisation op
init = tf.global_variables_initializer()
nn_model = FLANeuralNetwork(n_hidden_1, num_input, num_classes, tf.nn.ACTIVATION_SH, tf.nn.sigmoid)
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
        all_w, d = nn_model.one_sim(sess, num_walks, num_steps, bounds, step_size, "manhattan", get_data)
        g = fla.calc_grad(all_w, d, step_size, bounds)
        print("Avg Grad: ", g)
        grad_list[i] = g
        print("----------------------- Sim ",i+1," is done -------------------------------")
        
    print("Gradients across sims: ", grad_list)
    #print("FEM across sims: ", fem_list)
    #print("M1/M2 across sims: ", m_list)

    filename1 = "data/xor_gavg"
    if macro is True:
        filename1 = filename1 + "_macro"
    else:
        filename1 = filename1 + "_micro"
    filename1 = filename1 + "_ACTIVATION_SH"
    filename1 = filename1 + "_BOUNDS_SH.csv"

    filename2 = "data/xor_gdev"
    if macro is True:
        filename2 = filename2 + "_macro"
    else:
        filename2 = filename2 + "_micro"
    filename2 = filename2 + "_ACTIVATION_SH"
    filename2 = filename2 + "_BOUNDS_SH.csv"

    with open(filename1, "a") as f:
        np.savetxt(f, ["# (1) cross-entropy", "# (2) mse", "# (3) accuracy"], "%s")
        np.savetxt(f, grad_list[:, 0, :], delimiter=",")

    with open(filename2, "a") as f:
        np.savetxt(f, ["# (1) cross-entropy", "# (2) mse", "# (3) accuracy"], "%s")
        np.savetxt(f, grad_list[:, 1, :], delimiter=",")
