""" Sampling Iris for FLA.
Author: Anna Bosman
Project: https://github.com/arakitianskaia/fla-in-numpy
"""

from __future__ import print_function

import tensorflow as tf
import numpy as np
import pandas as pd
import fla_metrics as fla
import random_samplers as rs

from nn_for_fla import FLANeuralNetwork
from csv_data_reader import Data

# Input/output data
IRIS_TRAINING = "data/input/iris_training.csv"

data_reader = Data()
data_reader.load(IRIS_TRAINING)

data_reader.scale_features_to_range()

X_data = data_reader.training_features
Y_data = data_reader.training_labels_1hot

#print("X_data: ", X_data)
#print("Y_data: ", Y_data)

def get_data():
    return X_data, Y_data


# NN Parameters
num_steps = 1000  # FEM: 1000 steps; Neutrality: # steps proportionate to step size/search space
batch_size = X_data.shape[0]  # The whole data set; i.e. batch gradient descent.

# Sampling parameters
macro = MACRO_SH  # try micro and macro for all
bounds = BOUNDS_SH  # Also try: 0.5, 1
step_size = 0
if macro is True:
    step_size = (2 * bounds) * 0.1  # 10% of the search space
else:
    step_size = (2 * bounds) * 0.01  # 1% of the search space

num_walks = 35  # make it equal to num weights (i.e. dimension)
num_sims = 30  # 30 independent runs: for stats

# Network Parameters
n_hidden_1 = 4  # 1st layer number of neurons
num_input = 4  # num features
num_classes = 3  # 3 classes

# Define the initialisation op
init = tf.global_variables_initializer()
nn_model = FLANeuralNetwork(n_hidden_1, num_input, num_classes, tf.nn.ACTIVATION_SH, tf.nn.sigmoid)
X, Y = nn_model.X, nn_model.Y

# Do the sampling:
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    # Run the initializer
    tf.get_default_graph().finalize()
    sess.run(init)

    #grad_list = np.empty((num_sims, 2, 3))
    fem_list = np.empty((num_sims, 3))
    #m_list = np.empty((num_sims, 2, 3))

    for i in range(0, num_sims):
        all_w, d = nn_model.one_sim(sess, num_walks, num_steps, bounds, step_size, "progressive", get_data)
        #m1, m2 = fla.calc_ms(all_w)
        fem = fla.calc_fem(all_w)
        #print("Avg Grad: ", g)
        #print("Avg FEM for walk ", i+1, ": ", fem)
        #print("Avg M1: ", m1)
        #print("Avg M2: ", m2)
        # grad_list[i] = g
        #m_list[i][0] = m1
        #m_list[i][1] = m2
        fem_list[i] = fem
        print("----------------------- Sim ", i + 1, " is done -------------------------------")

    # print("Gradients across sims: ", grad_list)
    print("FEM across sims: ", fem_list)
    #print("M1/M2 across sims: ", m_list)

    filename = "data/output/iris/iris_fem"
    if macro is True:
        filename = filename + "_macro"
    else:
        filename = filename + "_micro"
    filename = filename + "_ACTIVATION_SH"
    filename = filename + "_BOUNDS_SH.csv"

    with open(filename, "a") as f:
        np.savetxt(f, ["# (1) cross-entropy", "# (2) mse", "# (3) accuracy"], "%s")
        np.savetxt(f, fem_list, delimiter=",")
