""" Sampling Diabetes for FLA.
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
from fla_for_nn import MetricGenerator
from csv_data_reader import Data

# Input/output data
TRAINING_DATA = "data/input/diabetes.csv"

data_reader = Data()
data_reader.load(TRAINING_DATA)

data_reader.scale_features_to_range()

X_data = data_reader.training_features
Y_data = data_reader.training_labels

#print("X_data: ", X_data)
#print("Y_data: ", Y_data)
#print("Num classes:", Y_data.shape[1])

def get_data():
    return X_data, Y_data

# Sampling parameters
num_steps = 1000  # FEM: 1000 steps; Neutrality: # steps proportionate to step size/search space
macro = True  # try micro and macro for all
bounds = 1  # Also try: 0.5, 1

num_walks = 81  # make it equal to num weights (i.e. dimension)
num_sims = 1 # 30 independent runs: for stats

# Network Parameters
n_hidden_1 = 8  # 1st layer number of neurons
num_input = X_data.shape[1]
num_classes = Y_data.shape[1]

# Define the initialisation op
init = tf.global_variables_initializer()
nn_model = FLANeuralNetwork(n_hidden_1, num_input, num_classes, tf.nn.sigmoid, tf.nn.sigmoid)
mgen = MetricGenerator(get_data=get_data, num_steps=num_steps, bounds=bounds, macro=macro, num_walks=num_walks, num_sims=num_sims, nn_model=nn_model, print_to_screen=True)

init = tf.global_variables_initializer()
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    tf.get_default_graph().finalize()
    sess.run(init)
    mgen.calculate_ruggedness_metrics("data/output/diab_test", sess)
    mgen.num_steps = 200
    mgen.calculate_neutrality_metrics("data/output/diab_test", sess)