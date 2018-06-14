""" Sampling Diabetes for FLA.
Author: Anna Bosman
Project: https://github.com/arakitianskaia/fla-in-numpy
"""

from __future__ import print_function

import tensorflow as tf
import numpy as np
import nn_for_fla_tf as nns

from nn_for_fla_tf import FLANeuralNetwork
from fla_for_nn_tf import MetricGenerator
from csv_data_reader import Data

debug = False

# Input/output data
TRAINING_DATA = "/mnt/lustre/users/abosman/fla/data/input/heart.csv"
#/mnt/lustre/users/abosman/fla/
data_reader = Data()
data_reader.load(TRAINING_DATA, header=False)

data_reader.scale_features_to_range()

X_data = data_reader.training_features
Y_data = data_reader.training_labels_1hot

# Network Parameters
num_input = X_data.shape[1]
n_hidden_1 = 9  # 1st layer number of neurons
num_classes = Y_data.shape[1]
dimension = nns.get_dimensionality(num_input, [n_hidden_1], num_classes)

if debug:
    print("Num in: ", num_input)
    print("Num out: ", num_classes)
    print("Dimensionality: ", dimension)
    print("Targets: ", Y_data)

batch_size = X_data.shape[0]  # Number of input patterns

_inputs = tf.placeholder(tf.float32, [None, num_input])
_outputs = tf.placeholder(tf.float32, [None, num_classes])

data = tf.data.Dataset.from_tensor_slices((_inputs, _outputs)).shuffle(batch_size).repeat().batch(batch_size)
iterator = tf.data.Iterator.from_structure(data.output_types, data.output_shapes)
next_batch = iterator.get_next()
data_init_op = iterator.make_initializer(data)

# Sampling parameters
macro = MACRO      # macro

if macro:
    num_steps = 100   # Macro: 100 steps, micro: 1000 steps
else:
    num_steps = 1000   # Macro: 100 steps, micro: 1000 steps

bounds = BOUNDS       # Variable: {1,10}

num_walks = dimension * 10   # make it equal to num weights (i.e. dimension)
num_sims = 1                 # Do 1 sim for now. See if we can get away with it. (Ask Prof!)

# Do the sampling!
nn_model = FLANeuralNetwork(input_tensor=next_batch[0], output_tensor=next_batch[1],
                            num_input=num_input, num_classes=num_classes, num_hidden=[n_hidden_1],
                            act_fn=tf.nn.sigmoid, out_act_fn=tf.nn.sigmoid, error_function="ERROR", compute_eigens=True)

mgen = MetricGenerator(nn_model, "unbounded_gradient", num_steps, num_walks, num_sims, bounds,
                       macro=macro, print_to_screen=False)

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
init = tf.global_variables_initializer()

with tf.Session(config=config) as sess:
    tf.get_default_graph().finalize()
    sess.run(init)
    sess.run(data_init_op, feed_dict={_inputs: X_data, _outputs: Y_data})
    mgen.write_walks_to_file_sequentially_one_at_a_time("/mnt/lustre/users/abosman/fla/data/output/heart/hessian/heart_hessian", sess)