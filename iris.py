""" Sampling XOR for FLA.
Author: Anna Bosman
Project: https://github.com/arakitianskaia/fla-in-numpy
"""

from __future__ import print_function

import tensorflow as tf
import numpy as np
import sklearn.datasets as sk
import sklearn.preprocessing as pp
import nn_for_fla_tf as nns

from nn_for_fla_tf import FLANeuralNetwork
from fla_for_nn_tf import MetricGenerator
from csv_data_reader import Data

# Input/output data
X_data, Y_data = sk.load_iris(return_X_y=True)
X_data = pp.scale(X_data)
Y_data = pp.label_binarize(Y_data, classes=np.unique(Y_data))
# print(X_data)
# print(Y_data)

# Network Parameters
n_hidden_1 = 4 # 1st layer number of neurons
num_input = 4 # two bits
num_classes = 3 # 1 bit
dimension = nns.get_dimensionality(num_input, [n_hidden_1], num_classes)

# NN Parameters
batch_size = X_data.shape[0]  # Number of input patterns
num_steps = 100    # Macro: 100 steps
macro = True       # macro
bounds = 1        # Variable: {1,10}

_inputs = tf.placeholder(tf.float32, [None, num_input])
_outputs = tf.placeholder(tf.float32, [None, num_classes])

data = tf.data.Dataset.from_tensor_slices((_inputs, _outputs)).shuffle(batch_size).repeat().batch(batch_size)
iterator = tf.data.Iterator.from_structure(data.output_types, data.output_shapes)
next_batch = iterator.get_next()
data_init_op = iterator.make_initializer(data)

# Sampling parameters
num_steps = 1000   # Macro: 100 steps, micro: 1000 steps
macro = False      # macro
bounds = 1       # Variable: {1,10}

num_walks = dimension * 10   # make it equal to num weights (i.e. dimension)
num_sims = 1                 # Do 1 sim for now. See if we can get away with it. (Ask Prof!)

# Do the sampling!
nn_model = FLANeuralNetwork(input_tensor=next_batch[0], output_tensor=next_batch[1],
                            num_input=num_input, num_classes=num_classes, num_hidden=[n_hidden_1],
                            act_fn=tf.nn.sigmoid, out_act_fn=tf.nn.sigmoid, error_function="mse", compute_eigens=True)

mgen = MetricGenerator(nn_model, "unbounded_gradient", num_steps, num_walks, num_sims, bounds,
                       macro=macro, print_to_screen=False)

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
init = tf.global_variables_initializer()

with tf.Session(config=config) as sess:
    tf.get_default_graph().finalize()
    sess.run(init)
    sess.run(data_init_op, feed_dict={_inputs: X_data, _outputs: Y_data})
    mgen.write_walks_to_file_sequentially("data/output/iris/hessian/iris_hessian", sess)
