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
X_data = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
Y_data = np.array([[0.], [1.], [1.], [0.]])

# Network Parameters
n_hidden_1 = 2 # 1st layer number of neurons
num_input = 2 # two bits
num_classes = 1 # 1 bit
dimension = nns.get_dimensionality(num_input, [n_hidden_1], num_classes)


_inputs = tf.placeholder(tf.float32, [None, num_input])
_outputs = tf.placeholder(tf.float32, [None, num_classes])

# zip the two datasets together
data = tf.data.Dataset.from_tensor_slices((_inputs, _outputs)).repeat().batch(4)
iterator = tf.data.Iterator.from_structure(data.output_types, data.output_shapes)
next_batch = iterator.get_next()
data_init_op = iterator.make_initializer(data)

# Sampling parameters
num_steps = 100   # Macro: 100 steps
macro = True      # macro
bounds = 10       # Variable: {1,10}

num_walks = dimension * 10   # make it equal to num weights (i.e. dimension)
num_sims = 1                 # Do 1 sim for now. See if we can get away with it. (Ask Prof!)

# Do the sampling!
nn_model = FLANeuralNetwork(input_tensor=next_batch[0], output_tensor=next_batch[1],
                            num_input=num_input, num_classes=num_classes, num_hidden=[n_hidden_1],
                            act_fn=tf.nn.sigmoid, out_act_fn=tf.nn.sigmoid, error_function="ce", compute_eigens=True)

mgen = MetricGenerator(nn_model, "unbounded_gradient", num_steps, num_walks, num_sims, bounds,
                       macro=macro, print_to_screen=False)

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
init = tf.global_variables_initializer()

with tf.Session(config=config) as sess:
    tf.get_default_graph().finalize()
    sess.run(init)
    sess.run(data_init_op, feed_dict={_inputs: X_data, _outputs: Y_data})
    mgen.write_walks_to_file_sequentially("data/output/xor/hessian/xor_hessian", sess)
