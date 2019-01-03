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
n_hidden_1 = HIDDEN # 1st layer number of neurons
num_input = 2 # two bits
num_classes = 1 # 1 bit
dimension = nns.get_dimensionality(num_input, [n_hidden_1], num_classes)

batch_size = 4  # Number of input patterns
train_data = tf.data.Dataset.from_tensor_slices((tf.cast(X_data, tf.float32), tf.cast(Y_data, tf.float32))).shuffle(batch_size).repeat().batch(batch_size)

iterator = tf.data.Iterator.from_structure(train_data.output_types, train_data.output_shapes) # reinitializable!
next_batch = iterator.get_next()

train_iterator_init_op = iterator.make_initializer(train_data)
iter_dict = {'train_init': train_iterator_init_op}

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
                            act_fn=tf.nn.elu, out_act_fn=tf.nn.sigmoid, error_function="ce", compute_eigens=True)

mgen = MetricGenerator(nn_model, "unbounded_gradient", num_steps, num_walks, num_sims, bounds,
                       macro=macro, print_to_screen=False)

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
init = tf.global_variables_initializer()

with tf.Session(config=config) as sess:
    tf.get_default_graph().finalize()
    sess.run(init)
    mgen.write_walks_to_file_sequentially("data/output/xor/hessian/xor_hessian_hHIDDEN", sess, iter_dict)
