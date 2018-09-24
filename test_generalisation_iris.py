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
from sklearn.model_selection import train_test_split

# Input/output data
X, Y = sk.load_iris(return_X_y=True)
X_data, X_test, Y_data, Y_test = train_test_split(X, Y, test_size=0.2)

scaler = pp.StandardScaler().fit(X_data)

X_data = scaler.transform(X_data)
X_test = scaler.transform(X_test)

classes = np.unique(Y_data)
Y_data = pp.label_binarize(Y_data, classes=classes)
Y_test = pp.label_binarize(Y_test, classes=classes)
# print(X_data)
# print(Y_data)

# Network Parameters
n_hidden_1 = 4 # 1st layer number of neurons
num_input = 4 # two bits
num_classes = 3 # 1 bit
dimension = nns.get_dimensionality(num_input, [n_hidden_1], num_classes)

# NN Parameters
batch_size = X_data.shape[0]  # Number of input patterns
test_size = X_test.shape[0]

#_inputs = tf.placeholder(tf.float32, [None, num_input])
#_outputs = tf.placeholder(tf.float32, [None, num_classes])

#train_data = tf.data.Dataset.from_tensor_slices((_inputs, _outputs)).shuffle(batch_size).repeat().batch(batch_size)
train_data = tf.data.Dataset.from_tensor_slices((tf.cast(X_data, tf.float32), tf.cast(Y_data, tf.float32))).shuffle(batch_size).repeat().batch(batch_size)
test_data = tf.data.Dataset.from_tensor_slices((tf.cast(X_test, tf.float32), tf.cast(Y_test, tf.float32))).shuffle(test_size).repeat().batch(test_size)

#iterator = tf.data.Iterator.from_structure(data.output_types, data.output_shapes)
#next_batch = iterator.get_next()
#data_init_op = iterator.make_initializer(data)

iterator = tf.data.Iterator.from_structure(train_data.output_types, train_data.output_shapes) # reinitializable!
next_batch = iterator.get_next()

train_iterator_init_op = iterator.make_initializer(train_data) #train_iterator.make_initializer(train_data)#
test_iterator_init_op = iterator.make_initializer(test_data)
#test_iterator_init_op = test_dataset.make_initializable_iterator()
iter_dict = {'train_init': train_iterator_init_op, 'test_init': test_iterator_init_op}

# Sampling parameters
macro = False      # macro
if macro:
    num_steps = 100  # Macro: 100 steps, micro: 1000 steps
else:
    num_steps = 1000  # Macro: 100 steps, micro: 1000 steps

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


    #train_handle = sess.run(train_iterator_handle_op)
    #validation_handle = sess.run(validation_iterator.string_handle())

    #sess.run(train_iterator_init_op, feed_dict={_inputs: X_data, _outputs: Y_data})
    #sess.run(iter_dict['train_init'])
    #sess.run(data_init_op, feed_dict={_inputs: X_data, _outputs: Y_data})
    mgen.write_walks_to_file_sequentially_one_at_a_time("data/output/iris/hessian/TEST_iris_hessian", sess, iter_dict)
