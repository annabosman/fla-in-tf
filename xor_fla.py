""" Neural Network.
A 2-Hidden Layers Fully Connected Neural Network (a.k.a Multilayer Perceptron)
implementation with TensorFlow. This example is using the MNIST database
of handwritten digits (http://yann.lecun.com/exdb/mnist/).
Links:
    [MNIST Dataset](http://yann.lecun.com/exdb/mnist/).
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""

from __future__ import print_function

import tensorflow as tf
import numpy as np
import fla_metrics as fla
import random_samplers as rs

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data

# Input/output data
X_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y_data = np.array([[0], [1], [1], [0]])

# NN Parameters
learning_rate = 0.1
num_steps = 50
batch_size = X_data.shape[0]  # The whole dataset; i.e. batch gradient descent.
display_step = 100

# Sampling parameters
step_size = 0.5
bounds = 5
num_walks = 10

# Network Parameters
n_hidden_1 = 2 # 1st layer number of neurons
#n_hidden_2 = 256 # 2nd layer number of neurons
num_input = 2 # two bits
num_classes = 1 # 1 bit

# Initialise random mask & weights
current_weights = {
    'h1': np.empty([num_input, n_hidden_1]),
    #'h2': np.empty([n_hidden_1, n_hidden_2]),
    'out': np.empty([n_hidden_1, num_classes]),
    'b1': np.empty([n_hidden_1]),
    #'b2': np.empty([n_hidden_2]),
    'b3': np.empty([num_classes])
}

all_weights = np.concatenate([v.flatten() for k, v in sorted(current_weights.items())])

print (all_weights)

start = rs.progressive_mask_tf(all_weights.shape)

all_weights = rs.init_progressive_mask(start, bounds)
print (all_weights)

i = 0
for k in sorted(current_weights):
    length = current_weights[k].size
    shape = current_weights[k].shape
    current_weights[k] = all_weights[i:i+length].reshape(shape)
    i += length

# tf Graph input
X = tf.placeholder(tf.float64, [None, num_input])
Y = tf.placeholder(tf.float64, [None, num_classes])

# Store layers weight & bias
weights = {
    'h1': tf.Variable(current_weights['h1'], name="h1", dtype=tf.float64),
    'b1': tf.Variable(current_weights['b1'], name="b1", dtype=tf.float64),
    #'h2': tf.Variable(current_weights['h2'], name="h2", dtype=tf.float64),
    #'b2': tf.Variable(current_weights['b2'], name="b2", dtype=tf.float64),
    'out': tf.Variable(current_weights['out'], name="out", dtype=tf.float64),
    'b3': tf.Variable(current_weights['b3'], name="b3", dtype=tf.float64)
}


# Create model
def neural_net(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), weights['b1'])
    # Hidden fully connected layer with 256 neurons
    #layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), weights['b2'])
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_1, weights['out']) + weights['b3']
    return out_layer

# Construct model
logits = neural_net(X)
prediction = tf.nn.sigmoid(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    logits=logits, labels=Y))

# Evaluate model
correct_pred = tf.equal(tf.floor(prediction+0.5), Y)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)
    all_walks = []
    for walk_counter in range(0, num_walks):
        error_history_py = []

        for step in range(1, num_steps+1):
            batch_x = X_data
            batch_y = Y_data
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x, Y: batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))
            all_weights, start = rs.progressive_random_step_tf(all_weights, start, step_size, bounds)
            i = 0
            for k in sorted(current_weights):
                length = current_weights[k].size
                shape = weights[k].shape
                current_weights[k] = all_weights[i:i + length].reshape(shape)
                ### ASSIGN updated values to the TF variables
                sess.run(tf.assign(weights[k], current_weights[k]))
                i += length

            # And now: update the weight variables!
            error_history_py.append([loss, acc])
        print("Done with walk number ", walk_counter)
        all_walks.append(error_history_py)

        start = rs.progressive_mask_tf(all_weights.shape) # next mask
        all_weights = rs.init_progressive_mask(start, bounds) # new initi = 0
        i = 0
        for k in sorted(current_weights):
            length = current_weights[k].size
            shape = weights[k].shape
            current_weights[k] = all_weights[i:i + length].reshape(shape)
            ### ASSIGN updated values to the TF variables
            sess.run(tf.assign(weights[k], current_weights[k]))
            i += length

    print("All random walks are done now.")

print(all_walks)

# COMPUTE GRADIENT MEASURE
# (1) calculate the fitness differences:
err_diff = np.diff(error_history_py, axis=0)

x1, x2 = fla.compute_grad(np.asarray(error_history_py)[:, 0], all_weights.shape[0], step_size, bounds)
print ("Grad2: ", x1, x2)

# COMPUTE RUGGEDNESS MEASURE
print(err_diff[:, 0])
print ("and FEM is: ", fla.compute_fem(err_diff[:, 0]))

# COMPUTE NEUTRALITY MEASURE
print ("Neutrality M1: ", fla.compute_m1(fla.scale_walk(np.asarray(error_history_py)[:, 0]), 1.0e-8))
print ("Neutrality M2: ", fla.compute_m2(fla.scale_walk(np.asarray(error_history_py)[:, 0]), 1.0e-8))