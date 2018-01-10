""" Neural Network.
A 1-Hidden Layers Fully Connected Neural Network (a.k.a Multilayer Perceptron)
implementation with TensorFlow. This example is using the XOR gate dataset.
Author: Anna Bosman
Project: https://github.com/arakitianskaia/fla-in-numpy
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
num_steps = 100
batch_size = X_data.shape[0]  # The whole dataset; i.e. batch gradient descent.
display_step = 100

# Sampling parameters
bounds = 5
step_size = bounds * 0.1
num_walks = 9 # make it equal to num weights (i.e. dimension)
num_sims = 2#30

# Network Parameters
n_hidden_1 = 2 # 1st layer number of neurons
#n_hidden_2 = 256 # 2nd layer number of neurons
num_input = 2 # two bits
num_classes = 1 # 1 bit
# Store layers weight & bias
weights = {
    'h1': tf.Variable(np.empty([num_input, n_hidden_1]), name="h1", dtype=tf.float64, trainable=False),
    'b1': tf.Variable(np.empty([n_hidden_1]), name="b1", dtype=tf.float64, trainable=False),
    #'h2': tf.Variable(current_weights['h2'], name="h2", dtype=tf.float64),
    #'b2': tf.Variable(current_weights['b2'], name="b2", dtype=tf.float64),
    'out': tf.Variable(np.empty([n_hidden_1, num_classes]), name="out", dtype=tf.float64, trainable=False),
    'b3': tf.Variable(np.empty([num_classes]), name="b3", dtype=tf.float64, trainable=False)
}

weight_placeholders = {
    'h1': tf.placeholder(name="h1_ph", dtype=tf.float64, shape=weights['h1'].shape),
    'b1': tf.placeholder(name="b1_ph", dtype=tf.float64, shape=weights['b1'].shape),
    #'h2': tf.Variable(current_weights['h2'], name="h2", dtype=tf.float64),
    #'b2': tf.Variable(current_weights['b2'], name="b2", dtype=tf.float64),
    'out': tf.placeholder(name="out_ph", dtype=tf.float64, shape=weights['out'].shape),
    'b3': tf.placeholder(name="b3_ph", dtype=tf.float64, shape=weights['b3'].shape)
}


### ASSIGN updated values to the TF variables
weight_upd_ops = []
for k in sorted(weights):
    weight_upd_ops.append(tf.assign(weights[k], weight_placeholders[k]))


def assign_upd_weights(session, current_weights, all_weights):
    i = 0
    j = 0
    for k in sorted(current_weights):
        length = current_weights[k].size
        shape = current_weights[k].shape
        current_weights[k] = all_weights[i:i + length].reshape(shape)
        ### ASSIGN updated values to the TF variables
        if session is not None:
            session.run(weight_upd_ops[j], feed_dict={weight_placeholders[k]: current_weights[k]})
            j = j + 1
        i += length

# tf Graph input
X = tf.placeholder(tf.float64, [None, num_input])
Y = tf.placeholder(tf.float64, [None, num_classes])


# Create model
def neural_net(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.nn.xw_plus_b(x, weights['h1'], weights['b1'])
    # Output fully connected layer with a neuron for each class
    out_layer = tf.nn.xw_plus_b(layer_1, weights['out'], weights['b3'])
    return out_layer

# Construct model
logits = neural_net(X)
prediction = tf.nn.sigmoid(logits)

# Define loss
cross_entropy_op = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    logits=logits, labels=Y))
mse_op = tf.reduce_mean(tf.square(prediction - Y))

# Evaluate model
correct_pred = tf.equal(tf.floor(prediction+0.5), Y)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()


def calculate_metrics(all_walks, dim):
    # Work with ALL walks:
    all_err_diff = np.diff(all_walks, axis=1)
    # (1) Gradients [NB: requires a different walk!]:
    my_grad = np.apply_along_axis(fla.compute_grad, 1, all_walks, dim, step_size, bounds)
    print("Grad on ALL: ", my_grad)  # each column corresponds to outputs per walk
    # (2) Ruggedness:
    my_rugg = np.apply_along_axis(fla.compute_fem, 1, all_err_diff)
    print("FEM on ALL: ", my_rugg)  # each column corresponds to outputs per walk
    print("Max FEM: ", np.amax(my_rugg, 0))
    print("Min FEM: ", np.amin(my_rugg, 0))
    print("Avg FEM: ", np.average(my_rugg, 0))
    # (3) Neutrality:
    my_neut1 = np.apply_along_axis(fla.compute_m1, 1, all_err_diff, 1.0e-8)
    my_neut2 = np.apply_along_axis(fla.compute_m2, 1, all_err_diff, 1.0e-8)
    print("M1 on ALL: ", my_neut1)  # each column corresponds to outputs per walk
    print("M2 on ALL: ", my_neut2)  # each column corresponds to outputs per walk
    return my_grad, my_rugg, my_neut1, my_neut2


def one_sim(sess):
    all_walks = np.empty((num_walks, num_steps, 3))
    for walk_counter in range(0, num_walks):
        error_history_py = np.empty((num_steps, 3))  # dimensions: x -> steps, y -> error metrics

        # weights for the current walk:
        current_weights = {
            'h1': np.empty(weights['h1'].shape),
            # 'h2': np.empty([n_hidden_1, n_hidden_2]),
            'out': np.empty(weights['out'].shape),
            'b1': np.empty(weights['b1'].shape),
            # 'b2': np.empty([n_hidden_2]),
            'b3': np.empty(weights['b3'].shape)
        }

        all_weights = np.concatenate([v.flatten() for k, v in sorted(current_weights.items())])
        start = rs.progressive_mask_tf(all_weights.shape)
        all_weights = rs.init_progressive_mask(start, bounds)
        assign_upd_weights(sess, current_weights, all_weights)

        for step in range(0, num_steps):
            # Calculate batch loss and accuracy
            ce, mse, acc = sess.run([cross_entropy_op, mse_op, accuracy], feed_dict={X: X_data, Y: Y_data})
            if step % display_step == 0:
                print("Step " + str(step) + ", Cross-entropy Loss = " + \
                      "{:.4f}".format(ce) + ", MSE Loss = " + \
                      "{:.4f}".format(mse) + ", Training Accuracy = " + \
                      "{:.3f}".format(acc))
            all_weights, start = rs.progressive_random_step_tf(all_weights, start, step_size, bounds)
            assign_upd_weights(sess, current_weights, all_weights)

            # And now: update the weight variables!
            error_history_py[step] = [ce, mse, acc]
        print("Done with walk number ", walk_counter)
        all_walks[walk_counter] = error_history_py

    print("All random walks are done now.")
    print("Calculating FLA metrics...")
    print("Dimensionality is: ", all_weights.shape[0])

    grad, fem, m1, m2 = calculate_metrics(all_walks, all_weights.shape[0])
    return np.average(grad, 0), np.average(fem, 0), np.average(m1, 0), np.average(m2, 0)


# Start training
with tf.Session() as sess:
    # Run the initializer
    tf.get_default_graph().finalize()
    sess.run(init)

    grad_list = np.empty((num_sims, 2, 3))
    fem_list = np.empty((num_sims, 3))
    m_list = np.empty((num_sims, 2, 3))

    for i in range(0, num_sims):
        g, fem, m1, m2 = one_sim(sess)
        print("Avg Grad: ", g)
        print("Avg FEM: ", fem)
        print("Avg M1: ", m1)
        print("Avg M2: ", m2)
        grad_list[i] = g
        fem_list[i] = fem
        m_list[i][0] = m1
        m_list[i][1] = m2
        print("----------------------- Sim ",i," is done -------------------------------")
        
    print("Gradients across sims: ", grad_list)
    print("FEM across sims: ", fem_list)
    print("M1/M2 across sims: ", m_list)

    g_avg = grad_list[:,0,:]
    print("g_avg: ", g_avg)


    g_dev = grad_list[:,1,:]
    print("g_dev: ", g_dev)

    np.savetxt("data/xor_g_avg_test.csv", g_avg, delimiter=",")
