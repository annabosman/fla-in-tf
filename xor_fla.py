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

# Input/output data
X_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y_data = np.array([[0], [1], [1], [0]])

# NN Parameters
num_steps = 1000    # FEM: 1000 steps; Neutrality: # steps proportionate to step size/search space
batch_size = X_data.shape[0]  # The whole data set; i.e. batch gradient descent.
display_step = 100

# Sampling parameters
macro = False                                       # try micro and macro for all
bounds = 1                                          # Also try: 0.5, 1
step_size = 0
if macro is True: step_size = (2 * bounds) * 0.1    # 10% of the search space
else: step_size = (2 * bounds) * 0.01               # 1% of the search space

num_walks = 1   # make it equal to num weights (i.e. dimension)
num_sims = 1   # 30 independent runs: for stats

# Network Parameters
num_input = 2 # two bits
n_hidden_1 = 3 # 1st layer number of neurons
#n_hidden_2 = 256 # 2nd layer number of neurons
num_classes = 1 # 1 bit

# tf Graph input: data input (X) and output (Y)
X = tf.placeholder(tf.float64, [None, num_input])
Y = tf.placeholder(tf.float64, [None, num_classes])

# Define layers weights & biases
weights = {
    'h1': tf.Variable(np.empty([num_input, n_hidden_1]), name="h1", dtype=tf.float64, trainable=False),
    'b1': tf.Variable(np.empty([n_hidden_1]), name="b1", dtype=tf.float64, trainable=False),
    'out': tf.Variable(np.empty([n_hidden_1, num_classes]), name="out", dtype=tf.float64, trainable=False),
    'b3': tf.Variable(np.empty([num_classes]), name="b3", dtype=tf.float64, trainable=False)
}

weight_placeholders = {
    'h1': tf.placeholder(name="h1_ph", dtype=tf.float64, shape=weights['h1'].shape),
    'b1': tf.placeholder(name="b1_ph", dtype=tf.float64, shape=weights['b1'].shape),
    'out': tf.placeholder(name="out_ph", dtype=tf.float64, shape=weights['out'].shape),
    'b3': tf.placeholder(name="b3_ph", dtype=tf.float64, shape=weights['b3'].shape)
}

### Define a list of operations to ASSIGN updated values to the TF variables
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

# Create model
def neural_net(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.nn.sigmoid(tf.nn.xw_plus_b(x, weights['h1'], weights['b1'])) ######## SIGMOID
    # Output fully connected layer with a neuron for each class
    out_layer = tf.nn.xw_plus_b(layer_1, weights['out'], weights['b3'])
    return out_layer

# Construct model
logits = neural_net(X)
prediction = tf.nn.sigmoid(logits) ############### SIGMOID

# Define losses
cross_entropy_op = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    logits=logits, labels=Y))
mse_op = tf.reduce_mean(tf.square(prediction - Y))

# Define gradient, hessian:
gradients = tf.gradients(mse_op, [weights['h1'], weights['b1'], weights['out'], weights['b3']])
hessians = tf.hessians(mse_op, [weights['h1'], weights['b1'], weights['out'], weights['b3']])

eigenvals1 = tf.self_adjoint_eigvals(tf.reshape(hessians[0], [tf.size(weights['h1']), tf.size(weights['h1'])]))
eigenvals2 = tf.self_adjoint_eigvals(tf.reshape(hessians[1], [tf.size(weights['b1']), tf.size(weights['b1'])]))
eigenvals3 = tf.self_adjoint_eigvals(tf.reshape(hessians[2], [tf.size(weights['out']), tf.size(weights['out'])]))
eigenvals4 = tf.self_adjoint_eigvals(tf.reshape(hessians[3], [tf.size(weights['b3']), tf.size(weights['b3'])]))

# Evaluate model
correct_pred = tf.equal(tf.floor(prediction+0.5), Y)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Define the initialisation op
init = tf.global_variables_initializer()

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
            ce, mse, acc, grads, hess, eigens = sess.run([cross_entropy_op, mse_op, accuracy, gradients, hessians, eigenvals1], feed_dict={X: X_data, Y: Y_data})
            if step % display_step == 0:
                print("Step " + str(step) + ", Cross-entropy Loss = " + \
                      "{:.4f}".format(ce) + ", MSE Loss = " + \
                      "{:.4f}".format(mse) + ", Training Accuracy = " + \
                      "{:.3f}".format(acc))
               # print("Grads shape: " + str(grads))
                #print("Hessians shape: " + str(hess))
                print("Eigens: " + str(eigens))
            # And now: update the weight variables!
            all_weights, start = rs.progressive_random_step_tf(all_weights, start, step_size, bounds)
            assign_upd_weights(sess, current_weights, all_weights)

            error_history_py[step] = [ce, mse, acc]
        print("Done with walk number ", walk_counter)
        all_walks[walk_counter] = error_history_py

    print("All random walks are done now.")
    print("Calculating FLA metrics...")
    print("Dimensionality is: ", all_weights.shape[0])

    return all_walks, all_weights.shape[0]


# Start training
with tf.Session() as sess:
    # Run the initializer
    tf.get_default_graph().finalize()
    sess.run(init)

    grad_list = np.empty((num_sims, 2, 3))
    fem_list = np.empty((num_sims, 3))
    m_list = np.empty((num_sims, 2, 3))

    for i in range(0, num_sims):
        all_w, d = one_sim(sess)
        g, fem, m1, m2 = fla.calculate_metrics(all_w, d, step_size, bounds)
        #fem = calc_fem(all_w)
        # print("Avg Grad: ", g)
        #print("Avg FEM for walk ", i+1, ": ", fem)
        # print("Avg M1: ", m1)
        # print("Avg M2: ", m2)
        # grad_list[i] = g
        m_list[i][0] = m1
        m_list[i][1] = m2
        print("----------------------- Sim ",i+1," is done -------------------------------")
        
    #print("Gradients across sims: ", grad_list)
    #print("FEM across sims: ", fem_list)
    print("M1/M2 across sims: ", m_list)

    m1 = m_list[:,0,:]
    print("m1: ", m1)

    m2 = m_list[:,1,:]
    print("m2: ", m2)

    # g_avg = grad_list[:,0,:]
    # print("g_avg: ", g_avg)
    #
    # g_dev = grad_list[:,1,:]
    # print("g_dev: ", g_dev)

    #with open("data/xor_fem_micro_1.csv", "a") as f:
    #    np.savetxt(f, ["cross-entropy", "mse", "accuracy"], "%s")
    #    np.savetxt(f, fem_list, delimiter=",")
