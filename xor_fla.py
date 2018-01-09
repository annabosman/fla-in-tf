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
num_steps = 50
batch_size = X_data.shape[0]  # The whole dataset; i.e. batch gradient descent.
display_step = 10

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

# Store layers weight & bias
weights = {
    'h1': tf.Variable(current_weights['h1'], name="h1", dtype=tf.float64, trainable=False),
    'b1': tf.Variable(current_weights['b1'], name="b1", dtype=tf.float64, trainable=False),
    #'h2': tf.Variable(current_weights['h2'], name="h2", dtype=tf.float64),
    #'b2': tf.Variable(current_weights['b2'], name="b2", dtype=tf.float64),
    'out': tf.Variable(current_weights['out'], name="out", dtype=tf.float64, trainable=False),
    'b3': tf.Variable(current_weights['b3'], name="b3", dtype=tf.float64, trainable=False)
}

weight_placeholders = {
    'h1': tf.placeholder(name="h1_ph", dtype=tf.float64, shape=weights['h1'].shape),
    'b1': tf.placeholder(name="b1_ph", dtype=tf.float64, shape=weights['b1'].shape),
    #'h2': tf.Variable(current_weights['h2'], name="h2", dtype=tf.float64),
    #'b2': tf.Variable(current_weights['b2'], name="b2", dtype=tf.float64),
    'out': tf.placeholder(name="out_ph", dtype=tf.float64, shape=weights['out'].shape),
    'b3': tf.placeholder(name="b3_ph", dtype=tf.float64, shape=weights['b3'].shape)
}

all_weights = np.concatenate([v.flatten() for k, v in sorted(current_weights.items())])

print (all_weights)

start = rs.progressive_mask_tf(all_weights.shape)

all_weights = rs.init_progressive_mask(start, bounds)

print (all_weights)

### ASSIGN updated values to the TF variables
weight_upd_ops = []
for k in sorted(current_weights):
    weight_upd_ops.append(tf.assign(weights[k], weight_placeholders[k]))

def assign_upd_weights(session):
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

assign_upd_weights(None)

# tf Graph input
X = tf.placeholder(tf.float64, [None, num_input])
Y = tf.placeholder(tf.float64, [None, num_classes])




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

# Define loss
loss_op = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    logits=logits, labels=Y))
tf.summary.scalar('loss', loss_op)

# Evaluate model
correct_pred = tf.equal(tf.floor(prediction+0.5), Y)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
tf.summary.scalar('accuracy', accuracy)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('./graphs', sess.graph)
    # Run the initializer
    tf.get_default_graph().finalize()
    sess.run(init)
    all_walks = np.empty((num_walks, num_steps, 2))
    for walk_counter in range(0, num_walks):
        error_history_py = np.empty((num_steps, 2)) # dimensions: x -> steps, y -> error metrics

        for step in range(0, num_steps):
            # Calculate batch loss and accuracy
            summ, loss, acc = sess.run([merged, loss_op, accuracy], feed_dict={X: X_data, Y: Y_data})
            writer.add_summary(summ, step)
            if step % display_step == 0:
                print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))
            all_weights, start = rs.progressive_random_step_tf(all_weights, start, step_size, bounds)
            assign_upd_weights(sess)

            # And now: update the weight variables!
            error_history_py[step] = [loss, acc]
        print("Done with walk number ", walk_counter)
        all_walks[walk_counter] = error_history_py

        start = rs.progressive_mask_tf(all_weights.shape) # next mask
        all_weights = rs.init_progressive_mask(start, bounds) # new init = 0

        assign_upd_weights(sess)

    print("All random walks are done now.")
    writer.close()

print(all_walks)

# COMPUTE GRADIENT MEASURE
# (1) calculate the fitness differences:
err_diff = np.diff(error_history_py, axis=0)

x1, x2 = fla.compute_grad(np.asarray(error_history_py)[:, 0], all_weights.shape[0], step_size, bounds)
print ("Grad2: ", x1, x2)

# COMPUTE RUGGEDNESS MEASURE
print(err_diff[:, 0])
print ("and FEM is: ", fla.compute_fem(err_diff[:, 0]))
print ("and FEM is: ", fla.compute_fem(err_diff[:, 1]))

# COMPUTE NEUTRALITY MEASURE
print ("Neutrality M1: ", fla.compute_m1(fla.scale_walk(np.asarray(error_history_py)[:, 0]), 1.0e-8))
print ("Neutrality M2: ", fla.compute_m2(fla.scale_walk(np.asarray(error_history_py)[:, 0]), 1.0e-8))