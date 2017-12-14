# coding: utf-8

# In[17]:
import math
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt  # Matplotlib is used to generate plots of data.
import fla_metrics as fla
import random_samplers as rs

np.random.seed(1)

### HYPERPARAMETERS
training_iterations = 50  # 'epochs'
display_step = 1  # How often should we print our results
###

# Network Parameters
num_input = 2  # 2-dimensional input data
num_hidden = 2
num_outputs = 2  # 0 or 1
step_size = 0.5
bounds = 5

tf.reset_default_graph()


def cross_entropy_tf(predictions, targets):
    """Calculate the cross entropy loss given some model predictions and target (true) values."""
    return tf.reduce_mean(-tf.reduce_sum(targets * tf.log(predictions), axis=1))


def mean_squared_error_tf(predictions, targets):
    return tf.reduce_mean(tf.reduce_sum(tf.pow(predictions - targets, 2), axis=1))

##############################
X_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y_data = np.array([0, 1, 1, 0])

batch_size = X_data.shape[0]  # The whole dataset; i.e. batch gradient descent.

starting_zone = tf.placeholder(tf.int32, [None])  # bit mask for progressive random walk

x_tf = tf.placeholder(tf.float32, [None, num_input], name="inputs")
y_tf = tf.placeholder(tf.int32, [None], name="outputs")

W1 = tf.placeholder(tf.float32, [num_input, num_hidden], name="W1")
b1 = tf.placeholder(tf.float32, [num_hidden], name="b1")

W2 = tf.placeholder(tf.float32, [num_hidden, num_outputs], name="W2")
b2 = tf.placeholder(tf.float32, [num_outputs], name="b2")

# model

Z = tf.nn.sigmoid(tf.matmul(x_tf, W1) + b1)
Y = tf.nn.softmax(tf.matmul(Z, W2) + b2)  # Compute the predictions of the model

labels = tf.one_hot(y_tf, num_outputs)

# loss functions
cross_entropy = cross_entropy_tf(Y, labels)
mse = mean_squared_error_tf(Y, labels)

# acc_tf = tf.metrics.accuracy(labels, Y)
# mse_tf = tf.metrics.mean_squared_error(labels, Y)

# % of correct answers found in batch
is_correct = tf.equal(tf.argmax(Y, 1), tf.argmax(labels, 1))  # Is the model's prediction correct?
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))  # Compute the average accuracy

init = tf.global_variables_initializer()  # Create an op that initializes these parameters

sess = tf.Session()  # Create a session
writer = tf.summary.FileWriter('./graphs', sess.graph)
sess.run(init)  # Initialize the variables

avg_cost = 0.
avg_accuracy = 0.

current_w1 = np.empty(W1.shape)
current_b1 = np.empty(b1.shape)
current_w2 = np.empty(W2.shape)
current_b2 = np.empty(b2.shape)

all_weights = np.hstack((current_w1.flatten(), current_b1.flatten(), current_w2.flatten(), current_b2.flatten()))

start = rs.progressive_mask_tf(all_weights.shape)

all_weights = rs.init_progressive_mask(start, bounds)

current_w1, current_b1, current_w2, current_b2 = np.hsplit(all_weights, [current_w1.flatten().size,
                                                                         current_w1.flatten().size + current_b2.flatten().size,
                                                                         current_w1.flatten().size + current_b2.flatten().size + current_w2.flatten().size])
current_w1 = current_w1.reshape(W1.shape)
current_b1 = current_b1.reshape(b1.shape)
current_w2 = current_w2.reshape(W2.shape)
current_b2 = current_b2.reshape(b2.shape)

error_history_py = []
b1_history = np.array([current_b1])
b2_history = np.array([current_b2])

print(current_w1)
for iteration in range(training_iterations):
    avg_cost = 0.
    total_batch = int(X_data.shape[0] / batch_size)

    # Loop over all batches.
    for i in range(total_batch):
        batch_x = X_data[i * batch_size: (i + 1) * batch_size, :]
        batch_y = Y_data[i * batch_size: (i + 1) * batch_size]
        # print("Batch x: " + str(batch_x))
        # print("Batch y: " + str(batch_y))

        a, c, m = sess.run([accuracy, cross_entropy, mse],
                           feed_dict={W1: current_w1, b1: current_b1, W2: current_w2, b2: current_b2, x_tf: batch_x,
                                      y_tf: batch_y})

        # all_weights, start = progressive_manhattan_random_step_tf(all_weights, start)
        all_weights, start = rs.progressive_random_step_tf(all_weights, start, step_size, bounds)

        current_w1, current_b1, current_w2, current_b2 = np.hsplit(all_weights, [current_w1.flatten().size,
                                                                                 current_w1.flatten().size + current_b2.flatten().size,
                                                                                 current_w1.flatten().size + current_b2.flatten().size + current_w2.flatten().size])
        current_w1 = current_w1.reshape(W1.shape)
        current_b1 = current_b1.reshape(b1.shape)
        current_w2 = current_w2.reshape(W2.shape)
        current_b2 = current_b2.reshape(b2.shape)

        b1_history = np.append(b1_history, [current_b1], axis=0)
        b2_history = np.append(b2_history, [current_b2], axis=0)
        error_history_py.append([c, m, a])

        # print(current_w1)
        # print(current_b1)
        # print(current_w2)
        # print(current_b2)
        # print(w)
        # Compute average loss.
        avg_accuracy += a / total_batch
        avg_cost += c / total_batch

        print "Step", i, "Current CE cost, MSE cost, and accuracy: ", c, m, a

print("Completed Training")

# COMPUTE GRADIENT MEASURE
# (1) calculate the fitness differences:
err_diff = np.diff(error_history_py, axis=0)

x1, x2 = fla.compute_grad(np.asarray(error_history_py)[:, 0], all_weights.shape[0], step_size, bounds)
print "Grad2: ", x1, x2

# COMPUTE RUGGEDNESS MEASURE
print(err_diff[:, 0])
print "and FEM is: ", fla.compute_fem(err_diff[:, 0])

# COMPUTE NEUTRALITY MEASURE
print "Neutrality M1: ", fla.compute_m1(fla.scale_walk(np.asarray(error_history_py)[:, 0]), 1.0e-8)
print "Neutrality M2: ", fla.compute_m2(fla.scale_walk(np.asarray(error_history_py)[:, 0]), 1.0e-8)

# DRAW FIGURES
fig = plt.figure()

plt.scatter(b1_history[:, 0], b1_history[:, 1])
plt.plot(b1_history[:, 0], b1_history[:, 1])

plt.scatter(b2_history[:, 0], b2_history[:, 1])
plt.plot(b2_history[:, 0], b2_history[:, 1])
# error_history = np.array(error_history_py)
# print(error_history.shape[0])

# plt.plot(np.arange(0,error_history.shape[0]),error_history[:,0])
# plt.plot(np.arange(0,error_history.shape[0]),error_history[:,1])
# plt.plot(np.arange(0,error_history.shape[0]),error_history[:,2])

# plt.axis('equal')
plt.show()

writer.close()

# In[ ]:
