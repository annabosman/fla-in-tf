# coding: utf-8

# In[17]:
import math
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # Matplotlib is used to generate plots of data.

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


def random_step_tf(weights):
    return np.add(weights, np.random.uniform(-step_size, step_size, weights.shape))


def progressive_manhattan_random_step_tf(weights, mask):
    random_steps = np.zeros(weights.shape)
    shape = weights.shape
    random_steps = random_steps.flatten()
    ind = np.random.choice(weights.size, size=1)
    random_steps[ind] = step_size  # Manhattan: step size is constant
    random_steps = random_steps.reshape(shape)
    mask = bounds_check(weights, mask, random_steps * mask)
    return np.add(weights, random_steps * mask), mask


def progressive_random_step_tf(weights, mask):
    random_steps = np.random.uniform(0, step_size, weights.shape)
    mask = bounds_check(weights, mask, random_steps * mask)
    return np.add(weights, random_steps * mask), mask


def progressive_mask_tf(shape):
    start = np.ones(shape) * -1
    mask = np.random.randint(0, 2, shape)
    start = start ** mask
    print(start)
    return start


def bounds_check(inputs, mask, step):
    conds = [np.absolute(inputs + step) > bounds, np.absolute(inputs + step) <= bounds]
    funcs = [lambda mask: -mask, lambda mask: mask]
    return np.piecewise(mask, conds, funcs)  # return mask


def init_progressive_mask(mask):
    random_nums = np.random.uniform(0, bounds, mask.shape)
    conds = [mask == 1, mask == -1]
    funcs = [lambda x: bounds - x, lambda x: x - bounds]
    return np.piecewise(random_nums, conds, funcs)  # return initialised random numbers


def compute_s(epsilon, walk_diff):
    """Computer the symbolic array {-1,0,1} for the FEM metrics."""
    conds = [walk_diff < -epsilon, (walk_diff >= -epsilon) & (walk_diff <= epsilon), walk_diff > epsilon]
    funcs = [-1, 0, 1]
    return np.piecewise(walk_diff, conds, funcs)


def compute_h(symbolic_s):
    """Computer the entropy of the three-point objects in symbolic array {-1,0,1}."""
    bin_count = [0., 0., 0., 0., 0., 0.]
    p = 0
    q = 1
    num_symbols = len(symbolic_s)
    while q < (num_symbols - 1):
        if symbolic_s[p] != symbolic_s[q]:
            if symbolic_s[p] == 0:
                if (symbolic_s[q] == 1):
                    bin_count[0] += 1
                else:
                    bin_count[1] += 1
            if symbolic_s[p] == 1:
                if (symbolic_s[q] == 0):
                    bin_count[2] += 1
                else:
                    bin_count[3] += 1
            if symbolic_s[p] == -1:
                if (symbolic_s[q] == 0):
                    bin_count[4] += 1
                else:
                    bin_count[5] += 1
        p += 1
        q += 1

    entropy = 0.
    for i in range(0, 5):
        if (bin_count[i] != 0):
            bin_count[i] = bin_count[i] / (num_symbols - 2.)
            entropy -= bin_count[i] * math.log(bin_count[i], 6)
    return entropy


def is_flat(s):
    """Given symbolic array S, return true is all symbols = 0, and false otherwise."""
    for i in range(0, len(s)):
        if s[i] != 0:
            return False
    return True


def get_epsilon_star(walk_diff):
    """Calculate epsilon* for the FEM metrics."""
    eps_base = 10.
    eps_step = 10.
    eps = 0.
    eps_order = 0.
    not_found = True
    # Quickly find the order:
    while (not_found):
        symbolic_s = compute_s(eps, walk_diff)
        if is_flat(symbolic_s):
            not_found = False
            eps_step = eps_step / eps_base
        else:
            eps = eps_step
            eps_order += 1
            eps_step *= eps_base
    small_step = 0.01 * (10 ** eps_order)
    not_found = True
    while (not_found):
        symbolic_s = compute_s(eps, walk_diff)
        if is_flat(symbolic_s):
            if eps_step <= small_step:
                not_found = False
            else:
                eps -= eps_step
                eps_step /= eps_base
                eps += eps_step
        else:
            eps += eps_step
    return eps  # this is epsilon*


def compute_fem(walk_diff):
    """Calculate the FEM metric."""
    eps_star = get_epsilon_star(walk_diff)
    incr = 0.05 * eps_star
    h_max = 0.
    for i in np.arange(0., eps_star, incr):
        cur_h = compute_h(compute_s(i, walk_diff))
        if cur_h > h_max: h_max = cur_h
    return h_max


def scale_walk(walk):
    """Scales the walk of arbitrary fitness range to [0,1]"""
    min_f = min(walk)
    max_f = max(walk)
    diff = max_f - min_f
    return (walk - min_f) / diff


def compute_m1(walk, epsilon):
    """Calculate the neutrality metric M1. Input: progressive random walk (fitness values scaled to [0,1])"""
    m1 = 0.
    len_3p_walk = len(walk) - 2
    for i in range(0, len_3p_walk):
        min_f = min(walk[i:i + 3])
        max_f = max(walk[i:i + 3])
        if max_f - min_f <= epsilon: m1 += 1.

    return m1 / len_3p_walk


def compute_m2(walk, epsilon):
    """Calculate the neutrality metric M2. Input: progressive random walk (fitness values scaled to [0,1])"""
    m2 = 0.
    temp = 0.
    len_3p_walk = len(walk) - 2
    for i in range(0, len_3p_walk):
        min_f = min(walk[i:i + 3])
        max_f = max(walk[i:i + 3])
        if max_f - min_f <= epsilon:  # is neutral
            temp += 1.0
        elif temp > 0:
            if temp > m2: m2 = temp
            temp = 0

    return m2 / len_3p_walk

def compute_grad(walk, n, step_size):
    # (1) calculate the fitness differences:
    err_diff = np.diff(walk)
    # (2) calculate the difference between max and min fitness:
    fit_diff = np.amax(walk) - np.amin(walk)
    # (3) calculate the total manhattan distance between the bounds of the search space
    manhattan_diff = n * 2 * bounds
    scaled_step = step_size / manhattan_diff
    # (4) calculate g(t) for t = 1..T
    g_t = (err_diff / fit_diff) / scaled_step
    # (5) calculate G_avg
    return np.mean(np.absolute(g_t)), np.std(np.absolute(g_t))

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

start = progressive_mask_tf(all_weights.shape)

all_weights = init_progressive_mask(start)

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
        all_weights, start = progressive_random_step_tf(all_weights, start)

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
# (2) calculate the difference between max and min fitness:
fit_diff = np.amax(error_history_py, axis=0) - np.amin(error_history_py, axis=0)
# (3) calculate the total manhattan distance between the bounds of the search space
manhattan_diff = all_weights.shape[0] * 2 * bounds
scaled_step = step_size / manhattan_diff
# (4) calculate g(t) for t = 1..T
g_t = (err_diff / fit_diff) / scaled_step
# (5) calculate G_avg
g_avg = np.mean(np.absolute(g_t), axis=0)
g_dev = np.std(np.absolute(g_t), axis=0)
print "Grad1: ", (g_avg, g_dev)

x1, x2 = compute_grad(np.asarray(error_history_py)[:, 0], all_weights.shape[0], step_size)
print "Grad2: ", x1, x2

# COMPUTE RUGGEDNESS MEASURE
print(err_diff[:, 0])
ss = compute_s(0.1, err_diff[:, 0])
print(ss)
print(compute_h(ss))
print(get_epsilon_star(err_diff[:, 0]))
print "and FEM is: ", compute_fem(err_diff[:, 0])

# COMPUTE NEUTRALITY MEASURE
print "Neutrality M1: ", compute_m1(scale_walk(np.asarray(error_history_py)[:, 0]), 1.0e-8)
print "Neutrality M2: ", compute_m2(scale_walk(np.asarray(error_history_py)[:, 0]), 1.0e-8)

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
