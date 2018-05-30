""" Sampling XOR for FLA.
Author: Anna Bosman
Project: https://github.com/arakitianskaia/fla-in-numpy
"""

from __future__ import print_function
from cycler import cycler

import tensorflow as tf
import numpy as np
from nn_for_fla_tf import FLANeuralNetwork
from fla_for_nn_tf import MetricGenerator
import matplotlib.pyplot as plt  # Matplotlib is used to generate plots of data.

# Input/output data
X_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y_data = np.array([[0], [1], [1], [0]])


def get_data():
    return X_data, Y_data


def coverage_hist(all_p):
    mean = np.mean(all_p.ravel())
    std = np.std(all_p.ravel())

    n, bins, patches = plt.hist(all_p.ravel(), 100, normed=1, facecolor='k', alpha=0.75, histtype='bar')
    tit = '$\mu=$' + ("%.3f" % mean) + ', $\sigma=$' + ("%.3f" % std)
    plt.title(tit)
    plt.show()


def plot_scatter(all_p):
    # DRAW FIGURES (for debugging)
    # Create cycler object. Use any styling from above you please
    monochrome = (
        cycler('color', ['k']) * (cycler('linestyle', ['--', ':', '-.', '-', '--'])) * cycler('marker', ['+', '.', '*']))# * cycler('marker', [',', '+'])) #


    fig, ax = plt.subplots()  # fig = plt.figure()
    ax.set_prop_cycle(monochrome)
    walk = all_p[0]
    walk2 = all_p[1]

    #ax.scatter(walk[:, 0], walk[:, 1])
    ax.plot(walk[:, 0], walk[:, 1])

    #ax.scatter(walk[:, 2], walk[:, 3])
    ax.plot(walk[:, 2], walk[:, 3])

    #ax.scatter(walk[:, 4], walk[:, 5])
    ax.plot(walk[:, 4], walk[:, 5])

    #ax.scatter(walk2[:, 0], walk2[:, 1])
    ax.plot(walk2[:, 0], walk2[:, 1])

    #ax.scatter(walk2[:, 2], walk2[:, 3])
    ax.plot(walk2[:, 2], walk2[:, 3])

    #ax.scatter(walk2[:, 4], walk2[:, 5])
    ax.plot(walk2[:, 4], walk2[:, 5])

    #ax.xlabel('x1')
    #ax.ylabel('x2')
    #plt.xlim(-10, 10)
    #plt.ylim(-10, 10)
    # ax.axis('equal')
    plt.show()

# Sampling parameters
num_steps = 1000   # FEM: 1000 steps; Neutrality: # steps proportionate to step size/search space
macro = True                                       # try micro and macro for all
bounds = 10                                          # Also try: 0.5, 1                                      # try micro and macro for all
step_size = 0.3 # NOT USED

num_walks = 1   # make it equal to num weights (i.e. dimension)
num_sims = 1   # 30 independent runs: for stats

# Network Parameters
n_hidden_1 = 2 # 1st layer number of neurons
num_input = 2 # two bits
num_classes = 1 # 1 bit

# Do the sampling!
nn_model = FLANeuralNetwork(num_input=num_input, num_classes=num_classes, num_hidden=[n_hidden_1],
                            act_fn=tf.nn.sigmoid, out_act_fn=tf.nn.sigmoid, compute_eigens=True)
#nn_model.build_random_walk_graph(walk_type="progressive", step_size=step_size, bounds=bounds)
macro = False
mgen = MetricGenerator(nn_model, get_data, "unbounded_gradient", num_steps, num_walks, num_sims, bounds,
                       macro=macro, print_to_screen=True)

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
init = tf.global_variables_initializer()

with tf.Session(config=config) as sess:
    tf.get_default_graph().finalize()
    sess.run(init)
    mgen.get_neutrality_and_ruggedness_metrics_only(sess=sess, filename_header="data/output/xor/gecco/xor_unbounded_grad")
    #mgen.calculate_ruggedness_metrics(sess=sess, filename_header="data/output/xor/TEST_xor")
    #all_w, all_p = mgen.do_the_walks(sess=sess)
    #plot_scatter(all_p)
    #coverage_hist(all_p)

    #mse = all_w[:,:,2].ravel()
    #mean = np.mean(mse)
    #std = np.std(mse)

    #values1, counts1 = np.unique(mse, return_counts=True)
    #print(values1, counts1)

    #coverage_hist(all_w[:,:,1])

    #all_w = nn_model.one_sim(sess, num_walks, num_steps, get_data, print_to_screen=True)
    #sess.run(init)

#mgen = MetricGenerator(get_data=get_data, num_steps=num_steps, bounds=bounds, macro=macro, num_walks=num_walks, num_sims=num_sims, nn_model=nn_model, print_to_screen=True)
#mgen.calculate_neutrality_metrics("data/output/xor/xor")

