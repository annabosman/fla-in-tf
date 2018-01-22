""" Generic Neural Network model suitable for Fitness Landscape Analysis
Author: Anna Bosman
Based on the original code by: Aymeric Damien
Project: https://github.com/arakitianskaia/fla-in-numpy
"""

from __future__ import print_function

import tensorflow as tf
import numpy as np
import random_samplers as rs


class FLANeuralNetwork(object):
    def __init__(self, n_hidden_1, num_input, num_classes, act_fn, out_act_fn):
        self.n_hidden_1 = n_hidden_1  # 1st layer number of neurons ##### TODO: replace with arbitrary number of layers
        self.num_input = num_input
        self.num_classes = num_classes

        self.act_fn = act_fn
        self.out_act_fn = out_act_fn

        self.X = tf.placeholder(tf.float64, [None, self.num_input])
        self.Y = tf.placeholder(tf.float64, [None, self.num_classes])

        # Store layers weight & bias
        self.weights = {
            'h1': tf.Variable(np.empty([num_input, n_hidden_1]), name="h1", dtype=tf.float64, trainable=False),
            'b1': tf.Variable(np.empty([n_hidden_1]), name="b1", dtype=tf.float64, trainable=False),
            'out': tf.Variable(np.empty([n_hidden_1, num_classes]), name="out", dtype=tf.float64, trainable=False),
            'b3': tf.Variable(np.empty([num_classes]), name="b3", dtype=tf.float64, trainable=False)
        }

        self.weight_placeholders = {
            'h1': tf.placeholder(name="h1_ph", dtype=tf.float64, shape=self.weights['h1'].shape),
            'b1': tf.placeholder(name="b1_ph", dtype=tf.float64, shape=self.weights['b1'].shape),
            'out': tf.placeholder(name="out_ph", dtype=tf.float64, shape=self.weights['out'].shape),
            'b3': tf.placeholder(name="b3_ph", dtype=tf.float64, shape=self.weights['b3'].shape)
        }

        # weights for the current walk:
        self.current_weights = {
            'h1': np.empty(self.weights['h1'].shape),
            'out': np.empty(self.weights['out'].shape),
            'b1': np.empty(self.weights['b1'].shape),
            'b3': np.empty(self.weights['b3'].shape)
        }

        ### Define a list of operations to ASSIGN updated values to the TF variables
        self.weight_upd_ops = []
        for k in sorted(self.weights):
            self.weight_upd_ops.append(tf.assign(self.weights[k], self.weight_placeholders[k]))
        self.build_model()

    def assign_upd_weights(self, all_weights):
        i = 0
        j = 0
        for k in sorted(self.current_weights):
            length = self.current_weights[k].size
            shape = self.current_weights[k].shape
            self.current_weights[k] = all_weights[i:i + length].reshape(shape)
            ### ASSIGN updated values to the TF variables
            self.sess.run(self.weight_upd_ops[j], feed_dict={self.weight_placeholders[k]: self.current_weights[k]})
            j = j + 1
            i += length

    # Create model
    def neural_net(self, x):
        # Hidden fully connected layer with 256 neurons
        layer_1 = self.act_fn(tf.nn.xw_plus_b(x, self.weights['h1'], self.weights['b1']))
        # Output fully connected layer with a neuron for each class
        out_layer = tf.nn.xw_plus_b(layer_1, self.weights['out'], self.weights['b3'])
        return out_layer

    def build_model(self):
        self.logits = self.neural_net(self.X)
        self.prediction = self.out_act_fn(self.logits)
        self.ce_op = self.cross_entropy()
        self.mse_op = self.mse()
        self.acc_op = self.accuracy()

    def get_hidden_act(self):
        if self.act_fn == tf.nn.sigmoid:
            return "sigmoid"
        elif self.act_fn == tf.nn.tanh:
            return "tanh"
        elif self.act_fn == tf.nn.relu:
            return "relu"
        return "unknown"

    def cross_entropy(self):
        ce_op = None
        if self.out_act_fn == tf.nn.sigmoid:
            ce_op = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.Y))
        elif self.out_act_fn == tf.nn.softmax:
            ce_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y))
        return ce_op

    def mse(self):
        mse_op = tf.reduce_mean(tf.square(self.prediction - self.Y))
        return mse_op

    def accuracy(self):
        if self.num_classes == 1:
            correct_pred = tf.equal(tf.floor(self.prediction + 0.5), self.Y)
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            return accuracy
        else:
            correct_pred = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self.Y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            return accuracy

    def one_sim(self, sess, num_walks, num_steps, bounds, step_size, walk_type, data_generator, print_to_screen=False):
        self.sess = sess
        all_walks = np.empty((num_walks, num_steps, 3))
        for walk_counter in range(0, num_walks):
            error_history_py = np.empty((num_steps, 3))  # dimensions: x -> steps, y -> error metrics

            all_weights = np.concatenate([v.flatten() for k, v in sorted(self.current_weights.items())])
            start = rs.progressive_mask_tf(all_weights.shape)
            all_weights = rs.init_progressive_mask(start, bounds)
            self.assign_upd_weights(all_weights)

            for step in range(0, num_steps):
                batch_x, batch_y = data_generator() ########## Provide correct generator from outside
                # Calculate batch loss and accuracy
                ce, mse, acc = sess.run([self.ce_op, self.mse_op, self.acc_op], feed_dict={self.X: batch_x, self.Y: batch_y})
                if step % (num_steps/10) == 0 and print_to_screen is True:
                    print("Step " + str(step) + ", Cross-entropy Loss = " + \
                          "{:.4f}".format(ce) + ", MSE Loss = " + \
                          "{:.4f}".format(mse) + ", Training Accuracy = " + \
                          "{:.3f}".format(acc))
                if walk_type == "progressive":
                    all_weights, start = rs.progressive_random_step_tf(all_weights, start, step_size, bounds)
                elif walk_type == "manhattan":
                    all_weights, start = rs.progressive_manhattan_random_step_tf(all_weights, start, step_size, bounds)
                self.assign_upd_weights(all_weights)
                error_history_py[step] = [ce, mse, acc]
            if print_to_screen is True: print("Done with walk number ", walk_counter+1)
            all_walks[walk_counter] = error_history_py

        if print_to_screen is True:
            print("All random walks are done now.")
            print("Dimensionality is: ", all_weights.shape[0])

        return all_walks, all_weights.shape[0]
