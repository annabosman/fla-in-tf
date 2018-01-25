""" Generic Neural Network model suitable for Fitness Landscape Analysis
Author: Anna Bosman
Based on the original code by: Aymeric Damien
Project: https://github.com/arakitianskaia/fla-in-numpy
"""

from __future__ import print_function

import tensorflow as tf
import numpy as np
import random_samplers_tf as rs


def dense_linear_layer(inputs, layer_name, input_size, output_size):
    """
    Builds a layer that takes a batch of inputs of size `input_size` and returns
    a batch of outputs of size `output_size`.

    Args:
        inputs: A `Tensor` of shape [batch_size, input_size].
        layer_name: A string representing the name of the layer.
        input_size: The size of the inputs
        output_size: The size of the outputs

    Returns:
        out, weights, biases: layer outputs, weights and biases.

    """
    # Name scopes allow us to logically group together related variables.
    # Setting reuse=False avoids accidental reuse of variables between different runs.
    with tf.variable_scope(layer_name, reuse=False):
        # Create the weights for the layer
        layer_weights = tf.get_variable("weights",
                                        shape=[input_size, output_size],
                                        dtype=tf.float32,
                                        initializer=tf.random_uniform_initializer())
        # Create the biases for the layer
        layer_bias = tf.get_variable("biases",
                                     shape=[output_size],
                                     dtype=tf.float32,
                                     initializer=tf.random_uniform_initializer())

        outputs = tf.nn.xw_plus_b(inputs, layer_weights, layer_bias)

    return outputs, layer_weights, layer_bias


class FLANeuralNetwork(object):
    def __init__(self, num_input=1, num_hidden=[], num_classes=1, act_fn=tf.nn.sigmoid, out_act_fn=tf.nn.sigmoid):
        self.num_input = num_input
        self.num_classes = num_classes
        self.num_hidden = num_hidden  # list of hidden weight layers

        self.act_fn = act_fn
        self.out_act_fn = out_act_fn

        self.X = tf.placeholder(tf.float32, [None, self.num_input])
        self.Y = tf.placeholder(tf.float32, [None, self.num_classes])

        # Store layers weight & bias
        self.all_weights = []
        self.build_model()

    def build_model(self):
        self.logits = self.neural_net(self.X)
        self.prediction = self.out_act_fn(self.logits)
        self.ce_op = self.cross_entropy()
        self.mse_op = self.mse()
        self.acc_op = self.accuracy()

    # Create model
    def neural_net(self, x):
        prev_layer = self.X
        prev_size = self.num_input
        for layer_num, size in enumerate(self.num_hidden):
            layer_name = "layer_" + str(layer_num)
            layer, weights, biases = dense_linear_layer(prev_layer, layer_name, prev_size, size)

            self.all_weights.append(weights)
            self.all_weights.append(biases)
            layer = self.act_fn(layer) # Making it non-linear
            prev_layer, prev_size = layer, size

        # Output fully connected layer with a neuron for each class
        out_layer, weights, biases = dense_linear_layer(prev_layer, "output", prev_size, self.num_classes)
        self.all_weights.append(weights)
        self.all_weights.append(biases)
        return out_layer

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

    def build_random_walk_graph(self, walk_type, step_size, bounds):
        self.walk_type = walk_type
        self.weight_init_ops = []
        self.weight_upd_ops = []
        #self.weight_init_ops.append(tf.global_variables_initializer())
        if self.walk_type == "random":
            for w in self.all_weights:
                self.weight_upd_ops.append(rs.random_step_tf(w, step_size))
        if self.walk_type == "progressive":
            for w in self.all_weights:
                mask = rs.progressive_mask_tf(w.name[:len(w.name)-2], w.shape)
                # self.all_masks.append(mask)
                self.weight_upd_ops.append(rs.progressive_random_step_tf(w, mask, step_size, bounds))
                self.weight_init_ops.append(rs.reinit_progressive_pos(w, mask, bounds))
        elif self.walk_type == "manhattan":
            for w in self.all_weights:
                mask = rs.progressive_mask_tf(w.name[:len(w.name)-2], w.shape)
                # self.all_masks.append(mask)
                self.weight_upd_ops.append(rs.progressive_manhattan_random_step_tf(w, mask, step_size, bounds))
                self.weight_init_ops.append(rs.reinit_progressive_pos(w, mask, bounds))

    def one_sim(self, sess, num_walks, num_steps, data_generator, print_to_screen=False):
        all_walks = np.empty((num_walks, num_steps, 3))
        for walk_counter in range(0, num_walks):
            error_history_py = np.empty((num_steps, 3))  # dimensions: x -> steps, y -> error metrics

            sess.run(self.weight_init_ops) # initialise all weights/masks

            for step in range(0, num_steps):
                batch_x, batch_y = data_generator() ########## Provide correct generator from outside
                # Calculate batch loss and accuracy
                ce, mse, acc = sess.run([self.ce_op, self.mse_op, self.acc_op], feed_dict={self.X: batch_x, self.Y: batch_y})
                if step % (num_steps/10) == 0 and print_to_screen is True:
                    print("Step " + str(step) + ", Cross-entropy Loss = " + \
                          "{:.4f}".format(ce) + ", MSE Loss = " + \
                          "{:.4f}".format(mse) + ", Training Accuracy = " + \
                          "{:.3f}".format(acc))
                if self.walk_type == "random" or self.walk_type == "progressive":
                    sess.run(self.weight_upd_ops)
                elif self.walk_type == "manhattan":
                    i = np.random.randint(0, len(weight_upd_ops), 1)
                    sess.run(self.weight_upd_ops[i])
                error_history_py[step] = [ce, mse, acc]
            if print_to_screen is True:
                print("Done with walk number ", walk_counter+1)
            all_walks[walk_counter] = error_history_py

        if print_to_screen is True:
            print("All random walks are done now.")

        return all_walks
