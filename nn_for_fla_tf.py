""" Generic Neural Network model suitable for Fitness Landscape Analysis
Author: Anna Bosman
Based on the original code by: Aymeric Damien
Project: https://github.com/arakitianskaia/fla-in-numpy
"""

from __future__ import print_function

import tensorflow as tf
import numpy as np
import random_samplers_tf as rs


class Curvature:
    convex, concave, saddle, singular = range(1, 5)


def get_curvature(eigens_vector):
    """
    Calculates the "state" of the Hessian by analysing the vector of eigenvalues.

    Args:
        eigens_vector: A `Tensor` of rank 1 (vector) containing eigenvalues.

    Returns:
        curvature: Return the corresponding curvature from the Curvature class.
    """
    max_element = eigens_vector[tf.argmax(eigens_vector, axis=0)]
    min_element = eigens_vector[tf.argmin(eigens_vector, axis=0)]
    #max_element = tf.Print(max_element, [max_element], "Max element: ")
    #min_element = tf.Print(min_element, [min_element], "Min element: ")
    non_zero = tf.count_nonzero(eigens_vector)
    total = tf.size(input=eigens_vector, out_type=tf.int64)

    def ret_concave(): return tf.constant(Curvature.concave)
    def ret_convex(): return tf.constant(Curvature.convex)
    def ret_saddle(): return tf.constant(Curvature.saddle)
    def ret_singular(): return tf.constant(Curvature.singular)
    def ret_default(): return tf.constant(value=-1)

    comparison = tf.case([
        (tf.less(non_zero, total), ret_singular),
        (tf.logical_and(tf.logical_and(tf.greater(max_element, 0), tf.less(min_element, 0)), tf.equal(non_zero, total)),
            ret_saddle),
        (tf.greater(min_element, 0), ret_convex),
        (tf.less(max_element, 0), ret_concave)
    ], default=ret_default, exclusive=True)

    #ratio = tf.abs(min_element) / tf.abs(max_element)

    return comparison#, ratio


# Get dimensionality of a NN
def get_dimensionality(input, hidden, output):
    total = 0
    previous = input
    for i in hidden:
        total += (previous + 1) * i
        previous = i
    total += (previous + 1) * output
    return total


def get_random_mask(scope, shape):
    with tf.variable_scope(scope, reuse=False):
        mask = tf.get_variable("mask",
                                shape=shape,
                                dtype=tf.float32,
                                initializer=tf.ones_initializer())
    return mask


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


# Compute, but do not assign. Work with tensors, not variables.
def dense_linear_layer_symbolic(inputs, layer_weights, layer_bias):
    outputs = tf.nn.xw_plus_b(inputs, layer_weights, layer_bias)
    return outputs


# Compute, but do not assign. Work with tensors, not variables.
def neural_net_symbolic(x, all_weights, hidden_act_fn, out_act_fn):
    prev_layer = x
    for i in range(0, len(all_weights)):
        layer = tf.nn.xw_plus_b(prev_layer, all_weights[2*i], all_weights[2*i + 1])
        if i < len(all_weights) - 1:
            layer = hidden_act_fn(layer)  # Making it non-linear
        else:
            layer = out_act_fn(layer)
        prev_layer = layer

    # Output fully connected layer with a neuron for each class
    return prev_layer


def cross_entropy_symbolic(logits, y, act_fn):
    ce_op = None
    if act_fn == tf.nn.sigmoid:
        ce_op = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits, y))
    elif act_fn == tf.nn.softmax:
        ce_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, y))
    return ce_op


def mse_symbolic(pred, y):
    mse_op = tf.reduce_mean(tf.square(pred - y))
    return mse_op


def accuracy_symbolic(pred, y, num_classes):
    if num_classes == 1:
        correct_pred = tf.equal(tf.floor(pred + 0.5), y)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        return accuracy
    else:
        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        return accuracy


class FLANeuralNetwork(object):
    def __init__(self, input_tensor, output_tensor,
                 num_input=1, num_hidden=[], num_classes=1, act_fn=tf.nn.sigmoid, out_act_fn=tf.nn.sigmoid,
                 error_function="mse", compute_acuracy=True, compute_grad_norm=False, compute_eigens=False,
                 implicit_loop=False):
        self.num_input = num_input
        self.num_classes = num_classes
        self.num_hidden = num_hidden  # list of hidden weight layers

        self.act_fn = act_fn
        self.out_act_fn = out_act_fn

        self.X = input_tensor
        self.Y = output_tensor

        # Store layers weight & bias
        self.all_weights = []
        if not implicit_loop:
            self.build_model(error_function) ### Otherwise, build the model inside while_loop
        self.compute_grad_norm = compute_grad_norm
        self.compute_eigens = compute_eigens
        if compute_eigens:
            self.compute_grad_norm = True
            self.eigenvalues, self.curvature = self.calc_hess_eigens()
        self.compute_accuracy = compute_acuracy

    def build_model(self, error_function):
        self.logits = self.neural_net()
        self.prediction = self.out_act_fn(self.logits)

        self.acc_op = self.accuracy()

        self.error_descr = error_function
        if error_function is "mse":
            self.mse_op = self.mse()
            self.error = self.mse_op
        elif error_function is "ce":
            self.ce_op = self.cross_entropy()
            self.error = self.ce_op
        else:
            raise ValueError('Invalid error function provided. Use either "mse" or "ce" code.')


    # Create model
    def neural_net(self):
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

    def get_output_act(self):
        if self.out_act_fn == tf.nn.sigmoid:
            return "sigmoid"
        elif self.out_act_fn == tf.nn.softmax:
            return "softmax"
        return "unknown"

    def get_error_descr(self):
        return self.error_descr

    def cross_entropy(self):
        if self.out_act_fn == tf.nn.sigmoid:
            ce_op = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.Y))
        elif self.out_act_fn == tf.nn.softmax:
            ce_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y))
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
        self.mask_init_ops = []
        self.weight_upd_ops = []
        self.init = tf.global_variables_initializer()
        if self.walk_type == "random":
            for w in self.all_weights:
                mask = get_random_mask(w.name[:len(w.name)-2], w.shape)#rs.progressive_mask_tf(w.name[:len(w.name)-2], w.shape)
                self.mask_init_ops.append(rs.reinit_progressive_mask_tf(mask))
                self.weight_init_ops.append(rs.reinit_progressive_pos(w, mask, bounds))
                self.weight_upd_ops.append(rs.bounded_random_step_tf(w, step_size, bounds))
        if self.walk_type == "progressive":
            for w in self.all_weights:
                mask = get_random_mask(w.name[:len(w.name)-2], w.shape)#rs.progressive_mask_tf(w.name[:len(w.name)-2], w.shape)
                self.mask_init_ops.append(rs.reinit_progressive_mask_tf(mask))
                self.weight_init_ops.append(rs.reinit_progressive_pos(w, mask, bounds))
                self.weight_upd_ops.append(rs.progressive_random_step_tf(w, mask, step_size, bounds))
        elif self.walk_type == "manhattan":
            for w in self.all_weights:
                mask = rs.progressive_mask_tf(w.name[:len(w.name)-2], w.shape)
                # self.all_masks.append(mask)
                self.weight_upd_ops.append(rs.progressive_manhattan_random_step_tf(w, mask, step_size, bounds))
                self.weight_init_ops.append(rs.reinit_progressive_pos(w, mask, bounds))
        elif self.walk_type == "gradient":
            for w in self.all_weights:
                mask = get_random_mask(w.name[:len(w.name)-2], w.shape)#rs.progressive_mask_tf(w.name[:len(w.name)-2], w.shape)
                self.mask_init_ops.append(rs.reinit_progressive_mask_tf(mask))
                self.weight_init_ops.append(rs.reinit_progressive_pos(w, mask, bounds))

            self.gradients = tf.gradients(ys=self.error, xs=self.all_weights)
            for g, w in zip(self.gradients, self.all_weights):
                new_step = self.convert_to_random_step(w, g, step_size, bounds)
                self.weight_upd_ops.append(tf.assign_add(w, new_step))

        elif self.walk_type == "unbounded_gradient":
            for w in self.all_weights:
                mask = get_random_mask(w.name[:len(w.name)-2], w.shape)#rs.progressive_mask_tf(w.name[:len(w.name)-2], w.shape)
                self.mask_init_ops.append(rs.reinit_progressive_mask_tf(mask))
                self.weight_init_ops.append(rs.reinit_progressive_pos(w, mask, bounds))

            gradients = tf.gradients(ys=self.error, xs=self.all_weights)
            gradlist = []
            for g, w in zip(gradients, self.all_weights):
                new_step = self.convert_to_unbounded_random_step(w, g, step_size)
                self.weight_upd_ops.append(tf.assign_add(w, new_step))
                gradlist.append(tf.reshape(g, [-1]))
            grad_vector = tf.concat(gradlist, 0)
            # gradient size (this should not _always_ be computed!):
            if self.compute_grad_norm:
                self.grad_norm = tf.norm(grad_vector)


        elif self.walk_type == "gd":
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=step_size).minimize(self.error)
            self.weight_upd_ops.append(optimizer)

        elif self.walk_type == "adam":
            optimizer = tf.train.AdamOptimizer(learning_rate=step_size)
            self.weight_upd_ops.append(optimizer.minimize(self.error))

            #opt = tf.train.GradientDescentOptimizer(learning_rate=1)
            #gradients = opt.compute_gradients(self.mse(), self.all_weights)

            #randomised_grads = [(rs.bounds_check_no_mask(v, (g + tf.random_uniform(g.shape, 0.01, 0.1)), bounds), v)
            #                    for g, v in gradients]

            #randomised_grads = [(self.convert_to_random_step(v, g, step_size, bounds), v) for g, v in gradients]
            #self.weight_upd_ops.append(opt.apply_gradients(randomised_grads))

    # Calculate the hessian eigenvalues
    def calc_hess_eigens(self):
        eigens = []
        hessians = tf.hessians(ys=self.error, xs=self.all_weights)
        for h, w in zip(hessians, self.all_weights):
            eigens.append(tf.reshape(tf.self_adjoint_eigvals(tf.reshape(h, [tf.size(w), tf.size(w)])), [-1]))
        eigens_vector = tf.concat(eigens, 0)
        curvature = get_curvature(eigens_vector)
        return eigens_vector, curvature

    def convert_to_random_step(self, inputs, grad, step_size, bounds):
        is_positive = grad > 0
        new_mask = tf.cast((1 + (-2) * tf.cast(is_positive, dtype=tf.int32)), dtype=tf.float32)
        # Now, the gradient vector has been interpreted as "direction", and reversed (follow negative gradient!).
        step = rs.progressive_random_step_calc_tf(inputs, new_mask, step_size, bounds)
        return step

    def convert_to_unbounded_random_step(self, inputs, grad, step_size):
        #nonzero = tf.count_nonzero(grad)
        #nonzero = tf.Print(nonzero, [nonzero], message="non-zero count: ")

        #grad = self.grad_debug(grad)

        is_positive = grad > 0
        new_mask = tf.cast((1 + (-2) * tf.cast(is_positive, dtype=tf.int32)), dtype=tf.float32)
        #nonzero = tf.count_nonzero(grad)
        #nonzero = self.grad_debug(nonzero)
        # Now, the gradient vector has been interpreted as "direction", and reversed (follow negative gradient!).
        #with tf.control_dependencies([nonzero]):
        step = rs.unbounded_progressive_random_step_calc_tf(inputs, new_mask, step_size)
        return step

    def grad_debug(self, g):
        g = tf.Print(g,[g],'G: ')
        return g

    def body(self, i, out_array):
        self.logits = self.neural_net()
        self.prediction = self.out_act_fn(self.logits)
        ce_op = self.cross_entropy()
        mse_op = self.mse()
        acc_op = self.accuracy()
        with tf.control_dependencies([ce_op, mse_op, acc_op]):
            out_array = out_array.write(i, tf.stack([ce_op, mse_op, acc_op], 0))
        self.build_random_walk_graph(self.walk_type, self.step_size, self.bounds)
        if self.walk_type == "manhattan":
            with tf.control_dependencies(self.weight_upd_ops):
                return i + 1, out_array
        else:
            rand_ind = tf.random_uniform(shape=[], minval=0, maxval=len(self.weight_upd_ops), dtype=tf.int32)
            chosen_upd = tf.gather(self.weight_upd_ops, rand_ind)
            with tf.control_dependencies([chosen_upd]):
                return i + 1, out_array

    def build_random_walk_loop(self, walk_type, step_size, bounds, num_steps):
        self.walk_type = walk_type
        self.step_size = step_size
        self.num_steps = num_steps
        self.bounds = bounds
        output_array = tf.TensorArray(dtype=tf.float32, size=num_steps)

        def cond(i, out_array):
            return tf.less(i, num_steps)

        _, self.walk_op = tf.while_loop(cond, self.body, [tf.constant(0), output_array])
        self.run_random_walk_loop_op = self.run_random_walk_loop()

    def run_random_walk_loop(self):
        walk_tensor = self.walk_op.stack()
        return walk_tensor

    def one_sim_tf(self, sess, num_walks, print_to_screen=False):
        all_walks = np.empty((num_walks, num_steps, 3))
        for walk_counter in range(0, num_walks):
            sess.run(self.init) # initialise all weights/masks
            one_walk = sess.run(self.run_random_walk_loop_op)
            if print_to_screen is True:
                print("Done with walk number ", walk_counter+1)
            all_walks[walk_counter] = one_walk

        return all_walks


    # def build_model(self):
    #     self.logits = self.neural_net(self.X)
    #     self.prediction = self.out_act_fn(self.logits)
    #     self.ce_op = self.cross_entropy()
    #     self.mse_op = self.mse()
    #     self.acc_op = self.accuracy()

    def one_sim_pos(self, sess, num_walks, num_steps, print_to_screen=False):
        all_walks = np.empty((num_walks, num_steps, 3))
        all_walks_pos = np.empty((num_walks, num_steps, self.all_weights[0].shape[0] * self.all_weights[0].shape[1]
                        + self.all_weights[1].shape[0] + self.all_weights[2].shape[0] * self.all_weights[2].shape[1]
                        + self.all_weights[3].shape[0]))

        for walk_counter in range(0, num_walks):
            error_history_py = np.empty((num_steps, 3))  # dimensions: x -> steps, y -> error metrics

            # FOR PLOTTING
            position_history_py = np.empty((num_steps, self.all_weights[0].shape[0] * self.all_weights[0].shape[1]
                  + self.all_weights[1].shape[0] + self.all_weights[2].shape[0] * self.all_weights[2].shape[1]
                  + self.all_weights[3].shape[0]))

            sess.run(self.init) # initialise all weights/masks
            sess.run(self.mask_init_ops)
            sess.run(self.weight_init_ops)
            for step in range(0, num_steps):
                # Calculate batch loss and accuracy
                ce, mse, acc = sess.run([self.ce_op, self.mse_op, self.acc_op])
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
                elif self.walk_type == "gradient" or self.walk_type == "unbounded_gradient":
                    sess.run(self.weight_upd_ops)

                weight_matrix_1 = sess.run(self.all_weights[0])
                weight_matrix_2 = sess.run(self.all_weights[1])
                weight_matrix_3 = sess.run(self.all_weights[2])
                weight_matrix_4 = sess.run(self.all_weights[3])
                plot_weights = np.concatenate((weight_matrix_1.ravel(), weight_matrix_2, weight_matrix_3.ravel(), weight_matrix_4))
                #print(plot_weights)
                error_history_py[step] = [ce, mse, acc]
                position_history_py[step] = plot_weights
            if print_to_screen is True:
                print("Done with walk number ", walk_counter+1)
            all_walks[walk_counter] = error_history_py
            all_walks_pos[walk_counter] = position_history_py

        if print_to_screen is True:
            print("All random walks are done now.")

        return all_walks, all_walks_pos

    def one_sim(self, sess, num_walks, num_steps, print_to_screen=False):
        # 3rd dimension: number of variables calculated/returned per step.
        # In case of saddle metrics, need 3: fitness (always put fitness first!), curvature, gradient magnitude
        # In case of other metrics, all you need is the fitness. 1 parameter.
        # For debugging and other purpose, will be nice to also record classification error.
        num_vars = 1
        if self.compute_grad_norm: num_vars += 1
        if self.compute_eigens: num_vars += 1
        if self.compute_accuracy: num_vars += 1

        header = self.get_header()

        all_walks = np.empty((num_walks, num_steps, num_vars))

        for walk_counter in range(0, num_walks):
            error_history_py = np.empty((num_steps, num_vars))  # dimensions: x -> steps, y -> error metrics

            sess.run(self.init) # initialise all weights/masks
            sess.run(self.mask_init_ops)
            sess.run(self.weight_init_ops)
            for step in range(0, num_steps):
                single_step = []
                # Calculate batch loss and accuracy
                if self.compute_eigens and self.compute_accuracy:
                    error, acc, eigenvalues, curvature, grad_norm = \
                        sess.run([self.error, self.acc_op, self.eigenvalues, self.curvature,
                                  self.grad_norm])
                elif self.compute_eigens:
                    error, eigenvalues, curvature, grad_norm = \
                        sess.run([self.error, self.eigenvalues, self.curvature,
                                  self.grad_norm])
                elif self.compute_accuracy and self.compute_grad_norm:
                    error, acc, grad_norm = sess.run([self.error, self.acc_op, self.grad_norm])
                elif self.compute_grad_norm:
                    error, grad_norm = sess.run([self.error, self.grad_norm])
                elif self.compute_accuracy:
                    error, acc = sess.run([self.error, self.acc_op])
                else:
                    error = sess.run([self.error])

                single_step.append(error)
                if self.compute_accuracy:
                    single_step.append(acc)
                if self.compute_eigens:
                    single_step.append(curvature)
                if self.compute_grad_norm:
                    single_step.append(grad_norm)

                if step % (num_steps/10) == 0 and print_to_screen is True:
                    print("Step " + str(step) + ", Error = " +
                          "{:.4f}".format(error))
                    if self.compute_accuracy:
                        print("Accuracy: " + str(acc))
                    if self.compute_eigens:
                        print("Eigenvalues: " + str(eigenvalues))
                        print("Curvature: " + str(curvature))
                    if self.compute_grad_norm:
                        print("Gradient norm: " + str(grad_norm))
                if self.walk_type == "random" or self.walk_type == "progressive":
                    sess.run(self.weight_upd_ops)
                elif self.walk_type == "manhattan":
                    i = np.random.randint(0, len(weight_upd_ops), 1)
                    sess.run(self.weight_upd_ops[i])
                elif self.walk_type == "gradient" or self.walk_type == "unbounded_gradient"\
                        or self.walk_type == "adam_ce" or self.walk_type == "gd_ce":
                    sess.run(self.weight_upd_ops)

                error_history_py[step] = single_step #[ce, mse, acc]
            print("Done with walk number ", walk_counter + 1)
            all_walks[walk_counter] = error_history_py

        if print_to_screen is True:
            print("All random walks are done now.")

        return all_walks, header

    def one_walk(self, sess, num_steps, print_to_screen=False):
        # 3rd dimension: number of variables calculated/returned per step.
        # In case of saddle metrics, need 3: fitness (always put fitness first!), curvature, gradient magnitude
        # In case of other metrics, all you need is the fitness. 1 parameter.
        # For debugging and other purpose, will be nice to also record classification error.
        num_vars = 1
        if self.compute_grad_norm: num_vars += 1
        if self.compute_eigens: num_vars += 1
        if self.compute_accuracy: num_vars += 1

        error_history_py = np.empty((num_steps, num_vars))  # dimensions: x -> steps, y -> error metrics

        sess.run(self.init) # initialise all weights/masks
        sess.run(self.mask_init_ops)
        sess.run(self.weight_init_ops)
        for step in range(0, num_steps):
            single_step = []
            # Calculate batch loss and accuracy
            if self.compute_eigens and self.compute_accuracy:
                error, acc, eigenvalues, curvature, grad_norm = \
                    sess.run([self.error, self.acc_op, self.eigenvalues, self.curvature,
                              self.grad_norm])
            elif self.compute_eigens:
                error, eigenvalues, curvature, grad_norm = \
                    sess.run([self.error, self.eigenvalues, self.curvature,
                              self.grad_norm])
            elif self.compute_accuracy and self.compute_grad_norm:
                error, acc, grad_norm = sess.run([self.error, self.acc_op, self.grad_norm])
            elif self.compute_grad_norm:
                error, grad_norm = sess.run([self.error, self.grad_norm])
            elif self.compute_accuracy:
                error, acc = sess.run([self.error, self.acc_op])
            else:
                error = sess.run([self.error])

            single_step.append(error)
            if self.compute_accuracy:
                single_step.append(acc)
            if self.compute_eigens:
                single_step.append(curvature)
            if self.compute_grad_norm:
                single_step.append(grad_norm)

            if step % (num_steps/10) == 0 and print_to_screen is True:
                print("Step " + str(step) + ", Error = " +
                      "{:.4f}".format(error))
                if self.compute_accuracy:
                    print("Accuracy: " + str(acc))
                if self.compute_eigens:
                    print("Eigenvalues: " + str(eigenvalues))
                    print("Curvature: " + str(curvature))
                if self.compute_grad_norm:
                    print("Gradient norm: " + str(grad_norm))
            if self.walk_type == "random" or self.walk_type == "progressive":
                sess.run(self.weight_upd_ops)
            elif self.walk_type == "manhattan":
                i = np.random.randint(0, len(weight_upd_ops), 1)
                sess.run(self.weight_upd_ops[i])
            elif self.walk_type == "gradient" or self.walk_type == "unbounded_gradient"\
                    or self.walk_type == "adam_ce" or self.walk_type == "gd_ce":
                sess.run(self.weight_upd_ops)

            error_history_py[step] = single_step

        return error_history_py

    def get_header(self):
        header = []
        header.append("error")
        if self.compute_accuracy:
            header.append("accuracy")
        if self.compute_eigens:
            header.append("curvature")
        if self.compute_grad_norm:
            header.append("grad_norm")

        return header



if __name__ == '__main__':
    a = tf.placeholder(dtype=tf.float32, shape=[6])
    b = get_curvature(a)
    # Start training
    with tf.Session() as sess:
        # Run the initializer
        tf.get_default_graph().finalize()
        r1 = sess.run(b, feed_dict={a: [0,0,1,2,3,5]})
        r2 = sess.run(b, feed_dict={a: [2,6,1,2,3,5]})
        r3 = sess.run(b, feed_dict={a: [0,0,-1,2,3,5]})
        r4 = sess.run(b, feed_dict={a: [-1,-1,-1,-2,-3,-5]})
        r5 = sess.run(b, feed_dict={a: [1,-7,-1,2,3,5]})

        if(r1 == Curvature.singular):
            print("Correctly identified flat + positive")
            #print("Ratio is ", str(r1r))
        if(r2 == Curvature.convex):
            print("Correctly identified convexity")
            #print("Ratio is ", str(r2r))
        if(r3 == Curvature.singular):
            print("Correctly identified flat + positive + negative")
            #print("Ratio is ", str(r3r))
        if(r4 == Curvature.concave):
            print("Correctly identified concavity")
            #print("Ratio is ", str(r4r))
        if(r5 == Curvature.saddle):
            print("Correctly identified saddle")
            #print("Ratio is ", str(r5r))
