from __future__ import print_function

import numpy as np
import tensorflow as tf


def random_step_tf(weights, step_size):
    new_weight = tf.add(weights, tf.random_uniform(weights.shape, -step_size, step_size))
    return new_weight


def progressive_random_step_tf(weights, mask, step_size, bounds):
    random_steps = tf.random_uniform(weights.shape, 0, step_size)
    mask = bounds_check(weights, mask, random_steps * mask, bounds)
    new_weight = tf.add(weights, random_steps * mask)
    return new_weight, mask


def progressive_manhattan_random_step_tf(weights, mask, step_size, bounds):
    random_steps = tf.zeros(weights.shape)
    shape = weights.shape
    random_steps = tf.reshape(random_steps, [-1])

    ind = tf.random_uniform([], 0, weights.size)
    random_steps[ind] = step_size  # Manhattan: step size is constant
    random_steps = tf.reshape(random_steps, shape)
    masked_step = random_steps * mask
    mask = bounds_check(weights, mask, masked_step, bounds)
    return tf.add(weights, random_steps * mask), mask


def progressive_mask_tf(shape): # turn into a generator?
    start = tf.ones(shape) * -1
    mask = tf.random_uniform(shape, 0, 2, dtype=tf.int32)
    start = start ** mask
    return start


def bounds_check(inputs, mask, step, bounds):
    conds = [np.absolute(inputs + step) > bounds, np.absolute(inputs + step) <= bounds]
    funcs = [lambda mask: -mask, lambda mask: mask]
    return np.piecewise(mask, conds, funcs)  # return mask


def init_progressive_mask(mask, bounds):
    random_nums = np.random.uniform(0, bounds, mask.shape)
    conds = [mask == 1, mask == -1]
    funcs = [lambda x: bounds - x, lambda x: x - bounds]
    return np.piecewise(random_nums, conds, funcs)  # return initialised random numbers


if __name__ == '__main__':
    my_init = np.empty(5) # test
    my_step = 0.1
    my_bounds = 1

    start = progressive_mask_tf(my_init.shape)
    print("Start mask: ", start)

    my_init = init_progressive_mask(start, my_bounds)
    print("Initial point: ", my_init)

    print("Next step, random: ", random_step_tf(my_init, my_step))
    print("Next step + mask, progressive random: ", progressive_random_step_tf(my_init, start, my_step, my_bounds))
    print("Next step + mask, progressive manhattan random: ", progressive_manhattan_random_step_tf(my_init, start, my_step, my_bounds))