from __future__ import print_function

import numpy as np


def random_step_tf(weights, step_size):
    return np.add(weights, np.random.uniform(-step_size, step_size, weights.shape))


def progressive_random_step_tf(weights, mask, step_size, bounds):
    random_steps = np.random.uniform(0, step_size, weights.shape)
    mask = bounds_check(weights, mask, random_steps * mask, bounds)
    return np.add(weights, random_steps * mask), mask


def progressive_manhattan_random_step_tf(weights, mask, step_size, bounds):
    random_steps = np.zeros(weights.shape)
    shape = weights.shape
    random_steps = random_steps.flatten()
    ind = np.random.choice(weights.size, size=1)
    random_steps[ind] = step_size  # Manhattan: step size is constant
    random_steps = random_steps.reshape(shape)
    mask = bounds_check(weights, mask, random_steps * mask, bounds)
    return np.add(weights, random_steps * mask), mask


def progressive_mask_tf(shape): # turn into a generator?
    start = np.ones(shape) * -1
    mask = np.random.randint(0, 2, shape)
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
    np.random.seed(123)
    my_init = np.empty(5) # test
    my_step = 0.3
    my_bounds = 1

    start = progressive_mask_tf(my_init.shape)
    print("Initial mask: ", start)

    my_init = init_progressive_mask(start, my_bounds)
    print("Initial point: ", my_init)

    print("Next step, random: ", random_step_tf(my_init, my_step))
    print("Next step + mask, progressive random: ", progressive_random_step_tf(my_init, start, my_step, my_bounds))
    print("Next step + mask, progressive random: ", progressive_random_step_tf(my_init, start, my_step, my_bounds))
    print("Next step + mask, progressive manhattan random: ", progressive_manhattan_random_step_tf(my_init, start, my_step, my_bounds))
    print("Next step + mask, progressive manhattan random: ", progressive_manhattan_random_step_tf(my_init, start, my_step, my_bounds))
    print("Next step + mask, progressive manhattan random: ", progressive_manhattan_random_step_tf(my_init, start, my_step, my_bounds))