from __future__ import print_function

import numpy as np
import tensorflow as tf
#import matplotlib.pyplot as plt  # Matplotlib is used to generate plots of data.


def random_walk(num_steps, step_size, shape):
    with tf.variable_scope("pos_var", reuse=tf.AUTO_REUSE):
        current_pos = tf.get_variable("pos", dtype=tf.float32, initializer=tf.random_uniform(shape, -bounds, bounds, dtype=tf.float32))
    init_state = (0, current_pos)
    condition = lambda i, _: i < num_steps
    body = lambda i, pos: (i + 1, random_step_tf(pos, step_size))
    loop = tf.while_loop(condition, body, init_state)
    return loop


def random_step_tf(weights, step_size):
    new_weight = tf.assign_add(weights, tf.random_uniform(weights.shape, -step_size, step_size))
    return new_weight


def progressive_random_step_tf(weights, mask, step_size, bounds):
    random_steps = tf.random_uniform(weights.shape, 0, step_size)
    mask_op = tf.assign(mask, bounds_check(weights, mask, random_steps * mask, bounds))
    with tf.control_dependencies([mask_op]):
        weights_op = tf.assign_add(weights, random_steps * mask)
    return weights_op


def progressive_manhattan_random_step_tf(weights, mask, step_size, bounds):
    random_steps = tf.zeros(weights.shape)
    shape = weights.shape
    random_steps = tf.reshape(random_steps, [-1])

    ind = tf.random_uniform([], 0, random_steps.shape[0], dtype=tf.int32)
    #ind = tf.Print(ind, [ind], message="ind: ")
    add_step = tf.one_hot(ind, on_value=step_size, depth=random_steps.shape[0])  # Manhattan: step size is constant
    #add_step = tf.Print(add_step, [add_step], message="add_step: ")
    random_steps = tf.reshape(add_step, shape)
    masked_step = random_steps * mask
    mask_op = tf.assign(mask, bounds_check(weights, mask, masked_step, bounds))
    with tf.control_dependencies([mask_op]):
        weights_op = tf.assign_add(weights, random_steps * mask)
    return weights_op


def progressive_mask_tf(scope, shape): # turn into a generator?
    start = tf.ones(shape) * -1
    mask_base = tf.random_uniform(shape, 0, 2, dtype=tf.int32)
    init_mask = start ** tf.cast(mask_base, tf.float32)
    with tf.variable_scope(scope):
        with tf.variable_scope("mask_var", reuse=False):
            mask = tf.get_variable("mask", dtype=tf.float32, initializer=init_mask)
    return mask


def reinit_progressive_mask_tf(mask):
    start = tf.ones(mask.shape) * -1
    mask_base = tf.random_uniform(mask.shape, 0, 2, dtype=tf.int32)
    init_mask = start ** tf.cast(mask_base, tf.float32)
    mask = tf.assign(mask, init_mask)
    return mask


def bounds_check(inputs, mask, step, bounds):
    new_inputs = inputs + step
    over_bounds = tf.abs(new_inputs) > bounds
    #over_bounds = tf.Print(over_bounds, [over_bounds], message="over_bounds: ")
    new_mask = mask * tf.cast((1 + (-2) * tf.cast(over_bounds, dtype=tf.int32)), dtype=tf.float32)
    return new_mask


def init_progressive_pos(scope, mask, bounds):
    random_nums = tf.random_uniform(mask.shape, 0, bounds, dtype=tf.float32)
    prog_nums = random_nums - bounds
    masked_nums = mask * prog_nums
    with tf.variable_scope(scope):
        with tf.variable_scope("pos_var", reuse=False):
            current_pos = tf.get_variable("pos", dtype=tf.float32, initializer=masked_nums)
    return current_pos  # return initialised random numbers


def reinit_progressive_pos(pos, mask, bounds):
    random_nums = tf.random_uniform(mask.shape, 0, bounds, dtype=tf.float32)
    prog_nums = random_nums - bounds
    masked_nums = mask * prog_nums
    pos = tf.assign(pos, masked_nums)
    return pos  # return initialised random numbers

if __name__ == '__main__':
    np.random.seed(123)
    tf.set_random_seed(123)
    current_pos = tf.zeros(2) # test
    my_step = tf.constant(0.3, dtype=tf.float32)
    my_bounds = tf.constant(1, dtype=tf.float32)

    mask = progressive_mask_tf("my_mask", current_pos.shape)         # Variable: mask
    current_pos = init_progressive_pos("my_pos", mask, my_bounds)    # Variable: pos
    # # We define a "shape-able" Variable
    # walk = tf.Variable( ##################### Replace with tensor array!
    #     [], # A list of scalars
    #     dtype=tf.float32,
    #     validate_shape=False, # By "shape-able", i mean we don't validate the shape so we can change it
    #     trainable=False
    # )
    # # Build the walk:
    # walk_concat = tf.concat([walk, current_pos], 0)
    # walk_assign_op = tf.assign(walk, walk_concat, validate_shape=False)  # We force TF to skip the shape validation step
    #step = tf.Variable(0, name='step', trainable=False, dtype=tf.int32)
    #increment_step = tf.assign(step, step+1)

    #walk = tf.TensorArray(dtype=tf.float32, size=7, dynamic_size=True)
    #with tf.control_dependencies([increment_step]):
    #    walk = walk.write(step, current_pos)

#    with tf.control_dependencies([walk_op]):
    random_step_op = random_step_tf(current_pos, my_step)
    prog_random_step_op = progressive_random_step_tf(current_pos, mask, my_step, my_bounds)
    manhattan_step_op = progressive_manhattan_random_step_tf(current_pos, mask, my_step, my_bounds)
    #my_walk = walk.stack()

    init = tf.global_variables_initializer()
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        tf.get_default_graph().finalize()
        sess.run(init)
        print("Initial mask: ", sess.run(mask))
        print("Initial point: ", sess.run(current_pos))

        sess.run(init)
        print("Re-initialised mask: ", sess.run(mask))
        print("Re-initialised point: ", sess.run(current_pos))
        print("Next step, random: ", sess.run(random_step_op))
        prog_s = sess.run(prog_random_step_op)
        print("Next step + mask, progressive random: ", prog_s, sess.run(mask))
        prog_s = sess.run(prog_random_step_op)
        print("Next step + mask, progressive random: ", prog_s, sess.run(mask))

        sess.run(init) # re-init
        walk = np.array([sess.run(current_pos)])
        for i in range(10):
            step = sess.run(manhattan_step_op)
            walk = np.append(walk, [sess.run(current_pos)], axis=0)
        print("Manhattan progressive walk: ", walk)

        # DRAW FIGURES (for debugging)
        #fig = plt.figure()

#        plt.scatter(walk[:, 0], walk[:, 1])
 #       plt.plot(walk[:, 0], walk[:, 1])

        # plt.axis('equal')
  #      plt.show()