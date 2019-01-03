from __future__ import print_function

import math
import numpy as np
# import matplotlib.pyplot as plt
from io import StringIO


def read_walks(file_name, num_s=0):
    if num_s is 0:
        tokens = str.split(file_name, "_")
        step_arr = str.split(tokens[len(tokens) - 1], ".")
        step_arr = str.split(step_arr[0], "s")
        #print(step_arr)
        num_steps = int(step_arr[1])
    else:
        num_steps = num_s

    data = np.loadtxt(fname=file_name, delimiter=",", skiprows=1)  # skip the header
    # Reshape the data into 3d array, using num_steps

    valid_size = data.shape[0] - (data.shape[0] % num_steps)
    data = data[:valid_size, :]

    data = data.reshape((-1, num_steps, data.shape[1]))
    return data


def read_walks_with_header(file_name, num_s=0):
    header =""
    if num_s is 0:
        tokens = str.split(file_name, "_")
        header += tokens[3] # micro/macro
        header = header + " " + tokens[6] # micro/macro

        step_arr = str.split(tokens[len(tokens) - 1], ".")
        step_arr = str.split(step_arr[0], "s")
        num_steps = int(step_arr[1])
    else:
        num_steps = num_s

    data = np.loadtxt(fname=file_name, delimiter=",", skiprows=1)  # skip the header
    # Reshape the data into 3d array, using num_steps

    valid_size = data.shape[0] - (data.shape[0] % num_steps)
    data = data[:valid_size, :]

    data = data.reshape((-1, num_steps, data.shape[1]))

    return data, header


def write_fla_metrics_to_file(filename, files):
    header = "setting,fem,m1,m2,gavg,gdev,last_t,best_t,last_g,best_g"
    with open(filename, "w") as f:
        f.write(header + "\n")
    for fname in files:
        print("Processing file ", fname, "...")
        arr, line = read_walks_with_header(fname)

        fem, _ = calc_fem(arr)  # discard stdev
        m1, _, m2, _ = calc_ms(arr)  # discard stdevs
        gavg = np.average(arr[:, :, arr.shape[2] - 1])
        gdev = np.std(arr[:, :, arr.shape[2] - 1])
        last_t = np.average(arr[:, arr.shape[1] - 1, 2])
        best_t = np.average(np.amax(arr[:, :, 2], axis=1))
        last_g = np.average(arr[:, arr.shape[1] - 1, 3])
        best_g = np.average(np.amax(arr[:, :, 3], axis=1))

        line = line + "," + str(fem[0]) + "," + str(m1[0]) + "," + str(m2[0])
        line = line + "," + str(gavg) + "," + str(gdev) + "," + str(last_t) + "," + str(best_t)
        line = line + "," + str(last_g) + "," + str(best_g)

        with open(filename, "a") as f:
            f.write(line + "\n")

# Exponentially weighted moving average
# Borrowed from: https://stackoverflow.com/questions/42869495/numpy-version-of-exponential-weighted-moving-average-equivalent-to-pandas-ewm
#
def ewma(data, window=-1):
    if window is -1:
        window = data.shape[0]

    alpha = 2 /(window + 1.0)
    alpha_rev = 1-alpha
    n = data.shape[0]

    pows = alpha_rev**(np.arange(n+1))
    if not np.all(pows[:-1]):
        print("alpha, window, n, pows: ", alpha, window, n, pows)
    scale_arr = 1/pows[:-1]
    offset = data[0]*pows[1:]
    pw0 = alpha*alpha_rev**(n-1)

    mult = data*pw0*scale_arr
    cumsums = mult.cumsum()
    out = offset + cumsums*scale_arr[::-1]
    return out


# This answer may seem irrelevant. But, for those who also need to calculate the exponentially weighted variance
# (and also standard deviation) with numpy, the following solution will be useful:
def ew(a, winSize):
    alpha = 2 /(winSize + 1.0)
    _alpha = 1 - alpha
    ws = _alpha ** np.arange(winSize)
    w_sum = ws.sum()
    ew_mean = np.convolve(a, ws)[winSize - 1] / w_sum
    bias = (w_sum ** 2) / ((w_sum ** 2) - (ws ** 2).sum())
    ew_var = (np.convolve((a - ew_mean) ** 2, ws)[winSize - 1] / w_sum) * bias
    ew_std = np.sqrt(ew_var)
    return ew_mean, ew_var, ew_std


def moving_average(a, n=4) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def smoothed_ma(a, windowSize=4) :
    # calculate the smoothed moving average
    weights = np.repeat(1.0, windowSize) / windowSize
    yMA = np.convolve(a[:], weights, 'valid')
    return yMA


def moving_stdev(a, W=4):
    nrows = a.size - W + 1
    n = a.strides[0]
    a2D = np.lib.stride_tricks.as_strided(a, shape=(nrows, W), strides=(n, n))
    return np.std(a2D, 1)


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
                if symbolic_s[q] == 1:
                    bin_count[0] += 1
                else:
                    bin_count[1] += 1
            if symbolic_s[p] == 1:
                if symbolic_s[q] == 0:
                    bin_count[2] += 1
                else:
                    bin_count[3] += 1
            if symbolic_s[p] == -1:
                if symbolic_s[q] == 0:
                    bin_count[4] += 1
                else:
                    bin_count[5] += 1
        p += 1
        q += 1

    entropy = 0.
    for i in range(0, 5):
        if bin_count[i] != 0:
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
    while not_found:
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
    while not_found:
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
    if diff == 0:
        return walk
    return (walk - min_f) / diff


def get_stationary(walk):
    """Try different window sizes; Return the values with the largest flatness length length."""
    walk = scale_walk(walk)
    archive_length = 0
    avg_length = 0
    win = 20
    window = 20
    while window > 4:
        #moving_exp_avg,_,moving_std_exp_avg = ew(walk, int(window))
        #print("moving_exp_avg, moving_std_exp_avg", moving_exp_avg,moving_std_exp_avg)
        moving_exp_avg = ewma(walk, int(window))
        #moving_std_exp_avg = pd.ewmstd(walk, span=int(window))
        moving_std_exp_avg = moving_stdev(moving_exp_avg, int(window))
        y = compute_stationary(moving_std_exp_avg, np.std(moving_std_exp_avg))
        #print("x, y, z, w: ", x, y, z, window)
        if not y:
            avg = 0
        else:
            avg = np.average(y)
        #print("window, avg: ", window, avg)
        if avg > avg_length:
            archive_length = len(y)
            avg_length = avg
            win = window
        window -= 2

    return archive_length, avg_length, win


def compute_stationary(walk, epsilon):
    """Calculate the number of times a given walk gets "stuck". Input: diff of a scaled random walk"""
    """The function assumes standard deviation sequence is sent through; difference from epsilon
    is calculated."""
    stuck = False
    length_stuck = 0
    archive = []

    if epsilon < 1e-8:
        return [len(walk)]

    for i in range(0, len(walk)):
        if stuck:
            if walk[i] > epsilon:
                stuck = False
                if length_stuck > 5:
                    archive.append(length_stuck)
                    length_stuck = 0
            else:
                length_stuck += 1
        else:
            if walk[i] < epsilon:
                stuck = True
                length_stuck += 1

    if length_stuck > 5:
        archive.append(length_stuck)
    return archive


def compute_m1(walk, epsilon):
    """Calculate the neutrality metric M1. Input: progressive random walk (will be scaled to [0,1])"""
    walk = scale_walk(walk)
    m1 = 0.
    len_3p_walk = len(walk) - 2
    for i in range(0, len_3p_walk):
        min_f = min(walk[i:i + 3])
        max_f = max(walk[i:i + 3])
        if max_f - min_f <= epsilon: m1 += 1.

    return m1 / len_3p_walk


def compute_m2(walk, epsilon):
    """Calculate the neutrality metric M2. Input: progressive random walk (will be scaled to [0,1])"""
    walk = scale_walk(walk)
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


def compute_grad(walk, n, step_size, bounds):
    # (1) calculate the fitness differences:
    err_diff = np.diff(walk)
    # (2) calculate the difference between max and min fitness:
    fit_diff = np.amax(walk) - np.amin(walk)
    # (3) calculate the total manhattan distance between the bounds of the search space
    manhattan_diff = n * 2 * bounds
    scaled_step = step_size / manhattan_diff
    # (4) calculate g(t) for t = 1..T
    g_t = np.zeros_like(err_diff)
    if fit_diff != 0:
        g_t = (err_diff / fit_diff) / scaled_step
    # (5) calculate G_avg
    return np.mean(np.absolute(g_t)), np.std(np.absolute(g_t))


# This method must be run after a simulation, to calculate the necessary metrics on the obtained walks
def calculate_metrics(all_walks, dim, step_size, bounds):
    # Work with ALL walks:
    # (1) Gradients [NB: requires a Manhattan walk!]:
    my_grad = calc_grad(all_walks, dim, step_size, bounds)
    print("Average Gavg and Gdev: ", my_grad)  # each column corresponds to outputs per walk
    # (2) Ruggedness [NB: requires a ]:
    my_rugg = calc_fem(all_walks)
    print("Average FEM: ", my_rugg)  # each column corresponds to outputs per walk
    #print("Max FEM: ", np.amax(my_rugg, 0))
    # (3) Neutrality:
    my_neut1, my_neut2 = calc_ms(all_walks)
    print("Average M1: ", my_neut1)  # each column corresponds to outputs per walk
    print("Average M2: ", my_neut2)  # each column corresponds to outputs per walk
    return my_grad, my_rugg, my_neut1, my_neut2


def calc_stationary(all_walks):
    # calc avg over multiple walks; assume single variable, i.e 2d array
    my_stat_x = []
    my_stat_y = []
    my_stat_w = []
    for i in range(all_walks.shape[0]):
        x, y, w = get_stationary(all_walks[i,:])
        my_stat_x.append(x)
        my_stat_y.append(y)
        my_stat_w.append(w)
    return my_stat_x, my_stat_y, my_stat_w


def calc_grad(all_walks, dim, step_size, bounds):
    # (1) Gradients [NB: requires a Manhattan walk!]:
    my_grad = np.apply_along_axis(compute_grad, 1, all_walks, dim, step_size, bounds)
    return np.average(my_grad, 0)


def calc_fem(all_walks):
    # (2) Ruggedness [NB: requires a progressive random walk]:
    my_rugg = np.apply_along_axis(compute_fem, 1, np.diff(all_walks, axis=1))
    return np.average(my_rugg, 0), np.std(my_rugg, 0)


def calc_ms(all_walks):
    # (3) Neutrality [NB: requires a progressive random walk which does not cross the search space more than once]:
    all_err_diff = np.diff(all_walks, axis=1)
    my_neut1 = np.apply_along_axis(compute_m1, 1, all_err_diff, 1.0e-8)
    my_neut2 = np.apply_along_axis(compute_m2, 1, all_err_diff, 1.0e-8)
    return np.average(my_neut1, 0), np.std(my_neut1, 0), np.average(my_neut2, 0), np.std(my_neut2, 0)


if __name__ == '__main__':
    my_walk = np.array([0.,1.,0.,1.,1.,1.,1.,0.,1.,0.,1]) # test
    print("M1: ", compute_m1(my_walk, 1e-8))
    print("M2: ", compute_m2(my_walk, 1e-8))
    print("FEM: ", compute_fem(np.diff(my_walk)))
    print("Grad: ", compute_grad(my_walk, 1, 0.1, 0.5))

    my_walk = scale_walk(np.array([0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.1,0.3,0.5,0.7,0.8,0.9,1.,1.1,1.,1.,1.2,1,1.3,1.,1.,1.,1.1,
                        1,1.,0.9,1.,1.1,1.,1,1.1,1.,0.9,1.,1.,1.,1.2,1.3,1.6,1.8,1.9,2.0,2.1,2.0,2.05,1.9,2.,2.1,1.9,
                        2.0,2.1,2.0,2.05,1.9,2.,2.1,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.1,0.3,0.5,0.7,0.8,0.9,1.,1.1,1.,
                        1.,1.2,1,1.3,1.,1.,1.,1.1,1,1.,0.9,1.,1.1,1.,1,1.1,1.,0.9,1.,1.,1.,1.2,1.3,1.6,1.8,1.9,2.0,2.1,
                        2.0,2.05,1.9,2.,2.1,1.9,2.0,2.1,2.0,2.05,1.9,2.,2.1])) # test


    my_walk = scale_walk(np.array([0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.1,0.3,0.5,0.7,0.8,0.9,1.,1.1,1.,1.,1.5,1,1.3,1.,1.,1.,1.5,1.5,1,1.3,1.,1.,1.,1.5,
                        1,1.,0.9,1.,1.1,1.,1,1.1,1.,0.9,1.,1.,1.,1.2,1.3,1.6,1.8,1.9,2.0,2.1,2.0,2.05,1.9,2.,2.1,1.9,2.1,2.0,2.05,1.9,2.,2.1,1.9,
                        2.0,2.1,2.0,2.05,1.9,2.,2.1,0.,0.,0.4,0.2,0.,0.3,0.,0.,0.,0.,0.1,0.3,0.5,0.7,0.8,0.9,1.,1.1,1.,1.,1.2,1,1.3,1.,1.,1.,1.1,1,
                        1.,1.2,1,1.3,1.,1.,1.,1.1,1,1.,0.9,1.,1.1,1.,1,1.1,1.,0.9,1.,1.,1.,1.2,1.3,1.6,1.8,1.9,2.0,2.1,1.2,1.3,1.6,1.8,1.9,2.0,2.1,
                        2.0,2.05,1.9,2.,2.1,1.9,2.0,2.1,2.0,2.05,1.9,2.,2.1,1.9,2.0,2.1,2.0,2.05,1.9,2.,2.1])) # test
    np.random.seed(0)
    my_walk = np.append(np.ones(33)*3, [np.ones(33)*2, np.ones(33)])
    my_walk = my_walk + np.random.normal(0, 0.15, 99)
    print("Walk length: ", len(my_walk))
    window = 6  # Test different window values; check the lengths; max length is the one to use
    # Use exponential avg
    moving_avg = smoothed_ma(my_walk, window)
    moving_exp_avg = ewma(my_walk, window)
    moving_std = moving_stdev(my_walk, window)
    moving_std_avg = moving_stdev(moving_avg, window)
    moving_std_exp_avg = moving_stdev(moving_exp_avg, window)
    y = compute_stationary(moving_std_exp_avg, np.std(moving_std_exp_avg))
    std_graph = np.repeat(np.std(moving_std_exp_avg), len(my_walk))

    print("stuck, length: ", len(y), np.average(y))

    x, y, w = get_stationary(my_walk)
    print("AUTO stuck, length, window: ", x, y, w)
    #plt.show()
    arr = read_walks("data/output/xor/hessian/xor_hessian_ce_micro_sigmoid_sigmoid_b10_s1000.csv")
    #arr = read_walks("xor_hessian_mse_macro_sigmoid_sigmoid_b10_s100.csv")
    #arr = read_walks("xor_TEST_many_s1000.csv")
    #print(arr[:,:,0].shape)
    sub = arr[:,:,0] # for a single walk: [i,:]
    #print(sub[0,:])
    #x, y, w = calc_stationary(sub)
    print("Averages:", np.average(x), np.average(y), np.average(w))
    print("Std devs:", np.std(x), np.std(y), np.std(w))
    print("Min:", np.min(x), np.min(y), np.min(w))
    print("Max:", np.max(x), np.max(y), np.max(w))

    print("arxive", x)
    print("length", y)
    print("window", w)
    #my_walk = scale_walk(sub[79,:])
    #my_walk = sub[3, :]
    #print("walk:", my_walk)

    # window = 4  # Test different window values; check the lengths; max length is the one to use
    # # Use exponential avg
    moving_avg = smoothed_ma(my_walk, window)
    moving_exp_avg = ewma(my_walk, window)
    moving_std = moving_stdev(my_walk, window)
    moving_std_avg = moving_stdev(moving_avg, window)
    moving_std_exp_avg = moving_stdev(moving_exp_avg, window)#
    y = compute_stationary(moving_std_exp_avg, np.std(moving_std_exp_avg))
    std_graph = np.repeat(np.std(moving_std_exp_avg), len(my_walk))
    print("std of the walk: ", np.std(moving_std_exp_avg))
    if y : avrg = np.average(y)
    else: avrg = 0
    print("stuck, archive, avg: ", len(y), y, avrg)

    plt.plot(my_walk, label='The walk')
    plt.plot(std_graph, label="Std dev")
    plt.plot(moving_exp_avg, label='Exp moving avg')
    plt.plot(moving_std_exp_avg, label='Moving std dev')
    plt.legend(loc=0, borderaxespad=0.)
    #plt.savefig("walk_12.eps", bbox_inches='tight')
    plt.show()