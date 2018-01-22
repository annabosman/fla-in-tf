from __future__ import print_function

import tensorflow as tf
import numpy as np
import fla_metrics as fla
from nn_for_fla import FLANeuralNetwork


class MetricGenerator:
    """
    This class provides an easy-to-use interface to combine NNs with FLA metrics
    """
    def __init__(self, get_data, num_steps, bounds, macro, num_walks, num_sims, nn_model, print_to_screen=False):
        self.get_data = get_data
        self.num_steps = num_steps
        self.bounds = bounds
        self.macro = macro
        if self.macro is True:
            self.step_size = (2 * self.bounds) * 0.1  # 10% of the search space
        else:
            self.step_size = (2 * self.bounds) * 0.01  # 1% of the search space
        self.num_walks = num_walks
        self.num_sims = num_sims
        self.nn_model = nn_model
        self.print_to_screen = print_to_screen

    def calculate_neutrality_metrics(self, filename_header):
        init = tf.global_variables_initializer()
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            # Run the initializer
            tf.get_default_graph().finalize()
            sess.run(init)

            m_list = np.empty((self.num_sims, 2, 3))

            for i in range(0, self.num_sims):
                all_w, d = self.nn_model.one_sim(sess, self.num_walks, self.num_steps, self.bounds, self.step_size, "progressive", self.get_data, self.print_to_screen)
                m1, m2 = fla.calc_ms(all_w)
                if self.print_to_screen is True:
                    print("Avg M1: ", m1)
                    print("Avg M2: ", m2)
                m_list[i][0] = m1
                m_list[i][1] = m2
                print("----------------------- Sim ", i + 1, " is done -------------------------------")

            if self.print_to_screen is True:
                print("M1/M2 across sims: ", m_list)

            filename1 = filename_header + "_m1"
            if self.macro is True:
                filename1 = filename1 + "_macro_"
            else:
                filename1 = filename1 + "_micro_"
            filename1 = filename1 + self.nn_model.get_hidden_act()
            filename1 = filename1 + "_" + str(self.bounds) + ".csv"

            filename2 = filename_header + "_m2"
            if self.macro is True:
                filename2 = filename2 + "_macro_"
            else:
                filename2 = filename2 + "_micro_"
            filename2 = filename2 + self.nn_model.get_hidden_act()
            filename2 = filename2 + "_" + str(self.bounds) + ".csv"

            with open(filename1, "a") as f:
                np.savetxt(f, ["# (1) cross-entropy", "# (2) mse", "# (3) accuracy"], "%s")
                np.savetxt(f, m_list[:, 0, :], delimiter=",")

            with open(filename2, "a") as f:
                np.savetxt(f, ["# (1) cross-entropy", "# (2) mse", "# (3) accuracy"], "%s")
                np.savetxt(f, m_list[:, 1, :], delimiter=",")
