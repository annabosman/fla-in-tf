from __future__ import print_function

import tensorflow as tf
import numpy as np
import fla_metrics as fla
from nn_for_fla_tf import FLANeuralNetwork


class MetricGenerator:
    """
    This class provides an easy-to-use interface to combine NNs with FLA metrics
    """
    def __init__(self, nn_model, walk_type, num_steps, num_walks, num_sims, bounds, macro=True, step_size=0, print_to_screen=False):
        self.walk_type = walk_type
        self.num_steps = num_steps
        self.num_walks = num_walks
        self.num_sims = num_sims
        self.bounds = bounds
        self.macro = macro
        if step_size == 0:
            if self.macro is True:
                self.step_size = (2 * self.bounds) * 0.1  # 10% of the search space
            else:
                self.step_size = (2 * self.bounds) * 0.01  # 1% of the search space

        self.nn_model = nn_model
        self.nn_model.build_random_walk_graph(self.walk_type, self.step_size, self.bounds)
        self.print_to_screen = print_to_screen

    # Return all walks performed for each simulation
    def do_the_sims(self, sess):
        all_sims = []
        for i in range(0, self.num_sims):
            all_w = self.nn_model.one_sim(sess, self.num_walks, self.num_steps, self.print_to_screen)
            all_sims.append(all_w)
        return all_sims

    # Return all walks performed for a single simulation
    def do_the_walks(self, sess, dict):
        all_w = self.nn_model.one_sim(sess, dict, self.num_walks, self.num_steps, self.print_to_screen)
        return all_w

    def write_walks_to_file_sequentially(self, filename_header, sess, dict):
        all_w, header = self.do_the_walks(sess, dict)
        all_w = np.reshape(all_w, [all_w.shape[0] * all_w.shape[1], all_w.shape[2]])

        filename = filename_header + "_" + self.nn_model.get_error_descr()
        if self.macro is True:
            filename += "_macro_"
        else:
            filename += "_micro_"
        filename = filename + self.nn_model.get_hidden_act()
        filename = filename + "_" + self.nn_model.get_output_act()
        filename = filename + "_b" + str(self.bounds)
        filename = filename + "_s" + str(self.num_steps) + ".csv"
        with open(filename, "w") as f:
            np.savetxt(f, [header], "%s", delimiter=",")
            np.savetxt(f, all_w, delimiter=",")

    def write_walks_to_file_sequentially_one_at_a_time(self, filename_header, sess, init_ops):

        header = self.nn_model.get_header()

        filename = filename_header + "_" + self.nn_model.get_error_descr()
        if self.macro is True:
            filename += "_macro_"
        else:
            filename += "_micro_"
        filename = filename + self.nn_model.get_hidden_act()
        filename = filename + "_" + self.nn_model.get_output_act()
        filename = filename + "_b" + str(self.bounds)
        filename = filename + "_s" + str(self.num_steps) + ".csv"
        with open(filename, "w") as f:
            np.savetxt(f, [header], "%s", delimiter=",")

        with open(filename, "a") as f:
            for walk_counter in range(0, self.num_walks):
                walk = self.nn_model.one_walk(sess, init_ops, self.num_steps, self.print_to_screen)
                print("Done with walk ", walk_counter)
                np.savetxt(f, walk, delimiter=",")

    def get_neutrality_and_ruggedness_metrics_only(self, filename_header, sess):
        m_list = np.empty((self.num_sims, 2, 3))
        fem_list = np.empty((self.num_sims, 3))
        for i in range(0, self.num_sims):
            all_w = self.nn_model.one_sim(sess, self.num_walks, self.num_steps, self.print_to_screen)

            m1, m2 = fla.calc_ms(all_w)
            if self.print_to_screen is True:
                print("Avg M1: ", m1)
                print("Avg M2: ", m2)
            m_list[i][0] = m1
            m_list[i][1] = m2

            fem = fla.calc_fem(all_w)
            if self.print_to_screen is True:
                print("Avg FEM: ", fem)
            fem_list[i] = fem

            print("----------------------- Sim ", i + 1, " is done -------------------------------")

        if self.print_to_screen is True:
            print("FEM across sims: ", fem_list)

        filename1 = filename_header + "_fem"
        if self.macro is True:
            filename1 = filename1 + "_macro_"
        else:
            filename1 = filename1 + "_micro_"
        filename1 = filename1 + self.nn_model.get_hidden_act()
        filename1 = filename1 + "_" + str(self.bounds) + ".csv"

        with open(filename1, "w") as f:
            np.savetxt(f, ["# (1) cross-entropy", "# (2) mse", "# (3) accuracy"], "%s")
            np.savetxt(f, fem_list, delimiter=",")

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

    def calculate_neutrality_metrics(self, filename_header, sess):
        m_list = np.empty((self.num_sims, 2, 3))

        for i in range(0, self.num_sims):
            all_w = self.nn_model.one_sim(sess, self.num_walks, self.num_steps, self.print_to_screen)

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

    def calculate_ruggedness_metrics(self, filename_header, sess):
        fem_list = np.empty((self.num_sims, 3))

        for i in range(0, self.num_sims):
            all_w = self.nn_model.one_sim(sess, self.num_walks, self.num_steps, self.print_to_screen)
            fem = fla.calc_fem(all_w)
            if self.print_to_screen is True:
                print("Avg FEM: ", fem)
            fem_list[i] = fem
            print("----------------------- Sim ", i + 1, " is done -------------------------------")

        if self.print_to_screen is True:
            print("FEM across sims: ", fem_list)

        filename1 = filename_header + "_fem"
        if self.macro is True:
            filename1 = filename1 + "_macro_"
        else:
            filename1 = filename1 + "_micro_"
        filename1 = filename1 + self.nn_model.get_hidden_act()
        filename1 = filename1 + "_" + str(self.bounds) + ".csv"

        with open(filename1, "a") as f:
            np.savetxt(f, ["# (1) cross-entropy", "# (2) mse", "# (3) accuracy"], "%s")
            np.savetxt(f, fem_list, delimiter=",")

    def calculate_gradient_metrics(self, filename_header, sess):
        if self.walk_type != "manhattan":
            raise ValueError('Gradient metric can only be obtained using a Manhattan progressive random walk')
        grad_list = np.empty((self.num_sims, 2, 3))

        for i in range(0, self.num_sims):
            all_w = self.nn_model.one_sim(sess, self.num_walks, self.num_steps, self.print_to_screen)
            g = fla.calc_grad(all_w, d, self.step_size, self.bounds)
            if self.print_to_screen is True:
                print("Avg Grad: ", g)
            grad_list[i] = g
            print("----------------------- Sim ", i + 1, " is done -------------------------------")

        if self.print_to_screen is True:
            print("Grad across sims: ", grad_list)

        filename1 = filename_header + "_gavg"
        if self.macro is True:
            filename1 = filename1 + "_macro_"
        else:
            filename1 = filename1 + "_micro_"
        filename1 = filename1 + self.nn_model.get_hidden_act()
        filename1 = filename1 + "_" + str(self.bounds) + ".csv"

        filename2 = filename_header + "_gdev"
        if self.macro is True:
            filename2 = filename2 + "_macro_"
        else:
            filename2 = filename2 + "_micro_"
        filename2 = filename2 + self.nn_model.get_hidden_act()
        filename2 = filename2 + "_" + str(self.bounds) + ".csv"

        with open(filename1, "a") as f:
            np.savetxt(f, ["# (1) cross-entropy", "# (2) mse", "# (3) accuracy"], "%s")
            np.savetxt(f, grad_list[:, 0, :], delimiter=",")

        with open(filename2, "a") as f:
            np.savetxt(f, ["# (1) cross-entropy", "# (2) mse", "# (3) accuracy"], "%s")
            np.savetxt(f, grad_list[:, 1, :], delimiter=",")
