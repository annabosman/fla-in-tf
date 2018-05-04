import matplotlib as mp
mp.use('PS')
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
import pandas as pd

mp.rcParams['ps.useafm'] = True
mp.rcParams['pdf.use14corefonts'] = True
mp.rcParams['text.usetex'] = True
mp.rcParams['hatch.linewidth'] = 1.5
mp.rcParams['hatch.linewidth'] = 1.5
mp.rcParams.update({'font.size': 12})

data = [0.5,0.5,0.5,0.25,0.75,0.75]
other_data = [0.5,0.5,0.5,0.25,0.75,0.75,0.75,0.75,0.25]
data = np.array(data)
other_data = np.array(other_data)

macro_prog_u = [0., 0.25, 0.5, 0.75, 1.]
macro_prog_count = [7, 556, 8868, 560, 9]

macro_rand_u = [0., 0.25, 0.5, 0.75, 1.]
macro_rand_count = [4, 645, 8878, 470, 3]

macro_grad_u = [0., 0.25, 0.5, 0.75, 1.]
macro_grad_count = [0, 31, 2107, 5810, 2052]

micro_grad_u = [0., 0.25, 0.5, 0.75, 1.]
micro_grad_count = [4, 89, 2943, 56442, 40522]

micro_prog_u = [0., 0.25, 0.5, 0.75, 1.]
micro_prog_count = [113, 5854, 88144, 5761, 128]

micro_rand_u = [0., 0.25, 0.5,  0.75, 1.]
micro_rand_count = [31, 5942, 88810, 5217, 0]

micro_grad_unb_u = [0, 0.25, 0.5, 0.75, 1.]
micro_grad_unb_count = [0, 77, 3950, 56693, 39280]

macro_grad_unb_u = [0, 0.25, 0.5, 0.75, 1.]
macro_grad_unb_count = [0, 45, 1571, 5110, 3274]

values, counts = np.unique(data, return_counts=True)
print(values, counts)
values1, counts1 = np.unique(other_data, return_counts=True)
print(values1, counts1)

N = 5
rand_micro_means = np.array(micro_rand_count, dtype=float)
rand_micro_means /= sum(rand_micro_means)

prog_micro_means = np.array(micro_prog_count, dtype=float)
prog_micro_means /= sum(prog_micro_means)

grad_micro_means = np.array(micro_grad_count, dtype=float)
grad_micro_means /= sum(grad_micro_means)

grad_unb_micro_means = np.array(micro_grad_unb_count, dtype=float)
grad_unb_micro_means /= sum(grad_unb_micro_means)

rand_macro_means = np.array(macro_rand_count, dtype=float)
rand_macro_means /= sum(rand_macro_means)

prog_macro_means = np.array(macro_prog_count, dtype=float)
prog_macro_means /= sum(prog_macro_means)

grad_macro_means = np.array(macro_grad_count, dtype=float)
grad_macro_means /= sum(grad_macro_means)

grad_unb_macro_means = np.array(macro_grad_unb_count, dtype=float)
grad_unb_macro_means /= sum(grad_unb_macro_means)

ind = np.arange(N)  # the x locations for the groups
width = 0.2       # the width of the bars

bar_cycle = (cycler('hatch', ['///', '...', 'xxx', '\\\\']) + cycler('color', ['c', 'r', 'y', 'm']) *cycler('zorder', [10]))
styles = bar_cycle()

fig, ax = plt.subplots()
rects1 = ax.bar(ind - width - width/2, rand_micro_means, width, linewidth=1.5, edgecolor='black', **next(styles))
rects2 = ax.bar(ind - width/2, prog_micro_means, width, linewidth=1.5, edgecolor='black', **next(styles))
rects3 = ax.bar(ind + width/2, grad_micro_means, width, linewidth=1.5, edgecolor='black', **next(styles))
rects4 = ax.bar(ind + width + width/2, grad_unb_micro_means, width, linewidth=1.5, edgecolor='black', **next(styles))

# add some text for labels, title and axes ticks
# ax.set_ylabel('Proportion')
# ax.set_title('Scores by group and gender')
ax.set_xticks(ind)
ax.set_xticklabels(('0.0', '0.25', '0.5', '0.75', '1.0'))

ax.legend((rects1[0], rects2[0], rects3[0], rects4[0]), ('Random', 'Progressive', 'Gradient', 'Unbounded Gradient'))

plt.savefig('test.pdf', dpi=600)
plt.show()