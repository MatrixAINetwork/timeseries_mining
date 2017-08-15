from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ffx_learn import learn_ffx
from deap_learn import learn_deap


def plot_dataset(data, names):
    num_col = len(names)
    f, axis = plt.subplots(num_col, 1, sharex=True, figsize=(10, 8))
    for i, ax, name in zip(range(num_col), axis, names):
        y = data[:, i]
        ax.plot(y)
        ax.set_ylabel(name)
        ax.set_ylim(min(y)-np.std(y), max(y)+np.std(y))


def test_ffx_no_delay():
    results = learn_ffx.learn_nodelay(data, var_num_learn, varnames)
    learn_ffx.result_summary(results, varnames[0:var_num_learn])
    learn_ffx.compare_plot_nodelay([4, 7, 4, 5, 6, 5], data, results, varnames, var_num_learn)
    plt.show()


def test_ffx_delay():
    seq_timedelay = np.arange(16)
    results = learn_ffx.learn_delay(data, seq_timedelay, var_num_learn, varnames)
    learn_ffx.result_summary(results, varnames[0:var_num_learn])
    learn_ffx.compare_plot_delay([8, 4, 3, 4, 6, 6], data, seq_timedelay, results, varnames, var_num_learn)
    plt.show()


def test_deap_no_delay():
    results = learn_deap.learn_nodelay(data, var_num_learn, varnames)
    learn_deap.result_summary(results, varnames)
    learn_deap.compare_plot_nodelay([0, 0, 0, 0, 0, 0], data, results, varnames, var_num_learn)
    plt.show()


def test_deap_delay():
    seq_timedelay = np.arange(16)
    results = learn_deap.learn_delay(data, seq_timedelay, var_num_learn, varnames)
    learn_deap.result_summary(results, varnames)
    learn_deap.compare_plot_delay_deap([0, 0, 0, 0, 0, 0], data, seq_timedelay, results, varnames, var_num_learn)

data = pd.read_table("data/163_0151_2016-06-30_80_allAxisTemperature_merge2_1_reg_axl1.txt", sep=" ", header=None)
print("the information of data")
print(data.info())
print(data.head())

data = data.as_matrix()
varnames = np.array(['ZX_WD_1', 'ZX_WD_2', 'ZX_WD_3', 'ZX_WD_4', 'ZX_WD_5', 'ZX_WD_6',
                    'ZX_HW_1', 'ZX_HW_2',
                     'ZD_LCG', 'ZD_TFG', 'ZD_JHG', 'ZD_LLJ', 'ZD_SPEED'])
var_num = 13
var_num_learn = 6


## test ffx
#test_ffx_no_delay()
#test_ffx_delay()

## test deap
#test_deap_no_delay()
#test_deap_delay()

