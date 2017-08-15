from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

from preprocess import base
from ffx_learn import learn_ffx
from ffx_learn import call_ffx


def norm_2(x):
    return (x-np.mean(x))/(x.max()-x.min())
    #return (x-x.mean()) / np.sqrt(x*x)

root_dir = os.getcwd()
data_dir = root_dir + "/data_0134/integrated_temp_sdmc_mean/"
filelist = base.get_files_csv(data_dir)
print(filelist)

file = filelist[0]
data = pd.read_csv(data_dir+file)
del data['BTSJ']
print(data.columns)
#data = data.apply(norm_2, axis=1)
# ax = data.plot()
# ax.set_title(file)
# #ax.set_ylim([-1, 1])
# plt.show()

# varnames = data.columns
# X = data.as_matrix()
# y = np.zeros(X.shape[0])
# print(X.shape)
# print(y.shape)
# var_num_learn = 1
# results = list()
# result = call_ffx.run_ffx_main_half(X, y, varnames)
# results.insert(len(results), result)
# learn_ffx.result_summary(results, 'y')

y = data['hw_mean'].as_matrix()
del data['hw_mean']
varnames = data.columns
X = data.as_matrix()
print(X.shape)
print(y.shape)
var_num_learn = 1
results = list()
result = call_ffx.run_ffx_main_half(X, y, varnames)
results.insert(len(results), result)
learn_ffx.result_summary(results, 'y')

# from scipy.signal import savgol_filter
#
# df = data[['hw_mean', 'zw_mean']].copy()
# df['hw_mean'] = savgol_filter(df['hw_mean'], 101, 4)
# df['zw_mean'] = savgol_filter(df['zw_mean'], 101, 4)
# y = df['hw_mean'].diff()
# X = df.as_matrix()
# print(y.shape)
# print(X.shape)
# plt.plot(y)
# plt.show()
# varnames = df.columns
# results = list()
# result = call_ffx.run_ffx_main_half(X, y, varnames)
# results.insert(len(results), result)
# learn_ffx.result_summary(results, 'y')
