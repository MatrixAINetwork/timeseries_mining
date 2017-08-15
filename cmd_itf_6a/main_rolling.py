import numpy as np
import pandas as pd
from preprocess import base
import statsmodels.tsa.stattools as tsm
from scipy import stats
import os


def cal_rolling_stats(file_in, file_out, func, win_rate):
    data = pd.read_csv(file_in)
    win = int(len(data)*win_rate)
    start = 0
    end = start + win - 1
    names = list(data.columns)
    data_stats = pd.DataFrame(columns=names)
    names.remove('BTSJ')

    i = 0
    while end < len(data):
        subset = data[start:end]
        subset = subset[names]
        values = subset.apply(func)
        data_stats.loc[i, 'BTSJ'] = data.loc[i, 'BTSJ']
        data_stats.loc[i, names] = values
        start += 1
        end += 1
        i += 1
    data_stats.to_csv(file_out, index=False)


def cal_rolling_path(path_in, path_out, func, win_rate):
    filist =base.get_files_csv(path_in)
    for file in filist:
        print("caculate for file {}".format(file))
        cal_rolling_stats(file_in=path_in + file,
                          file_out=path_out + file,
                          func=func,
                          win_rate=win_rate)


def acf_lag1(x):
    ret = tsm.acf(x)
    return ret[1]


root_dir = os.getcwd()
data_root_dir = root_dir + "/data_0192/"
data_in_dir = data_root_dir + "smooth_mean_interpolate_bin_mean/"
win_rate = 0.3
out_root = data_root_dir + "rolling_result/rolling_analysis_win{}/".format(win_rate)

print("calculating mean......")
if not os.path.exists(out_root+"mean/"):
    os.makedirs(out_root+"mean/")
cal_rolling_path(data_in_dir,
                 out_root + "mean/",
                 np.mean, win_rate)

print("calculating std......")
if not os.path.exists(out_root+"std/"):
    os.makedirs(out_root+"std/")
cal_rolling_path(data_in_dir,
                 out_root + "std/",
                 np.std, win_rate)

print("calculating skew......")
if not os.path.exists(out_root+"skew/"):
    os.makedirs(out_root+"skew/")
cal_rolling_path(data_in_dir,
                 out_root+"skew/",
                 stats.skew, win_rate)

print("calculating kurtosis......")
if not os.path.exists(out_root+"kurtosis/"):
    os.makedirs(out_root+"kurtosis/")
cal_rolling_path(data_in_dir,
                 out_root+"kurtosis/",
                 stats.kurtosis, win_rate)

print("calculating acf ......")
if not os.path.exists(out_root+"acf/"):
    os.makedirs(out_root+"acf/")
cal_rolling_path(data_in_dir,
                 out_root+"acf/",
                 acf_lag1, win_rate)



