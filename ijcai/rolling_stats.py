import numpy as np
import pandas as pd
import preprocess
import statsmodels.tsa.stattools as tsm
from scipy import stats


def cal_rolling_stats(file_in, file_out, func, win):

    data = pd.read_csv(file_in)
    start = 0
    end = start+win-1
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


def cal_rolling_path(path_in, path_out, func, win):
    filist = preprocess.get_filelist(path_in)
    for file in filist:
        print("caculate for file {}".format(file))
        cal_rolling_stats(file_in=path_in+file,
                         file_out=path_out+file,
                         func=func,
                         win=win)


def acf_lag1(x):
    ret = tsm.acf(x)
    return ret[1]
    

win = 2500
cal_rolling_path("data/interpolation_10s_int/",
                 "result/rolling_analysis_win{}/mean/".format(win),
                 np.mean, win)
cal_rolling_path("data/interpolation_10s_int/",
                 "result/rolling_analysis_win{}/std/".format(win),
                 np.std, win)
cal_rolling_path("data/interpolation_10s_int/",
                 "result/rolling_analysis_win{}/skew/".format(win),
                 stats.skew, win)
cal_rolling_path("data/interpolation_10s_int/",
                 "result/rolling_analysis_win{}/kurtosis/".format(win),
                 stats.kurtosis, win)
cal_rolling_path("data/interpolation_10s_int/",
                 "result/rolling_analysis_win{}/acf/".format(win),
                 acf_lag1, win)



