import pandas as pd
from common import *


def zw_original(path):
    data = pd.read_csv(path)
    name_list = get_name_zw()
    sub_data = data[name_list]
    return sub_data


def hw_original(path):
    data = pd.read_csv(path)
    name_list = get_name_hw()
    sub_data = data[name_list]
    return sub_data


def sdmc_original(path):
    data = pd.read_csv(path)
    name_list = get_name_sdmc()
    sub_data = data[name_list]
    return sub_data


def zw_sub_hw(path):
    data = pd.read_csv(path)
    temp_diff = pd.DataFrame()
    for i in np.arange(1, 7):
        name_hw = "ZX_HW{0}_{1}".format(2, i)
        for j in np.arange(1, 7):
            name_zw = "ZX_WD_{0}_{1}".format(i, j)
            temp_diff[name_zw] = data[name_zw] - data[name_hw]
    return temp_diff


def get_temp_mean(path):
    zws = zw_original(path)
    hws = hw_original(path)
    zw_mean = zws.mean(axis=1)
    hw_mean = hws.mean(axis=1)
    d = {'zw_mean': zw_mean,
         'hw_mean': hw_mean}
    df = pd.DataFrame(d)
    return df


def get_temp_sdmc_mean(path):
    zws = zw_original(path)
    hws = hw_original(path)
    sdmcs = sdmc_original(path)
    zw_mean = zws.mean(axis=1)
    hw_mean = hws.mean(axis=1)
    sdmc_mean = sdmcs.mean(axis=1)
    d = {'zw_mean': zw_mean,
         'hw_mean': hw_mean,
         'sdmc_mean': sdmc_mean}
    df = pd.DataFrame(d)
    return df


def get_piece_corr_segment(data, win):
    start = 0
    end = start + win
    corr_value = []
    while end < len(data):
        sub = data.loc[start:end,:]
        corr_value.append(abs(sub.corr()).sum().sum()/data.shape[1])
        end += win
        start += win
    return corr_value


def get_piece_corr_rolling(data, win):
    start = 0
    end = start + win
    corr_value = []
    while end<len(data):
        sub = data.loc[start:end,:]
        corr_value.append(abs(sub.corr()).sum().sum()/data.shape[1])
        end += 1
        start += 1
    return corr_value


def get_dist_p(X, Y, p):
    return np.power(X-Y, p).sum()**(1/p)


def get_piece_dist_segment(X, Y, win, p):
    if(len(X) != len(Y)):
        print("not equal length series!!!!!!")
        return
    start = 0
    end = start + win
    results = []
    while end<len(X):
        results.append(get_dist_p(X[start:end], Y[start:end], p))
        start += win
        end += win
    return results