import numpy as np


def cal_nrmse(model, X, y):
    return np.sqrt(np.nanmean((model.simulate(X)-y)**2 / (max(y) - min(y))))


def cal_rmse(model, X, y):
    return np.sqrt(np.nanmean((model.simulate(X)-y)**2))


def get_name_zw():
    name_list = list()
    for i in np.arange(1, 7):
        for j in np.arange(1, 7):
            name_zw = "ZX_WD_{0}_{1}".format(i, j)
            name_list.insert(len(name_list), name_zw)
    return name_list


def get_name_hw():
    name_list = list()
    for i in np.arange(1, 7):
        for j in np.arange(1, 3):
            name_hw = "ZX_HW{0}_{1}".format(j, i)
            name_list.insert(len(name_list), name_hw)
    return name_list


def get_name_sdmc():
    name_list = list()
    for i in np.arange(1, 7):
        name = "ZX_SDMC_{}".format(i)
        name_list.append(name)
    return name_list
