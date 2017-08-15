import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import var_name
from preprocess import base


# ------------------------- ZH_HW
def plot_box_ZX_HW(data):
    plt.figure(figsize=(13, 2))
    name = var_name.name_HW1 + var_name.name_HW2
    plt.boxplot(data[name].as_matrix())
    plt.xticks(np.arange(1, 1 + len(name)), name)


def plot_stats_ZX_HW(data, func, title):
    plt.figure(figsize=(13, 2))
    name = var_name.name_HW1 + var_name.name_HW2
    stds = data[name].apply(func)
    plt.plot(stds.values, 'o-')
    plt.xticks(np.arange(stds.count()), stds.index)
    plt.title(title)


def plot_ZX_HW(data):
    for i in np.arange(1, 3):
        plt.figure(figsize=(13, 2))
        plt.title("ZX_HW{}".format(i))
        for j in np.arange(1, 7):
            name = "ZX_HW{0}_{1}".format(i, j)
            plt.plot(data[name], label=name)
        plt.legend(loc='best', prop={'size': 5})
    plot_box_ZX_HW(data)
    plot_stats_ZX_HW(data, np.std, "std")
    plot_stats_ZX_HW(data, np.mean, "mean")


def plot_box_ZX_HW_batch(path):
    filelist = base.get_files_csv(path)
    name = var_name.name_HW1 + var_name.name_HW2
    for file in filelist:
        plt.figure(figsize=(13, 2))
        file_path = path + file
        data = pd.read_csv(file_path)
        plt.boxplot(data[name].as_matrix())
        plt.xticks(np.arange(1, 1+len(name)), name)
        plt.title(file)


def plot_stats_ZX_HW_batch(path, func):
    plt.figure(figsize=(15, 5))
    name = var_name.name_HW1 + var_name.name_HW2
    filelist = base.get_files_csv(path)
    for file in filelist:
        file_path = path + file
        data = pd.read_csv(file_path)
        stds = data[name].apply(func)
        plt.plot(stds.values, 'o-', label=file)
        plt.xticks(np.arange(stds.count()), stds.index)

    plt.legend(loc='best', prop={'size': 8})
    plt.title("std of ZH_HW")


############################### ZX_WD_No1: ZX_WD_1_*


def plot_stats_ZX_WD_No1(data):
    data_std = pd.DataFrame()
    data_mean = pd.DataFrame()
    _, axs = plt.subplots(6, 1, figsize=(13, 10))
    for i in np.arange(1, 7):
        names = list()
        title = "ZX_WD_{}".format(i)
        for j in np.arange(1, 7):
            name = "ZX_WD_{0}_{1}".format(i, j)
            names.insert(len(names), name)
        axs[i-1].set_ylabel(title)
        data[names].boxplot(return_type='dict', ax=axs[i-1])

        stds = data[names].apply(np.std)
        means = data[names].apply(np.mean)
        data_std[title] = stds.values
        data_mean[title] = means.values

    axstd = data_std.plot(figsize=(13, 2), title="std")
    axstd.legend(loc='best', prop={'size': 8})

    axmean = data_mean.plot(figsize=(13, 2), title="mean")
    axmean.legend(loc='best', prop={'size': 8})


def plot_ZX_WD_No1(data):
    for i in np.arange(1, 7):
        plt.figure(figsize=(10, 2))
        plt.title("ZX_WD_{}".format(i))
        for j in np.arange(1, 7):
            name = "ZX_WD_{0}_{1}".format(i, j)
            plt.plot(data[name], label=name)
        plt.legend(loc='best', prop={'size': 5})

#################################### ZX_WD_No2: ZX_WD_*_1

def plot_stats_ZX_WD_No2(data):
    data_std = pd.DataFrame()
    data_mean = pd.DataFrame()
    _, axs = plt.subplots(6, 1, figsize=(13, 10))
    for i in np.arange(1, 7):
        names = list()
        title = "ZX_WD_*_{}".format(i)
        for j in np.arange(1, 7):
            name = "ZX_WD_{0}_{1}".format(j, i)
            names.insert(len(names), name)
        axs[i - 1].set_ylabel(title)
        data[names].boxplot(return_type='dict', ax=axs[i - 1])

        stds = data[names].apply(np.std)
        means = data[names].apply(np.mean)
        data_std[title] = stds.values
        data_mean[title] = means.values

    axstd = data_std.plot(figsize=(13, 2), title="std")
    axstd.legend(loc='best', prop={'size': 8})

    axmean = data_mean.plot(figsize=(13, 2), title="mean")
    axmean.legend(loc='best', prop={'size': 8})


def plot_ZX_WD_No2(data):
    for i in np.arange(1, 7):
        plt.figure(figsize=(10, 2))
        plt.title("ZX_WD_*_{}".format(i))
        for j in np.arange(1, 7):
            name = "ZX_WD_{0}_{1}".format(j, i)
            plt.plot(data[name])
        plt.legend(loc='best', prop={'size': 5})
