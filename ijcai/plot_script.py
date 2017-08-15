import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import preprocess

#########################  ZH_HW
def plot_ZH_HW_box(data):
    plt.figure(figsize=(13, 2))
    name = ['ZX_HW1_1', 'ZX_HW1_2', 'ZX_HW1_3', 'ZX_HW1_4', 'ZX_HW1_5', 'ZX_HW1_6',
            'ZX_HW2_1', 'ZX_HW2_2', 'ZX_HW2_3', 'ZX_HW2_4', 'ZX_HW2_5', 'ZX_HW2_6']
    plt.boxplot(data[name].as_matrix())
    plt.xticks(np.arange(1, 1 + len(name)), name)


def plot_ZH_HW_stats(data, func, title):
    plt.figure(figsize=(13, 2))
    name = ['ZX_HW1_1', 'ZX_HW1_2', 'ZX_HW1_3', 'ZX_HW1_4', 'ZX_HW1_5', 'ZX_HW1_6',
            'ZX_HW2_1', 'ZX_HW2_2', 'ZX_HW2_3', 'ZX_HW2_4', 'ZX_HW2_5', 'ZX_HW2_6']
    stds = data[name].apply(func)
    plt.plot(stds.values, 'o-')
    plt.xticks(np.arange(stds.count()), stds.index)
    plt.title(title)


def plot_ZH_HW(data):
    for i in np.arange(1, 3):
        plt.figure(figsize=(13, 2))
        plt.title("ZX_HW{}".format(i))
        for j in np.arange(1, 7):
            name = "ZX_HW{0}_{1}".format(i, j)
            plt.plot(data[name], label=name)
        plt.legend(loc='best', prop={'size': 5})
    plot_ZH_HW_box(data)
    plot_ZH_HW_stats(data, np.std, "std")
    plot_ZH_HW_stats(data, np.mean, "mean")


def plot_ZH_HW_box_all(path):
    filelist = preprocess.get_filelist(path)
    name = ['ZX_HW1_1', 'ZX_HW1_2', 'ZX_HW1_3', 'ZX_HW1_4', 'ZX_HW1_5', 'ZX_HW1_6',
            'ZX_HW2_1', 'ZX_HW2_2', 'ZX_HW2_3', 'ZX_HW2_4', 'ZX_HW2_5', 'ZX_HW2_6']
    for file in filelist:
        plt.figure(figsize=(13, 2))
        file_path = path + file
        data = pd.read_csv(file_path)
        plt.boxplot(data[name].as_matrix())
        plt.xticks(np.arange(1, 1+len(name)), name)
        plt.title(file)


def plot_ZH_HW_stats_all(path, func):
    plt.figure(figsize=(15, 5))
    name = ['ZX_HW1_1', 'ZX_HW1_2', 'ZX_HW1_3', 'ZX_HW1_4', 'ZX_HW1_5', 'ZX_HW1_6',
            'ZX_HW2_1', 'ZX_HW2_2', 'ZX_HW2_3', 'ZX_HW2_4', 'ZX_HW2_5', 'ZX_HW2_6']
    filelist = preprocess.get_filelist(path)
    filelist = np.sort(filelist)
    for file in filelist:
        file_path = path + file
        data = pd.read_csv(file_path)
        stds = data[name].apply(func)
        plt.plot(stds.values, 'o-', label=file)
        plt.xticks(np.arange(stds.count()), stds.index)

    plt.legend(loc='best', prop={'size': 8})
    plt.title("std of ZH_HW")


############################### ZX_WD_No1: ZX_WD_1_*


def plot_ZX_WD_No1_stats(data):
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

def plot_ZX_WD_No2_stats(data):
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


def plot_other(data):
    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(421)
    ax.plot(data['ZD_FLAG'], 'o')
    ax.set_ylabel("ZD_FLAG")
    ax.set_ylim(-1, 2)

    ax = fig.add_subplot(422)
    ax.plot(data['ZD_ALT'], 'o')
    ax.set_ylabel("ZD_ALT")
    ax.set_ylim(-1, 2)

    ax = fig.add_subplot(423)
    ax.plot(data['ZD_CNT'], 'o')
    ax.set_ylabel("ZD_CNT")

    ax = fig.add_subplot(424)
    ax.plot(data['ZD_LCG'], 'o')
    ax.set_ylabel("ZD_LCG")

    ax = fig.add_subplot(425)
    ax.plot(data['ZD_TFG'], 'o')
    ax.set_ylabel("ZD_TFG")

    ax = fig.add_subplot(426)
    ax.plot(data['ZD_JHG'], 'o')
    ax.set_ylabel("ZD_JHG")

    ax = fig.add_subplot(427)
    ax.plot(data['ZD_LLJ'], 'o')
    ax.set_ylabel("ZD_LLJ")

    ax = fig.add_subplot(428)
    ax.plot(data['ZD_SPEED'], 'o')
    ax.set_ylabel("ZD_SPEED")
