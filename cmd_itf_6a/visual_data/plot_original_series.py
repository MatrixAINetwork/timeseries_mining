import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_data(data):
    x = len(data)

    # ZD_Info
    names_zd = ['ZD_CNT', 'ZD_LCG', 'ZD_TFG', 'ZD_JHG', 'ZD_LLJ', 'ZD_SPEED']
    fig, axis = plt.subplots(len(names_zd), 1, figsize=(12, 5), sharex=True)
    axis[0].set_title("ZD_Info")
    x = np.arange(len(data))
    for name, ax in zip(names_zd, axis):
        series = data[name]
        ax.plot(x, series, '-o')
        ax.set_ylabel(name)

    # ZX_BJ_Info : ZX_BJ_1 ~ ZX_BJ_6
    plt.figure(figsize=(12, 2))
    for i in np.arange(1, 7):
        name = "ZX_BJ_{}".format(i)
        series = data[name]
        plt.plot(x, series, '-o', label=name)
    plt.title("BJ_Info")
    plt.legend(loc='best', prop={'size': 5})

    # 'ZX_SDMC_1'
    plt.figure(figsize=(12, 2))
    for i in np.arange(1, 7):
        name = 'ZX_SDMC_{}'.format(i)
        series = data[name]
        plt.plot(x, series, '-o', label=name)
    plt.title('ZX_SDMC Info')
    plt.legend(loc='best', prop={'size': 5})

    # Bearing temperature info
    for f in np.arange(1, 7):
        # ZX_WD_1
        plt.figure(figsize=(12, 2))
        for i in np.arange(1, 7):
            name = "ZX_WD_{0}_{1}".format(f, i)
            series = data[name]
            plt.plot(x, series, '-o', label=name)
        plt.legend(loc='best', prop={'size': 5})
        plt.title("ZX_WD_{}".format(f))

        # ZX_HW1_1, ZX_HW2_1
        plt.figure(figsize=(12, 2))
        for i in [1, 2]:
            name = 'ZX_HW{0}_{1}'.format(i, f)
            plt.plot(x, data[name], '-o', label=name)
        plt.title("ZX_HW_{}".format(f))
        plt.legend(loc='best', prop={'size': 5})


def plot_data2(data):
    x = len(data)

    # ZD_Info
    names_zd = ['ZD_CNT', 'ZD_LCG', 'ZD_TFG', 'ZD_JHG', 'ZD_LLJ', 'ZD_SPEED']
    fig, axis = plt.subplots(len(names_zd), 1, figsize=(12, 5), sharex=True)
    axis[0].set_title("ZD_Info")
    x = np.arange(len(data))
    for name, ax in zip(names_zd, axis):
        series = data[name]
        ax.plot(x, series, '-o')
        ax.set_ylabel(name)

    # 'ZX_SDMC_1'
    plt.figure(figsize=(12, 2))
    for i in np.arange(1, 7):
        name = 'ZX_SDMC_{}'.format(i)
        series = data[name]
        plt.plot(x, series, '-o', label=name)
    plt.title('ZX_SDMC Info')
    plt.legend(loc='best', prop={'size': 5})

    plt.figure(figsize=(12, 2))
    for i in [1, 2]:
        for j in np.arange(1, 7):
            name = "ZX_HW{0}_{1}".format(i, j)
            plt.plot(x, data[name], '-o', label=name)
    plt.title("ZX_HW")
    plt.legend(loc='best', prop={'size': 5})

    # Bearing temperature info
    for f in np.arange(1, 7):
        # ZX_WD_1
        plt.figure(figsize=(12, 2))
        for i in np.arange(1, 7):
            name = "ZX_WD_{0}_{1}".format(f, i)
            series = data[name]
            plt.plot(x, series, '-o', label=name)
        plt.legend(loc='best', prop={'size': 5})
        plt.title("ZX_WD_{}".format(f))

        # ZX_BJ_Info : ZX_BJ_1 ~ ZX_BJ_6
    plt.figure(figsize=(12, 2))
    for i in np.arange(1, 7):
        name = "ZX_BJ_{}".format(i)
        series = data[name]
        plt.plot(x, series, '-o', label=name)
    plt.title("BJ_Info")
    plt.legend(loc='best', prop={'size': 5})

