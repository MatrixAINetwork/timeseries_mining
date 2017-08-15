import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import preprocess



def var_corr_significant(path):
    filelist = preprocess.get_filelist(path)
    filelist = np.sort(filelist)
    corr_criterion = 0.5
    log_file = open("result/corr_log_{0}_{1}.txt".format(path.split('/')[-2], corr_criterion), "w")
    print("********************** file correlation large than {} ***************".format(corr_criterion),
          file=log_file)
    for file in filelist:
        print("\n\n*************************** file: {} **********************".format(file), file=log_file)
        data = pd.read_csv(path+file, index_col=0)
        names = data.columns
        for col_name in names[1::]:
            series = data[col_name]
            filter = series.abs() > corr_criterion
            print("--------------- {} ------------------".format(col_name), file=log_file)
            for index, value in zip(names[filter], series[filter]):
                print("{0}\t\t{1}".format(index, value), file=log_file)


def cal_correlation(path_in, path_out):
    filelist = preprocess.get_filelist(path_in)
    for file in filelist:
        print(file)
        data = pd.read_csv(path_in + file, index_col=0)
        result = data.corr()
        result.to_csv(path_out + file)



def select_spercific():
    path = "result/corr_interpolation_10s_int/"
    filelist = preprocess.get_filelist(path)
    filelist = np.sort(filelist)
    file_log = open("result/corr_zw.txt", "w")
    for file in filelist:
        print("**************** result of file: {} **************".format(file), file=file_log)
        data = pd.read_csv(path+file, index_col=0)
        for i in np.arange(1, 7):
            for j in np.arange(1, 7):
                name = 'ZX_WD_{0}_{1}'.format(i, j)
                print("-------- {} --------------".format(name), file=file_log)
                names_corr = list()
                for k in np.arange(1, 7):
                    names_corr.insert(len(names_corr), 'ZX_WD_{0}_{1}'.format(i, k))
                for k in np.arange(1, 7):
                    if k == i:
                        continue
                    names_corr.insert(len(names_corr), 'ZX_WD_{0}_{1}'.format(k, j))
                for index, value in zip(names_corr, data.loc[names_corr, name]):
                    print("{0}\t\t{1}".format(index, value), file=file_log)
        print("***********************************************************************\n\n", file=file_log)


def cal_corr_segments():
    data = pd.read_csv("data/interpolation_10s_int/0394.csv")
    segs = np.arange(1, len(data), 2000)
    file_log = open("result/seg_corr_0394.txt", "w")
    for p in np.arange(1, len(segs)):
        subset = data[segs[p-1]:segs[p]]
        corrset = subset.corr()
        print("**************** result of seg: {0}:{1} **************".format(segs[p-1], segs[p]), file=file_log)
        for i in np.arange(1, 7):
            for j in np.arange(1, 7):
                name = 'ZX_WD_{0}_{1}'.format(i, j)
                print("-------- {} --------------".format(name), file=file_log)
                names_corr = list()
                for k in np.arange(1, 7):
                    names_corr.insert(len(names_corr), 'ZX_WD_{0}_{1}'.format(i, k))
                for k in np.arange(1, 7):
                    if k == i:
                        continue
                    names_corr.insert(len(names_corr), 'ZX_WD_{0}_{1}'.format(k, j))
                for index, value in zip(names_corr, corrset.loc[names_corr, name]):
                    print("{0}\t\t{1}".format(index, value), file=file_log)
        print("***********************************************************************\n\n", file=file_log)

def cal_corr_segments_mean():
    data = pd.read_csv("data/interpolation_10s_int/0394.csv")
    segs = np.arange(1, len(data), 2000)
    file_log = open("result/seg_corr_mean_0394.txt", "w")
    for p in np.arange(1, len(segs)):
        subset = data[segs[p - 1]:segs[p]]
        corrset = subset.corr()
        print("**************** result of seg: {0}:{1} **************".format(segs[p - 1], segs[p]), file=file_log)
        for i in np.arange(1, 7):
            indicators = list()
            for j in np.arange(1, 7):
                name = 'ZX_WD_{0}_{1}'.format(i, j)
                names_corr = list()
                for k in np.arange(1, 7):
                    names_corr.insert(len(names_corr), 'ZX_WD_{0}_{1}'.format(i, k))
                for k in np.arange(1, 7):
                    if k == i:
                        continue
                    names_corr.insert(len(names_corr), 'ZX_WD_{0}_{1}'.format(k, j))
                indicators.insert(len(indicators), np.mean(corrset.loc[names_corr, name]))
            print(indicators, file = file_log)
        print("***********************************************************************\n\n", file=file_log)

def cal_corr_segments_rolling_mean():
    data = pd.read_csv("data/interpolation_10s_int/0500.csv")
    win = 2000
    start = 0
    end = start + win-1
    file_log = open("result/seg_corr_rolling_mean_0500.txt", "w")
    indicators_seg = list()
    while end < len(data):
        subset = data[start:end]
        corrset = subset.corr()
        print("**************** result of seg: {0}:{1} **************".format(start, end), file=file_log)
        indicators_metrix = list()
        for i in np.arange(1, 7):
            indicators = list()
            for j in np.arange(1, 7):
                name = 'ZX_WD_{0}_{1}'.format(i, j)
                names_corr = list()
                for k in np.arange(1, 7):
                    names_corr.insert(len(names_corr), 'ZX_WD_{0}_{1}'.format(i, k))
                for k in np.arange(1, 7):
                    if k == i:
                        continue
                    names_corr.insert(len(names_corr), 'ZX_WD_{0}_{1}'.format(k, j))
                indicators.insert(len(indicators), np.mean(corrset.loc[names_corr, name]))
            print(indicators, file=file_log)
            indicators_metrix.insert(len(indicators_metrix), np.mean(indicators))
        print("***********************************************************************\n\n", file=file_log)
        indicators_seg.insert(len(indicators_seg), np.mean(indicators_metrix))
        start = start + 1
        end = end + 1
    plt.plot(indicators_seg)
    plt.show()

cal_corr_segments_rolling_mean()
#cal_corr_segments_mean()
#cal_corr_segments()
#select_spercific()
#var_corr_significant("result/ZWNo1/")
# cal_correlation("data/interpolation_10s_int/", "result/corr_interpolation_10s_int/")
# var_corr_significant("result/corr_interpolation_10s_int/")
