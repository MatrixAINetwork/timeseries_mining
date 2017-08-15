
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from common import *
from preprocess import base
import sys


def DTWDistance1(s1, s2):
    DTW = {}

    for i in range(len(s1)):
        DTW[(i, -1)] = float('inf')
    for i in range(len(s2)):
        DTW[(-1, i)] = float('inf')
    DTW[(-1, -1)] = 0

    for i in range(len(s1)):
        for j in range(len(s2)):
            dist = (s1[i] - s2[j]) ** 2
            DTW[(i, j)] = dist + min(DTW[(i - 1, j)], DTW[(i, j - 1)], DTW[(i - 1, j - 1)])

    return np.sqrt(DTW[len(s1) - 1, len(s2) - 1])


def DTWDistanceW(s1, s2, w):
    DTW = {}

    w = max(w, abs(len(s1) - len(s2)))

    for i in range(-1, len(s1)):
        for j in range(-1, len(s2)):
            DTW[(i, j)] = float('inf')
    DTW[(-1, -1)] = 0

    for i in range(len(s1)):
        for j in range(max(0, i - w), min(len(s2), i + w)):
            dist = (s1[i] - s2[j]) ** 2
            DTW[(i, j)] = dist + min(DTW[(i - 1, j)], DTW[(i, j - 1)], DTW[(i - 1, j - 1)])

    return np.sqrt(DTW[len(s1) - 1, len(s2) - 1])


def cluster(data, threshold=1000):
    varnames = data.columns
    i_clust = 0
    clust = {}
    for name in varnames:
        result = []
        for name2 in varnames:
            result.append(DTWDistanceW(data[name], data[name2], 10))
        name_list = list()
        dist_list = list()
        for i in np.arange(len(result)):
            if result[i] < threshold:
                name_list.append(varnames[i])
                dist_list.append(result[i])

        find = False
        for key in clust:
            if set(name_list) == set(clust[key]):
                find = True
                break
        if find == False:
            clust[i_clust] = name_list
            i_clust += 1

    return clust


def cluster_batch(data_dir):
    filelist = base.get_files_csv(data_dir)
    stdout_original = sys.stdout
    log_file = open(data_dir+"dtw_cluster.txt", "w")
    sys.stdout = log_file
    for file in filelist:
        data = pd.read_csv(data_dir + file)
        del data['BTSJ']
        #data = (data-data.mean())/(data.std())
        clust = cluster(data)
        print("-------------------------------------------------------")
        print("********process file: {}*********".format(file))
        print(clust)
        sys.stdout = stdout_original
        log_file.close()


root_dir = os.getcwd()
data_no_list = ['data_0134', 'data_0141', 'data_0192', 'data_0394', 'data_0790']
for data_no in data_no_list:
    data_dir = root_dir + "/{}/smooth_mean_interpolate_bin_mean/".format(data_no)
    cluster_batch(data_dir)
    print("finish to process dir：{}".format(data_no))
