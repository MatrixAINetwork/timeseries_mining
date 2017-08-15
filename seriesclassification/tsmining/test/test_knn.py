import time

import numpy as np
from sklearn.metrics import classification_report
from classifier import knnclassify as TSC
from utils import distance as DISTFunc

DATA_ROOT = "/home/happyling/workspace/timeseries/data/testset/"


def test_KNNClassify():
    originalTrainArr = np.genfromtxt(DATA_ROOT + 'train.csv', delimiter='\t', dtype='float')
    originalTestArr = np.genfromtxt(DATA_ROOT + 'test.csv', delimiter='\t', dtype='float')
    trainArr = originalTrainArr[:, :-1]
    testArr = originalTestArr[:, :-1]
    trainY = originalTrainArr[:, -1]
    testY = originalTestArr[:, -1]
    print(trainArr.shape)
    print("class: ", np.unique(np.vstack((trainY, testY))))
    print("classification start........")

    predList = TSC.KNNClassify(trainArr, trainY, testArr, 1, DISTFunc.dist_win_dtw, 50)
    print(classification_report(testY, predList))


def test_KNNClassifyParallelJobLib():
    startTime = time.time()
    originalTrainArr = np.genfromtxt(DATA_ROOT + 'train.csv', delimiter='\t', dtype='float')
    originalTestArr = np.genfromtxt(DATA_ROOT + 'test.csv', delimiter='\t', dtype='float')
    trainArr = originalTrainArr[:, :-1]
    testArr = originalTestArr[:, :-1]
    trainY = originalTrainArr[:, -1]
    testY = originalTestArr[:, -1]
    print(trainArr.shape)
    print("class: ", np.unique(np.vstack((trainY, testY))))
    win = int(len(trainArr)*0.2)
    nJobs = 10
    print("classification start........")
    predList = TSC.KNNClassifyParallelJobLib(trainArr, trainY, testArr, 1, nJobs,
                                             DISTFunc.dist_win_dtw, win)
    endTime = time.time()
    wastTime = endTime - startTime
    print ("used time is ", wastTime)
    print(classification_report(testY, predList))
