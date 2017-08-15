import numpy as np
import joblib


def KNNClassify(trainArr, trainY, testArr, K, distFunc, *distFuncArgs):
    assert len(trainArr) == len(trainY), 'the length of instance and labels in train set are not equal!!'

    K = min(len(trainArr), K)
    predList = []
    for ind, vecTest in enumerate(testArr):
        distList = []
        for vecTrain in trainArr:
            dist = distFunc(vecTest, vecTrain, *distFuncArgs)
            distList.append(dist)
        sortedIndexs = np.argsort(distList)
        predDic = {}
        for i in range(K):
            voteLabel = trainY[sortedIndexs[i]]
            predDic[voteLabel] = predDic.get(voteLabel, 0) + 1
        sortedClassCount = sorted(predDic.items(), key=lambda p: p[1], reverse=True)
        predList.append(sortedClassCount[0][0])

    return predList


def KNNClassifyParallelJobLib(trainArr, trainY, testArr, k, nJobs, distFunc, *distFuncArgs):
    assert len(trainArr) == len(trainY), 'the length of instance and labels in train set are not equal!!'

    k = min(len(trainArr), k)
    predList = []
    for ind, vecTest in enumerate(testArr):
        distList = joblib.Parallel(n_jobs=nJobs)(joblib.delayed(distFunc)(vecTest, vecTrain, *distFuncArgs)
                                                 for vecTrain in trainArr)
        sortedIndexs = np.argsort(distList)
        predDic = {}
        for i in range(k):
            voteLabel = trainY[sortedIndexs[i]]
            predDic[voteLabel] = predDic.get(voteLabel, 0) + 1

        sortedClassCount = sorted(predDic.items(), key=lambda p: p[1], reverse=True)
        predList.append(sortedClassCount[0][0])
    return predList