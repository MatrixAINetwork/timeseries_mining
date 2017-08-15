import argparse
import numpy as np
from classifier import knnclassify as TSC
from utils import distance as DISTFunc
from sklearn.metrics import classification_report

def parseDataSet(filename):
    originDataArr = np.genfromtxt(filename, delimiter=',', dtype='float')
    instanceArr = originDataArr[:, 1::]
    classLabelArr = originDataArr[:, 0]

    return instanceArr, classLabelArr

def UCRClassify():
    argparser = argparse.ArgumentParser(description=' KNN Time series Classification')
    argparser.add_argument('trainFilename', help='training set directory')
    argparser.add_argument('testFilename', help='test set directory')
    argparser.add_argument('--i', type=int, default=1,
                           help='the number of distance function, default is [1], euclidean distance. \n'
                                '[1] euclidean distance,\n'
                                '[2] DTW distance, \n'
                                '[3] Speeded DTW distance, node: a window should be set\n'
                                '[4] LBKeogh, node: a window should be set\n')
    argparser.add_argument('--win', type=int, default=50,
                           help='the win size in distance calculation function Speeded DTW or LBKeogh, default is 50')
    argparser.add_argument('--K', type=int, default=1,
                           help='K parameter in KNN classification, defatu is 1')
    args = argparser.parse_args()

    trainDataArr, trainLabelArr = parseDataSet(args.trainFilename)
    testDataArr, testLabelArr = parseDataSet(args.testFilename)
    print("class label: ", np.unique(trainLabelArr))
    print("start to classify ...... ")

    nJobs = 10
    isParalell = True

    predList = []
    if args.i == 1:
        print("start to classify with Euclidean distance function ..... ")
        if isParalell:
            predList = TSC.KNNClassifyParallelJobLib(trainDataArr, trainLabelArr, testDataArr,
                                                     args.K, nJobs, DISTFunc.dist_euclidean)
        else:
            predList = TSC.KNNClassify(trainDataArr, trainLabelArr, testDataArr, testLabelArr,
                                       args.K, DISTFunc.dist_euclidean)
    elif args.i == 2:
        print("start to classify with DTW distance function ..... ")
        if isParalell:
            predList = TSC.KNNClassifyParallelJobLib(trainDataArr, trainLabelArr, testDataArr,
                                                     args.K, nJobs, DISTFunc.dist_basic_dtw)
        else:
            predList = TSC.KNNClassify(trainDataArr, trainLabelArr, testDataArr, testLabelArr,
                                       args.K, DISTFunc.dist_basic_dtw)
    elif args.i == 3:
        print("start to classify with speeded DTW distance function ..... ")
        if isParalell:
            predList = TSC.KNNClassifyParallelJobLib(trainDataArr, trainLabelArr, testDataArr,
                                                     args.K, nJobs, DISTFunc.dist_win_dtw, (args.win))
        else:
            predList = TSC.KNNClassify(trainDataArr, trainLabelArr, testDataArr, testLabelArr,
                                       args.K, DISTFunc.dist_win_dtw, (args.win))
    elif args.i == 4:
        print("start to classify with LBKeogh distance function ..... ")
        if isParalell:
            predList = TSC.KNNClassifyParallelJobLib(trainDataArr, trainLabelArr, testDataArr,
                                                     args.K, nJobs, DISTFunc.dist_LBKeogh, (args.win))
        else:
            predList = TSC.KNNClassify(trainDataArr, trainLabelArr, testDataArr,
                                       args.K, DISTFunc.dist_LBKeogh, (args.win))
    else:
        print("the number of distance function can not unidentifiable!")

    report = classification_report(testLabelArr, predList)

    OUT_ROOT = '../../data/output/'
    print(report)
    fileOutName = OUT_ROOT + "%s" % args.trainFilename.split('/')[-2] + '.txt'
    print(fileOutName)
    fout = open(fileOutName, 'w')
    fout.write(report)
    fout.close()


if __name__ == '__main__':
    UCRClassify()
