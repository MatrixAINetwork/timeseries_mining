
from classifier.KNNClassifier import *
from utils import distance as DISTFunc
from utils import data_parser
from utils import validation
from classifier.KNNClassifierVariousDim import KNeighborsClassifierVariousDim
import os
import time

DATA_ROOT = "/home/happyling/workspace/timeseries/data/weikang"

if __name__ == '__main__':
    dataset = "set"
    file_train = os.path.join(DATA_ROOT, dataset, dataset+'_'+"TRAIN")
    file_test = os.path.join(DATA_ROOT, dataset, dataset+'_'+"TEST")

    print("file_train", file_train)
    print("file_test", file_test)

    X_train, y_train = data_parser.load_list_data(file_train)
    X_test, y_test = data_parser.load_list_data(file_test)

    print('test class', np.unique(y_train))
    print('train class', np.unique(y_test))

    # print(X_train[0])
    # print(X_test[0])

    # z_normalize
    for id, iTrain in enumerate(X_train):
        X_train[id] = (iTrain - np.mean(iTrain)) / (np.sqrt(np.var(iTrain)) + 1e-9)

    for id, iTest in enumerate(X_test):
        X_test[id] = (iTest - np.mean(iTest)) / (np.sqrt(np.var(iTest)) + 1e-9)

    # print(X_train[0])
    # print(X_test[0])

    n_neighbors = 1
    n_jobs = 10
    knn_clf = KNeighborsClassifierVariousDim(n_neighbors=n_neighbors, distfunc=DISTFunc.dist_basic_dtw)
    knn_clf.fit(X_train, y_train)
    predList = knn_clf.predict(X_test)
    acc = validation.cal_accuracy(predList, y_test)
    print(acc)






