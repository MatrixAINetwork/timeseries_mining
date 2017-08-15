import os

import numpy as np
from tsmining.classifier.KNNClassifier import KNeighborsClassifier
from tsmining.utils import data_parser
from tsmining.utils import validation

from base import *


def test_shapelet(dataset,
                  shapelet_cls,
                  min_shapelet_length=20,
                  max_shapelet_length=0.7,
                  num_shapelet=0.5,
                  length_increment=20,
                  position_increment=20,
                  n_neighbors=1,
                  n_jobs=10,
                  log_dir=None):

    print("="*80)
    print("\n")
    print("testing ", shapelet_cls.__name__)
    print("\n")

    # load data
    file_train = os.path.join(UCR_DATA_ROOT, dataset, dataset+'_TRAIN')
    file_test = os.path.join(UCR_DATA_ROOT, dataset, dataset+'_TEST')
    X_train, y_train = data_parser.load_ucr(file_train)
    X_test, y_test = data_parser.load_ucr(file_test)

    # z normalization
    X_train = data_parser.z_normalize(X_train)
    X_test = data_parser.z_normalize(X_test)

    print("="*80)
    print("basic information:  \n")
    print("file_train: ", file_train)
    print("file_test: ", file_test)
    print("X_train", X_train.shape)
    print("y_train", y_train.shape)
    print("X_test", X_test.shape)
    print("y_test", y_test.shape)
    print("="*80)


    # shapelet setup

    if max_shapelet_length < 1:
        max_shapelet_length = int(X_train.shape[1] * 0.7)
    if num_shapelet:
        num_shapelet = int(X_train.shape[1] * 0.5)

    callSTS = shapelet_cls(min_shapelet_length=min_shapelet_length,
                           max_shapelet_length=max_shapelet_length,
                           length_increment=length_increment,
                           position_increment=position_increment)
    callSTS.fit(X_train, y_train)

    print("="*80)
    print("set up shapelet learning and transformation: \n")
    print("series_list", np.shape(callSTS.series_list))
    print("class_list", np.shape(callSTS.class_list))
    print("min_shapelet_length", callSTS.min_shapelet_length)
    print("max_shapelet_length", callSTS.max_shapelet_length)
    print("length_increment", callSTS.length_increment)
    print("position_increment", callSTS.position_increment)
    print("distance function", str(callSTS.dist_func.__name__))
    print("distance parameter", callSTS.dist_func_params)
    if hasattr(callSTS, 'class_distribution'):
        print("class distribution", callSTS.class_distribution)
    print("="*80)

    print("="*80)
    print("shapelet learning ......")

    bestk_shapelets = callSTS.find_best_shapelets(num_shapelet)

    print("best k shapelet: ", np.shape(bestk_shapelets))
    print("="*80)

    print("="*80)
    print("shapelet transformation ......")

    X_train_transform = callSTS.transform(bestk_shapelets, X_train)
    X_test_transform = callSTS.transform(bestk_shapelets, X_test)
    X_train_transform = np.array(X_train_transform)
    X_test_transform = np.array(X_test_transform)

    print("X_train_transform: ", np.shape(X_train_transform))
    print("X_test_transform: ", np.shape(X_test_transform))

    # knn classification
    print("="*80)
    print("knn classification ......")

    KNNC = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=n_jobs)
    KNNC.fit(X_train_transform, y_train)
    y_pred = KNNC.predict(X_test_transform)
    acc = validation.cal_accuracy(y_test, y_pred)
    print("accuracy: ", acc)

    # log shapelet
    if log_dir is not None:
        file_shapelet = os.path.join(log_dir,
                                     "%s_minlen-%s_maxlen-%s" % (dataset, min_shapelet_length, max_shapelet_length))
        print("saving shapelet to %s........." % file_shapelet)
        callSTS.save_shapelet(bestk_shapelets, file_shapelet)

    return acc, bestk_shapelets