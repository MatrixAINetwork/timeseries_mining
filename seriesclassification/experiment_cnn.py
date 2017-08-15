# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 10:43:29 2016

@author: Rob Romijnders
"""
import os

import numpy as np
from tsmining.classifier.cnn_classifier import cnn_classifier
from tsmining.shapelet.basic import ShapeletTransformBasic
from tsmining.utils import data_parser

from base import *


def experiment_cnn(dataset, k=5, k_val=1):
    file_train = os.path.join(UCR_DATA_ROOT, dataset, dataset + '_TRAIN')
    file_test = os.path.join(UCR_DATA_ROOT, dataset, dataset + '_TEST')
    X_train, y_train = data_parser.load_ucr(file_train)
    X_test, y_test = data_parser.load_ucr(file_test)

    X_train, y_train, X_val, y_val = data_parser.k_fold_validation_balance(X_train, y_train, k, k_val)
    acc = cnn_classifier(X_train, y_train, X_test, y_test, X_val, y_val)

    print("="*80)
    print("\n")
    print("data set: ")
    print("train set", np.shape(X_train), np.shape(y_train))
    print("validation set", np.shape(X_val), np.shape(y_val))
    print("test set", np.shape(X_test), np.shape(y_test))
    print(acc)
    print("=" * 80)
    return acc


def experiment_cnn_shapelet(dataset, k=5, k_val=1):
    file_train = os.path.join(UCR_DATA_ROOT, dataset, dataset + '_TRAIN')
    file_test = os.path.join(UCR_DATA_ROOT, dataset, dataset + '_TEST')
    X_train, y_train = data_parser.load_ucr(file_train)
    X_test, y_test = data_parser.load_ucr(file_test)

    # shapelet transformation
    min_shapelet_length = 8
    max_shapelet_length = 30
    num_shapelet = int(X_train.shape[1] * 0.5)
    length_increment = 10
    position_increment = 20
    callSTS = ShapeletTransformBasic(n_shapelets=num_shapelet,
                                     min_shapelet_length=min_shapelet_length,
                                     max_shapelet_length=max_shapelet_length,
                                     length_increment=length_increment,
                                     position_increment=position_increment)

    print("shapelet learning.....")
    callSTS.fit(X_train, y_train)
    bestk_shapelets = callSTS.train()
    print("bestk_shapelets: ", np.shape(bestk_shapelets))

    print("shapelet transformation.....")
    X_train_ = callSTS.transform(X_train)
    X_test_ = callSTS.transform(X_test)
    X_train_ = np.array(X_train_)
    X_test_ = np.array(X_test_)


    # cnn classifier
    X_train_, y_train, X_val_, y_val = data_parser.k_fold_validation_balance(X_train_, y_train, k, k_val)
    acc = cnn_classifier(X_train_, y_train, X_test_, y_test, X_val_, y_val)

    # print result
    print("=" * 80)
    print("\n")
    print("original data set: ")
    print("train set", np.shape(X_train), np.shape(y_train))
    print("test set", np.shape(X_test), np.shape(y_test))
    print("shapelet transformed set: ")
    print("learned shapelets: ", np.shape(bestk_shapelets))
    print("train set: ", np.shape(X_train_), np.shape(y_train))
    print("validation set: ", np.shape(X_val_), np.shape(y_val))
    print("test set: ", np.shape(X_test_), np.shape(y_test))
    print("\n")
    print("shapelets learning parameter: ")
    print("minShapeletLength", min_shapelet_length)
    print("max_shapelet_length", max_shapelet_length)
    print("length_increment", length_increment)
    print("position_increment", position_increment)
    print("\n")

    print(acc)
    print("=" * 80)
    return acc


if __name__ == '__main__':

    dataset = "Beef"
    result = []
    for i in range(10):
        #acc = experiment_cnn(dataset)
        acc = experiment_cnn_shapelet(dataset)
        result.append(acc)

    print("\n")
    print(np.mean(result))
    print(result)
