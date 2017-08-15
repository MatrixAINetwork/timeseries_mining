"""

    an experiment to test all whole time series similarity measure function
    author: fanling huang
    
"""
from __future__ import print_function

import logging
import os
from time import time

from tsmining.classifier.KNNClassifier import *
from tsmining.utils import data_parser
from tsmining.utils import distance
from tsmining.utils import validation

from base import *

if __name__ == '__main__':

    print(__doc__)

    # Display progress logs on stdout
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')

    # Load data set
    dataset = "BeetleFly"
    file_train = os.path.join(UCR_DATA_ROOT, dataset, dataset+'_'+'TRAIN')
    file_test = os.path.join(UCR_DATA_ROOT, dataset, dataset+'_'+'TEST')

    X_train, y_train = data_parser.load_ucr(file_train)
    X_test, y_test = data_parser.load_ucr(file_test)

    print("="*80)
    print("** data description: ")
    print("train set: ", np.shape(X_train))
    print("test set: ", np.shape(X_test))
    print("class label:", np.unique(y_train))


    def benchmark(clf, distf, distparams=None):
        t0 = time()
        clf.set_distfunc(distfunc=distf, distfunc_params=distparams)
        y_pred = clf.predict(X_test)

        duration = time() - t0
        acc = validation.cal_accuracy(y_pred, y_test)
        return acc, duration

    # Set up the knn classifier
    n_neighbors = 1
    n_jobs = 10
    knn_clf = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=n_jobs)
    knn_clf.fit(X_train, y_train)

    print("** knn parameter")
    print("n_neighbors: ", n_neighbors)
    print("n_jobs: ", n_jobs)

    # test distance function map
    dist_map = {'manhattan': {'func': distance.manhattan, 'params': None},
                'euclidean': {'func': distance.euclidean, 'params': None},
                'infinity': {'func': distance.infinity, 'params': None},
                'DTW': {'func': distance.dtw_basic, 'params': None},
                'wDTW': {'func': distance.dtw_win, 'params': {'win': int(0.1 * len(y_train))}},
                'LBKeogh': {'func': distance.LBKeogh, 'params': {'win': int(0.1*len(y_train))}},
                'weightedDTW': {'func': distance.dtw_weighted, 'params': {'g': 0.5}},
                'lcss': {'func': distance.lcss, 'params': {'epsilon': 0.5}},
                'twe': {'func': distance.twe, 'params': {'lambda_': 0.5, 'v': 0.5}},
                'msm': {'func': distance.msm, 'params': {'c': 0.1}},
                'cid': {'func': distance.cid, 'params': {'distfunc': distance.dtw_win,
                                                         'distfunc_params': {'win': int(0.1*len(y_train))}}},
                'ddtw': {'func': distance.ddtw, 'params': {'alpha': 0.5,
                                                           'distfunc': distance.dtw_win,
                                                           'distfunc_params': {'win': int(0.1 * len(y_train))}}},
                'dtdc': {'func': distance.dtdc, 'params': {'alpha': 0.5,
                                                           'beta': 0.3,
                                                           'distfunc': distance.dtw_win,
                                                           'distfunc_params': {'win': int(0.1 * len(y_train))}}}
                }

    print("="*80)
    print("*** knn + various distance function")

    result = []
    for key, value in dist_map.items():
        print(key)
        acc, t = benchmark(knn_clf, value['func'], value['params'])
        result.append((key, acc, t, str(value['params'])))
        print(key, acc, t)

    print("="*80)
    print("*** result")

    for r in result:
        print(r)


