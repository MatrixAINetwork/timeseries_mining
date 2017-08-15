"""

    an experiment to test all whole time series similarity measure function
    author: fanling huang
    
"""
from __future__ import print_function

import logging
import os
from time import time

from classifier.KNNClassifier import *
from utils import data_parser
from utils import distance as DISTFunc
from utils import validation

from base import *

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
dist_map = {'manhattan': {'func': DISTFunc.dist_manhattan, 'params': None},
            'euclidean': {'func': DISTFunc.dist_euclidean, 'params': None},
            'infinity': {'func': DISTFunc.dist_infinity, 'params': None},
            'DTW': {'func': DISTFunc.dist_basic_dtw, 'params': None},
            'wDTW': {'func': DISTFunc.dist_win_dtw, 'params': {'win': int(0.1 * len(y_train))}},
            'LBKeogh': {'func': DISTFunc.dist_LBKeogh, 'params': {'win': int(0.1*len(y_train))}},
            'weightedDTW': {'func': DISTFunc.dist_weighted_dtw, 'params': {'g': 0.5}},
            'lcss': {'func': DISTFunc.dist_lcss, 'params': {'epsilon': 0.5}},
            'twe': {'func': DISTFunc.dist_twe, 'params': {'lambda_': 0.5, 'v': 0.5}},
            'msm': {'func': DISTFunc.dist_msm, 'params': {'c': 0.1}},
            'cid': {'func': DISTFunc.dist_cid, 'params': {'distfunc': DISTFunc.dist_win_dtw,
                                                          'distfunc_params': {'win': int(0.1*len(y_train))}}},
            'ddtw': {'func': DISTFunc.dist_ddtw, 'params': {'alpha': 0.5,
                                                            'distfunc': DISTFunc.dist_win_dtw,
                                                            'distfunc_params': {'win': int(0.1 * len(y_train))}}},
            'dtdc': {'func': DISTFunc.dist_dtdc, 'params': {'alpha': 0.5,
                                                            'beta': 0.3,
                                                            'distfunc': DISTFunc.dist_win_dtw,
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


