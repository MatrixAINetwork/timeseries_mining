"""

    this is experiment_parallel_subseq_distance.py.
    an experiment test test the performance of parallel through sub sequence distance calculation function.
    
"""

import os
from time import time

import numpy as np
from tsmining.tools.output import table2markdown
from tsmining.utils import data_parser
from tsmining.utils import distance
from tsmining.utils import pairwise

from base import OUT_ROOT
from base import UCR_DATA_ROOT
from base import dataset

position_increment = 10
dist_func = distance.euclidean
dist_func_params = {}

params = {'position_increment': position_increment,
          'dist_func': dist_func,
          'dist_func_params': dist_func_params}

print("="*80)
print(__doc__)
print("\n")

result = []
result.append(("data set name", "time", "time of parallel", "size",
               "length of sub sequence", "length of distance list"))

for set_name in dataset:
    file_train = os.path.join(UCR_DATA_ROOT, set_name, set_name+'_TRAIN')
    file_test = os.path.join(UCR_DATA_ROOT, set_name, set_name+'_TEST')
    X_train, _ = data_parser.load_ucr(file_train)
    X_test, _ = data_parser.load_ucr(file_test)
    X = np.vstack([X_train, X_test])
    m, n = X.shape
    sublen = int(0.5*n)
    subseq = X[0][:sublen]

    t_start = time()
    distance_list = []
    for i in range(m):
        dist = distance.dist_subsequence(subsequence=subseq,
                                         wholeseries=X[i],
                                         **params)
        distance_list.append(dist)
    t_end = time()
    t_non_parallel = t_end - t_start

    t_start_parallel = time()
    n_jobs = 10
    distance_list = pairwise.parallel_pairwise(subseq,
                                               X,
                                               n_jobs=n_jobs,
                                               func=distance.dist_subsequence,
                                               func_params=params)
    t_end_parallel = time()
    t_parallel = t_end_parallel - t_start_parallel

    result.append((set_name, t_non_parallel, t_parallel, m, sublen, len(distance_list)))
    print((set_name, t_non_parallel, t_parallel, m, sublen, len(distance_list)))

file_out = os.path.join(OUT_ROOT, "parallel_performance.md")
table2markdown(file_out, result, description=__doc__)


