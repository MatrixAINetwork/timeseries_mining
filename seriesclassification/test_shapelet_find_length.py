import os

from shapelet.transform_basic import *
from utils import data_parser

from base import UCR_DATA_ROOT

dataset = 'Beef'

file_train = os.path.join(UCR_DATA_ROOT, dataset, dataset+'_TRAIN')
file_test = os.path.join(UCR_DATA_ROOT, dataset, dataset+'_TEST')
X_train, y_train = data_parser.load_ucr(file_train)
X_test, y_test = data_parser.load_ucr(file_test)

print("estimating min length and max length of shapelet learning...")
min_length, max_length = estimate_min_max_length(X_train, y_train)
print(min_length)
print(max_length)


