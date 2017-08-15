import os

from base import UCR_DATA_ROOT
from tsmining.utils import data_parser

dataset = "50words"


def test_k_fold_split():
    print("="*80)
    print("\n")
    file_train = os.path.join(UCR_DATA_ROOT, dataset, dataset + '_TRAIN')
    file_test = os.path.join(UCR_DATA_ROOT, dataset, dataset + '_TEST')
    X_train, y_train = data_parser.load_ucr(file_train)
    X_test, y_test = data_parser.load_ucr(file_test)

    k = 5
    k_val = 1
    X_train, y_train, X_val, y_val = data_parser.k_fold_validation_balance(X_train, y_train, k, k_val)

    print(X_train.shape, y_train.shape)
    print(X_val.shape, y_val.shape)

    print("="*80)
    print("\n")

if __name__ == '__main__':
    test_k_fold_split()