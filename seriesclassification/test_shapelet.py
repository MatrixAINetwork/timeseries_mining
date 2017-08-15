import os

from tsmining.shapelet.basic import ShapeletTransformBasic
from base import dataset_100
import pipline_shapelet
from tsmining.shapelet.param_estimate import estimate_min_max_length
from tsmining.utils import data_parser
from base import UCR_DATA_ROOT
from base import OUT_ROOT


#################################
# ShapeletTransformBasic
#################################


def test_basic():
    print("=" * 80)
    print("this is test_one_dataset() ...")
    dataset = 'Beef'
    out_dir = os.path.join(OUT_ROOT, 'test')
    acc, _ = pipline_shapelet.run(dataset,
                                  ShapeletTransformBasic,
                                  length_increment=50,
                                  position_increment=50,
                                  log_dir=out_dir)


def test_basic_dataset_100():
    print("="*80)
    print("this is test_dataset_100(): ")
    result = []
    for name in dataset_100:
        acc, _ = pipline_shapelet.run(name, ShapeletTransformBasic)
        result.append((name, acc))
    print("result.....")
    for name, acc in result:
        print(name, acc)

#################################
# ShapeletTransformBasic
#################################


def test_shapelet_version2():
    from tsmining.shapelet.accelerate import ShapeletTransformPruning
    dataset = 'Beef'
    acc, _ = pipline_shapelet.run(dataset,
                                  ShapeletTransformPruning,
                                  length_increment=5,
                                  position_increment=10)


#################################
# parameter selection
#################################


def test_find_length():
    dataset = 'Beef'

    file_train = os.path.join(UCR_DATA_ROOT, dataset, dataset + '_TRAIN')
    file_test = os.path.join(UCR_DATA_ROOT, dataset, dataset + '_TEST')
    X_train, y_train = data_parser.load_ucr(file_train)
    X_test, y_test = data_parser.load_ucr(file_test)

    print("estimating min length and max length of shapelet learning...")
    min_length, max_length = estimate_min_max_length(X_train, y_train)
    print(min_length)
    print(max_length)

if __name__ == '__main__':
    test_shapelet_version2()

