import os
from time import time

from tsmining.classifier.KNNClassifier import KNeighborsClassifier
from sklearn.metrics import classification_report
from tsmining.utils import data_parser
from tsmining.utils import distance
from tsmining.utils import validation

from base import UCR_DATA_ROOT


def run(dataset, classifier_cls, dist_func=distance.euclidean, dist_func_params=None, is_normalize=False):
    print("="*80)
    print("\n")
    print("testing ", classifier_cls.__name__)
    print("="*80)

    # load data
    file_train = os.path.join(UCR_DATA_ROOT, dataset, dataset + '_TRAIN')
    file_test = os.path.join(UCR_DATA_ROOT, dataset, dataset + '_TEST')
    X_train, y_train = data_parser.load_ucr(file_train)
    X_test, y_test = data_parser.load_ucr(file_test)

    # z-normalization
    if is_normalize:
        X_train = data_parser.z_normalize(X_train)
        X_test = data_parser.z_normalize(X_test)

    print("=" * 60)
    print("basic information:")
    print("file_train", file_train)
    print("file_test", file_test)
    print("X_train", X_train.shape)
    print("y_train", y_train.shape)
    print("X_test", X_test.shape)
    print("y_test", y_test.shape)
    print("=" * 60)

    print("=" * 60)
    print("knn classification ")

    # set 1-nn classifier
    n_jobs = 10
    n_neighbors = 1
    classifier = classifier_cls(n_neighbors=n_neighbors,
                                distfunc=dist_func,
                                distfunc_params=dist_func_params,
                                n_jobs=n_jobs)
    print("=" * 60)
    print("classifier parameters")
    print("distance function: ", classifier.distfunc.__name__)
    print("distance parameters: ", classifier.distfunc_params)
    print("n_jobs: ", classifier.n_jobs)
    print("n_neighbors: ", classifier.n_neighbors)
    print("=" * 60)

    time_start = time()

    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    report = classification_report(y_test, y_pred)
    print(report)
    time_end = time()

    acc = validation.cal_accuracy(y_test, y_pred)

    print("time: ", time_end - time_start)
    print("accuracy: ", acc)


if __name__ == '__main__':
    run("Beef", KNeighborsClassifier, dist_func=distance.dtw_win, dist_func_params={'win': 100})

