"""

    this is experiment_knn_ed.py
    an experiment to test validation the 1nn + euclidean distance classifier on different data set

"""
import os

from tsmining.classifier.KNNClassifier import KNeighborsClassifier
from tsmining.tools.output import table2markdown
from tsmining.utils import data_parser
from tsmining.utils import distance
from tsmining.utils import validation

from base import OUT_ROOT
from base import UCR_DATA_ROOT

dataset_ = ['50words', 'Adiac', 'Beef', 'Car', 'CBF', 'ChlorineConcentration', 'CinC_ECG_torso', 'Coffee',
           'DiatomSizeReduction', 'ECG200', 'ECGFiveDays', 'FaceFour', 'FacesUCR', 'FISH', 'Gun_Point',
           'Haptics', 'InlineSkate', 'ItalyPowerDemand', 'Lighting2', 'Lighting7', 'MALLAT', 'MedicalImages',
           'MoteStrain', 'OliveOil', 'OSULeaf', 'Plane', 'SonyAIBORobotSurface', 'SonyAIBORobotSurfaceII',
           'StarLightCurves', 'SwedishLeaf', 'Symbols', 'synthetic_control', 'Trace', 'TwoLeadECG',
           'Two_Patterns', 'wafer', 'WordsSynonyms', 'yoga']

if __name__ == '__main__':

    print("="*80)
    print("knn + euclidean classification ")
    print("data set: ")
    print(dataset_)
    print("\n")

    # check file
    for name in dataset_:
        file_train = os.path.join(UCR_DATA_ROOT, name, name+'_TRAIN')
        file_test = os.path.join(UCR_DATA_ROOT, name, name+'_TEST')
        if not os.path.exists(file_train):
            print(file_train + "  not exit !!")
        if not os.path.exists(file_test):
            print(file_test + "  not exit !!")


    # parameter
    n_neighbors = 1
    n_jobs = 10
    k_fold = 10
    distfunc = distance.euclidean

    # set up knn classifier
    knn_clf = KNeighborsClassifier(n_neighbors=n_neighbors,
                                   n_jobs=n_jobs,
                                   distfunc=distfunc)

    result = [['dataset name', 'accuracy']]
    for name in dataset_:
        # load data set
        file_train = os.path.join(UCR_DATA_ROOT, name, name + '_TRAIN')
        file_test = os.path.join(UCR_DATA_ROOT, name, name + '_TEST')
        X_train, y_train = data_parser.load_ucr(file_train)
        X_test, y_test = data_parser.load_ucr(file_test)
        # normalize
        X_train = data_parser.z_normalize(X_train)
        X_test = data_parser.z_normalize(X_test)

        # k fold validation
        acc_sum = 0
        for i in range(k_fold):
            knn_clf.fit(X_train, y_train)
            y_pred = knn_clf.predict(X_test)
            acc_sum += validation.cal_accuracy(y_test, y_pred)
            X_train, y_train, X_test, y_test = data_parser.resample_data(X_train, y_train, X_test, y_test)

        acc = acc_sum / k_fold
        result.append((name, acc))
        print(name, acc)

    file_out = os.path.join(OUT_ROOT, "%dnn_%dfold_ed_normalize.md" % (n_neighbors, k_fold))
    table2markdown(file_out, result, description=__doc__)

