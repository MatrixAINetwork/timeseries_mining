import os

from classifier.KNNClassifier import *
from shapelet.transform_basic import *
from sklearn.metrics import classification_report
from utils import data_parser
from utils import validation

from base import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def knn_experiment(knnClassifier, X_train, y_train, X_test, y_test, num_resample=1, verbose=True):
    sum_accuracy = 0
    for i in range(num_resample):
        knnClassifier.fit(X_train, y_train)
        predList = knnClassifier.predict(X_test)
        iaccuracy = validation.cal_accuracy(y_test, predList)
        if verbose:
            print(iaccuracy)
        sum_accuracy += iaccuracy
        X_train, y_train, X_test, y_test = data_parser.resample_data(X_train, y_train, X_test, y_test)

    return sum_accuracy / num_resample


def knn_shapelet_experiment(knnClassifier, X_train, y_train, X_test, y_test, num_resample=1, verbose=True):
    minShapeletLength = 30
    maxShapeletLength = int(X_train.shape[1] * 0.7)
    numShapelet = int(X_train.shape[1] * 0.5)
    lengthIncrement = 20
    positionIncrement = 20
    sum_accuracy = 0
    for i in range(num_resample):
        callSTS = ShapeletTransformSimplicity(X_train, y_train,
                                              minShapeletLength, maxShapeletLength,
                                              lengthIncrement, positionIncrement)
        bestKShapelet = callSTS.train(numShapelet)
        print("bestKShapelet: ", np.shape(bestKShapelet))
        X_train_ = callSTS.transform(bestKShapelet, X_train)
        X_test_ = callSTS.transform(bestKShapelet, X_test)
        X_train_ = np.array(X_train_)
        X_test_ = np.array(X_test_)

        knnClassifier.fit(X_train_, y_train)
        predList = knnClassifier.predict(X_test_)
        iaccuracy = validation.cal_accuracy(y_test, predList)
        sum_accuracy += iaccuracy
        if verbose:
            print(iaccuracy)
        X_train, y_train, X_test, y_test = data_parser.resample_data(X_train, y_train, X_test, y_test)

    return sum_accuracy / num_resample


def main_resample_experiment():
    dataset = "BeetleFly"
    file_train = os.path.join(UCR_DATA_ROOT, dataset, dataset+'_TRAIN')
    file_test = os.path.join(UCR_DATA_ROOT, dataset, dataset+'_TEST')
    trainX, trainY = data_parser.load_ucr(file_train)
    testX, testY = data_parser.load_ucr(file_test)

    print("******** data info *********")
    print("trainData", trainX.shape, trainY.shape)
    print("testData", testX.shape, testY.shape)
    print("****************************")

    print("************* KNN Classification with ED distance *********************")
    print("knn with original time series ")
    KNNC = KNeighborsClassifier(n_neighbors=1, n_jobs=10)
    avg_acc = knn_experiment(KNNC, trainX, trainY, testX, testY, 20)
    print("average accuracy: ", avg_acc)

    print("knn with shapelet:")
    avg_acc = knn_shapelet_experiment(KNNC, trainX, trainY, testX, testY, 20)
    print("average accuracy: ", avg_acc)

    print("*****************************************************")


def main_fixed_experiment():
    dataset = "BeetleFly"
    filename = UCR_DATA_ROOT + dataset + "/"
    trainX, trainY = data_parser.load_ucr(filename + '{}_TRAIN'.format(dataset))
    testX, testY = data_parser.load_ucr(filename + '{}_TEST'.format(dataset))

    trainX, trainY, testX, testY = data_parser.resample_data(trainX, trainY, testX, testY)

    print("******** data info *********")
    print("trainData", trainX.shape, trainY.shape)
    print("testData", testX.shape, testY.shape)
    print("****************************")

    print("************** shapelet transforming ******************")

    minShapeletLength = 30
    maxShapeletLength = int(trainX.shape[1] * 0.7)
    numShapelet = int(trainX.shape[1] * 0.5)
    lengthIncrement = 20
    positionIncrement = 20
    callSTS = ShapeletTransformSimplicity(trainX, trainY,
                                          minShapeletLength, maxShapeletLength,
                                          lengthIncrement, positionIncrement)

    bestKShapelet = callSTS.train(numShapelet)
    print("bestKShapelet: ", np.shape(bestKShapelet))
    trainTransformX = callSTS.transform(bestKShapelet, trainX)
    testTransformX = callSTS.transform(bestKShapelet, testX)
    trainTransformX = np.array(trainTransformX)
    testTransformX = np.array(testTransformX)

    print("*******************************************************")
    print("original data:", np.shape(trainX), np.shape(testX))
    print("transform data: ", np.shape(trainTransformX), np.shape(testTransformX))
    print("*** parameter: ")
    print("minShapeletLength: ", minShapeletLength)
    print("maxShapeletLength: ", maxShapeletLength)
    print("numShapelet: ", numShapelet)
    print("lengthIncrement: ", lengthIncrement)
    print("positionIncrement: ", positionIncrement)
    print("*******************************************************")

    print("************* KNN Classification *********************")
    KNNC = KNeighborsClassifier(n_jobs=10)

    print("knn with original time series.....")
    KNNC.fit(trainX, trainY)
    predList = KNNC.predict(testX)
    report = classification_report(testY, predList)
    print(report)
    print("accuracy: ", validation.calAccuracy(testY, predList))

    print("knn with shapelet ....")
    KNNC.fit(trainTransformX, trainY)
    predList = KNNC.predict(testTransformX)
    report = classification_report(testY, predList)
    print(report)
    print("accuracy: ", validation.calAccuracy(testY, predList))

    print("knn with shapelet and z-normalization...")
    trainTransformXZNorm = data_parser.z_normalize(trainTransformX)
    testTransformXZNorm = data_parser.z_normalize(testTransformX)
    KNNC.fit(trainTransformXZNorm, trainY)
    predList = KNNC.predict(testTransformX)
    report = classification_report(testY, predList)
    print(report)
    print("accuracy: ", validation.calAccuracy(testY, predList))

    print("*****************************************************")


if __name__ == '__main__':
    print("classification experiment")
    main_resample_experiment()
