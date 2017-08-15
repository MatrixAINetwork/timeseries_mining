import os

import numpy as np
from classifier.KNNClassifier import KNeighborsClassifier
from shapelet.transform_fast import ShapeletTransformFast
from utils import data_parser
from utils import validation

from base import *

dataset_small = ['Beef', 'BeetleFly', 'BirdChicken', 'Coffee', 'OliveOil']  # 40<=n_samples<=

dataset = " liveOil" # small data set: OliveOil
file_train = os.path.join(UCR_DATA_ROOT, dataset, dataset+'_TRAIN')
file_test = os.path.join(UCR_DATA_ROOT, dataset, dataset+'_TEST')
X_train, y_train = data_parser.load_ucr(file_train)
X_test, y_test = data_parser.load_ucr(file_test)

print("******** data info *********")
print("trainData", X_train.shape, y_train.shape)
print("testData", X_test.shape, y_test.shape)
print("****************************")

print("************** shapelet transforming ******************")

# parameter setting
minShapeletLength = 30
maxShapeletLength = int(X_train.shape[1] * 0.7)
numShapelet = int(X_train.shape[1] * 0.5)
lengthIncrement = 20
positionIncrement = 20
callSTS = ShapeletTransformFast(X_train, y_train,
                                minShapeletLength, maxShapeletLength,
                                lengthIncrement, positionIncrement)

best_k_shapelets = callSTS.find_best_shapelets(numShapelet)
print("bestKShapelet: ", np.shape(best_k_shapelets))

X_train_transform = callSTS.transform(best_k_shapelets, X_train)
X_test_transform = callSTS.transform(best_k_shapelets, X_test)
X_train_transform = np.array(X_train_transform)
X_test_transform = np.array(X_test_transform)

print("*******************************************************")
print("original data:", np.shape(X_train), np.shape(X_test))
print("transform data: ", np.shape(X_train_transform), np.shape(X_test_transform))
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
KNNC.fit(X_train, y_train)
y_pred = KNNC.predict(X_test)
print("accuracy: ", validation.cal_accuracy(y_test, y_pred))

print("knn with shapelet ....")
KNNC.fit(X_train_transform, y_train)
y_pred = KNNC.predict(X_test_transform)
print("accuracy: ", validation.cal_accuracy(y_test, y_pred))


print("*****************************************************")

