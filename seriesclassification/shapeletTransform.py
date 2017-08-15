from shapelet.transform_basic import *
from utils import *
import os


def out_data(filename, dataArr, LabelArr):
    assert len(dataArr) == len(LabelArr), "the instance set and label set not alignment"
    m, n = dataArr.shape
    fw = open(filename, "w")
    for i in range(m):
        str_row = str(LabelArr[i])
        for j in range(n):
            str_row = str_row + "," + str(dataArr[i][j])
        fw.write(str_row+"\n")
    fw.close()

""" Load data set """

UCR = True
if UCR:
    dataset = "ChlorineConcentration"
    datadir = "/home/happyling/workspace/timeseries/data/UCR_TS_Archive_2015/" + dataset +'/'
    trainDataArr, trainLabelArr = dataParser.parse_ucr(datadir+dataset+"_TRAIN")
    testDataArr, testLabelArr = dataParser.parse_ucr(datadir+dataset+"_TEST")

print("******** data info *********")
print("trainData", trainDataArr.shape, trainLabelArr.shape)
print("testData", testDataArr.shape, testLabelArr.shape)
print("****************************")

""" Hyperparameter """

minShapeletLength = 30
maxShapeletLength = int(trainDataArr.shape[1]*0.7)
numShapelet = int(trainDataArr.shape[1]*0.7)
lengthIncrement = 10
positionIncrement = 20

# minShapeletLength = 100
# maxShapeletLength = 150
# numShapelet = int(trainDataArr.shape[1]*0.5)
# lengthIncrement = 50
# positionIncrement = 50

""" shapelet transformation """
callSTS = ShapeletTransformSimplicity(trainDataArr, trainLabelArr,
                                      minShapeletLength, maxShapeletLength,
                                      lengthIncrement, positionIncrement)

print("*************shapelet transform....  ***************")

bestKShapelet = callSTS.find_best_shapelets(numShapelet)
print("bestKShapelet: ", np.shape(bestKShapelet))

trainTransform = callSTS.transform(bestKShapelet, trainDataArr)
testTransform = callSTS.transform(bestKShapelet, testDataArr)
trainTransform = np.array(trainTransform)
testTransform = np.array(testTransform)
print("transform data: ", np.shape(trainTransform), np.shape(testTransform))


""" out put """
out_root = "/home/happyling/workspace/timeseries/output/"
out_dir = "shapelet/" + dataset + "/"
# out_file = str(minShapeletLength) + '_' + str(maxShapeletLength) + '_' + str(numShapelet)
# + '_' + str(lengthIncrement) + '_' + str(positionIncrement)

if not os.path.exists(out_root + out_dir):
    os.makedirs(out_root + out_dir)

out_file = out_root + out_dir + dataset

print("********* out put transform data set *************")
print("out data dir : {}".format(out_file))
out_data(out_file+"_TRAIN", trainTransform, trainLabelArr)
out_data(out_file+"_TEST", testTransform, testLabelArr)


