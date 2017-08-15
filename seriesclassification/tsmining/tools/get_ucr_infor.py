########################################################################
# this script can be used to obtain the information of data set
# and save as markdown file.
########################################################################

from os import listdir
from os.path import isfile, join
import numpy as np

# Note : data set directory should be updated when using
DATA_ROOT = "../../data/UCR_TS_Archive_2015/"
OUT_DIR = "../../data/UCR_TS_Archive_2015_info.md"


def getDataInfo(filename):
    originDataArr = np.genfromtxt(filename, delimiter=',', dtype='float')
    instanceArr = originDataArr[:, 1::]
    classLabelArr = originDataArr[:, 0]

    size, length = instanceArr.shape
    numClass = np.unique(classLabelArr)

    return size, length, numClass


def main():
    fout = open(OUT_DIR, 'w')
    fout.write("Data Set | total data size | Data Length | num of class |Training size | Testing size \n")
    fout.write("---|---|---|---|---|---\n")
    file_dirs =[f for f in listdir(DATA_ROOT) if isfile(f) is False]
    for fdir in file_dirs:
        fileList = listdir(join(DATA_ROOT, fdir))
        infoDic = {}
        for file in fileList:
            tail = file.split('_')[-1]
            if tail == 'TEST':
                filename = join(DATA_ROOT, fdir, file)
                infoDic['test'] = getDataInfo(filename)
            elif tail == 'TRAIN':
                filename = join(DATA_ROOT, fdir, file)
                infoDic['train'] = getDataInfo(filename)

        strOut = "%s | %d | %d | %d | %d | %d \n" \
                 % (fdir, infoDic['test'][0] + infoDic['train'][0], infoDic['train'][1], len(infoDic['train'][2]),
                    infoDic['train'][0], infoDic['test'][0])
        fout.write(strOut)

if __name__ == '__main__':
    main()



