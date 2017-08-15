import pandas as pd
import numpy as np
import os


def get_filelist(path):
    filelist = list()
    for file in os.listdir(path):
        if file.endswith(".csv"):
            filelist.insert(len(filelist), file)
    return filelist


def converDateTime(str_date):
    str_time = str_date.split(' ')[1]
    str_time = str_time.split(':')
    return str_date.split(' ')[0] + ' ' + str_time[0] + ':' + str_time[1] + ':00'


def removeConstantValue(path):
    filelist = get_filelist(path)
    for file in filelist:
        print("process file: {}".format(file))
        data = pd.read_csv(path+file)
        ## the 7 and 8 bearing
        for i in np.arange(1, 7):
            for j in [7, 8]:
                name = "ZX_WD_{0}_{1}".format(i, j)
                data.drop(name, axis=1, inplace=True)
        data.to_csv(path+file)




def calMeanBy1m():
    filelist = get_filelist("data/original")
    print(filelist)

    for file in filelist:
        print("********** start to process file:{} ************".format(file))
        path_in = "data/original/" + file
        path_out = "data/mean_1m/" + file
        path_out_int ="data/mean_1m_int/" + file

        data = pd.read_csv(path_in)
        data['BTSJ'] = data['BTSJ'].apply(converDateTime)
        new_data = data.groupby('BTSJ').mean()

        new_data.to_csv(path_out)
        new_data_int = new_data.apply(round)
        new_data_int = new_data_int.astype(int)
        new_data_int.to_csv(path_out_int)


def roundInterpolation():
    filelist = get_filelist("data/interpolation_10s/")
    print(filelist)

    for file in filelist:
        print("********** start to process file:{} ************".format(file))
        path_in = "data/interpolation_10s/" + file
        path_out = "data/interpolation_10s_int/" + file

        data = pd.read_csv(path_in)
        names = list(data.columns)
        names.remove('BTSJ')
        data[names] = (data[names].apply(round)).astype(int)
        data.to_csv(path_out, index=False)


def dataSummary(path_in, path_out):
    filelist = get_filelist(path_in)
    file_log = open(path_out+"data_summary.txt", "w")
    for file in filelist:
        data = pd.read_csv(path_in+file)
        print("--------------- summary of file: {} -------------".format(file), file=file_log)
        print(data.describe(), file=file_log)
        print(data.info(), file=file_log)
        print("---------------------------------------------------\n\n", file=file_log)



dataSummary("data/data_CMD_ITF_6/", "data/data_CMD_ITF_6/")


#roundInterpolation()

#removeConstantValue("data/interpolation_10s/")
