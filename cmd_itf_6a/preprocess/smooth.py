import numpy as np
import pandas as pd
from preprocess import  base


def cut_sencond(date_time):
    date, time = date_time.split(" ")
    h, m, s = time.split(":")
    return date+" "+h+':'+m+':'+'00'


def sm_mean_1m(data):
    data['BTSJ'] = data['BTSJ'].apply(cut_sencond)
    data_sm = data.groupby('BTSJ').mean()
    data_sm['BTSJ'] = data_sm.index
    names = list(data_sm.columns)
    names.remove("BTSJ")
    names.insert(0, 'BTSJ')
    data_sm = data_sm[names]

    print("original length : {}".format(len(data)))
    print("new dataset length: {}".format(len(data_sm)))
    #print(data_sm.head())
    return data_sm


def sm_mean_1m_batch(path_in, path_out):
    filelist = base.get_files_csv(path_in)
    for file in filelist:
        print("-------- processing file: {}".format(file))
        data = pd.read_csv(path_in+file)
        data = sm_mean_1m(data)
        data.to_csv(path_out+file, index=False)


def interpolate_bin_mean(data, bin_size=5):
    names = list(data.columns)
    names.remove('BTSJ')
    for name in names:
        indexs = list(data[name].index[data[name].isnull()])
        for i in indexs:
            start = i-bin_size
            if start < 0:
                start = 0
            end = i + bin_size
            if end > len(data):
                end = len(data)
            data.loc[i, name] = data.loc[start:end, name].mean()
    return data


def interpolate_bin_mean_batch(path_in, path_out, bin_size=5):
    filelist = base.get_files_csv(path_in)
    for file in filelist:
        print("processing file: {}".format(file))
        data = pd.read_csv(path_in+file)
        data = interpolate_bin_mean(data, bin_size)
        data.to_csv(path_out+file, index=False)
