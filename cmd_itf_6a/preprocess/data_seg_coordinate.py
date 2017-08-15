import pandas as pd
import numpy as np
import sys
import os
from preprocess import base


# ---------------------------------------------check data segment flag

def _check_flag(df_flag):
    df_flag =df_flag[df_flag.notnull()]
    if (df_flag == 1).all():
        print("{} always on".format(df_flag.name))
    elif (df_flag == 0).all():
        print("{} always off".format(df_flag.name))
    else:
        percent = (df_flag == 1).sum() / len(df_flag)
        print("{0} sometimes on, percentage:{1}".format(df_flag.name, percent))

def check_seg_flag(file):
    data = pd.read_csv(file)
    names_flag = ['XD_FLAG', 'ZD_FLAG', 'JY_FLAG', 'SP_FLAG', 'ZX_FLAG', 'FH_FLAG']
    for name in names_flag:
        flag_series = data[name]
        _check_flag(flag_series)


def check_seg_flag_batch(path, log_dir):
    filelist = base.get_files_csv(path)
    stdout_origin = sys.stdout
    f_log = open(log_dir, "w")
    sys.stdout = f_log

    for file in filelist:
        print("**************** file:{} ***************".format(file))
        check_seg_flag(path+file)
        print("****************************************\t")

    sys.stdout = stdout_origin
    f_log.close()


# ------------------------------------- segment coordinate

def seg_flag_cor(data, names):
    for name in names:
        data = data.loc[data[name] == 1, :]
    return data


def seg_flag_cor_batch(path_in, path_out, names):
    filelist = base.get_files_csv(path_in)
    for file in filelist:
        print("processing file: {}".format(file))
        data = pd.read_csv(path_in+file)
        data = seg_flag_cor(data, names)
        data.to_csv(path_out+file, index=False)




