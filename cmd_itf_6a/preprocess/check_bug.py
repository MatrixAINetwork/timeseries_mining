from preprocess import base
import numpy as np
import pandas as pd
import sys


def check_bug(file):
    data = pd.read_csv(file)
    # ZD
    # ZD_CNT, 255 invalid
    zd_cnt = data[['BTSJ', 'ZD_CNT']]
    zd_cnt = zd_cnt[zd_cnt['ZD_CNT'].dropna(axis=0) != 255]
    print("************* ZD_CNT valid interval:")
    print(zd_cnt)

    # ZD_LCG, pip pressure, -1 breakdown
    zd_lcg = data[['BTSJ', 'ZD_LCG']]
    zd_lcg = zd_lcg[zd_lcg['ZD_LCG'].dropna(axis=0) == -1]
    print("************* ZD_LCG breakdown interval")
    print(zd_lcg)

    # ZD_TFG, stopping cylinder pressure, -1 breakdown
    zd_tfg = data[['BTSJ', 'ZD_TFG']]
    zd_tfg = zd_tfg[zd_tfg['ZD_TFG'].dropna(axis=0) == -1]
    print("************* ZD_TFG breakdown interval")
    print(zd_tfg)

    # ZD_JHG, average cylinder pressure, -1 breakdown
    zd_jhg = data[['BTSJ', 'ZD_JHG']]
    zd_jhg = zd_jhg[zd_jhg['ZD_JHG'].dropna(axis=0) == -1]
    print("************* ZD_JHG breakdown interval")
    print(zd_jhg)

    # ZD_LLJ, pip flow, -1 breakdown
    zd_llj = data[['BTSJ', 'ZD_LLJ']]
    zd_llj = zd_llj[zd_llj['ZD_LLJ'].dropna(axis=0) == -1]
    print("************* ZD_LLJ breakdown interval")
    print(zd_llj)

    # ZX
    # ZX_BJ_1, bearing temperature, NOT 0 alert
    # ZX_BJ_2, bearing temperature, NOT 0 alert
    # ZX_BJ_3, bearing temperature, NOT 0 alert
    # ZX_BJ_4, bearing temperature, NOT 0 alert
    # ZX_BJ_5, bearing temperature, NOT 0 alert
    # ZX_BJ_6, bearing temperature, NOT 0 alert
    for i in np.arange(1, 7):
        names = list(['BTSJ'])
        name = "ZX_BJ_{}".format(i)
        names.insert(len(names), name)
        zx_ = data[names]
        zx_ = zx_.dropna(axis=0)
        zx_ = zx_[zx_[name] != 0]
        print("************** {} breakdown interval:".format(name))
        print(zx_)


def check_bug_batch(path_in):
    filelist = base.get_files_csv(path_in)
    f_log = open(path_in+"check_bug_log.txt", "w")
    stdout_origin = sys.stdout
    sys.stdout = f_log

    for file in filelist:
        print("--------------- file: {}".format(file))
        check_bug(path_in+file)
        print("---------------\n\n")

    f_log.close()
    sys.stdout = stdout_origin
