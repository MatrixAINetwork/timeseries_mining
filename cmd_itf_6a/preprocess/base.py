import pandas as pd
import numpy as np
import os
import sys


def get_files_csv(path):
    file_list = list()
    for file in os.listdir(path):
        if file.endswith(".csv"):
            file_list.insert(len(file_list), file)
    return np.sort(file_list)


def summary_data(path_in, path_out):
    files = get_files_csv(path_in)
    log_file = open(path_out+"data_summary.txt", "w")
    stdout_original = sys.stdout
    sys.stdout = log_file
    for file in files:
        print("{0} summary of file: {1} {0}".format('-'*10, file))
        data = pd.read_csv(path_in+file)
        print(data.info())
        print(data.describe())
        print(data.head())
        print("{0}".format('-'*25))
    sys.stdout = stdout_original
    log_file.close()


def scale_data(path_in, path_out):
    files = get_files_csv(path_in)
    log_file = open(path_out+"scale_data.txt", "w")
    stdout_original = sys.stdout
    sys.stdout = log_file
    for file in files:
        data = pd.read_csv(path_in+file)
        print("{0}:\t{1}, {2}".format(file, len(data), len(data.columns)))
    sys.stdout = stdout_original
    log_file.close()


def clean_useless_null_constant(path_in, path_out):
    files = get_files_csv(path_in)
    stdout_original = sys.stdout
    log_file = open("data/log_clean_useless_null_constant.txt", "w")
    sys.stdout = log_file
    for file in files:
        print("-------------------------------------------------------")
        print("********process file: {}*********".format(file))
        data = pd.read_csv(path_in+file)
        names_useless = ['IDX', 'T_TYPE_ID', 'LOCO_NO', 'AB', 'RKSJ', 'XTSJ',
                         'XD_CARNO', 'XD_CENVER']
        data.drop(names_useless, inplace=True, axis=1)  # no use
        # del FH_TTZT, FH_TTLX
        data.drop(['FH_FLAG', 'FH_TTZT', 'FH_TTLX'], axis=1, inplace=True)  # not a constant in all file
        print("del useless variable:")
        print(names_useless)
        print(['FH_FLAG', 'FH_TTZT', 'FH_TTLX'])
        num_del = len(names_useless) + 3
        for name in data.columns:
            if data[name].notnull().sum() == 0:  # null variable
                print("del null variable {}".format(name))
                data.drop(name, inplace=True, axis=1)
                num_del += 1
                continue
            if (data[name] == data.loc[0, name]).all(skipna=True):
                print("del constant variable {0} and value is {1}".format(name, data.loc[0, name]))
                data.drop(name, inplace=True, axis=1)
                num_del += 1

        print("**** del {} variables in tall ****".format(num_del))
        print("-------------------------------------------------------\n\n")

        data.to_csv(path_out+file, index=False)

    sys.stdout = stdout_original
    log_file.close()


def clean_useless_null(path_in, path_out):
    files = get_files_csv(path_in)
    stdout_original = sys.stdout
    log_file = open("data/log_clean_useless_null.txt", "w")
    sys.stdout = log_file
    for file in files:
        print("-------------------------------------------------------")
        print("********process file: {}*********".format(file))
        data = pd.read_csv(path_in+file)
        names_useless = ['IDX', 'T_TYPE_ID', 'LOCO_NO', 'AB', 'RKSJ', 'XTSJ',
                         'XD_CARNO', 'XD_CENVER']
        data.drop(names_useless, inplace=True, axis=1)  # no use
        print("del no use variable:")
        print(names_useless)
        num_del = len(names_useless)
        for name in data.columns:
            if data[name].notnull().sum() == 0:  # null variable
                print("del null variable {}".format(name))
                data.drop(name, inplace=True, axis=1)
                num_del += 1
        print("**** del {} variables in tall ****".format(num_del))
        print("-------------------------------------------------------\n\n")

        data.to_csv(path_out+file, index=False)

    sys.stdout = stdout_original
    log_file.close()


def check_constant(path_in):
    filelist = get_files_csv(path_in)
    stdout_origin = sys.stdout
    f_log = open(path_in+"constant_info.txt", "w")
    sys.stdout = f_log
    print("check constant columns:")
    data = pd.read_csv(path_in+filelist[0])
    name_constant_set = set(data.columns)
    for file in filelist:
        name_set = set()
        print("********************** processing file: {}".format(file))
        data = pd.read_csv(path_in+file)
        for name in data.columns:
            if (data[name] == data.loc[0, name]).all():
                name_set.add(name)
        name_constant_set &= name_set

    name_constant_list = np.sort(list(name_constant_set))
    print("\n\n **************constant variable:")
    print(name_constant_list)
    print("**************************************\n\n")

    for file in filelist:
        print("---------- {}".format(file))
        data = pd.read_csv(path_in+file)
        #data.dropna(axis=0, inplace=True)
        for name in name_constant_list:
            print("{0}\t{1}".format(name, data.loc[0, name]))

        print("-----------\n")

    f_log.close()
    sys.stdout = stdout_origin

    return name_constant_list


def del_uselss_columns(path_in, path_out, names_del):
    filelist = get_files_csv(path_in)
    for file in filelist:
        print("processing file: {}".format(file))
        data = pd.read_csv(path_in + file)
        data.drop(names_del, axis=1, inplace=True)
        data.to_csv(path_out + file, index=False)





#summary_data("data/original_data/", "data/original_data/")
#clean("data/original_data/", "data/cleaned_data/")
#summary_data("data/cleaned_data/", "data/cleaned_data/")
#scale_data("data/cleaned_data/", "data/cleaned_data/")

# clean_useless_null_constant("data/original_data/", "data/cleaned_data2/")
# scale_data("data/cleaned_data2/", "data/cleaned_data2/")

# clean_useless_null("data/original_data/", "data/cleaned_data/")
# scale_data("data/cleaned_data/", "data/cleaned_data/")
