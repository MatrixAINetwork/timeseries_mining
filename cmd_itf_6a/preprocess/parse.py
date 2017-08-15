import numpy as np
import pandas as pd

from preprocess import base


def hex2dec(hex_str):
    hex_list = list()
    for i in np.arange(len(hex_str)):
        c = hex_str[i]
        num = ord(c) - ord('0')
        if num > 9:
            num = 10 + ord(c) - ord('A')
        hex_list.insert(len(hex_list), num)

    sum = 0

    for i, num in enumerate(reversed(hex_list)):
        sum += num*np.power(16, i)
    return sum


def hex2dec_main(hex_str, i):
    if hex_str == 'nan':
        return np.nan
    seg = 2
    start = i*seg
    end = start + seg
    sub_str = hex_str[start:end]
    num = hex2dec(sub_str)
    return num


def parse_temperature(data):

    names = list()
    for i in np.arange(1, 7):
        name = "ZX_WD_{}".format(i)
        names.insert(len(names), name)
        temp_series = data[name]
        for j in np.arange(1, 7):
            new_series = temp_series.apply(lambda x: hex2dec_main(hex_str=str(x), i=(j - 1)))
            new_name = "ZX_WD_{0}_{1}".format(i, j)
            data[new_name] = new_series
    data.drop(names, axis=1, inplace=True)

    return data


def parse_temp_HW(data):
    for i in [1, 2]:
        for j in np.arange(1, 7):
            name = "ZX_HW{0}_{1}".format(i, j)
            data[name] = data[name].replace(0, np.nan)
    return data


def parse_temp_batch(path_in, path_out):
    filelist = base.get_files_csv(path_in)
    for file in filelist:
        print("processing file {}".format(file))
        data = pd.read_csv(path_in+file)
        if len(data) < 1000:  # discard the dataset with too less records
            print("!!!!!! without parsing file:{} because the number of records too small!".format(file))
            continue
        data = parse_temperature(data)
        data = parse_temp_HW(data)
        data.to_csv(path_out+file, index=False)


def getdays(month):
    day_sum = 0
    for i in np.arange(1, (month+1)):
        if i in [1, 3, 5, 7, 8, 10, 12]:
            day_sum += 31
        elif i in [4, 6, 9, 11]:
            day_sum += 30
        elif i == 2:
            day_sum += 29
    return day_sum


def parse_time(time_list):
    date, time = time_list[0].split(" ")
    year_1, month_1, day_1 = date.split("-")
    hour_1, minute_1, second_1 = time.split(":")
    year_1, month_1, day_1 = int(year_1), int(month_1), int(day_1)
    hour_1, minute_1, second_1 = int(hour_1), int(minute_1), int(second_1)
    days_1 = getdays(month_1-1) + day_1
    secs_1 = hour_1*60*60 + minute_1*60 + second_1
    timestamp = 0
    timestamp_list = list()
    timestamp_list.insert(len(timestamp_list), timestamp)
    for str_time in time_list[1::]:
        date, time = str_time.split(" ")
        year_2, month_2, day_2 = date.split("-")
        hour_2, minute_2, second_2 = time.split(":")
        year_2, month_2, day_2 = int(year_2), int(month_2), int(day_2)
        hour_2, minute_2, second_2 = int(hour_2), int(minute_2), int(second_2)
        days_2 = getdays(month_2-1) + day_2
        secs_2 = hour_2*60*60 + minute_2*60 + second_2

        sub_time = (days_2-days_1)*24*60*60 + (secs_2-secs_1)
        timestamp += sub_time
        timestamp_list.insert(len(timestamp_list), timestamp)

        days_1 = days_2
        secs_1 = secs_2

    return timestamp_list


def parse_time_batch(path_in, path_out):
    filelist = base.get_files_csv(path_in)
    for file in filelist:
        print("process file: {}".format(file))
        data = pd.read_csv(path_in+file)
        data['BTSJ_I'] = parse_time(list(data['BTSJ']))
        names = list(data.columns)
        names.remove('BTSJ_I')
        names.insert(1, 'BTSJ_I')
        data = data[names]
        data.to_csv(path_out+file, index=False)


def del_dup(data):
    def is_number(s):
        try:
            t = int(s)
            if t != 0:
                return 1
            else:
                return 0
        except ValueError:
            return 0

    def del_dup_(data):
        num_zeros_list = list()
        for k in np.arange(len(data)):
            num = 0
            series = data.iloc[k]
            for value in series.values:
                num += is_number(value)
            num_zeros_list.insert(len(num_zeros_list), num)
        return num_zeros_list.index(min(num_zeros_list))

    time_stamps = list(data['BTSJ'])
    time_stamps_unique = np.unique(time_stamps)
    data_new = pd.DataFrame(columns=data.columns)
    for i in time_stamps_unique:
        subset = data.loc[data['BTSJ'] == i, :]
        if len(subset) > 1:
            k = del_dup_(subset)
            data_new.loc[len(data_new)] = subset.iloc[k]
        else:
            data_new.loc[len(data_new)] = subset.iloc[0]

    if len(np.unique(data_new['BTSJ'])) != len(data_new['BTSJ']):
        print("!!!!!!!!!!!!!!!!!! there still exist duplicate records")

    return data_new


def del_dup_batch(path_in, path_out):
    filelist = base.get_files_csv(path_in)
    for file in filelist:
        print("processing file: {}".format(file))
        data = pd.read_csv(path_in+file)
        data_new = del_dup(data)
        data_new.to_csv(path_out+file, index=False)

