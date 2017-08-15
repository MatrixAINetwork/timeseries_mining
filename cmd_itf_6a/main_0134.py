from preprocess.parse import *
from preprocess.base import *
from preprocess.data_seg_coordinate import *
from preprocess import check_bug
import numpy as np
from preprocess import smooth

root_dir = os.getcwd()
data_dir = root_dir + "/data_0134/"


# -------- clean
# clean_useless_null(data_dir+"original_data/", data_dir+"cleaned_data/")
# scale_data(data_dir+"cleaned_data/", data_dir+"cleaned_data/")

# ------ parse temperature
# parse_temp_batch(data_dir+"cleaned_data/", data_dir+"pharsed_data/")
# scale_data(data_dir+"pharsed_data/", data_dir+"pharsed_data/")

# ------- check segment flag
# check_seg_flag_batch(data_dir+"pharsed_data/",
#                      data_dir+"log_check_seg_flag.txt")

# ------------------------- coordinate data set based on each segment flag

# print("******* processing ZD_FLAG ****")
# dir_zd_flag = "coordinated_zd_flag/"
# seg_flag_cor_batch(data_dir + "pharsed_data/",
#                    data_dir + dir_zd_flag,
#                    ['ZD_FLAG'])
# check_seg_flag_batch(data_dir + dir_zd_flag,
#                      data_dir + dir_zd_flag + "log_check_seg_flag.txt")
# scale_data(data_dir + dir_zd_flag,
#            data_dir + dir_zd_flag)
#
# print("******* processing SP_FLAG ****")
# dir_sp_flag = "coordinated_sp_flag/"
# seg_flag_cor_batch(data_dir + "pharsed_data/",
#                    data_dir + dir_sp_flag,
#                    ['SP_FLAG'])
# check_seg_flag_batch(data_dir + dir_sp_flag,
#                      data_dir + dir_sp_flag + "log_check_seg_flag.txt")
# scale_data(data_dir + dir_sp_flag,
#            data_dir + dir_sp_flag)
#
#
# print("******* processing ZX_FLAG ****")
# dir_zx_flag = "coordinated_zx_flag/"
# seg_flag_cor_batch(data_dir + "pharsed_data/",
#                    data_dir + dir_zx_flag,
#                    ['ZX_FLAG'])
# check_seg_flag_batch(data_dir + dir_zx_flag,
#                      data_dir + dir_zx_flag + "log_check_seg_flag.txt")
# scale_data(data_dir + dir_zx_flag,
#            data_dir + dir_zx_flag)
#
# print("******* processing FH_FLAG ****")
# dir_fh_flag = "coordinated_fh_flag/"
# seg_flag_cor_batch(data_dir+"pharsed_data/",
#                    data_dir + dir_fh_flag,
#                    ['FH_FLAG'])
# check_seg_flag_batch(data_dir + dir_fh_flag,
#                      data_dir + dir_fh_flag + "log_check_seg_flag.txt")
# scale_data(data_dir + dir_fh_flag,
#            data_dir + dir_fh_flag)
#
#
# print("******* processing ZD+ZX_FLAG ****")
# dir_zd_zx_flag = "coordinated_zd_zx_flag/"
# seg_flag_cor_batch(data_dir+"pharsed_data/",
#                    data_dir + dir_zd_zx_flag,
#                    ['ZD_FLAG', 'ZX_FLAG'])
# check_seg_flag_batch(data_dir + dir_zd_zx_flag,
#                      data_dir + dir_zd_zx_flag + "log_check_seg_flag.txt")
# scale_data(data_dir + dir_zd_zx_flag,
#            data_dir + dir_zd_zx_flag)
#
#
# print("******* processing ZD+ZX+SP_FLAG ****")
# dir_zd_zx_sp_flag = "coordinated_zd_zx_sp_flag/"
# seg_flag_cor_batch(data_dir+"pharsed_data/",
#                    data_dir + dir_zd_zx_sp_flag,
#                    ['ZD_FLAG', 'ZX_FLAG', 'SP_FLAG'])
# check_seg_flag_batch(data_dir + dir_zd_zx_sp_flag,
#                      data_dir + dir_zd_zx_sp_flag + "log_check_seg_flag.txt")
# scale_data(data_dir + dir_zd_zx_sp_flag,
#            data_dir + dir_zd_zx_sp_flag)


# ------------------------ get final dataset
# names_useless = ['XD_FLAG',
#                  'JY_FLAG', 'JY_JCDY', 'JY_MXDY', 'JY_BJZT', 'JY_CSZT', 'JY_ZJGZ',
#                  'FH_FLAG', 'FH_ZXKL', 'FH_ZXDL', 'FH_TTZT', 'FH_TTLX']
# del_uselss_columns(data_dir + "coordinated_zd_zx_sp_flag/",
#                    data_dir + "work_data_constant/",
#                    names_useless)
#
# names_constant = check_constant(data_dir + "work_data_constant/")
# del_uselss_columns(data_dir + "work_data_constant/",
#                    data_dir + "work_data/",
#                    names_constant)
#
# scale_data(data_dir + "work_data/",
#            data_dir + "work_data/")
# parse_time_batch(data_dir + "work_data/",
#                  data_dir + "work_data/")
#
# del_dup_batch(data_dir + "work_data/",
#               data_dir + "work_data_unique/")


# ------------------------- check breakdown

# check_bug.check_bug_batch(data_dir + "work_data/")


# ------------------------ parse time test

# data = pd.read_csv(root_dir + "/data/work_data/" + "01_233_0143_CC-66666_2016-05-20.csv")
# timestamp_list = parse_time(list(data['BTSJ']))
# data['BTSJ_I'] = timestamp_list
# names = list(data.columns)
# names.remove('BTSJ_I')
# names.insert(1, 'BTSJ_I')
# data = data[names]
# # d = {'old_time': list(data['BTSJ']),
# #      'new_time': timestamp_list}
# # df_time = pd.DataFrame(data=d)
# # df_time.to_csv(root_dir+"/data/test.csv")
# data.to_csv(root_dir+"/data/test.csv", index=False)


# ------------------ calculate mean in 1 minute

smooth.sm_mean_1m_batch(data_dir + "work_data_unique/",
                        data_dir + "smooth_mean/")

smooth.interpolate_bin_mean_batch(data_dir + "smooth_mean/",
                                  data_dir + "smooth_mean_interpolate_bin_mean/")
