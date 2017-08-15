import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import preprocess
import static_variable


def pca_plot(data):
    pca = PCA(n_components=data.shape[1], whiten=False)
    pca.fit(data)
    print(pca.explained_variance_ratio_)
    print(pca.explained_variance_)
    pca_transfored = pca.fit_transform(data)
    plt.figure(figsize=(13, 3))
    for i in np.arange(0, pca_transfored.shape[1]):
        plt.plot(pca_transfored[:, i], label="pca{}".format(i))
    plt.legend(loc='best', prop={'size': 8})


def pca_transformed(data, *args):
    pca = PCA(n_components=data.shape[1], *args)
    pca.fit(data)
    data_transformed = pca.transform(data)
    return pca.explained_variance_ratio_, data_transformed


name_Others = ['BTSJ', 'ZD_CNT', 'ZD_LCG', 'ZD_TFG', 'ZD_JHG', 'ZD_LLJ', 'ZD_SPEED']
def pca_transformed_main():
    path_in = "data/interpolation_10s_int/"
    path_out = "data/pca1/"
    file_log = open(path_out+"log.txt", "w")
    filelist = preprocess.get_filelist(path_in)

    for file in filelist:
        print("------------- start to process {} ------------".format(file), file=file_log)
        data = pd.read_csv(path_in+file)
        data_new = pd.DataFrame()

        # other variables
        for name in name_Others:
            data_new[name] = data[name]

        # ZX_HW1 and ZX_HW2
        print("ZX_HW:", file=file_log)
        for i in [1, 2]:
            names = list()
            for j in np.arange(1, 7):
                name = "ZX_HW{0}_{1}".format(i, j)
                names.insert(len(names), name)
            ratio, value_transformed = pca_transformed(data[names])
            print(ratio, file=file_log)
            for k in np.arange(len(names)):
                data_new[names[k]] = value_transformed[:, k]

        # ZX_WD_No1: ZX_WD_1_*
        print("ZX_WD_No1: ", file=file_log)
        name_prefix_no1 = "No1_"
        for i in np.arange(1, 7):
            names = list()
            for j in np.arange(1, 7):
                name = "ZX_WD_{0}_{1}".format(i, j)
                names.insert(len(names), name)
            ratio, value_transformed = pca_transformed(data[names])
            print(ratio, file=file_log)
            for k in np.arange(len(names)):
                data_new[name_prefix_no1+names[k]] = value_transformed[:, k]

        #  ZX_WD_No2: ZX_WD_ * _1
        name_prefix_no2 = "No2_"
        print("ZX_WD_No2: ", file=file_log)
        for i in np.arange(1, 7):
            names = list()
            for j in np.arange(1, 7):
                name = "ZX_WD_{0}_{1}".format(j, i)
                names.insert(len(names), name)
            ratio, value_transformed = pca_transformed(data[names])
            print(ratio, file=file_log)
            for k in np.arange(len(names)):
                data_new[name_prefix_no2+names[k]] = value_transformed[:, k]

        data_new.to_csv(path_out+file, index=False)
        print("----------------  end ----------------------------", file=file_log)


def pca_dataset_plot(path):

    def pd_plot(data):
        ax = data_plot.plot(figsize=(13, 2))
        ax.legend(loc='best', prop={'size': 6})

    filelist = preprocess.get_filelist(path)
    for file in filelist:
        print("---------------{}----------".format(file))
        data = pd.read_csv(path + file)

        # other
        data_plot = data[static_variable.name_pca_other]
        axs = data_plot.plot(subplots=True, layout=(np.int(data_plot.shape[1] / 2), 2), figsize=(15, 8), style='o')
        # HW1
        data_plot = data[static_variable.name_HW1]
        pd_plot(data_plot)
        # HW2
        data_plot = data[static_variable.name_HW2]
        pd_plot(data_plot)
        # ZW
        data_plot = data[static_variable.name_pca_ZWNo1_1]
        pd_plot(data_plot)
        data_plot = data[static_variable.name_pca_ZWNo1_2]
        pd_plot(data_plot)
        data_plot = data[static_variable.name_pca_ZWNo1_3]
        pd_plot(data_plot)
        data_plot = data[static_variable.name_pca_ZWNo1_4]
        pd_plot(data_plot)
        data_plot = data[static_variable.name_pca_ZWNo1_5]
        pd_plot(data_plot)
        data_plot = data[static_variable.name_pca_ZWNo1_6]
        pd_plot(data_plot)

pca_transformed_main()
