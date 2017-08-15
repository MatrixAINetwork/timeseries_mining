import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def base(data, whiten=False):
    pca = PCA(n_components=data.shape[1])
    pca.fit(data, whiten)
    data_transformed = pca.transform(data)
    return data_transformed, pca.explained_variance_ratio_, pca.explained_variance_


def transform(file_path, name_list, threshold=0.9, whiten=False, is_plot=False):
    title = file_path.split("/")[-1]
    print("************* processing {}".format(title))
    data = pd.read_csv(file_path)
    data_sub = data[name_list].copy()
    data_transformed, variance_ratio_, variance_ = base(data_sub, whiten=whiten)
    score = np.cumsum(variance_ratio_)
    for i in np.arange(0, len(score)):
        if score[i] > threshold:
            break
    print("variance_ratio_: ")
    print(variance_ratio_[:(i + 1)])
    print("variance:")
    print(variance_[:(i + 1)])

    if is_plot:
        plot(title, data_transformed[:, :(i + 1)])
    return np.array(data_transformed[:, :(i + 1)])


def plot(title, data):
    plt.figure(figsize=(13, 2))
    plt.title(title)
    for j in np.arange(0, data.shape[1]):
        plt.plot(data[:, j], label="pca{}".format(j))
    plt.legend(loc='best', prop={'size': 8})


