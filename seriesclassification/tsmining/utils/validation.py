import numpy as np


def cal_accuracy(pred_list, label_list):
    assert len(pred_list) == len(label_list), "predict list and the label list did not match"
    n = len(pred_list)
    return np.sum(pred_list == label_list) * 1.0 / np.float(n)
