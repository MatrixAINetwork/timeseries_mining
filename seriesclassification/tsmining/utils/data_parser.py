import numpy as np
from . import Bunch


def load_ucr(filename, return_X_y=True):
    """
    parse UCR time series: 
    1. each data set have two file, *_TRAIN and *_TEST
    2. delimiter is ','
    3. first column is class label
    
    Parameters
    ----------
    :param filename: 
    :param return_X_y: boolean, default=True.
        If True, returns ``(data, target)`` instead of a Bunch object which is a dict object.
        
    :return: 
    """
    data_target = np.genfromtxt(filename, delimiter=',', dtype='float')
    data = data_target[:, 1::]
    target = data_target[:, 0]

    if return_X_y:
        return data, target
    else:
        return Bunch(data=data, target=target)  # data description can be added later


def load_list_data(filename, delimiter=',', return_X_y=True):
    """
    
    :param filename: 
    :param delimiter: 
    :param return_X_y: 
    :return: 
        data list, target array
    """
    fr = open(filename)
    data_list = []
    target_list = []
    for line in fr.readlines():
        str_list = line.strip().split(delimiter)
        vec = np.array(map(float, str_list))
        data_list.append(vec[1::])
        target_list.append(vec[0])

    if return_X_y:
        return data_list, np.array(target_list)
    else:
        return Bunch(data=data_list, target=np.array(target_list))


def cal_distribution(y):
    """
    calculate the distribution for the class list of y
    
    :param y: class label list or array 
    :return: dic{label : number of instance}
    """
    classes = np.unique(y)
    dic = {}
    for c in classes:
        dic[c] = np.sum(y == c)
    return dic


def seg_by_class(X, y):
    """
    
    :param X:  data , array-like, (n_samples, n_features)  
    :param y:  target value, (n_samples, )
    :return: dic{label : data segment array}
    """
    classes = np.unique(y)
    dic = {}
    for c in classes:
        dic[c] = X[y == c]
    return dic


def resample_data(X_train, y_train, X_test, y_test):
    X_all = np.vstack([X_train, X_test])
    y_all = np.hstack([y_train, y_test])
    distr_train = cal_distribution(y_train)

    X_train_new = []
    y_train_new = []
    X_test_new = []
    y_test_new = []

    # shuffle
    indexs = np.array(range(len(y_all)))
    np.random.shuffle(indexs)
    X_all = X_all[indexs]
    y_all = y_all[indexs]

    data_seg = seg_by_class(X_all, y_all)
    for key, seg in data_seg.items():
        num_train = distr_train[key]
        seg_X_train = seg[:num_train]
        seg_X_test = seg[num_train::]
        seg_y_train = [key]*len(seg_X_train)
        seg_y_test = [key]*len(seg_X_test)

        X_train_new.append(seg_X_train)
        y_train_new.append(seg_y_train)

        X_test_new.append(seg_X_test)
        y_test_new.append(seg_y_test)

    return np.vstack(X_train_new), np.hstack(y_train_new), np.vstack(X_test_new), np.hstack(y_test_new)


def k_fold_split_balance(X, y, k=10, shuffle=False):
    if shuffle:
        indexs = np.array(range(len(y)))
        np.random.shuffle(indexs)
        X = X[indexs]
        y = y[indexs]

    data_seg = seg_by_class(X, y)
    dic = {}
    n_fail = 0
    keys_fail = []
    for key, seg in data_seg.items():
        n_samples = len(seg)
        n_sub_samples = int(n_samples / k)

        if n_sub_samples == 0:
            n_fail += 1
            keys_fail.append(key)

        kk = k
        if n_sub_samples == 0:
            kk = k
            while n_sub_samples == 0:
                kk -= 1
                n_sub_samples = int(n_samples / kk)

        remainder = n_samples % kk
        if remainder == 0:
            dic[key] = np.vsplit(seg, kk)
        else:
            temp = seg[:-remainder]
            dic[key] = np.vsplit(temp, kk)
            # the remainder will be added to the last segment
            dic[key][-1] = np.vstack([dic[key][-1], seg[-remainder:]])

    print("="*80)
    print("print from function: k_fold_split_balance")
    if n_fail == 0:
        print("success to split data set in balance")
    else:
        print("some class label can not balance !!")
        print("there are %d classes and %d fail to split in balance"
              % (len(np.unique(y)), n_fail))
        print("the keys list as follow:  ")
        print(keys_fail)
    print("=" * 80)
    print("\n")

    return dic


def k_fold_validation_balance(X, y, k, k_val):
    assert k > k_val, \
        "(k=%d, k_val=%d), k should b larger than k_val " % \
        (k, k_val)
    dic = k_fold_split_balance(X, y, k=k)
    X_train = []
    y_train = []
    X_val = []
    y_val = []

    for key, seglist in dic.items():
        kk = len(seglist)
        if kk > k_val:
            temp_val = np.vstack(seglist[:k_val])
            temp_train = np.vstack(seglist[k_val:])
            X_val.append(temp_val)
            X_train.append(temp_train)
            y_val.append([key] * len(temp_val))
            y_train.append([key] * len(temp_train))
        else:
            temp = np.vstack(seglist)
            X_train.append(temp)
            y_train.append([key] * len(temp))

    X_train = np.vstack(X_train)
    X_val = np.vstack(X_val)
    y_train = np.hstack(y_train)
    y_val = np.hstack(y_val)
    return X_train, y_train, X_val, y_val


def z_normalize(data):
    norm_data = data.copy()
    mean = np.mean(norm_data, axis=0)
    variance = np.var(norm_data, axis=0)

    norm_data = norm_data - mean
    # The 1e-9 avoids dividing by zero
    norm_data = norm_data / (np.sqrt(variance) + 1e-9)

    return norm_data
