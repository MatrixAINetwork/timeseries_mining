import random
from common import *
from . import core


def run_ffx(train_X, train_y, test_X, test_y, varnames, verbose=None):
    #models = ffx.run(train_X=train_X, train_y=train_y, test_X=test_X, test_y=test_y, varnames=varnames, verbose=verbose)
    models = core.MultiFFXModelFactory().build(train_X, train_y, test_X, test_y, varnames, verbose)
    numBase = [model.numBases() for model in models]
    test_nrmse = [cal_nrmse(model, test_X, test_y) for model in models]
    train_nrmse = [cal_nrmse(model, train_X, train_y) for model in models]
    test_rmse = [cal_rmse(model, test_X, test_y) for model in models]
    train_rmse = [cal_rmse(model, train_X, train_y) for model in models]

    return numBase, [test_nrmse, train_nrmse], [test_rmse, train_rmse], models


def run_ffx_main_tail(X, y, varnames, train_ratio, verbose=None):
    nlen = len(X)
    lag = np.round(train_ratio * nlen)
    train_X = X[1:(nlen-lag), :]
    test_X = X[(nlen-lag)::, :]
    train_y = y[1:(nlen-lag)]
    test_y = y[(nlen-lag)::]

    result = run_ffx(train_X, train_y, test_X, test_y, varnames, verbose)
    return result


def run_ffx_main_half(X, y, varnames, verbose=None):
    train_X = X[::2, :]
    test_X = X[1::2, :]
    train_y = y[::2]
    test_y = y[1::2]

    result = run_ffx(train_X, train_y, test_X, test_y, varnames, verbose)
    return result


def run_ffx_main_random(X, y, varnames, train_ratio, verbose=None):
    nlen = len(X)
    num_train = np.round(train_ratio * nlen)
    indexs_all = range(nlen)
    indexs_train = random.sample(indexs_all, num_train)
    indexs_train = np.sort(indexs_train)
    indexs_test = np.setdiff1d(indexs_all, indexs_train)
    train_X = X[indexs_train, :]
    test_X = X[indexs_test, :]
    train_y = y[indexs_train]
    test_y = y[indexs_test]

    result = run_ffx(train_X, train_y, test_X, test_y, varnames, verbose)
    return result







