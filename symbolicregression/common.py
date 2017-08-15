import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def cal_nrmse(model, X, y):
    return np.sqrt(np.nanmean((model.simulate(X)-y)**2 / (max(y) - min(y))))


def cal_rmse(model, X, y):
    return np.sqrt(np.nanmean((model.simulate(X)-y)**2))


def get_data_nodelay(data, y_i, cols):
    X = data[:, cols]
    y = data[:, y_i]

    return X, y


def get_data_delay(data, seq_timedelay, y_i, varnames):
    print("---------------prepare time delay data-------------")
    new_data = pd.DataFrame()
    tail = 10
    start = max(seq_timedelay) + 10
    varnum = len(varnames)
    cols = list(np.arange(varnum))
    y_name = "{0}_dt{1}".format(varnames[y_i], 0) # name can not contain character '.'
    for i_var in cols:
        for dt in seq_timedelay:
            name = "{0}_dt{1}".format(varnames[i_var], dt)
            new_data[name] = data[(start-dt):(-dt-tail), i_var]

    y_value = new_data[y_name]
    del new_data[y_name]

    names = list(new_data.columns)
    X = new_data.as_matrix()
    y = y_value.as_matrix()

    print("del the target variable %s" % (y_name))
    print("number of variable: %d" % (len(names)))
    print("the shape of X")
    print(np.shape(X))
    print("the shape of y")
    print(np.shape(y))

    return X, y, names


def compare_plot(y, yp, name):
    f, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 5))

    ax1.set_title('The result of {}'.format(name))
    ax1.plot(y)
    ax1.set_ylim(min(y) - np.std(y), max(y) + np.std(y))
    ax1.set_ylabel('Original value')
    ax2.plot(yp)
    ax2.set_ylim(min(yp) - np.std(yp), max(yp + np.std(yp)))
    ax2.set_ylabel('Predict Value')
    errors = yp - y
    ax3.plot(errors)
    ax3.set_ylabel('errors')
    ax3.set_ylim(min(errors) - np.std(errors), max(errors) + np.std(errors))
    plt.show()