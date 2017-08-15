from common import *
from . import call_ffx
import matplotlib.pyplot as plt

def result_summary(results, varnames):
    for result, name in zip(results, varnames):
        numBases, nrmses, rmses, models = result
        nrmses_test, nrmses_train = nrmses
        rmses_test, rmses_train = rmses

        print("{0} {1} {0}".format("*"*20, name))
        for model in models:
            print(str(model))
        print("{0}".format("-"*25))
        print("complexity\t\tnrmse_test\t\tnrmse_train\t\trmse_test\t\trmse_train")
        for com, nrmse_test, nrmse_train, rmse_test, rmse_train in \
                zip(numBases, nrmses_test, nrmses_train, rmses_test, rmses_train):
            print("{0}\t\t{1}\t\t{2}\t\t{3}\t\t{4}".format(com, nrmse_test, nrmse_train, rmse_test, rmse_train))
        print("{0}".format("*"*40))

    plt.figure()
    for result, name in zip(results, varnames):
        numBases, nrmses, rmses, _ = result
        nrmses_test, _ = nrmses
        plt.plot(numBases, nrmses_test, label=name)
    plt.xlabel("Complexity")
    plt.ylabel("Error")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure()
    for result, name in zip(results, varnames):
        numBases, nrmses, rmses, _ = result
        rmses_test, _ = rmses
        plt.plot(numBases, rmses_test, label=name)
    plt.xlabel("Complexity")
    plt.ylabel("Error")
    plt.legend()
    plt.tight_layout()
    plt.show()


def compare_plot_nodelay(model_indexs, data_learn, results, varnames, var_num_learn):
    var_num = len(varnames)
    for i, result in zip(np.arange(0, var_num_learn), results):
        coms, nrmses, rmses, models = result
        cols = list(np.arange(0, var_num))
        del cols[i]
        X, y = get_data_nodelay(data_learn, i, cols)
        model = models[model_indexs[i]]
        print(str(model))
        print()
        yp = model.simulate(X)
        compare_plot(y, yp, varnames[i])


def compare_plot_delay(model_indexs, data_learn, seq_timedelay, results, varnames, var_num_learn):
    var_num = len(varnames)
    for i, result in zip(np.arange(0, var_num_learn), results):
        coms, nrmses, rmses, models = result
        X, y, names = get_data_delay(data=data_learn, seq_timedelay=seq_timedelay, y_i=i, varnames=varnames)
        model = models[model_indexs[i]]
        print(str(model))
        print()
        yp = model.simulate(X)
        compare_plot(y, yp, varnames[i])


def learn_nodelay(data_learn, var_num_learn, varnames):
    var_num = len(varnames)
    results = list()
    for i in np.arange(0, var_num_learn):
        print("-------------------------learn %s------------" %varnames[i])
        cols = list(np.arange(0, var_num))
        del cols[i]
        X, y = get_data_nodelay(data_learn, i, cols)
        result = call_ffx.run_ffx_main_half(X, y, varnames[cols])
        results.insert(i, result)
        print("--------------------end----------------------------")
    return results


def learn_delay(data, seq_timedelay, var_num_learn, varnames):
    results = list()
    for i in np.arange(0, var_num_learn):
        print("-------------------------learn %s------------" %varnames[i])
        X, y, names = get_data_delay(data, seq_timedelay, i, varnames)
        result = call_ffx.run_ffx_main_half(X, y, names)
        results.insert(i, result)
        print("--------------------end----------------------------")
    return results

