from common import *
from .call_deap import *
import matplotlib.pyplot as plt


def result_summary(results, varnames):
    for result, name in zip(results, varnames):
        numBases, nrmses, rmses, models, _ = result
        nrmses_test, nrmses_train = nrmses
        rmses_test, rmses_train = rmses

        print("{0} {1} {0}".format("*"*20, name))
        print("original model:")
        for model in models:
            print(model)
        print("symplified model:")
        for model in models:
            print(simplify_this(model))

        print("{0}".format("-"*25))
        print("complexity\t\tnrmse_test\t\tnrmse_train\t\trmse_test\t\trmse_train")
        for com, nrmse_test, nrmse_train, rmse_test, rmse_train in \
                zip(numBases, nrmses_test, nrmses_train, rmses_test, rmses_train):
            print("{0}\t\t{1}\t\t{2}\t\t{3}\t\t{4}".format(com, nrmse_test, nrmse_train, rmse_test, rmse_train))
        print("{0}".format("*"*40))

    plt.figure()
    for result, name in zip(results, varnames):
        coms, nrmses, rmses, _, _ = result
        nrmses_test, _ = nrmses
        plt.plot(coms, nrmses_test, label=name)
    plt.xlabel("Complexity")
    plt.ylabel("Error")
    plt.legend()
    plt.tight_layout()

    plt.figure()
    for result, name in zip(results, varnames):
        coms, nrmses, rmses, _, _ = result
        rmses_test, _ = rmses
        plt.plot(coms, rmses_test, label=name)
    plt.xlabel("Complexity")
    plt.ylabel("Error")
    plt.legend()
    plt.tight_layout()


def compare_plot_nodelay(model_indexs, data, results, varnames, var_num_learn):
    var_num = len(varnames)
    for i, result in zip(np.arange(0, var_num_learn), results):
        coms, nrmses, rmses, models, models_bin = result
        cols = list(np.arange(0, var_num))
        del cols[i]
        X, y = get_data_nodelay(data, i, cols)
        model = models_bin[model_indexs[i]]
        yp = model(*X.T)
        print(simplify_this(models[model_indexs[i]]))
        print()
        compare_plot(y, yp, varnames[i])


def compare_plot_delay_deap(model_indexs, data, seq_timedelay, results, varnames, var_num_learn):
    for i, result, mindex in zip(np.arange(0, var_num_learn), results, model_indexs):
        coms, nrmses, rmses, models, models_bin = result
        X, y, names = get_data_delay(data=data, seq_timedelay=seq_timedelay, y_i=i, varnames=varnames)
        model = models_bin[model_indexs[i]]
        yp = model(*X.T)
        print(simplify_this(models[model_indexs[i]]))
        print()
        compare_plot(y, yp, varnames[i])


def learn_nodelay(data, var_num_learn, varnames):
    results = list()
    var_num = len(varnames)
    for i in np.arange(0, var_num_learn):
        print("-------------------------learn %s------------"%varnames[i])
        cols = list(np.arange(0, var_num))
        del cols[i]
        X, y = get_data_nodelay(data, i, cols)
        result = run_gp_main_half(X, y, varnames[cols])
        results.insert(i, result)
        print("--------------------end----------------------------")
    return results


def learn_delay(data_learn, seq_timedelay, var_num_learn, varnames):
    results = list()
    for i in np.arange(0, var_num_learn):
        print("-------------------------learn %s------------" %varnames[i])
        X, y, names = get_data_delay(data_learn, seq_timedelay, i, varnames)
        result = run_gp_main_half(X, y, names)
        results.insert(i, result)
        print("--------------------end----------------------------")
    return results
