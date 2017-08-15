from deap import tools
from deap import creator
from deap import base
import random
import operator
from .prepare_func_deap import *


def _evolve(toolbox, optimizer, seed, gen=0, mu=1, lambda_=1, cxpb=1, mutp=1):
    random.seed(seed)
    np.random.seed(seed)

    pop = remove_twins(toolbox.population(n=mu))
    pop = list(toolbox.map(optimizer, pop))

    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitness = list(toolbox.map(toolbox.evaluate, invalid_ind))
    for ind, fit in zip(invalid_ind, fitness):
        ind.fitness.values = fit

    pop = toolbox.select(pop, mu)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.nanmin, axis=0)
    stats.register("max", np.nanmax, axis=0)
    stats.register("diversity", lambda pop: len(set(map(str, pop))))

    logbook = tools.Logbook()
    logbook.header = "gen", 'evals', 'min', 'max', 'diversity'

    record = stats.compile(pop)
    logbook.record(gen=0, evals=(len(invalid_ind)), **record)
    print(logbook.stream)
    if record['min'][0] == 0.0:
        return pop, logbook

    for g in range(1, gen):
        offspring = tools.selRandom(pop, mu)
        offspring = [toolbox.clone(ind) for ind in offspring]

        # mate
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() <= cxpb:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        # mutate
        for mutant in offspring:
            if random.random() <= mutp:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        # re-evaluate
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitness = list(toolbox.map(toolbox.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, fitness):
            ind.fitness.values = fit

        # select
        pop = toolbox.select(remove_twins(pop + offspring), mu)
        record = stats.compile(pop)
        logbook.record(gen=g, evals=len(invalid_ind), **record)
        print(logbook.stream)
        if record['min'][0] < 1E-4:
            break

    return pop, logbook


def run_gp(train_X, trian_y, test_X, test_y, varnames, hyper, seed=43):

    hyper = {'gen': 92, 'mu': 500, 'cxpb': 0.5}
    seed = 43

    #strat = ['exp', 'symc', 'pot', 'trigo']
    strat =['pot']
    funcset = prepare_funcset(strat=strat, arity=train_X.shape[1])
    new_varnames = {'ARG%i' % i: var for i, var in enumerate(varnames)}
    funcset.renameArguments(**new_varnames)

    toolbox = base.Toolbox()
    toolbox.register("compile", gp.compile, pset=funcset)

    def _error(ind, compile, X, y):
        model = compile(expr=ind)
        yp = model(*X.T)
        er = np.sqrt(np.nanmean((y-yp)**2))
        return er

    def _evaluate(ind, compile, X, y):
        er = _error(ind, compile, X, y)
        com = len(ind)
        return er, com

    creator.create("Fitness", base.Fitness, weights=(-1.0, -1.0))
    creator.create("Individual", MyGPTree, fitness=creator.Fitness)

    toolbox.register("expr", gp.genHalfAndHalf, pset=funcset, min_=1, max_=4)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("select", tools.selNSGA2)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genHalfAndHalf, pset=funcset, min_=1, max_=4)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=funcset)
    toolbox.register("evaluate", _evaluate, compile=toolbox.compile, X=train_X, y=trian_y)

    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

    # constant optimization setting
    options = {'maxiter': 5}
    context = generate_context(funcset, train_X)
    constraints = ({'type': 'ineq', 'fun': lambda x: 10.0-np.abs(x)})
    cost = lambda args, yp: np.sum((trian_y - yp(*args))**2)
    optimize = lambda ind: optimize_constants(ind=ind, cost=cost, context=context,
                                              options=options, constraints=constraints)

    pop, log = _evolve(toolbox=toolbox, optimizer=optimize, seed=seed,  **hyper)
    pareto_front = tools.ParetoFront()
    pareto_front.update(pop)

    complexity = [ind.fitness.values[1] for ind in pareto_front]
    models_compiled = [toolbox.compile(m) for m in pareto_front]

    return complexity, pareto_front, models_compiled


def cal_nrmse_gp(model,X,y):
    return np.sqrt(np.nanmean((model(*X.T)-y)**2))/(max(y)-min(y))


def cal_rmse_gp(model,X,y):
    return np.sqrt(np.nanmean((model(*X.T)-y)**2))


def run_gp_main_tail(X, y, varnames, train_ratio, verbose=None):
    nlen = len(X)
    lag = np.round(train_ratio * nlen)
    train_X = X[1:(nlen-lag), :]
    test_X = X[(nlen-lag)::, :]
    train_y = y[1:(nlen-lag)]
    test_y = y[(nlen-lag)::]

    result = run_gp(train_X, train_y, test_X, test_y, varnames, verbose)
    complexity, pareto_front, models_compiled = result

    test_nrmse = [cal_nrmse_gp(model, test_X, test_y) for model in models_compiled]
    test_rmse = [cal_rmse_gp(model, test_X, test_y) for model in models_compiled]

    train_nrmse = [cal_nrmse_gp(model, train_X, train_y) for model in models_compiled]
    train_rmse = [cal_rmse_gp(model, train_X, train_y) for model in models_compiled]

    return complexity, [test_nrmse,train_nrmse], [test_rmse,train_rmse], pareto_front, models_compiled


def run_gp_main_half(X, y, varnames, verbose=None):
    train_X = X[::2, :]
    test_X = X[1::2, :]
    train_y = y[::2]
    test_y = y[1::2]

    result = run_gp(train_X, train_y, test_X, test_y, varnames, verbose)
    complexity, pareto_front, models_compiled = result

    test_nrmse = [cal_nrmse_gp(model, test_X, test_y) for model in models_compiled]
    test_rmse = [cal_rmse_gp(model, test_X, test_y) for model in models_compiled]

    train_nrmse = [cal_nrmse_gp(model, train_X, train_y) for model in models_compiled]
    train_rmse = [cal_rmse_gp(model, train_X, train_y) for model in models_compiled]

    return complexity, [test_nrmse, train_nrmse], [test_rmse, train_rmse], pareto_front, models_compiled


def run_gp_main_random(X, y, varnames, train_ratio, verbose=None):
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

    result = run_gp(train_X, train_y, test_X, test_y, varnames, verbose)
    complexity, pareto_front, models_compiled = result

    test_nrmse = [cal_nrmse_gp(model, test_X, test_y) for model in models_compiled]
    test_rmse = [cal_rmse_gp(model, test_X, test_y) for model in models_compiled]

    train_nrmse = [cal_nrmse_gp(model, train_X, train_y) for model in models_compiled]
    train_rmse = [cal_rmse_gp(model, train_X, train_y) for model in models_compiled]

    return complexity, [test_nrmse, train_nrmse], [test_rmse, train_rmse], pareto_front, models_compiled



