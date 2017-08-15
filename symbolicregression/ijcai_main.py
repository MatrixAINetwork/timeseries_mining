import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ffx_learn import call_ffx
from ffx_learn import learn_ffx


data_0134 = pd.read_csv("data_ijcai/interpolation_10s_int/0134.csv")
data_0134_pca = pd.read_csv("data_ijcai/pca1/0134.csv")

y = data_0134['ZX_WD_1_1']
X = data_0134[['ZX_WD_1_2', 'ZX_WD_1_3', 'ZX_WD_1_4', 'ZX_WD_1_5', 'ZX_WD_1_6']]
results = list()
result = call_ffx.run_ffx_main_half(X.as_matrix(), y.values, X.columns, verbose=False)
results.insert(len(results), result)
learn_ffx.result_summary(results, 'ZX_WD_1_1')
