import shelve
import numpy as np

from matplotlib import pyplot as plt

import synch_funcs

fs = 100
sigma = .4
gauss_step = 5
max_lag = 10
n = 1000

data = shelve.open('data')
all_correlations = {}
for cricket_1 in range(6):
    all_correlations['{0}. trace'.format(cricket_1)] = {}
    for cricket_2 in range(cricket_1, 6):
        all_correlations['{0}. trace'.format(cricket_1)]['{0}. trace'.format(cricket_2)] = {}
        times_1 = data['{0}. trace'.format(cricket_1)]['CallStart']
        times_2 = data['{0}. trace'.format(cricket_2)]['CallStart']
        lags, correlation = synch_funcs.calculate_cross_correlation(times_1, times_2, fs, sigma, gauss_step, max_lag)
        bootstrap_correlations = synch_funcs.bootstrap_cross_correlation(times_1, times_2, n, fs, sigma, gauss_step, max_lag)
        all_correlations['{0}. trace'.format(cricket_1)]['{0}. trace'.format(cricket_2)]['correlation'] = correlation
        all_correlations['{0}. trace'.format(cricket_1)]['{0}. trace'.format(cricket_2)]['bootstrapping'] = bootstrap_correlations
all_correlations['lags'] = lags
data.close()

myShelf = shelve.open('myShelf')
myShelf['my dictionary'] = all_correlations
myShelf.close()